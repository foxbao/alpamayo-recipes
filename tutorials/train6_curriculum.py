"""train6: 三段课程与训练技巧

train4/5 讲的是「一步训练里发生什么」。这一课讲**训练是怎么编排的** ——
内容全部来自 `recipes/alpamayo1_5_sft/configs/` 里那几个 yaml。

三段课程（每段一个 config）：

    stage1_lingoqa  数据: LingoQA（VQA 问答）
                    components_order: [image, question, answer]
                    label_components: [answer]
                    → 先建立「看图说话」的视觉-语言对齐

    stage1_nav      数据: PAI nav + 导航标注
                    components_order: [image, traj_history, route, prompt, traj_future]
                    label_components: [traj_future]
                    技巧: lr_multiplier {vlm.model.visual: 0.1}
                    → 学驾驶推理（CoC）+ 未来轨迹的离散 token

    stage2_nav      模型: TrainableAlpamayoR1（从 Alpamayo-1.5-10B 起）
                    注入: stage1_vlm_checkpoint_path ← stage1 训出的 VLM
                    cotrain_vlm: false
                    技巧: deepspeed: null
                    → 冻结 VLM，训 action expert（train5 讲的就是这步的内部）

★ 本课能实测的三件事：
    ① 同一套权重，两个阶段吃的是【不同形态】的数据（打印两条序列对比）
    ② lr_multiplier 的机制 —— 最长前缀匹配 + 实际 lr（照抄 `trainer.py:58`）
    ③ checkpoint 交接 —— stage1 训出的 VLM 怎么进 stage2

⚠️ 本课**演示不了**「为什么先 LingoQA 再 nav 更好」。
   课程学习的好处来自【数据规模和分布】，玩具只有 3 条样本、怎么排都能过拟合。
   这里只讲清课程【是什么、怎么实现】，不讲【效果有多大】。

对应真实代码：
    recipes/alpamayo1_5_sft/configs/sft_stage1_lingoqa.yaml
    recipes/alpamayo1_5_sft/configs/sft_stage1_nav.yaml
    recipes/alpamayo1_5_sft/configs/sft_stage2_nav.yaml
    recipes/alpamayo1_5_sft/configs/models/ar1_5_{base,expert}.yaml
    recipes/alpamayo1_5_sft/trainer.py:58  create_optimizer（lr_multiplier）

【简化】纯 CPU、秒级。
"""

import torch
import torch.nn as nn

from train5_sft_stage2 import (
    Stage2Model, TinyVLM, make_sample, train_vlm, V, VOCAB, DATA,
)


# ═══════════════════════════════════════════════════════════════
# 模拟真实 config 里的三份「数据形态」
# ═══════════════════════════════════════════════════════════════

# 来自 configs/vla_processor/{vqa,nav}.yaml
VQA_SPEC = {
    "components_order": ["image", "question", "answer"],
    "label_components": ["answer"],
}
NAV_SPEC = {
    "components_order": ["image", "traj_history", "route", "prompt", "traj_future"],
    "label_components": ["traj_future"],
}

# 来自 configs/sft_stage1_nav.yaml
LR_MULTIPLIER = {"vlm.model.visual": 0.1}
BASE_LR = 1e-5


# ═══════════════════════════════════════════════════════════════
# ② lr_multiplier：把 trainer.py:58 的逻辑照抄一份
# ═══════════════════════════════════════════════════════════════

class ToyStage2Params(nn.Module):
    """参数命名模仿真实模型，这样前缀匹配能对上。

    真实名字形如 `vlm.model.visual.blocks.0.attn.qkv.weight`、
    `vlm.model.language_model.layers.0...`、`expert.layers.0...`。
    """

    def __init__(self):
        super().__init__()
        self.vlm = nn.Module()
        self.vlm.model = nn.Module()
        self.vlm.model.visual = nn.Linear(4, 4)          # 视觉塔
        self.vlm.model.language_model = nn.Linear(4, 4)  # 语言塔
        self.expert = nn.Linear(4, 4)                    # action expert


def build_param_groups(model, lr_multiplier, base_lr):
    """照着 `trainer.py:58 create_optimizer` 的逻辑走一遍（去掉了 decay 分组）：

    对每个参数的【全名】找前缀匹配，取**最长的那个**；没匹配上的用 1.0。
    每组的 lr = base_lr × multiplier。
    """
    groups = {}          # prefix -> [(name, param)]
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        matched = None
        for prefix in lr_multiplier:
            if name.startswith(prefix):
                # ★ 最长前缀优先：否则 {vlm:..., vlm.model.visual:...} 会选错
                if matched is None or len(prefix) > len(matched):
                    matched = prefix
        groups.setdefault(matched or "default", []).append((name, param))

    out = []
    for key, params in groups.items():
        mult = 1.0 if key == "default" else lr_multiplier[key]
        out.append({"key": key, "mult": mult, "lr": base_lr * mult,
                    "params": [p for _, p in params],
                    "names": [n for n, _ in params]})
    return out


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 72)
    print("① 三段课程，和它们各自吃的「数据形态」")
    print("=" * 72)
    print("  真实里每个阶段的 components_order / label_components 都不一样：\n")
    for name, spec in [("stage1_lingoqa (VQA)", VQA_SPEC),
                       ("stage1_nav / stage2_nav", NAV_SPEC)]:
        print(f"  {name}")
        print(f"      components_order = {spec['components_order']}")
        print(f"      label_components = {spec['label_components']}   ← 只有这段算 loss")
    print("\n  → 同一套权重，两个阶段喂进去的序列【结构不同】：")
    print("     VQA 是「图 + 问题 + 答案」，nav 是「图 + 历史 + 导航 + 提示 + 未来轨迹」。")
    print("     先让模型在通用 VQA 上把「看图说话」练好，再学「看图开车」这个具体领域。\n")

    # 用 train5 的数据形态举个具体的序列例子
    s0 = make_sample(*DATA[0])[0]
    print(f"  nav 的一条真实序列长这样（train5 用的同款）：")
    print(f"     {' '.join(VOCAB[t] for t in s0.tolist())}")
    print(f"     对应的 components_order 是 image | traj_history | route | prompt | traj_future，")
    print(f"     但 toy 里只有其中三段（image / prompt / traj_future）。\n")

    # ── ② lr_multiplier ──
    print("=" * 72)
    print("② lr_multiplier：为什么视觉塔只给 1/10 学习率")
    print("=" * 72)
    print(f"  configs/sft_stage1_nav.yaml:  lr_multiplier: {LR_MULTIPLIER}")
    print(f"  假设 base lr = {BASE_LR:g}\n")

    model = ToyStage2Params()
    groups = build_param_groups(model, LR_MULTIPLIER, BASE_LR)
    print(f"  {'参数组(前缀)':<26}{'倍率':>6}{'实际 lr':>12}   参数")
    print("  " + "-" * 68)
    for g in groups:
        print(f"  {g['key']:<26}{g['mult']:>6.1f}{g['lr']:>12.2e}   {len(g['names'])} 个")
    print()
    print("  最长前缀匹配的意义：假如同时写了 {'vlm': 0.5, 'vlm.model.visual': 0.1}，")
    print("  视觉塔会命中【更长】的那条（0.1），而不是 vlm 那条。")
    g2 = build_param_groups(model, {"vlm": 0.5, "vlm.model.visual": 0.1}, BASE_LR)
    for g in g2:
        print(f"     {g['key']:<22} mult={g['mult']:<5} lr={g['lr']:.2e}")
    print("\n  → 为什么视觉塔要小 lr？视觉特征是从预训练来的、最通用也最容易被打坏。")
    print("     驾驶数据规模远小于预训练，大 lr 会把 ViT 拉偏，反而丢掉通用视觉能力。")
    print("     语言塔要学新的 CoC 表达，就得给足 lr。\n")

    # ── ③ checkpoint 交接 ──
    print("=" * 72)
    print("③ checkpoint 交接：stage1 训出的 VLM 怎么进 stage2")
    print("=" * 72)
    print("  真实是 stage2 的模型从 `nvidia/Alpamayo-1.5-10B` 起（含 expert），")
    print("  再用 `stage1_vlm_checkpoint_path` 把 stage1 训出的 VLM 权重【覆盖】进去。")
    print("  所以 stage2 拿到的是：新 VLM（来自 stage1）+ 旧 expert（来自发布模型）。\n")

    torch.manual_seed(0)
    samples = [make_sample(*d) for d in DATA]
    input_ids = torch.stack([x[0] for x in samples])
    img_feats = torch.stack([x[1] for x in samples])

    # 「stage1」：只训 VLM（train5 的 train_vlm 只动 .vlm 的参数）
    stage1_holder = Stage2Model()
    loss1 = train_vlm(stage1_holder, input_ids, img_feats)
    stage1_vlm = stage1_holder.vlm
    stage1_ckpt = {k: v.clone() for k, v in stage1_vlm.state_dict().items()}
    print(f"  stage1 训完：next-token loss = {loss1:.4f}")

    # 「stage2」：新模型（expert 随机）
    stage2 = Stage2Model(cotrain_vlm=False, stop_grad_from_vlm=True)
    expert_before = {k: v.clone() for k, v in stage2.expert.state_dict().items()}
    vlm_random_gap = max((stage2.vlm.state_dict()[k] - stage1_ckpt[k]).abs().max().item()
                         for k in stage1_ckpt)

    # 注入 stage1 的 VLM
    stage2.vlm.load_state_dict(stage1_ckpt)
    vlm_gap_after = max((stage2.vlm.state_dict()[k] - stage1_ckpt[k]).abs().max().item()
                        for k in stage1_ckpt)
    expert_gap = max((stage2.expert.state_dict()[k] - expert_before[k]).abs().max().item()
                     for k in expert_before)

    print(f"\n  {'检查项':<34}{'最大差异':>12}")
    print("  " + "-" * 48)
    print(f"  {'注入前 VLM 权重 vs stage1':<34}{vlm_random_gap:>12.4f}   ← 随机初始化的差距")
    print(f"  {'注入后 VLM 权重 vs stage1':<34}{vlm_gap_after:>12.4f}   ← 完全一致 ✓")
    print(f"  {'expert 权重（注入前后）':<34}{expert_gap:>12.4f}   ← expert 没被动过 ✓")
    print("\n  → 这就是 `stage1_vlm_checkpoint_path` 干的事：只换 VLM，expert 保持原样。")
    print("     （RL 那条线同理：RL 只导出 VLM 权重，拿回 SFT 时 expert 随机初始化）")
