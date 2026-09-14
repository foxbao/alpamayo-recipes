"""train8: 逐行对照真实代码 —— 收口

train1~7 每课讲一个机制。这一课把它们全部映射回真实文件，
并给出「打开哪个文件、看哪一段、对应哪一课」的清单。

用法：先看下面的表，再按顺序打开真实文件读。每读一段，回来跑一下本课，
      它会打印那一课最小可运行的等价代码，方便并排看。

真实代码的四个关键文件：

    recipes/alpamayo1_5_sft/train_hf.py                   入口（Hydra 组装）
    recipes/alpamayo1_5_sft/trainer.py                    Trainer 子类（lr_multiplier）
    recipes/alpamayo1_5_sft/models/sft_base_model.py      阶段一的模型 + 两段损失
    recipes/alpamayo1_5_sft/models/sft_alpamayo_r1.py     阶段二的模型（冻结 + expert）

    recipes/alpamayo1_5_sft/configs/                      三段课程的 yaml
    src/alpamayo/utils/get_label_mask.py                  标签掩码（两个函数）

【简化】纯 CPU、秒级。本课不训练，只做映射与对照。
"""

import inspect
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RECIPE = os.path.abspath(os.path.join(HERE, "..", "recipes", "alpamayo1_5_sft"))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))

# ═══════════════════════════════════════════════════════════════
# 对照表：教程机制 → 真实位置
# ═══════════════════════════════════════════════════════════════

MAPPING = [
    # (课, 机制, 真实文件, 定位, 说明)
    ("train1", "next-token 交叉熵", "models/sft_base_model.py", "_compute_next_token_loss",
     "阶段一本体。真实实现要处理变长 + ignore_index，核心和 train1 的 shift 一样"),
    ("train2", "标签掩码", "src/alpamayo/utils/get_label_mask.py",
     "fill_masks_between_special_tokens",
     "注释原文：'we should include the end token in the mask' —— 就是 train2 的 2-1"),
    ("train2", "assistant 的 <eos> 也监督", "src/alpamayo/utils/get_label_mask.py",
     "get_assistant_mask",
     "注释原文：'skip the bos + assistant\\n (always 3 tokens) and include the eos (+1)'"),
    ("train3", "两段损失拆分", "models/sft_base_model.py", "losses['future_traj'] / losses['others']",
     "按【token id 区间】判轨迹段：future_token_start_idx 起 traj_vocab_size 个"),
    ("train3", "两段相加不平均", "models/sft_base_model.py", "outputs.loss = sum(losses.values())",
     "就是 train3 说的「相加不是加权平均」"),
    ("train4", "轨迹 token 融进序列", "models/sft_base_model.py", "fuse_traj_tokens",
     "训练前把历史/未来的轨迹 token 填进占位符位置"),
    ("train4", "历史轨迹 token 化", "models/sft_base_model.py", "tokenize_history_trajectory", ""),
    ("train5", "阶段二 forward", "models/sft_alpamayo_r1.py", "forward（第 125 行附近）",
     "五步：fuse → labels → VLM(no_grad) → crop/detach → expert"),
    ("train5", "crop 掉真值轨迹", "models/sft_alpamayo_r1.py", "kv_cache.crop(last_traj_future_start_idx)",
     "train5 用「裁 vs 不裁」实测了不做这步的后果"),
    ("train5", "冻结 VLM 的开关", "models/sft_alpamayo_r1.py", "cotrain_vlm / stop_grad_from_vlm",
     "config 里 cotrain_vlm: false；train5 用梯度范数把这个开关打印出来了"),
    ("train5", "expert 非因果", "models/sft_alpamayo_r1.py", "expert_non_causal_attention",
     "对应 toy 里 CacheBlock 的 causal=False"),
    ("train5", "位置 id 的偏移", "models/sft_alpamayo_r1.py", "_process_position_ids_qwen2_5_vl",
     "rope_deltas + past_key_values.get_seq_length() —— toy 用可学习 pos_embed，省略了"),
    ("train6", "三段课程", "configs/sft_stage1_lingoqa.yaml", "defaults: 那几行",
     "三段各一个 yaml；对比 sft_stage1_nav.yaml / sft_stage2_nav.yaml 看它们换了什么"),
    ("train6", "lr_multiplier", "trainer.py", "create_optimizer（第 58 行附近）",
     "最长前缀匹配 → lr = base_lr × mult；train6 手写了一遍"),
    ("train6", "checkpoint 交接", "configs/models/ar1_5_expert.yaml", "stage1_vlm_checkpoint_path",
     "只换 VLM，expert 保持发布模型里那份"),
    ("train7", "ZeRO 配置", "configs/deepspeed/zero2.json", "zero_optimization.stage",
     "stage1 用 ZeRO-2；stage2 是 deepspeed: null"),
    ("train7", "DDP 未使用参数", "configs/sft_stage2_nav.yaml", "ddp_find_unused_parameters: true",
     "因为 VLM 冻结 → 有参数永远拿不到梯度"),
]

# ═══════════════════════════════════════════════════════════════
# 每课的最小等价代码（直接从教程文件里取，保证和真实跑的一致）
# ═══════════════════════════════════════════════════════════════

LESSONS = {
    "train1": ("train1_lm_loss.py", "TinyLM"),
    "train2": ("train2_label_mask.py", "get_label_mask"),
    "train5": ("train5_sft_stage2.py", "Stage2Model"),
    "train6": ("train6_curriculum.py", "build_param_groups"),
    "train7": ("train7_memory.py", "zero_per_gpu"),
}


def show_source(filename, symbol, max_lines=14):
    """打印教程里对应的那段最小代码。"""
    path = os.path.join(HERE, filename)
    if not os.path.exists(path):
        print(f"    （找不到 {filename}）")
        return
    src = open(path, encoding="utf-8").read().split("\n")
    start = None
    for i, line in enumerate(src):
        if line.startswith(f"def {symbol}") or line.startswith(f"class {symbol}"):
            start = i
            break
    if start is None:
        print(f"    （{filename} 里找不到 {symbol}）")
        return
    # 截到下一个顶层 def/class
    end = len(src)
    for j in range(start + 1, len(src)):
        if (src[j].startswith("def ") or src[j].startswith("class ")
                or src[j].startswith("if __name__")):
            end = j
            break
    body = src[start:min(end, start + max_lines)]
    for line in body:
        print(f"    │ {line}")
    if end - start > max_lines:
        print(f"    │ ... （以下省略 {end - start - max_lines} 行）")


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 78)
    print("① 教程机制 → 真实代码 对照表")
    print("=" * 78)
    cur = None
    for lesson, mech, rel, loc, note in MAPPING:
        if lesson != cur:
            print(f"\n  【{lesson}】")
            cur = lesson
        print(f"    {mech}")
        print(f"        文件: {rel}")
        print(f"        定位: {loc}")
        if note:
            print(f"        说明: {note}")

    print("\n" + "=" * 78)
    print("② 真实文件在不在？（能读到就能直接对照）")
    print("=" * 78)
    files = sorted({rel for *_, rel, _, _ in MAPPING})
    for rel in files:
        p = os.path.join(RECIPE, rel)
        if not os.path.exists(p):
            p = os.path.join(SRC, rel.replace("src/", ""))
        ok = os.path.exists(p)
        n = sum(1 for _ in open(p, encoding="utf-8")) if ok else 0
        print(f"  {'✅' if ok else '❌'} {rel:<52}{n:>5} 行")

    print("\n" + "=" * 78)
    print("③ 每课的最小等价代码（从教程文件里直接取，和跑起来的一致）")
    print("=" * 78)
    for lesson, (fname, symbol) in LESSONS.items():
        print(f"\n  ── {lesson}  ← {fname}::{symbol} ──")
        show_source(fname, symbol)

    print("\n" + "=" * 78)
    print("④ 建议的阅读顺序")
    print("=" * 78)
    order = [
        ("configs/sft_stage1_nav.yaml", "先看三段课程各自配了什么，建立全景"),
        ("src/alpamayo/utils/get_label_mask.py", "最短的一个文件，train2 的真实版"),
        ("models/sft_base_model.py", "阶段一：forward + 两段损失（train1~4）"),
        ("models/sft_alpamayo_r1.py", "阶段二：crop + 冻结 + expert（train5）"),
        ("trainer.py", "lr_multiplier（train6）"),
        ("configs/deepspeed/zero2.json", "ZeRO 配置（train7）"),
        ("train_hf.py", "最后看入口，把上面全串起来"),
    ]
    for i, (rel, why) in enumerate(order, 1):
        print(f"  {i}. {rel:<46}{why}")

    print("\n  ⚠️ 读的时候记住两件事（README 训练全景里也写了）：")
    print("     · 起点是 nvidia/Alpamayo-1.5-10B 的 VLM 权重，不是从头训")
    print("     · Alpamayo 有两条动作通路：离散 token（RL 走这条）和连续 expert（SFT 走这条）")
