"""train7: SFT 阶段二 —— 冻结 VLM，只训 action expert

★ 前置（两个，缺一个都会卡）：
    train6_flow_matching.py  —— 本课的 expert 就是用 flow matching 训的。
                                 没读的话，下面 `x_t = (1-t)·x0 + t·x1` 那两行看不懂。
    推理教程 stage13（尤其 Part 2）—— 本课就是它的真实版（KV cache 当条件）。

阶段一（train5）训 VLM：看图 → 写 CoC + 生成离散轨迹 token。
阶段二换一条通路：**VLM 不动**，用它留下的 KV cache 当条件，训一个连续动作的 expert。

模型与数据的定义在共享模块 `toy_stage2.py`（train8 也要用，所以抽出去了）；
本文件是【讲】这些的课——配着下面的注释和实测读。

真实代码 `recipes/alpamayo1_5_sft/models/sft_alpamayo_r1.py:125` 的 forward，五步：

    ① input_ids = self.fuse_traj_tokens(input_ids, traj_data)   轨迹 token 融进序列
    ② labels    = where(labels_mask, labels, IGNORE_INDEX)      train2 的掩码
    ③ with torch.no_grad():  vlm_outputs = self.vlm(input_ids, labels=labels, use_cache=True)
       → vlm_outputs.past_key_values                            缓存在这产生
    ④ kv_cache.crop(last_traj_future_start_idx)                 ★ 裁掉真值轨迹 token
       if stop_grad_from_vlm:
           layer.keys = layer.keys.detach(); layer.values = ...detach()
    ⑤ expert(inputs_embeds=action_embeds, past_key_values=kv_cache,
             is_causal=False)  → action_out_proj → flow matching loss

    if self.cotrain_vlm:
        loss += vlm_outputs.loss       # ← 只有开这个 VLM 才一起训（config 里默认 False）

★ 为什么必须有第 ④ 步的 crop？—— 本课会实测这个坑：
  训练时序列是 teacher forcing 的，`<traj_future_start>` 后面【跟着真值轨迹 token】。
  如果不裁掉，expert 的前缀里就装着标准答案，它只要抄就行了 —— **标签泄漏**。
  裁掉之后 expert 看不到未来，只能用 `<traj_future_start>` 之前的 CoC 去推。
  本课会把「裁」和「不裁」两个 expert 都训出来，再在【推理条件】（没有真值轨迹）下对比。

★ 本课和推理教程 stage13 的 Part 2 是同一件事，差别只有两点：
    1. cache 来自【真值 CoC】（teacher forcing），不是模型自己生成的
    2. 多了 crop，和 cotrain_vlm / stop_grad_from_vlm 两个开关

对应真实代码：recipes/alpamayo1_5_sft/models/sft_alpamayo_r1.py:125 `forward`
            recipes/alpamayo1_5_sft/configs/models/ar1_5_expert.yaml（cotrain_vlm: false）

【简化】纯 CPU、约 15 秒。VLM/expert 各 2 层、hidden 32、3 个场景、8 个 waypoint。
        真实是 36 层 / hidden 2048 的 expert + 数千 token 的前缀 + 64 个 waypoint。
        另外真实专家还要处理 RoPE 位置（`rope_deltas + past_key_values.get_seq_length()`），
        这里用可学习 pos_embed，位置从 0 开始，省略了这个偏移。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from toy_stage2 import (
    IGNORE_INDEX, HIDDEN, N_LAYERS, N_HEADS, IMG_DIM, ACTION_DIM,
    N_WAYPOINTS, N_STEPS, N_SAMPLES, BATCH_PER_SCENE, N_ITERS,
    VOCAB, V, VOCAB_SIZE, DATA, make_sample,
    CacheBlock, TinyVLM, crop_caches,
    ActionInProj, TinyExpert, Stage2Model, train_vlm,
)


def train_expert(model, input_ids, img_feats, target_actions, crop_at,
                 n_iters=N_ITERS, lr=5e-3):
    """阶段二：冻结 VLM，只按 flow matching 训 expert。`crop_at=None` 表示不裁（触发泄漏）。

    `target_actions` 形状 (B, N_WAYPOINTS, 2)，由轨迹 token 解码而来（见 make_sample）。
    """
    model.vlm.eval()
    for p in model.vlm.parameters():
        p.requires_grad_(model.cotrain_vlm)
    opt = torch.optim.Adam(model.expert.parameters(), lr=lr)
    # 把每个场景在 batch 维铺 BATCH_PER_SCENE 份（每份的 x0 / t 是独立采样的），
    # 这样一步就有 3×BATCH_PER_SCENE 条梯度信号，而不是 3 条。
    ids_t = input_ids.repeat(BATCH_PER_SCENE, 1)
    img_t = img_feats.repeat(BATCH_PER_SCENE, 1)
    act_t = target_actions.repeat(BATCH_PER_SCENE, 1, 1)
    B = ids_t.shape[0]
    tail = []
    for _ in range(n_iters):
        x1 = act_t                                       # 目标 = token 解码出来的动作
        x0 = torch.randn_like(x1)
        t = torch.rand(B)
        x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        caches, vlm_loss = model.cache_for_expert(ids_t, img_t, crop_at)
        fm_loss = F.mse_loss(model.expert(x_t, t, caches), x1 - x0)
        loss = fm_loss + (vlm_loss if model.cotrain_vlm else 0)
        opt.zero_grad(); loss.backward(); opt.step()
        if _ >= n_iters - 50:          # 只统计最后 50 步，单步的值噪声太大
            tail.append(fm_loss.item())
    return sum(tail) / len(tail)


@torch.no_grad()
def sample_kappa(expert, caches, n_steps=N_STEPS, n_samples=N_SAMPLES):
    """从噪声积分出轨迹，返回平均曲率。多次采样压掉随机性。"""
    acc = []
    for _ in range(n_samples):
        x = torch.randn(1, N_WAYPOINTS, ACTION_DIM)
        for s in range(n_steps):
            tt = torch.full((1,), s / n_steps)
            x = x + (1.0 / n_steps) * expert(x, tt, caches)
        acc.append(x[0, :, 1].mean().item())
    return sum(acc) / len(acc)


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    torch.manual_seed(0)
    samples = [make_sample(*d) for d in DATA]
    input_ids = torch.stack([s[0] for s in samples])          # (3, L)
    img_feats = torch.stack([s[1] for s in samples])          # (3, 3)
    target_actions = torch.stack([s[2] for s in samples])     # (3, 8, 2) 由 token 解码而来
    L = input_ids.shape[1]
    # 目标 κ 取解码后动作的最后一维（每个 waypoint 都一样，取第一个即可）
    TARGET = target_actions[:, 0, 1].tolist()

    print("阶段一的完整序列（teacher forcing，`<traj_future_start>` 后面是真值轨迹 token）：")
    for i, (d, ids) in enumerate(zip(DATA, input_ids)):
        print(f"  场景{i}: {' '.join(VOCAB[t] for t in ids.tolist())}")
    print(f"  L = {L} 个 token（前面还有 1 个图片 token）\n")

    # `<traj_future_start>` 在序列里的下标 → crop 位置
    pos = int((input_ids[0] == V["<traj_future_start>"]).nonzero()[0, 0])
    CROP_AT = pos + 2          # +1 是前面的图片 token，+1 是把 <traj_future_start> 本身留下
    print(f"  <traj_future_start> 在 token 下标 {pos} → kv_cache.crop({CROP_AT})")
    print(f"  crop 后 expert 能看到：{' '.join(VOCAB[t] for t in input_ids[0, :pos + 1].tolist())}")
    print(f"  crop 掉了（真值答案）：{' '.join(VOCAB[t] for t in input_ids[0, pos + 1:].tolist())}\n")

    # ── 训练两个 expert：一个 crop，一个不 crop ──
    torch.manual_seed(0)
    model = Stage2Model(cotrain_vlm=False, stop_grad_from_vlm=True)
    vlm_loss = train_vlm(model, input_ids, img_feats)
    print(f"【阶段一】VLM 训完，next-token loss = {vlm_loss:.4f}\n")

    print("【阶段二】冻结 VLM，只训 expert（flow matching）")
    print(f"  cotrain_vlm = {model.cotrain_vlm}     ← 真实 config 里是 false")
    print(f"  stop_grad_from_vlm = {model.stop_grad_from_vlm}\n")

    leaky = Stage2Model(cotrain_vlm=False, stop_grad_from_vlm=True)
    leaky.vlm.load_state_dict(model.vlm.state_dict())

    fm_crop = train_expert(model, input_ids, img_feats, target_actions, crop_at=CROP_AT)
    print(f"  ✅ 裁到 <traj_future_start>：末 50 步平均 fm_loss = {fm_crop:.4f}")
    fm_leak = train_expert(leaky, input_ids, img_feats, target_actions, crop_at=None)
    print(f"  ❌ 不裁（前缀里带真值轨迹 token）：末 50 步平均 fm_loss = {fm_leak:.4f}")
    print("  ⚠️ 注意两个 loss 几乎一样！—— 标签泄漏【不会】让训练 loss 变难看，")
    print("     它只是让 expert 学成「一半看 CoC、一半抄答案」。只有推理条件能暴露它。\n")

    # ── 在【推理条件】下评估：没有真值轨迹，前缀到 <traj_future_start> 为止 ──
    print("=" * 70)
    print("① 在推理条件下评估（前缀只到 <traj_future_start>，没有真值轨迹）")
    print("=" * 70)
    with torch.no_grad():
        infer_caches = crop_caches(model.vlm(input_ids, img_feats)[1], CROP_AT)
        infer_caches = [(k.detach(), v.detach()) for k, v in infer_caches]

    print(f"  {'场景':<8}{'目标 κ':>10}{'crop 版':>12}{'不裁版':>12}")
    for i, d in enumerate(DATA):
        c = [(k[i:i + 1], v[i:i + 1]) for k, v in infer_caches]
        a = sample_kappa(model.expert, c)
        b = sample_kappa(leaky.expert, c)
        print(f"  {['left','right','straight'][i]:<8}{TARGET[i]:>+10.3f}{a:>+12.3f}{b:>+12.3f}")

    # ── 决定性一测：固定 CoC，只换前缀里的真值轨迹 token ──
    print()
    print("=" * 70)
    print("② 决定性一测：CoC 固定成 'turn left'，只换前缀里的【真值轨迹 token】")
    print("=" * 70)
    # CoC 固定成 'turn left'，只换 <traj_future_start> 后面那【整串】16 个轨迹 token
    base_seq = input_ids[0, :pos + 1]
    traj_seqs = [input_ids[k, pos + 1:-2] for k in range(3)]   # 三个场景各自的 16 个 token
    fixed_coc = torch.stack([
        torch.cat([base_seq, traj_seqs[k],
                   torch.tensor([V["<traj_future_end>"], V["<eos>"]])])
        for k in range(3)
    ])
    with torch.no_grad():
        print(f"  {'前缀里塞的真值轨迹':<26}{'crop 版':>10}{'不裁版':>10}")
        for k, meaning in enumerate(["左转的 16 个 token", "右转的 16 个 token", "直行的 16 个 token"]):
            c_full = model.vlm(fixed_coc[k:k + 1], img_feats[:1])[1]
            c_crop = [(x.detach(), y.detach()) for x, y in crop_caches(c_full, CROP_AT)]
            c_none = [(x.detach(), y.detach()) for x, y in c_full]
            a = sample_kappa(model.expert, c_crop)
            b = sample_kappa(leaky.expert, c_none)
            print(f"  {meaning:<26}{a:>+10.3f}{b:>+10.3f}")
    print("  → crop 版：三个数几乎不变 —— 那 16 个 token 已经被裁掉了，它只能靠 CoC")
    print("     不裁版：塞「右转」的 token 时预测被拉低 → 它【确实在读前缀里的答案】")
    print("     这就是 `kv_cache.crop(...)` 存在的理由：训练时前缀里装着标准答案，")
    print("     expert 会去抄；裁掉它（真实里是 128 个 token），才逼着 expert 从 CoC 推。")
    print("     ⚠️ 注意「不裁版」是预先训练好的专家 —— 这里只换了前缀的内容。\n")

    # ── 条件验证三连（同推理教程 stage4-5 的验法）──
    print("=" * 70)
    print("③ 验证：条件（cache）真的被读进去了吗？")
    print("=" * 70)
    with torch.no_grad():
        print("  ① 各用各的 cache：")
        out = []
        for i in range(3):
            c = [(k[i:i + 1], v[i:i + 1]) for k, v in infer_caches]
            out.append(sample_kappa(model.expert, c))
        print(f"     κ = [{out[0]:+.3f}, {out[1]:+.3f}, {out[2]:+.3f}]   目标 = "
              f"[{TARGET[0]:+.3f}, {TARGET[1]:+.3f}, {TARGET[2]:+.3f}]")

        print("  ② cache 全置零（相当于没有条件）：")
        z = [(torch.zeros_like(k), torch.zeros_like(v)) for k, v in infer_caches]
        out0 = [sample_kappa(model.expert, [(k[i:i + 1], v[i:i + 1]) for k, v in z]) for i in range(3)]
        print(f"     κ = [{out0[0]:+.3f}, {out0[1]:+.3f}, {out0[2]:+.3f}]   ← 应该都趋近 0")

        print("  ③ 交叉换 cache（用别人的条件）：")
        for i, src in [(0, 1), (1, 0)]:
            c = [(k[src:src + 1], v[src:src + 1]) for k, v in infer_caches]
            y = sample_kappa(model.expert, c)
            print(f"     场景{i} 用场景{src} 的 cache → κ = {y:+.3f}"
                  f"   （场景{src} 的目标是 {TARGET[src]:+.3f}）")
        print("     → 预测跟着 cache 走，说明条件不是摆设。\n")

    # ── 用梯度范数看 VLM 冻没冻 ──
    print("=" * 70)
    print("cotrain_vlm 开关：用【梯度范数】看 VLM 到底冻没冻")
    print("=" * 70)
    for cotrain in (False, True):
        m = Stage2Model(cotrain_vlm=cotrain, stop_grad_from_vlm=not cotrain)
        m.train()
        x0 = torch.randn(len(samples), N_WAYPOINTS, ACTION_DIM)
        t = torch.rand(len(samples))
        caches, vlm_loss = m.cache_for_expert(input_ids, img_feats, CROP_AT)
        loss = F.mse_loss(m.expert(x0, t, caches), x0) + (vlm_loss if cotrain else 0)
        m.zero_grad(); loss.backward()

        def gnorm(mod):
            gs = [p.grad.norm().item() for p in mod.parameters() if p.grad is not None]
            return sum(gs) / len(gs) if gs else 0.0

        print(f"  cotrain_vlm={str(cotrain):<5}  VLM 梯度范数={gnorm(m.vlm):.3e}   "
              f"expert 梯度范数={gnorm(m.expert):.3e}")
    print("  → False 时 VLM 梯度精确为 0：`torch.no_grad()` + `cache.detach()` 双重保证。")
    print("    这和推理教程 stage13 的 Part 1 / Part 2 是同一个开关的两个状态。")
