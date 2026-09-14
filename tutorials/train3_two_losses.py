"""train3: 两段损失 —— 为什么 future_traj 和 others 要分开算

真代码（sft_base_model.py:388-407）把损失拆成两段再相加：
    losses["future_traj"] = 对轨迹 token 求平均
    losses["others"]      = 对文本 token 求平均
    outputs.loss = sum(两者)

本课演示：**「分开求平均再相加」和「合起来求平均」在数学上不一样**，
差别在于「两组 token 的权重」。

对应真实代码：recipes/alpamayo1_5_sft/models/sft_base_model.py:314 `_compute_next_token_loss`
             :388-407 两段损失的拆分
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = -100

# 词表：前半是文本，<i0>/<i1> 模拟"离散轨迹 token"
VOCAB = ["<pad>", "<bos>", "<eos>",
         "<cot_start>", "<cot_end>", "<traj_start>", "<traj_end>",
         "<i0>", "<i1>",
         "construction", "blocks", "right", "lane", "change", "left"]
V = {t: i for i, t in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)


class TinyLM(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, hidden=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids):
        return self.head(self.embed(input_ids))


if __name__ == "__main__":
    torch.manual_seed(0)
    tokens = ["<bos>",
              "<cot_start>", "construction", "blocks", "right", "lane", "change", "left", "<cot_end>",
              "<traj_start>", "<i0>", "<i1>", "<traj_end>",
              "<eos>"]
    input_ids = torch.tensor([[V[t] for t in tokens]])
    L = len(tokens)

    # ---- 两个区间：cot(文本) 和 traj(轨迹) ----
    cot = torch.zeros_like(input_ids, dtype=torch.bool)
    traj = torch.zeros_like(input_ids, dtype=torch.bool)
    cot[0, 1:9] = True     # <cot_start> ... <cot_end>  共 8 个
    traj[0, 9:13] = True   # <traj_start> ... <traj_end> 共 4 个

    print("序列划分：")
    for i, t in enumerate(tokens):
        tag = "cot(文本)" if cot[0, i] else ("traj(轨迹)" if traj[0, i] else "—")
        print(f"  {i:2d} {t:<14} {tag}")
    n_cot, n_traj = int(cot.sum()), int(traj.sum())
    print(f"\n  cot 段 {n_cot} 个 token，traj 段 {n_traj} 个 token")

    # ---- 构造 labels（两段都保留）----
    labels = torch.where(cot | traj, input_ids, IGNORE_INDEX)

    model = TinyLM()
    logits = model(input_ids)

    # ---- 逐位置的损失 ----
    shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)   # (L-1, V)
    shift_labels = labels[:, 1:].reshape(-1)                   # (L-1,)  索引 k 对应原始位置 k+1
    per_pos = F.cross_entropy(
        shift_logits, shift_labels, ignore_index=IGNORE_INDEX, reduction="none"
    )

    sup = shift_labels != IGNORE_INDEX
    cot_sel = torch.tensor([bool(cot[0, k + 1]) for k in range(L - 1)])
    traj_sel = torch.tensor([bool(traj[0, k + 1]) for k in range(L - 1)])

    # ---- 两种算法 ----
    loss_merged = per_pos[sup].mean()                 # 合起来：一组 mean
    loss_cot = per_pos[cot_sel].mean()
    loss_traj = per_pos[traj_sel].mean()
    loss_split = loss_cot + loss_traj                 # 拆开：两个 mean 相加

    print(f"\n{'='*58}")
    print("两种算法对比")
    print(f"{'='*58}")
    print(f"  合起来算:  mean(全部 {int(sup.sum())} 个)        = {loss_merged.item():.4f}")
    print(f"     ├ 其中 cot 段 {n_cot} 个，贡献权重 {n_cot/(n_cot+n_traj):.0%}")
    print(f"     └ 其中 traj 段 {n_traj} 个，贡献权重 {n_traj/(n_cot+n_traj):.0%}")
    print()
    print(f"  拆开算:")
    print(f"     cot 段:  mean({n_cot} 个)               = {loss_cot.item():.4f}")
    print(f"     traj 段: mean({n_traj} 个)               = {loss_traj.item():.4f}")
    print(f"     相加:                                    = {loss_split.item():.4f}")

    print(f"\n{'='*58}")
    print("关键差别：traj 段的『权重』")
    print(f"{'='*58}")
    print(f"  合起来算: traj 只占 {n_traj/(n_cot+n_traj):.0%}  ← 被 token 数量决定")
    print(f"  拆开算:   traj 占 50%            ← 强制与文本段平起平坐")

    # ---- 极端演示：让 traj 段变成"特别难"，看两种算法的敏感度 ----
    print(f"\n{'='*58}")
    print("极端演示：把 traj 段损失人为放大 10 倍，看总损失变化")
    print(f"{'='*58}")
    per_pos_bad = per_pos.clone()
    per_pos_bad[traj_sel] *= 10
    m = per_pos_bad[sup].mean().item()
    s = per_pos_bad[cot_sel].mean().item() + per_pos_bad[traj_sel].mean().item()
    print(f"  合起来算: {loss_merged.item():.4f} -> {m:.4f}  (变化 {m-loss_merged.item():+.3f})")
    print(f"  拆开算:   {loss_split.item():.4f} -> {s:.4f}  (变化 {s-loss_split.item():+.3f})")
    print("  → 拆开算对轨迹段的变化更敏感（因为它占的权重更大）")
