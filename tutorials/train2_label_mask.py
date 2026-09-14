"""train2: 标签掩码 —— 哪些 token 该算 loss，以及那个「错一位」

一条训练序列里，只有「模型该生成的那部分」算 loss；
提示词、图片等「给定的」部分用 IGNORE_INDEX=-100 屏蔽。

同一件事分三步看，顺序不要跳（先看形状，再看数值，最后看梯度）：
  ① 【对齐】位置 → token → mask → labels → 错一位：先搞清谁监督谁
⚠️ 本课 toy 的序列里**带了 CoC**（`<cot_start>…<cot_end>`），那是为了讲清机制。
   但 shipped 的 `nav.yaml` 里 `components_order` **不含 `cot`**——
   真实 nav 阶段的序列是「图 + 历史轨迹 + 导航指令 + 未来轨迹 token」，没有 CoC。
   （CoC 能力继承自发布 checkpoint；`cot` 字段在数据里有，但用在 RL 的 reward 上。）
   详见 README 的「CoC 到底在哪」一节。
  ② 【数值】加不加 mask，loss 差多少
  ③ 【梯度】被屏蔽的位置梯度必须为 0 —— 这才是「不算 loss」的真正含义

对应真实代码：src/alpamayo/utils/get_label_mask.py
             `fill_masks_between_special_tokens`
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = -100    # 和真实代码一致（sft_base_model.py:35）

VOCAB = ["<pad>", "<bos>", "<eos>", "<cot_start>", "<cot_end>",
         "you", "are", "a", "driver",
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


def get_label_mask(input_ids, start_id, end_id):
    """标记 [start_token, end_token] 区间内的 token 该算 loss（含 end）。

    这就是真实代码 fill_masks_between_special_tokens 的简化版。
    """
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    starts = (input_ids == start_id).nonzero()
    ends = (input_ids == end_id).nonzero()
    for (b, si), (_, ei) in zip(starts, ends):
        mask[b, si : ei + 1] = True    # 注意：含 end token（学会何时停）
    return mask


def show(tok_id):
    return "<ignored>" if tok_id == IGNORE_INDEX else VOCAB[tok_id]


if __name__ == "__main__":
    tokens = ["<bos>", "you", "are", "a", "driver",
              "<cot_start>", "construction", "blocks", "right", "lane",
              "<cot_end>", "<eos>"]
    input_ids = torch.tensor([[V[t] for t in tokens]])
    L = len(tokens)

    # ---- 造掩码：只覆盖 <cot_start> ... <cot_end> ----
    mask = get_label_mask(input_ids, V["<cot_start>"], V["<cot_end>"])
    labels = torch.where(mask, input_ids, IGNORE_INDEX)

    # ═══════════════════════════════════════════════════════════
    print("=" * 66)
    print("① 对齐：位置 → token → mask → labels")
    print("=" * 66)
    print(f"{'位置':>4}  {'token':<14} {'mask':>5}  {'labels':<14}")
    print("-" * 66)
    for i in range(L):
        kind = "  ← 提示词（给定）" if i <= 4 else ""
        print(f"{i:>4}  {tokens[i]:<14} {'✓' if mask[0, i] else '·':>5}  "
              f"{show(labels[0, i].item()):<14}{kind}")
    print("  ✓=算 loss   ·=屏蔽")

    # ═══════════════════════════════════════════════════════════
    print()
    print("=" * 66)
    print("② 那个「错一位」—— 预测来自位置 i，监督目标是位置 i+1 的 label")
    print("=" * 66)
    print(f"{'预测来自(位置 i)':<20} {'→ 目标是(位置 i+1)':<22} {'被监督':<6}")
    print("-" * 66)
    n_sup = 0
    for k in range(L - 1):
        sup = labels[0, k + 1].item() != IGNORE_INDEX
        n_sup += sup
        note = ""
        if k == 4:
            note = "  ← 从 driver 开始生成"
        if k == 9:
            note = "  ← 学会在这里停"
        print(f"  {k:>2} {tokens[k]:<16} → {k+1:>2} {tokens[k + 1]:<18} "
              f"{'✓' if sup else '✗'}{note}")
    print("-" * 66)
    print(f"共 {n_sup} 个 token 参与 loss（= mask 中 True 的个数）")

    # ═══════════════════════════════════════════════════════════
    model = TinyLM()
    logits = model(input_ids)

    def loss_of(use_mask):
        shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
        shift_labels = (labels if use_mask else input_ids)[:, 1:].reshape(-1)
        return F.cross_entropy(shift_logits, shift_labels, ignore_index=IGNORE_INDEX)

    loss_all = loss_of(use_mask=False)
    loss_masked = loss_of(use_mask=True)
    n_all = L - 1

    print()
    print("=" * 66)
    print("③ 数值：加不加 mask，loss 差多少")
    print("=" * 66)
    print(f"不加 mask: loss = {loss_all.item():.4f}   (在 {n_all} 个位置平均)")
    print(f"加  mask: loss = {loss_masked.item():.4f}   (只在 {n_sup} 个位置平均)")

    # ═══════════════════════════════════════════════════════════
    print()
    print("=" * 66)
    print("④ 梯度：屏蔽的位置必须【不产生梯度】—— 这才是「不算 loss」的真正含义")
    print("=" * 66)
    model.zero_grad()
    loss_of(use_mask=True).backward()
    g = model.embed.weight.grad
    for i in [2, 7]:   # 位置 2="are"(提示词, 屏蔽) / 位置 7="blocks"(答案, 算 loss)
        norm = g[input_ids[0, i]].norm().item()
        kind = "答案(算 loss)" if mask[0, i] else "提示词(被屏蔽)"
        print(f"  位置{i:2d} {tokens[i]:12s} {kind:16s} 梯度范数 = {norm:.4f}")
    print("  → 提示词位置梯度为 0，答案位置梯度非 0 ✓")
