"""train5a: causal self-attention 的最小、可观察实验。

本课只讲 attention 和 causal mask，不讲图片、loss 或 action expert。
你会看到：QK^T 的分数、mask 后的注意力权重，以及未来 token 泄漏的后果。
"""

import math

import torch


torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)


def causal_mask(length: int) -> torch.Tensor:
    """未来位置为 -inf；softmax 后这些位置的权重就是 0。"""
    return torch.triu(torch.full((length, length), float("-inf")), diagonal=1)


def self_attention(x: torch.Tensor, use_causal_mask: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """一个没有 batch/head 的 attention，便于把每一步打印出来。"""
    q, k, v = x, x, x
    scores = q @ k.T / math.sqrt(x.shape[-1])
    if use_causal_mask:
        scores = scores + causal_mask(x.shape[0])
    weights = scores.softmax(dim=-1)
    return weights @ v, weights


def print_run(x: torch.Tensor, masked: bool) -> None:
    output, weights = self_attention(x, masked)
    print(f"\nuse_causal_mask={masked}")
    print("attention weights（每一行是当前位置读取各位置的比例）:")
    print(weights)
    print("输出:")
    print(output)


if __name__ == "__main__":
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [3.0, 3.0]])
    print("输入 x（位置 0..3）:")
    print(x)
    print("\ncausal mask:")
    print(causal_mask(len(x)))
    print_run(x, masked=False)
    print_run(x, masked=True)

    changed = x.clone()
    changed[3] = torch.tensor([-20.0, -20.0])
    before_unmasked = self_attention(x, False)[0][1]
    after_unmasked = self_attention(changed, False)[0][1]
    before_masked = self_attention(x, True)[0][1]
    after_masked = self_attention(changed, True)[0][1]
    print("\n只改位置 3，再观察位置 1:")
    print(f"  无 mask: {before_unmasked.tolist()} -> {after_unmasked.tolist()}")
    print(f"  有 mask: {before_masked.tolist()} -> {after_masked.tolist()}")
    print("结论：训练语言模型时必须屏蔽未来，否则模型可以直接偷看答案。")
