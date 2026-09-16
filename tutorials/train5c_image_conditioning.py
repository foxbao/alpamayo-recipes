"""train5c: 图片作为前缀 token，以及 logits/labels 的对齐。

本课用一个很小的 causal Transformer 学习：图片是最前面的条件 token，文本位置可
读取图片；图片位置自身不预测语言 token。轨迹和 masked loss 留给 train5d。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(0)
VOCAB = ["<bos>", "left", "right", "<eos>"]
V = {token: i for i, token in enumerate(VOCAB)}
IMAGES = torch.eye(2)
TARGETS = [["<bos>", "left", "<eos>"], ["<bos>", "right", "<eos>"]]


def causal_mask(length: int) -> torch.Tensor:
    return torch.triu(torch.full((length, length), float("-inf")), diagonal=1)


class ImagePrefixLM(nn.Module):
    def __init__(self, hidden: int = 24):
        super().__init__()
        self.image_proj = nn.Linear(2, hidden)
        self.embed = nn.Embedding(len(VOCAB), hidden)
        self.attn = nn.MultiheadAttention(hidden, num_heads=2, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, len(VOCAB))

    def forward(self, input_ids: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        text = self.embed(input_ids)
        image_token = self.image_proj(image)[:, None, :]
        x = torch.cat([image_token, text], dim=1)
        attended = self.attn(x, x, x, attn_mask=causal_mask(x.shape[1]))[0]
        x = self.norm(x + attended)
        return self.head(x[:, 1:])  # 丢掉 [IMG] 的输出后，才和 labels 对齐。


def batch() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor([[V[t] for t in target] for target in TARGETS]), IMAGES


if __name__ == "__main__":
    input_ids, images = batch()
    model = ImagePrefixLM()
    print("input_ids shape:", tuple(input_ids.shape))
    print("加入图片后 transformer 序列长度:", input_ids.shape[1] + 1)
    logits = model(input_ids, images)
    print("切掉图片位置后的 logits shape:", tuple(logits.shape))
    print("labels shape:", tuple(input_ids.shape))

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
    for step in range(401):
        logits = model(input_ids, images)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, len(VOCAB)), input_ids[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in (0, 100, 400):
            print(f"step {step:3d} loss={loss.item():.4f}")

    with torch.no_grad():
        prefix = torch.tensor([[V["<bos>"]]])
        left_id = model(prefix, images[:1])[0, -1].argmax().item()
        right_id = model(prefix, images[1:])[0, -1].argmax().item()
    print("\n同一个 <bos>，交换图片后的预测:")
    print("  image[0] ->", VOCAB[left_id])
    print("  image[1] ->", VOCAB[right_id])
    print("图片位置提供条件；语言 loss 只比较文本位置的 logits 与文本 labels。")
