"""train5b: teacher forcing 与自回归生成。

本课不加入图片，也不加入轨迹 token，只把“输入”和“目标”对齐问题跑透：
训练时整条序列一次前向；生成时只能拿已经生成的前缀。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


VOCAB = ["<bos>", "turn", "left", "<eos>"]
V = {token: i for i, token in enumerate(VOCAB)}
SEQUENCE = ["<bos>", "turn", "left", "<eos>"]


class TinyLM(nn.Module):
    """为了突出对齐关系，使用 embedding + head，不引入 attention。"""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.embed = nn.Embedding(len(VOCAB), hidden)
        self.head = nn.Linear(hidden, len(VOCAB))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(input_ids))


def ids(tokens: list[str]) -> torch.Tensor:
    return torch.tensor([[V[token] for token in tokens]])


def next_token_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[:, :-1].reshape(-1, len(VOCAB)), input_ids[:, 1:].reshape(-1))


def same_position_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """故意写错：让模型复制刚刚读到的 token。"""
    return F.cross_entropy(logits.reshape(-1, len(VOCAB)), input_ids.reshape(-1))


@torch.no_grad()
def generate(model: nn.Module, steps: int) -> list[str]:
    generated = [V["<bos>"]]
    for _ in range(steps):
        next_id = model(torch.tensor([generated]))[0, -1].argmax().item()
        generated.append(next_id)
    return [VOCAB[token_id] for token_id in generated]


def train(loss_fn, seed: int) -> tuple[nn.Module, float]:
    torch.manual_seed(seed)
    input_ids = ids(SEQUENCE)
    model = TinyLM()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(300):
        loss = loss_fn(model(input_ids), input_ids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, float(loss.detach())


if __name__ == "__main__":
    print("完整序列:", SEQUENCE)
    print("\n正确对齐（位置 i -> 位置 i+1）:")
    for i in range(len(SEQUENCE) - 1):
        print(f"  logits[{i}] 读取 {SEQUENCE[i]:<6} -> label[{i}] = {SEQUENCE[i + 1]}")

    correct, correct_loss = train(next_token_loss, seed=0)
    wrong, wrong_loss = train(same_position_loss, seed=0)
    print("\n训练结果:")
    print(f"  正确 shift loss: {correct_loss:.4f}")
    print(f"  错误 same-position loss: {wrong_loss:.4f}")
    print("\n自回归生成（从 <bos> 开始，每次只拿最后一个 logits）:")
    print("  正确模型:", generate(correct, len(SEQUENCE) - 1))
    print("  错误模型:", generate(wrong, len(SEQUENCE) - 1))
    print("错误模型训练指标也能很好，但它学到的是复制输入，而不是预测下一个 token。")
