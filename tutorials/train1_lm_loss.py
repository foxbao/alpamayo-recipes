"""train1: 语言建模损失 —— next-token prediction（SFT 阶段一的本质）"""

import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN = 16

# 迷你词表：模拟一段 CoC 推理文本
VOCAB = ["<pad>", "<bos>", "<eos>", "<cot_start>", "<cot_end>",
        "construction", "blocks", "right", "lane", "change", "left"]
V = {t: i for i, t in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)


class TinyLM(nn.Module):
    """最小的语言模型：embedding + 线性头，输出每个位置的词表分布。"""
    def __init__(self, vocab_size=VOCAB_SIZE, hidden=HIDDEN):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids):
        return self.head(self.embed(input_ids))   # (B, L, V)


def encode(tokens):
    return torch.tensor([[V[t] for t in tokens]])


if __name__ == "__main__":
    # 一条训练样本：一段 CoC 推理
    tokens = ["<bos>", "<cot_start>", "construction", "blocks", "right",
            "lane", "change", "left", "<cot_end>", "<eos>"]
    input_ids = encode(tokens)
    B, L = input_ids.shape
    print("tokens:   ", tokens)
    print("input_ids:", input_ids.tolist(), f"  shape={tuple(input_ids.shape)}")

    model = TinyLM()
    logits = model(input_ids)
    print(f"\nlogits shape: {tuple(logits.shape)}   (B, L, 词表大小)")

    # ---- 关键：next-token 预测要「错一位」----
    # 位置 i 的输出，用来预测位置 i+1 的 token
    shift_logits = logits[:, :-1, :]      # 用位置 0..L-2 的预测
    shift_labels = input_ids[:, 1:]       # 去预测位置 1..L-1 的真实 token
    print(f"shift_logits: {tuple(shift_logits.shape)}  (预测)")
    print(f"shift_labels: {tuple(shift_labels.shape)}  (真值) = {shift_labels.tolist()}")

    loss = F.cross_entropy(
        shift_logits.reshape(-1, VOCAB_SIZE),
        shift_labels.reshape(-1),
    )
    import math
    ln_vocab = math.log(VOCAB_SIZE)
    print(f"\n初始 loss = {loss.item():.4f}  "
          f"(随机初始化，应接近 ln({VOCAB_SIZE}) = {ln_vocab:.2f})")

    # ---- 训练：让它学会预测这段文本 ----
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for it in range(300):
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, VOCAB_SIZE),
            input_ids[:, 1:].reshape(-1),
        )
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 50 == 0:
            print(f"iter {it:3d}  loss={loss.item():.4f}")

    # ---- 训练后：让模型自己「续写」 ----
    print("\n训练后，让模型自回归续写（贪心）：")
    gen = [V["<bos>"]]
    model.eval()
    with torch.no_grad():
        for _ in range(len(tokens) - 1):
            out = model(torch.tensor([gen]))
            nxt = out[0, -1].argmax().item()
            gen.append(nxt)
    print("  ", " ".join(VOCAB[i] for i in gen))
    print("   真实:", " ".join(tokens))
