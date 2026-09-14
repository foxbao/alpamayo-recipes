"""train4: 阶段一的【机制组装】—— 迷你 VLM 的监督微调

把 train1~3 的三个机制装起来：
  train1: next-token 交叉熵
  train2: 标签掩码（只有该生成的部分算 loss）
  train3: 两段损失（轨迹段 + 文本段）

再加一个极简的「视觉编码器」，让它真的「看图说话」。

⚠️ 这是【机制】上的完整，不是复现真实阶段一。差距见 README 的训练全景：
  - 真实阶段一从 nvidia/Alpamayo-1.5-10B 的 VLM 权重继续训，不是从零
  - 真实阶段一分两段课程：先 LingoQA(VQA)，再 PAI nav
  - 真实的「视觉编码器」是 Qwen3-VL 的 ViT，这里压缩成一个 Linear(3, H) → 1 个 token
  - 这里只有 3 条样本、18k 参数、没有历史轨迹 / 多相机 / 导航指令

对应真实代码：recipes/alpamayo1_5_sft/models/sft_base_model.py:359 `forward`
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = -100
HIDDEN = 32
IMG_DIM = 3          # 图片特征维度（3 种场景各一个 one-hot）

VOCAB = ["<pad>", "<bos>", "<eos>", "<cot_start>", "<cot_end>",
         "<traj_start>", "<traj_end>", "<i0>", "<i1>", "<i2>",
         "turn", "left", "right", "go", "straight"]
V = {t: i for i, t in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

# ---- 数据集：3 个"场景"，每个 = (图片, CoC 文本, 离散轨迹 token) ----
# 图片用 one-hot 表示"看到什么"；轨迹 token <i0>/<i1>/<i2> 代表左/右/直
DATA = [
    # (img_idx, cot 词, traj token)
    (0, ["turn", "left"], "<i0>"),        # 看到左转场景 -> 左转
    (1, ["turn", "right"], "<i1>"),       # 右转场景
    (2, ["go", "straight"], "<i2>"),      # 直行场景
]


def make_sample(img_idx, cot_words, traj_tok):
    """把一个场景拼成一条训练样本。三个入参就是 `DATA` 里的那个 tuple（调用处用 `*d` 拆开）。

    拼出来的序列（以样本 0 为例）：
        <bos> <cot_start>  turn   left  <cot_end> <traj_start> <i0> <traj_end> <eos>
        固定     固定      └── 变 ──┘    固定        固定      └变┘    固定     固定
                            ↑ CoC 文本                          ↑ 轨迹 token

    三个样本**只有这两处不同**，而且两者是【人为配对】的
    （左转图 ↔ `turn left` ↔ `<i0>`）—— 模型得自己学会这个对应关系。

    返回 3 个东西，作用完全不同，这也是这个函数看着绕的原因：
        [0] ids : torch 张量 (9,)  —— 真的喂给模型（走 `embed`）
        [1] img : torch 张量 (3,)  —— 图片的 one-hot，真的喂给模型（走 `img_proj`）
        [2] seq : list[str]        —— **不是张量**，只在最后打印 / 比对答案时用

    ⚠️ 模型里序列的【真实长相】见 `TinyVLM.forward`：transformer 收到的是 **10 个位置**
       —— 图片被拼在最前面 ——
            位置：   0       1          2          3     4    ...
                   [IMG]  [<bos>]  [<cot_start>]  turn  left  ...
        `head` 输出时做了 `x[:, 1:, :]`，把图片那一格切掉 → 剩 9 个 logits。
        切完才能和 `input_ids` 的 9 个 token 一一对齐，「错一位」才错得对。
    """
    seq = (["<bos>", "<cot_start>"] + cot_words + ["<cot_end>",
           "<traj_start>", traj_tok, "<traj_end>", "<eos>"])
    img = torch.zeros(IMG_DIM)
    img[img_idx] = 1.0
    return (torch.tensor([V[t] for t in seq]), img, seq)


class CausalBlock(nn.Module):
    """极简因果 transformer block（自注意力 + FFN）。"""
    def __init__(self, h=HIDDEN, n_heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(h, n_heads, batch_first=True)
        self.n1 = nn.LayerNorm(h)
        self.ffn = nn.Sequential(nn.Linear(h, h * 2), nn.SiLU(), nn.Linear(h * 2, h))
        self.n2 = nn.LayerNorm(h)

    def forward(self, x, mask):
        x = self.n1(x + self.attn(x, x, x, attn_mask=mask)[0])
        x = self.n2(x + self.ffn(x))
        return x


def causal_mask(L):
    return torch.triu(torch.full((L, L), float("-inf")), diagonal=1)


class TinyVLM(nn.Module):
    """迷你 VLM：视觉编码器 + 词嵌入 + 因果 transformer + 语言模型头。"""
    def __init__(self, vocab_size=VOCAB_SIZE, hidden=HIDDEN, n_blocks=2):
        super().__init__()
        self.img_proj = nn.Linear(IMG_DIM, hidden)      # ← 视觉编码器
        self.embed = nn.Embedding(vocab_size, hidden)
        self.blocks = nn.ModuleList([CausalBlock(hidden) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids, img_feat):
        emb = self.embed(input_ids)                          # (B, L, H)
        img = self.img_proj(img_feat)[:, None, :]            # (B, 1, H)
        x = torch.cat([img, emb], dim=1)                     # (B, 1+L, H)  图片在最前面
        for blk in self.blocks:
            x = blk(x, causal_mask(x.shape[1]))
        return self.head(x[:, 1:, :])                        # 去掉图片位置 -> (B, L, V)


def mask_between(input_ids, start_id, end_id):
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for b in range(input_ids.shape[0]):
        s = (input_ids[b] == start_id).nonzero()
        e = (input_ids[b] == end_id).nonzero()
        if len(s) and len(e):
            mask[b, s[0, 0] : e[0, 0] + 1] = True
    return mask


def masked_ce(logits, labels):
    """错一位 + 掩码的交叉熵（train1 + train2）。"""
    shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
    shift_labels = labels[:, 1:].reshape(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=IGNORE_INDEX)


def compute_loss(model, input_ids, img_feat):
    logits = model(input_ids, img_feat)

    cot = mask_between(input_ids, V["<cot_start>"], V["<cot_end>"])
    traj = mask_between(input_ids, V["<traj_start>"], V["<traj_end>"])

    # 段一：轨迹（train3）
    loss_traj = masked_ce(logits, torch.where(traj, input_ids, IGNORE_INDEX))
    # 段二：文本 + 结尾的 <eos>（让模型学会「收尾」）
    # 真实代码里对应 get_role_eos_mask / get_assistant_mask
    eos_mask = input_ids == V["<eos>"]
    others_mask = cot | eos_mask
    loss_others = masked_ce(logits, torch.where(others_mask, input_ids, IGNORE_INDEX))
    # 相加（train3）
    return loss_traj + loss_others, loss_traj, loss_others


@torch.no_grad()
def generate(model, img_feat, n_new=8):
    """给一张图 + <bos>，自回归生成后面的 token。"""
    ids = torch.tensor([[V["<bos>"]]])
    model.eval()
    for _ in range(n_new):
        logits = model(ids, img_feat)
        ids = torch.cat([ids, logits[0, -1].argmax().view(1, 1)], dim=1)
    return [VOCAB[i] for i in ids[0].tolist()]


if __name__ == "__main__":
    samples = [make_sample(*d) for d in DATA]
    input_ids = torch.stack([s[0] for s in samples])       # (3, L)
    img_feats = torch.stack([s[1] for s in samples])       # (3, IMG_DIM)
    seqs = [s[2] for s in samples]

    print("训练数据（3 个场景）：")
    for i, seq in enumerate(seqs):
        print(f"  场景{i} (图片 one-hot[{i}]): {' '.join(seq)}")

    model = TinyVLM()
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    print("\n训练（两段损失之和）：")
    for it in range(801):
        loss, lt, lo = compute_loss(model, input_ids, img_feats)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0:
            print(f"  iter {it:3d}  loss={loss.item():.4f}  "
                  f"(轨迹段 {lt.item():.4f} + 文本段 {lo.item():.4f})")

    print("\n训练后：给每张图，让模型自己生成完整回答：")
    for i, (name, seq) in enumerate(zip(["左转场景", "右转场景", "直行场景"], seqs)):
        gen = generate(model, img_feats[i : i + 1])
        ok = "✓" if gen[: len(seq)] == seq else "✗"
        print(f"\n  {name}:")
        print(f"    生成: {' '.join(gen)}")
        print(f"    真实: {' '.join(seq)}   {ok}")
