"""教程共享的阶段二 toy 模型与数据 —— train7 和 train8 共用。

真实对应：recipes/alpamayo1_5_sft/models/sft_alpamayo_r1.py
          （VLM + action expert + crop + 两个开关）

这里放的是一直在重复用、且两个文件都要用的东西：
    常量 / VOCAB / DATA / make_sample      —— 数据和词表
    CacheBlock / TinyVLM / crop_caches     —— 带 KV cache 的 VLM（真实是 Cosmos-Reason2）
    ActionInProj / TinyExpert / Stage2Model —— action expert 与它的组装
    train_vlm                              —— 阶段一：把 VLM 训到会写 CoC

train7 是【讲】这些的课（配着注释读它）；train8 只是【用】其中的模型做
checkpoint 交接演示。放这里是为了让 train8 不依赖 train7 的实现细节——
本文件改了 API，train7/train8 会一起编译不过，不会出现「一个悄悄坏掉」。

更完整的分工见 README 的课程路线。这个文件是共享实现，不是一节额外课程。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import trajectory_tokenizer as T

IGNORE_INDEX = -100
HIDDEN = 32
N_LAYERS = 2
N_HEADS = 2
IMG_DIM = 3
ACTION_DIM = 2
N_WAYPOINTS = T.N_WAYPOINTS
N_STEPS = 10          # flow matching 的积分步数
N_SAMPLES = 16        # 评估时每个场景采样几次（单次方差大）
# 训练规模。★ BATCH_PER_SCENE 比 N_ITERS 重要得多：只有 3 个场景时，一个 batch
# 只有 3 条的梯度方向噪声极大，堆步数救不回来。实测「每场景铺 16 份（各自独立
# 的噪声和时间步）+ 500 步」比「每场景 1 份 + 5000 步」又快（5s vs 33s）又好。
BATCH_PER_SCENE = 16
N_ITERS = 500

VOCAB = (["<pad>", "<bos>", "<eos>", "<cot_start>", "<cot_end>",
          "<traj_future_start>", "<traj_future_end>",
          "turn", "left", "right", "go", "straight"]
         + T.bin_names())        # <i0> … <i63>
V = {t: i for i, t in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

# ⚠️ 关于 <iK>（离散轨迹 token）—— 本教程压缩最狠的一处，必须知道
#
# 名字是从真实代码照搬的。真实里它们是 <i0> … <i3999>（release config 里 num_bins=3000）：
#     discrete_tokens = [f"<i{v}>" for v in range(self.traj_vocab_size)]
#   （src/alpamayo1_5/models/base_model.py:271）
#
# 关键：**一条轨迹是一串 token，不是一个**。每个 <iK> 是一个【量化刻度（bin）】，
# 单独看【没有任何语义】。量化四步见 train4_traj_tokens.py，本文件用同一套。
#
#   真实： <traj_future_start>  <i0> <i1> ... <i127>  <traj_future_end>
#          128 = 64 waypoint × 2 维，num_bins = 3000
#   toy：  <traj_future_start>  <i32> <i41> ...        <traj_future_end>
#          16 = 8 waypoint × 2 维，num_bins = 64
#          左转 κ=+0.10 → <i32> <i41> 重复；右转 → <i32> <i22>；直行 → 全是 <i32>
#
# ★ 这一课尤其相关：下面 `crop_at` 裁掉的「真值轨迹 token」，真实里就是这 128 个 <iK>。
#   裁掉它们 = 不让 expert 看到标准答案（本课会实测不裁的后果）。
#
# （顺带：这 3000 个 token 正是 RL 那条「离散通路」的载体；发布推理走连续 expert 时，
#   ExpertLogitsProcessor 会把它们全 mask 成 -inf，生成停在 <traj_future_start>。）

DATA = [
    # (img_idx, CoC 词, 目标 κ)
    (0, ["turn", "left"], +0.10),
    (1, ["turn", "right"], -0.10),
    (2, ["go", "straight"], 0.00),
]


def make_sample(img_idx, cot_words, kappa):
    """拼一条**阶段一风格**的完整序列 —— `<traj_future_start>` 后面跟着【16 个】真值轨迹 token。

    这正是阶段二训练时拿到的东西：标签是 teacher forcing 的，未来轨迹就在序列里。
    真实里这一段有 128 个 token，所以下面 crop 掉的是 128 个答案——不是一个。

    返回 (ids, img, target_action)：
        target_action 是【把这串 token 解码回来】的动作，也就是 expert 要学的目标。
        这样「序列里的答案」和「expert 的答案」是同一个东西——真实里也正是如此
        （VLM 预测的 token 和 expert 回归的动作，描述的是同一条轨迹）。
    """
    action = torch.zeros(1, T.N_WAYPOINTS, 2)
    action[0, :, 1] = kappa
    traj_ids = T.encode(action)[0] + V["<i0>"]            # bin 号 → 词表 id
    target_action = T.decode(traj_ids - V["<i0>"])[0]     # 量化后再解码 = 序列里那串的含义

    head = [V["<bos>"], V["<cot_start>"], *[V[w] for w in cot_words],
            V["<cot_end>"], V["<traj_future_start>"]]
    ids = torch.cat([torch.tensor(head), traj_ids,
                     torch.tensor([V["<traj_future_end>"], V["<eos>"]])])
    img = torch.zeros(IMG_DIM)
    img[img_idx] = 1.0
    return ids, img, target_action


# ═══════════════════════════════════════════════════════════════
# 迷你 VLM（带 KV cache）
# ═══════════════════════════════════════════════════════════════

class CacheBlock(nn.Module):
    """因果 transformer block，支持 KV cache —— 同推理教程 stage4 的 CacheBlock。"""

    def __init__(self, hidden=HIDDEN, n_heads=N_HEADS):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.SiLU(),
                                 nn.Linear(4 * hidden, hidden))

    def forward(self, x, cache=None, causal=True):
        B, T, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, self.n_heads, -1).transpose(1, 2) for t in (q, k, v))
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        attn = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        if causal:
            neg = torch.full((T, T), float("-inf"), device=x.device, dtype=attn.dtype)
            attn = attn + torch.triu(neg, diagonal=1)
        attn = attn.softmax(dim=-1)
        x = x + self.proj((attn @ v).transpose(1, 2).reshape(B, T, -1))
        x = x + self.ffn(self.ln2(x))
        return x, (k, v)


class TinyVLM(nn.Module):
    """极简因果 LM：图片 token + 文本 token → logits + 逐层 K/V。"""

    def __init__(self, hidden=HIDDEN, n_layers=N_LAYERS, max_len=64):
        super().__init__()
        self.img_proj = nn.Linear(IMG_DIM, hidden)
        self.embed = nn.Embedding(VOCAB_SIZE, hidden)
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden))
        self.blocks = nn.ModuleList([CacheBlock(hidden) for _ in range(n_layers)])
        self.head = nn.Linear(hidden, VOCAB_SIZE)

    def forward(self, input_ids, img_feat):
        """返回 `(logits, caches)` —— 对应真实代码的 `use_cache=True` / `past_key_values`。"""
        emb = self.embed(input_ids)
        img = self.img_proj(img_feat)[:, None, :]
        x = torch.cat([img, emb], dim=1) + self.pos[:, : input_ids.shape[1] + 1]
        caches = []
        for blk in self.blocks:
            x, c = blk(x)
            caches.append(c)
        # 切掉图片位置：logits[j] 由 [IMG] + tokens[0..j] 算出，用来预测 tokens[j+1]
        return self.head(x[:, 1:, :]), caches


def crop_caches(caches, n):
    """等价于真实代码的 `kv_cache.crop(n)`：只保留前 n 个位置的 K/V。"""
    return [(k[:, :, :n], v[:, :, :n]) for k, v in caches]


# ═══════════════════════════════════════════════════════════════
# 迷你 action expert（连续动作，flow matching）
# ═══════════════════════════════════════════════════════════════

class ActionInProj(nn.Module):
    """噪声动作 + 时间步 → 动作 embedding（对应真实 `action_in_proj`）。"""

    def __init__(self, hidden=HIDDEN, n_freqs=4):
        super().__init__()
        self.n_freqs = n_freqs
        self.mlp = nn.Sequential(nn.Linear(ACTION_DIM + 2 * n_freqs, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden))

    def forward(self, action, t):
        freqs = torch.arange(self.n_freqs, device=t.device, dtype=t.dtype)
        arg = t[:, None] * (2.0 ** freqs)[None, :] * math.pi
        te = torch.cat([arg.sin(), arg.cos()], dim=-1)      # (B, 2*n_freqs)
        x = torch.cat([action, te[:, None, :].expand(-1, action.shape[1], -1)], dim=-1)
        return self.mlp(x)


class TinyExpert(nn.Module):
    """动作 token 接在 VLM 的逐层 K/V 之后。**没有 cross-attention，没有 condition 参数。**"""

    def __init__(self, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.in_proj = ActionInProj(hidden)
        self.blocks = nn.ModuleList([CacheBlock(hidden) for _ in range(n_layers)])
        self.out_proj = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                      nn.Linear(hidden, ACTION_DIM))

    def forward(self, action, t, caches):
        x = self.in_proj(action, t)
        for i, blk in enumerate(self.blocks):
            # causal=False：8 个 waypoint 互相可见（轨迹是整体，见推理教程 stage4-2）
            x, _ = blk(x, cache=caches[i], causal=False)
        return self.out_proj(x)


class Stage2Model(nn.Module):
    """阶段二 = VLM + expert。`cotrain_vlm` / `stop_grad_from_vlm` 就是真实 config 的两个开关
    （`ar1_5_expert.yaml`: cotrain_vlm: false；源码里 stop_grad_from_vlm 默认 True）。
    """

    def __init__(self, cotrain_vlm=False, stop_grad_from_vlm=True):
        super().__init__()
        self.vlm = TinyVLM()
        self.expert = TinyExpert()
        self.cotrain_vlm = cotrain_vlm
        self.stop_grad_from_vlm = stop_grad_from_vlm

    def cache_for_expert(self, input_ids, img_feat, crop_at=None):
        """③④ 两步：VLM forward（可选 no_grad）→ crop → detach。"""
        context = torch.enable_grad() if self.cotrain_vlm else torch.no_grad()
        with context:
            logits, caches = self.vlm(input_ids, img_feat)
            vlm_loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB_SIZE),
                                       input_ids[:, 1:].reshape(-1))
        if crop_at is not None:
            caches = crop_caches(caches, crop_at)
        if self.stop_grad_from_vlm:
            caches = [(k.detach(), v.detach()) for k, v in caches]
        return caches, vlm_loss


def train_vlm(model, input_ids, img_feats, n_iters=300, lr=5e-3):
    """阶段一：把 VLM 训到会写 CoC（这里只关心它别是随机的）。"""
    opt = torch.optim.Adam(model.vlm.parameters(), lr=lr)
    for it in range(n_iters + 1):
        logits, _ = model.vlm(input_ids, img_feats)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB_SIZE),
                               input_ids[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()
