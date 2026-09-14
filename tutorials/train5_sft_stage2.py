"""train5: SFT 阶段二 —— 冻结 VLM，只训 action expert

阶段一（train4）训 VLM：看图 → 写 CoC + 生成离散轨迹 token。
阶段二换一条通路：**VLM 不动**，用它留下的 KV cache 当条件，训一个连续动作的 expert。

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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = -100
HIDDEN = 32
N_LAYERS = 2
N_HEADS = 2
IMG_DIM = 3
ACTION_DIM = 2
N_WAYPOINTS = 8
N_STEPS = 10          # flow matching 的积分步数
N_SAMPLES = 16        # 评估时每个场景采样几次（单次方差大）
# 训练规模。★ BATCH_PER_SCENE 比 N_ITERS 重要得多：只有 3 个场景时，一个 batch
# 只有 3 条的梯度方向噪声极大，堆步数救不回来。实测「每场景铺 16 份（各自独立
# 的噪声和时间步）+ 500 步」比「每场景 1 份 + 5000 步」又快（5s vs 33s）又好。
BATCH_PER_SCENE = 16
N_ITERS = 500

VOCAB = ["<pad>", "<bos>", "<eos>", "<cot_start>", "<cot_end>",
         "<traj_future_start>", "<traj_future_end>",
         "turn", "left", "right", "go", "straight", "<i0>", "<i1>", "<i2>"]
V = {t: i for i, t in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

DATA = [
    # (img_idx, CoC 词, 真值轨迹 token, 目标曲率)
    (0, ["turn", "left"], "<i0>", +0.10),
    (1, ["turn", "right"], "<i1>", -0.10),
    (2, ["go", "straight"], "<i2>", 0.00),
]


def make_sample(img_idx, cot_words, traj_tok, kappa):
    """拼一条**阶段一风格**的完整序列 —— 注意 `<traj_future_start>` 后面跟着真值轨迹 token。

    这正是阶段二训练时拿到的东西：标签是 teacher forcing 的，未来轨迹就在序列里。
    """
    seq = (["<bos>", "<cot_start>"] + cot_words
           + ["<cot_end>", "<traj_future_start>", traj_tok, "<traj_future_end>", "<eos>"])
    img = torch.zeros(IMG_DIM)
    img[img_idx] = 1.0
    return torch.tensor([V[t] for t in seq]), img, torch.tensor(kappa)


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


def train_expert(model, input_ids, img_feats, kappas, crop_at, n_iters=N_ITERS, lr=5e-3):
    """阶段二：冻结 VLM，只按 flow matching 训 expert。`crop_at=None` 表示不裁（触发泄漏）。"""
    model.vlm.eval()
    for p in model.vlm.parameters():
        p.requires_grad_(model.cotrain_vlm)
    opt = torch.optim.Adam(model.expert.parameters(), lr=lr)
    # 把每个场景在 batch 维铺 BATCH_PER_SCENE 份（每份的 x0 / t 是独立采样的），
    # 这样一步就有 3×BATCH_PER_SCENE 条梯度信号，而不是 3 条。
    ids_t = input_ids.repeat(BATCH_PER_SCENE, 1)
    img_t = img_feats.repeat(BATCH_PER_SCENE, 1)
    kap_t = kappas.repeat(BATCH_PER_SCENE)
    B = ids_t.shape[0]
    tail = []
    for _ in range(n_iters):
        x1 = torch.zeros(B, N_WAYPOINTS, ACTION_DIM)
        x1[:, :, 1] = kap_t[:, None]                      # 目标：每步曲率都是 kappa
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
    kappas = torch.stack([s[2] for s in samples])
    L = input_ids.shape[1]
    TARGET = kappas.tolist()

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

    fm_crop = train_expert(model, input_ids, img_feats, kappas, crop_at=CROP_AT)
    print(f"  ✅ 裁到 <traj_future_start>：末 50 步平均 fm_loss = {fm_crop:.4f}")
    fm_leak = train_expert(leaky, input_ids, img_feats, kappas, crop_at=None)
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
    base_seq = input_ids[0, :pos + 1]                     # <bos> ... <traj_future_start>（CoC 固定为 turn left）
    fixed_coc = torch.stack([
        torch.cat([base_seq, torch.tensor([V[tok], V["<traj_future_end>"], V["<eos>"]])])
        for tok in ["<i0>", "<i1>", "<i2>"]
    ])
    with torch.no_grad():
        print(f"  {'前缀里的真值 token':<20}{'crop 版':>12}{'不裁版':>12}")
        for k, (tok, meaning) in enumerate([("<i0>", "左"), ("<i1>", "右"), ("<i2>", "直")]):
            c_full = model.vlm(fixed_coc[k:k + 1], img_feats[:1])[1]
            c_crop = [(x.detach(), y.detach()) for x, y in crop_caches(c_full, CROP_AT)]
            c_none = [(x.detach(), y.detach()) for x, y in c_full]
            a = sample_kappa(model.expert, c_crop)
            b = sample_kappa(leaky.expert, c_none)
            print(f"  {tok}（真值={meaning}）{'':<8}{a:>+12.3f}{b:>+12.3f}")
    print("  → crop 版：三个数几乎不变（那个位置已经被裁掉了，它只能靠 CoC）")
    print("     不裁版：喂 <i1>（真值=右）时被拉低 → 它【确实在依赖前缀里的答案】")
    print("     这就是 `kv_cache.crop(...)` 存在的理由：训练时前缀里装着标准答案，")
    print("     expert 会去抄；裁掉它，才逼着 expert 从 CoC 推。\n")

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
