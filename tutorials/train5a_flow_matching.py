"""train5a: flow matching —— 模型怎么「生成」一个连续的东西

【为什么单独一课】
train5b（阶段二）要训一个「从噪声去噪出轨迹」的 expert，用的就是 flow matching。
但 flow matching 在前面几课里【一次都没出现过】——直接上 train5b 会卡在
`x_t = (1 - t) * x0 + t * x1` 那几行上。本课把它单独讲清楚。

（推理教程 `alpamayo1.5/tutorials/` 的 stage2 / stage6 也讲过，
  标题分别是「向量场采样直觉」和「训练（flow matching）」，可以对读。）

【要解决的问题】
想让模型输出一条轨迹（64 个 waypoint × 2 维 = 128 个数）。
「直接回归」不行——见推理教程 stage8：多个合理的未来会被平均成一条不存在的轨迹。
Flow matching 的做法是：**别让模型直接输出轨迹，让它输出一个「速度场」**，
然后从噪声出发、顺着速度场一步一步走过去。

【两步】

    训练：构造一条从噪声到数据的【直线】路径
        x_t = (1 - t) · x0  +  t · x1          t=0 是纯噪声，t=1 是数据
        让模型预测这条直线的方向：
        v_target = x1 - x0
        loss = MSE( v_model(x_t, t),  v_target )
        ★ 注意：v_target 和 t 无关——直线路径上速度恒定。这是 FM 比 DDPM 简单的地方。

    采样：从噪声出发，沿速度场积分
        x ← x + dt · v_model(x, t)             dt = 1/N_STEPS
        从 t=0 走到 t=1

【本课实测】一维上把 N(0,1) 推到 N(3, 0.5)，看分布真的变了。
            一维看得清：画不出图也能从均值和标准差看出来。

【对应 Alpamayo】
    x 就是轨迹动作 (64 waypoint, 2 维)；
    条件（图 + CoC）从哪来 —— 那是 train5b 的事（VLM 的 KV cache）。

★ 前置：不需要（本课自足）。想更细地看，推理教程的 stage2 / stage6 讲得更完整；
  本课只讲 train5b 用得到的那部分。

【简化】纯 CPU、秒级。一维数据、几十个参数的 MLP。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET_MEAN, TARGET_STD = 3.0, 0.5     # 要学出来的数据分布
NOISE_MEAN, NOISE_STD = 0.0, 1.0       # 起点（高斯噪声）
N_STEPS = 20                           # 采样时的积分步数
N_ITERS = 3000


class VelocityMLP(nn.Module):
    """预测「速度」的网络：吃 (x_t, t)，吐出速度。

    真实里它是 Alpamayo 的 action expert（36 层、hidden 2048），
    而且吃的不只是 (x_t, t)，还有从 VLM 来的条件——那是 train5b 的内容。
    """

    def __init__(self, hidden=64, n_freqs=8):
        super().__init__()
        self.n_freqs = n_freqs
        self.net = nn.Sequential(
            nn.Linear(1 + 2 * n_freqs, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, t):
        # t 做多频率编码再拼上（不然 MLP 学不好「不同 t 用不同速度」）
        freqs = torch.arange(self.n_freqs, dtype=t.dtype, device=t.device)
        arg = t[:, None] * (2.0 ** freqs)[None, :]
        feat = torch.cat([x, arg.sin(), arg.cos()], dim=-1)
        return self.net(feat)


def train(model, n_iters=N_ITERS, batch=256, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(n_iters):
        # ① 采两个端点：噪声 x0、数据 x1
        x0 = torch.randn(batch, 1) * NOISE_STD + NOISE_MEAN
        x1 = torch.randn(batch, 1) * TARGET_STD + TARGET_MEAN
        # ② 在它们之间取一个随机时刻 t，得到路径上的点 x_t
        t = torch.rand(batch, 1)
        x_t = (1 - t) * x0 + t * x1
        # ③ 让模型预测这条直线的方向
        loss = F.mse_loss(model(x_t, t.squeeze(-1)), x1 - x0)
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


@torch.no_grad()
def sample(model, n=2000, n_steps=N_STEPS, seed=0):
    """从噪声出发，沿速度场积分到 t=1。"""
    torch.manual_seed(seed)
    x = torch.randn(n, 1) * NOISE_STD + NOISE_MEAN          # t=0 的起点
    for s in range(n_steps):
        t = torch.full((n,), s / n_steps)
        x = x + (1.0 / n_steps) * model(x, t)               # ★ 欧拉积分，就这一行
    return x


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 72)
    print("① 训练目标：构造一条从噪声到数据的直线，让模型预测它的方向")
    print("=" * 72)
    print("     x_t = (1-t)·x0 + t·x1          t=0 噪声，t=1 数据")
    print("     v_target = x1 - x0              ★ 和 t 无关，直线路径速度恒定")
    print("     loss = MSE(v_model(x_t, t), v_target)\n")

    # 手动算一个例子让读者看清楚
    x0, x1, t = torch.tensor([0.0]), torch.tensor([3.0]), torch.tensor([0.25])
    print(f"  举个手算的例子：x0={x0.item():.1f}（噪声）, x1={x1.item():.1f}（数据）, t={t.item():.2f}")
    print(f"     x_t      = (1-{t.item():.2f})×{x0.item():.1f} + {t.item():.2f}×{x1.item():.1f}"
          f" = {((1-t)*x0 + t*x1).item():.2f}")
    print(f"     v_target = {x1.item():.1f} - {x0.item():.1f} = {(x1-x0).item():.1f}")
    print(f"     → 模型要学的是：在 x={((1-t)*x0 + t*x1).item():.2f}, t={t.item():.2f} 处吐出 3.0\n")

    # ── 训练 ──
    print("=" * 72)
    print("② 训练：把 N(0,1) 推向 N(3, 0.5)")
    print("=" * 72)
    model = VelocityMLP()
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  目标分布: 均值 {TARGET_MEAN}, 标准差 {TARGET_STD}")
    print(f"  起点分布: 均值 {NOISE_MEAN}, 标准差 {NOISE_STD}\n")

    torch.manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for it in range(0, N_ITERS + 1):
        x0 = torch.randn(256, 1) * NOISE_STD + NOISE_MEAN
        x1 = torch.randn(256, 1) * TARGET_STD + TARGET_MEAN
        t = torch.rand(256, 1)
        x_t = (1 - t) * x0 + t * x1
        loss = F.mse_loss(model(x_t, t.squeeze(-1)), x1 - x0)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 600 == 0:
            print(f"  iter {it:4d}  loss = {loss.item():.4f}")
    print("  （loss 不可能到 0：x_t 相同、t 相同时 x0/x1 有很多组合，")
    print("    模型学到的是它们的平均方向——这正是 flow matching 的估计量。）\n")

    # ── 采样 ──
    print("=" * 72)
    print("③ 采样：从噪声出发，沿速度场积分到 t=1")
    print("=" * 72)
    print(f"     x ← x + dt·v_model(x, t)       dt = 1/{N_STEPS}，共 {N_STEPS} 步")
    print("     ★ 采样时模型【只吃 x 和 t】，没有任何随机性——")
    print("       所以同样的起点一定得到同样的终点（要多样性就在【起点】上做）\n")

    x_start = torch.randn(2000, 1) * NOISE_STD + NOISE_MEAN
    x_end = sample(model, n=2000)
    print(f"  {'':<14}{'均值':>10}{'标准差':>10}")
    print("  " + "-" * 34)
    print(f"  {'起点 (t=0)':<14}{x_start.mean().item():>10.3f}{x_start.std().item():>10.3f}")
    print(f"  {'终点 (t=1)':<14}{x_end.mean().item():>10.3f}{x_end.std().item():>10.3f}")
    print(f"  {'目标':<14}{TARGET_MEAN:>10.3f}{TARGET_STD:>10.3f}")
    print("\n  → 分布确实被推过去了。这就是「生成」：不是输出一个值，")
    print("     而是学会一个【从噪声推过去的场】。")
    print("     之所以绕这一圈，是因为直接回归只会得到「条件均值」——")
    print("     多个合理的未来会被平均成一条不存在的轨迹（见推理教程 stage8）。\n")

    # ── 少走几步会怎样 ──
    print("=" * 72)
    print("④ 积分步数不够会怎样（真实里 N_STEPS 是个要调的参数）")
    print("=" * 72)
    print(f"  {'N_STEPS':>9}{'均值':>10}{'标准差':>10}   误差")
    print("  " + "-" * 40)
    for ns in [1, 2, 5, 20, 100]:
        e = sample(model, n=2000, n_steps=ns)
        err = abs(e.mean().item() - TARGET_MEAN)
        print(f"  {ns:>9}{e.mean().item():>10.3f}{e.std().item():>10.3f}   {err:.3f}")
    print("  → 步数越多越准（欧拉积分是一阶方法），但每步都要一次前向——")
    print("     真实里这是【速度 vs 精度】的权衡。")
    print("     train5b 里用 N_STEPS = 10（toy）；真实推理也是十几步量级。\n")

    print("=" * 72)
    print("⑤ 一个容易踩的坑：训练和采样的 t 范围必须一致")
    print("=" * 72)
    print("  训练时 t 从 U(0,1) 采，采样时也从 0 积到 1。")
    print("  如果采样只用 t ∈ [0, 0.5]，你只走完了路径一半——")
    print("  模型没错，是你没走到终点。推理教程 stage2 的 temperature")
    print("  就是这类「训练/推理必须对齐」的参数之一。\n")
    print("  下一课 train5b：把这个 expert 接到 VLM 的 KV cache 上——")
    print("  也就是「条件怎么进去」。本课的网络只吃 (x_t, t)，没有任何条件。")
