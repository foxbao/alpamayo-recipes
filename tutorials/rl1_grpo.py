"""rl1: GRPO 到底在算什么 —— 从 REINFORCE 到「用组内均值当 baseline」

【前置】只需要知道两件事：策略会采样，reward 是对采样结果的评分。
train1 的 log-prob / loss 和 train4 的离散轨迹 token 有帮助，但不需要先完成 SFT 全部 9 课。
真实 RL 确实接在 SFT checkpoint 后面；本课的 toy 不加载任何 SFT 模型。

【为什么需要 RL】
SFT 的损失是 teacher forcing 的 next-token 交叉熵——它优化的是「像不像真值」。
但「像真值」≠「车开得好」。这两个不是一回事，SFT 的 loss 里没有「好不好」这一项。
RL 就是把「好不好」（reward）变成梯度的办法。

【核心困难】reward 不可导
轨迹是用 vLLM 采样出来的，采样这一步不可导。所以拿不到 d(reward)/d(参数)。
解法是 **score function 估计量**（REINFORCE）：

    ∇J = E[ r · ∇log p(采样结果) ]

直觉：**采样到好的结果就提高它的概率，坏的就降低**。不用对采样求导，
只对「这个结果的对数概率」求导。

【REINFORCE 的问题】方差太大
直接用 r 乘，方差巨大——因为 r 的绝对值里有很多和「这个样本比其他样本好多少」
无关的成分。减掉一个 baseline b 就行：

    ∇J = E[ (r - b) · ∇log p ]

b 可以是学出来的 critic（PPO 的做法），**也可以就用这一组样本的平均值**——
后者就是 GRPO（Group Relative Policy Optimization）。

    A_i = (r_i - mean(r)) / std(r)          ← 组内归一化，这就是 GRPO 的「组」
    ∇J = E[ A_i · ∇log p_i ]

★ GRPO 的关键好处：**不需要 critic**（省掉一个和策略一样大的模型）。
  代价是每个 prompt 要采样一组（`n_generation = 12`），而不是一条。

【本课】用一个玩具把它跑出来：
  ① 梯度方差：REINFORCE vs GRPO（同样的参数、同样的采样预算）
  ② 组内 advantage 到底长什么样，以及组内奖励相同时会发生什么
  ③ 一个 toy 更新方向 sanity check（不是严谨的收敛速度 benchmark）
  ④ 真实代码里在这个骨架上加了什么（DAPO 风格的改动）

【简化】纯 CPU、秒级。策略是「单步选一个 bin」的玩具，不是 128 个 token 的自回归。
        真实是自回归生成 128 个 token，log p 是这些 token 的 log-prob 之和——
        「之和」这一步不影响算法本身，只是把标量换成求和。
"""

import torch
import torch.nn as nn

# 这个 toy 的网络很小；限制线程后，在共享 CPU 上不会把时间浪费在线程调度上。
torch.set_num_threads(4)

N_PROMPTS = 3          # 3 个场景（左/右/直）
N_BINS = 16            # 策略要选一个 bin（真实是生成 128 个 token）
G = 12                 # 每个 prompt 采样几条 —— 真实 rollout.n_generation = 12
N_STEPS = 300
STEP_SIZE = 0.35       # 固定步长（把梯度归一化后再乘）
TARGET_BINS = torch.tensor([12.0, 4.0, 8.0])   # 每个场景的「正确」bin


class Policy(nn.Module):
    """玩具策略：给一个 prompt，输出 N_BINS 个 bin 上的分布。

    真实里这是一个 10B 的自回归 VLM，输出的是词表上的分布；
    这里压成「单步、只选一个 bin」，算法的其余部分完全一样。
    """

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(N_PROMPTS, N_BINS) + 0.01)

    def dist(self, prompt_ids):
        return torch.distributions.Categorical(logits=self.logits[prompt_ids])


def reward_of(bins, prompt_ids):
    """玩具 reward：离目标 bin 越近越好（真实里是 -ADE + comfort，见 rl2）。

    注意它【不可导】——真实里 reward 也是从采样出来的轨迹上算的，整条链不可导。
    """
    return -(bins.float() - TARGET_BINS[prompt_ids]).abs()


def group_advantages(rewards):
    """Normalize rewards within each prompt group.

    A one-sample group has no relative information, so define its advantage
    as zero explicitly instead of asking ``torch.std`` for a one-element sample.
    """
    if rewards.shape[-1] < 2:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean(dim=-1, keepdim=True)) / (
        rewards.std(dim=-1, keepdim=True, unbiased=False) + 1e-6
    )


def grad_from_advantages(policy, prompt_ids, bins, advantages):
    """把 advantage 变成梯度：loss = -A · log p(采样的那个 bin)。

    这就是 score function 估计量。REINFORCE 和 GRPO 只差 advantage 怎么算。
    """
    logp = policy.dist(prompt_ids).log_prob(bins)
    loss = -(advantages.detach() * logp).mean()
    policy.zero_grad()
    loss.backward()
    return torch.cat([p.grad.flatten() for p in policy.parameters()])


def estimate_grad(policy, mode, n_repeat=200, seed0=0):
    """在【固定的参数】上重复估计梯度 n_repeat 次，返回它们的范数均值 / 方差。

    mode='reinforce' → 直接用 reward
    mode='grpo'      → 减组内均值、除组内标准差
    """
    norms, grads = [], []
    for k in range(n_repeat):
        torch.manual_seed(seed0 + k)
        pid = torch.arange(N_PROMPTS).repeat_interleave(G)        # 每个 prompt 采 G 条
        bins = policy.dist(pid).sample()
        r = reward_of(bins, pid)
        r = r.view(N_PROMPTS, G)
        if mode == "reinforce":
            adv = r.reshape(-1)                                    # ★ 不减 baseline
        else:
            adv = group_advantages(r).reshape(-1)
        g = grad_from_advantages(policy, pid, bins, adv)
        grads.append(g)
        norms.append(g.norm().item())
    Gmat = torch.stack(grads)
    return sum(norms) / len(norms), Gmat.var(dim=0).sum().item()


def train(policy, mode, n_steps=N_STEPS, seed=0):
    """跑训练，返回每个 prompt 的「期望 bin」随步数的轨迹。"""
    torch.manual_seed(seed)
    hist = []
    for step in range(n_steps):
        pid = torch.arange(N_PROMPTS).repeat_interleave(G)
        bins = policy.dist(pid).sample()
        r = reward_of(bins, pid).view(N_PROMPTS, G)
        if mode == "reinforce":
            adv = r.reshape(-1)
        else:
            adv = group_advantages(r).reshape(-1)
        g = grad_from_advantages(policy, pid, bins, adv)
        # ★ 把梯度归一化到固定范数再更新 —— 只比较【方向】的质量，不比较步幅。
        #   必须这么做：GRPO 除了组内 std，梯度范数本身就比 REINFORCE 小 ~5 倍，
        #   用同一个 lr 直接比会得出「GRPO 更慢」的错误结论（那只是步子小）。
        g = g / (g.norm() + 1e-8) * STEP_SIZE
        with torch.no_grad():
            for p in policy.parameters():
                p -= g.reshape(p.shape)
        with torch.no_grad():
            hist.append((policy.dist(torch.arange(N_PROMPTS)).probs
                         * torch.arange(N_BINS).float()).sum(-1).tolist())
    return hist


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 74)
    print("① 梯度估计的方差：REINFORCE vs GRPO")
    print("=" * 74)
    print(f"  在【同一个初始参数】上重复估计梯度 200 次（每个 prompt 采样 G={G} 条）")
    print(f"  {'':<14}{'梯度范数均值':>14}{'梯度方差(总和)':>18}")
    print("  " + "-" * 46)
    results = {}
    for mode, name in [("reinforce", "REINFORCE"), ("grpo", "GRPO")]:
        torch.manual_seed(0)
        p = Policy()
        mean_n, var = estimate_grad(p, mode)
        results[mode] = (mean_n, var)
        print(f"  {name:<14}{mean_n:>14.5f}{var:>18.6f}")
    ratio = results["reinforce"][1] / results["grpo"][1]
    print(f"\n  → GRPO 的梯度方差是 REINFORCE 的 1/{ratio:.1f}")
    print("     原因：减去组内均值【移除了「这一组整体好不好」这个共同成分】。")
    print("     那个成分对每个样本都一样，不携带「谁比谁好」的信息，")
    print("     但它会让梯度随机地整体放大缩小 —— 纯噪声。")
    print("     再除以组内标准差，等于把「这组的分辨率」归一化掉。\n")

    # ── ② advantage 长什么样 ──
    print("=" * 74)
    print("② 「组内归一化」到底做了什么 —— 看一组 advantage 的实际数值")
    print("=" * 74)
    torch.manual_seed(7)
    p = Policy()
    pid = torch.zeros(G, dtype=torch.long)          # 只看场景 0（目标 bin = 12）
    bins = p.dist(pid).sample()
    r = reward_of(bins, pid)
    mean, std = r.mean(), r.std()
    adv = (r - mean) / (std + 1e-6)
    print(f"  场景 0 采了 G={G} 条，目标 bin = {TARGET_BINS[0]:.0f}\n")
    print(f"  {'#':>3}{'采样到的 bin':>14}{'reward':>10}{'advantage':>12}")
    print("  " + "-" * 40)
    for i in range(G):
        print(f"  {i:>3}{bins[i].item():>14d}{r[i].item():>10.3f}{adv[i].item():>12.3f}")
    print("  " + "-" * 40)
    print(f"  {'':>3}{'均值':>14}{mean.item():>10.3f}{0.0:>12.3f}")
    print(f"  {'':>3}{'标准差':>14}{std.item():>10.3f}")
    print(f"\n  注意 advantage 的正负：**只有比组内平均好的才是正的**。")
    print(f"  最差的那条（离目标最远）advantage 约 {adv.min():.2f}，它会被【压低】；")
    print(f"  最好的那条约 {adv.max():.2f}，会被【抬高】。")

    # ★ 决定性演示：给所有 reward 加同一个常数，advantage 完全不变
    r_shift = r + 5.0
    adv_shift = (r_shift - r_shift.mean()) / (r_shift.std() + 1e-6)
    print(f"\n  ★ 把每条 reward 都 +5（整体变好），advantage 变了吗？")
    print(f"     原  advantage: {[round(v, 3) for v in adv[:4].tolist()]} ...")
    print(f"     加5 advantage: {[round(v, 3) for v in adv_shift[:4].tolist()]} ...")
    print(f"     最大差异 = {(adv - adv_shift).abs().max().item():.2e}   ← 完全一样")
    print("     这就是「相对」的含义：**整体好不好不进梯度，只有谁比谁好才进**。")
    print("     组内归一化相当于一个【自动的、不用学的 baseline】——这就是 GRPO")
    print("     能省掉 critic 的原因。\n")

    # ── ③ 组内没有相对差异时，GRPO 没有更新信号 ──
    print("=" * 74)
    print("③ 一个重要边界：组内 reward 相同，就没有相对学习信号")
    print("=" * 74)
    same_rewards = torch.full((G,), -2.0)
    same_adv = group_advantages(same_rewards.unsqueeze(0)).squeeze(0)
    one_sample = torch.tensor([[-2.0]])
    one_adv = group_advantages(one_sample).squeeze(0)
    print(f"  G={G} 且每条 reward 都是 -2 -> advantages = {same_adv.tolist()}")
    print(f"  G=1 时 -> advantage = {one_adv.tolist()}")
    print("  → GRPO 依赖组内比较：所有候选都一样好时，它无法判断该提高谁的概率。")
    print("    这也是 n_generation 不能随意设成 1 的原因；真实配置用 G=12。\n")

    print("  （附）顺手跑一遍，确认这个更新方向真的能训出来：")
    for mode, name in [("reinforce", "REINFORCE"), ("grpo", "GRPO")]:
        learned = torch.tensor(train(Policy(), mode)[-1])
        err = (learned - TARGET_BINS).abs().mean().item()
        vals = " ".join(f"{v:6.2f}" for v in learned.tolist())
        print(f"     {name:<10} 300 步后期望 bin = [{vals}]   平均偏差 {err:.3f}")
    print(f"     目标 bin                  = [{TARGET_BINS[0]:6.2f} {TARGET_BINS[1]:6.2f} "
          f"{TARGET_BINS[2]:6.2f}]")
    print("     两者都收敛了 —— 更新方向没问题。但这不是严格的收敛速度比较：")
    print("     train() 把每步梯度归一化成相同的固定步长，只比较方向而不是步幅。")
    print(f"     参数空间只有 {N_PROMPTS}×{N_BINS}，噪声几乎不影响收敛。")
    print("     GRPO 的收益要在【大参数空间 + 稀疏 reward】下才明显 ——")
    print("     ① 的 36 倍方差差才是它的真正价值。\n")

    # ── ④ 真实代码在这个骨架上加了什么 ──
    print("=" * 74)
    print("④ 真实 Alpamayo 的 GRPO 在这个骨架上加了什么")
    print("=" * 74)
    rows = [
        ("组大小", f"n_generation = {G}", "rollout.n_generation = 12（本课用的就是 12）"),
        ("critic", "不需要", "GRPO 用组内均值当 baseline，省掉一个 10B 的 value 网络"),
        ("重要性裁剪", "没有", "epsilon_low=0.2 / epsilon_high=0.28（★ 不对称，DAPO 风格）"),
        ("KL 惩罚", "没有", "kl_beta = 0.0 —— 完全不拉向参考模型"),
        ("内层迭代", "1 次", "mu_iterations = 1（PPO 那种多 epoch 这里没有）"),
        ("策略温度", "—", "rollout 采样 temperature=0.6, top_p=0.98"),
        ("更新尺度", f"固定步长 {STEP_SIZE}", "optm_lr = 2e-6（SFT 是 1e-5 量级，RL 要小得多）"),
        ("reward", "-|bin - target|", "ADE + comfort，带 gate（见 rl2）"),
    ]
    print(f"  {'要素':<12}{'本课玩具':<18}{'真实 Alpamayo'}")
    print("  " + "-" * 70)
    for a, b, c in rows:
        print(f"  {a:<12}{b:<18}{c}")

    print("\n  ★ 两点值得注意：")
    print("     · kl_beta = 0 意味着【不限制策略跑多远】——通常 DAPO 类做法还会配")
    print("       一个「正样本的 NLL 项」把好样本往回拉（真实 trainer 里就有）。")
    print("     · epsilon_high > epsilon_low（0.28 > 0.2）是【鼓励往上探】：")
    print("       好样本的概率被允许涨得比坏样本跌得快，避免过早收敛。")
    print("\n  下一课 rl2a/rl2b 讲运动 reward（ADE、comfort 和 gate），rl2c 再讲 reasoning reward。")
