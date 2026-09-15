"""rl1a: reward 不可导时，策略为什么仍然可以训练。

本课只讲一个公式：score-function estimator（REINFORCE）。
先采样一个动作，再用 ``reward * grad(log_prob)`` 更新策略。
暂时不讲 GRPO 的组内归一化；下一课再加上它。
"""

import torch
import torch.nn as nn

torch.set_num_threads(4)

N_BINS = 5
TARGET_BIN = 3
STEP_SIZE = 0.5


class Policy(nn.Module):
    """最小策略：对一个 prompt 输出 N_BINS 个动作的概率。"""

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(N_BINS))

    def dist(self):
        return torch.distributions.Categorical(logits=self.logits)


def reward_of(action):
    """奖励只依赖采样结果，不依赖 logits，因此不能沿采样过程反传。"""
    return -(action.float() - TARGET_BIN).abs()


def score_function_gradient(policy, action):
    """用 ``-reward * log_prob`` 构造一次 REINFORCE loss。"""
    log_prob = policy.dist().log_prob(action)
    reward = reward_of(action)
    loss = -reward.detach() * log_prob
    policy.zero_grad()
    loss.backward()
    return reward, log_prob, loss, policy.logits.grad.detach().clone()


if __name__ == "__main__":
    print("=" * 72)
    print("① 从一次采样到一次策略梯度")
    print("=" * 72)
    print("  目标 bin =", TARGET_BIN)
    print("  公式：∇J ≈ reward(action) · ∇log π(action)\n")

    torch.manual_seed(4)
    policy = Policy()
    before = policy.dist().probs.detach().clone()
    action = policy.dist().sample()
    reward, log_prob, loss, gradient = score_function_gradient(policy, action)

    print(f"  采样 action       = {action.item()}")
    print(f"  reward(action)     = {reward.item():.3f}")
    print(f"  log π(action)      = {log_prob.item():.3f}")
    print(f"  loss = -reward·logπ = {loss.item():.3f}")
    print(f"  reward 是否需要梯度 = {reward.requires_grad}")
    print(f"  logits 的梯度       = {gradient.tolist()}")

    with torch.no_grad():
        policy.logits -= STEP_SIZE * gradient
    after = policy.dist().probs.detach()
    print("\n  一次小更新后的概率：")
    for i, (p0, p1) in enumerate(zip(before, after)):
        print(f"    bin {i}: {p0:.3f} -> {p1:.3f}")

    print("\n  关键点：")
    print("    · 采样和 reward 计算不需要可导")
    print("    · 只对 log-prob 求导，所以仍能把好坏反馈给策略")
    print("    · 真实 Alpamayo 会对一整段 rollout 的 token log-prob 求和")
    print("\n  下一课 rl1b：同一个 prompt 采一组样本，如何得到 GRPO advantage。")
