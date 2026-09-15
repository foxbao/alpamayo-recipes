"""rl1c: 用实验比较 REINFORCE 与 GRPO 的梯度方差。

这是一个单步选 bin 的 toy，不是自回归 VLM，也不是严格的训练速度 benchmark。
它只回答一个问题：在相同参数和相同采样预算下，组内归一化是否让估计更稳定。
"""

import torch
import torch.nn as nn

torch.set_num_threads(4)

N_PROMPTS = 3
N_BINS = 16
GROUP_SIZE = 12
N_REPEAT = 200
N_STEPS = 300
STEP_SIZE = 0.35
TARGET_BINS = torch.tensor([12.0, 4.0, 8.0])


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(N_PROMPTS, N_BINS) + 0.01)

    def dist(self, prompt_ids):
        return torch.distributions.Categorical(logits=self.logits[prompt_ids])


def reward_of(bins, prompt_ids):
    return -(bins.float() - TARGET_BINS[prompt_ids]).abs()


def group_advantages(rewards):
    if rewards.shape[-1] < 2:
        return torch.zeros_like(rewards)
    std = rewards.std(dim=-1, keepdim=True, unbiased=False)
    return (rewards - rewards.mean(dim=-1, keepdim=True)) / (std + 1e-6)


def gradient_from(policy, prompt_ids, bins, advantages):
    log_prob = policy.dist(prompt_ids).log_prob(bins)
    loss = -(advantages.detach() * log_prob).mean()
    policy.zero_grad()
    loss.backward()
    return torch.cat([p.grad.flatten() for p in policy.parameters()])


def estimate_variance(policy, mode):
    gradients = []
    for seed in range(N_REPEAT):
        torch.manual_seed(seed)
        prompt_ids = torch.arange(N_PROMPTS).repeat_interleave(GROUP_SIZE)
        bins = policy.dist(prompt_ids).sample()
        rewards = reward_of(bins, prompt_ids).view(N_PROMPTS, GROUP_SIZE)
        advantages = rewards if mode == "reinforce" else group_advantages(rewards)
        gradients.append(gradient_from(policy, prompt_ids, bins, advantages.reshape(-1)))
    matrix = torch.stack(gradients)
    return matrix.var(dim=0, unbiased=False).sum().item()


def train(policy, mode):
    torch.manual_seed(0)
    for _ in range(N_STEPS):
        prompt_ids = torch.arange(N_PROMPTS).repeat_interleave(GROUP_SIZE)
        bins = policy.dist(prompt_ids).sample()
        rewards = reward_of(bins, prompt_ids).view(N_PROMPTS, GROUP_SIZE)
        advantages = rewards if mode == "reinforce" else group_advantages(rewards)
        gradient = gradient_from(policy, prompt_ids, bins, advantages.reshape(-1))
        # 固定方向步长，只做 sanity check，不比较真实 optimizer 的学习率。
        gradient = gradient / (gradient.norm() + 1e-8) * STEP_SIZE
        with torch.no_grad():
            for parameter in policy.parameters():
                parameter -= gradient.reshape(parameter.shape)
    probs = policy.dist(torch.arange(N_PROMPTS)).probs
    return (probs * torch.arange(N_BINS).float()).sum(-1)


if __name__ == "__main__":
    print("=" * 72)
    print("① 梯度方差：REINFORCE vs GRPO")
    print("=" * 72)
    print(f"  每个 prompt 采样 G={GROUP_SIZE} 条，固定初始参数，重复 {N_REPEAT} 次\n")

    variances = {}
    for mode in ("reinforce", "grpo"):
        torch.manual_seed(0)
        variances[mode] = estimate_variance(Policy(), mode)
        print(f"  {mode:<10} 梯度方差总和 = {variances[mode]:.6f}")
    print(f"\n  GRPO / REINFORCE 方差比例 = {variances['grpo'] / variances['reinforce']:.4f}")
    print("  → 当前 toy 中，组内 baseline 去掉了共同的整体波动。")

    print("\n② 更新方向 sanity check（不是收敛速度比较）：")
    for mode in ("reinforce", "grpo"):
        learned = train(Policy(), mode)
        print(f"  {mode:<10} 训练后期望 bin = {[round(x, 2) for x in learned.tolist()]}")
    print("  train() 每步都把梯度缩放成相同的固定步长；真实训练还会有 AdamW、clip、")
    print("  token-level log-prob 和分布式 rollout。真实配置细节放到 rl3b。")
