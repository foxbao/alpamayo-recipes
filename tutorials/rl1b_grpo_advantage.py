"""rl1b: GRPO 的组内 advantage。

本课把 REINFORCE 的标量 reward 换成同一个 prompt 的一组相对分数：

    A_i = (r_i - mean(r)) / std(r)

重点观察三件事：加同一个常数不改变 advantage、所有 reward 相同时没有信号、
G=1 时没有可比较的候选。本课还没有方差实验；那是 rl1c 的主题。
"""

import torch


def group_advantages(rewards):
    """按最后一维做组内归一化；一条样本时显式返回零。"""
    if rewards.shape[-1] < 2:
        return torch.zeros_like(rewards)
    # 这里用总体标准差，避免小组时出现无意义的 degrees-of-freedom 警告。
    std = rewards.std(dim=-1, keepdim=True, unbiased=False)
    return (rewards - rewards.mean(dim=-1, keepdim=True)) / (std + 1e-6)


if __name__ == "__main__":
    print("=" * 72)
    print("① 一组 reward 如何变成 GRPO advantage")
    print("=" * 72)

    rewards = torch.tensor([-1.0, -4.0, -2.0, -7.0, -1.0, -3.0])
    advantages = group_advantages(rewards.unsqueeze(0)).squeeze(0)
    print("  同一个 prompt 采了", len(rewards), "条候选：\n")
    print(f"  {'#':>3}{'reward':>10}{'advantage':>14}")
    print("  " + "-" * 29)
    for i, (reward, advantage) in enumerate(zip(rewards, advantages)):
        print(f"  {i:>3}{reward.item():>10.2f}{advantage.item():>14.3f}")
    print("  " + "-" * 29)
    print(f"  {'均值':>3}{rewards.mean().item():>10.2f}{advantages.mean().item():>14.3f}")

    shifted = group_advantages((rewards + 5).unsqueeze(0)).squeeze(0)
    print("\n② 给所有 reward 都加 5：")
    print("  最大 advantage 差异 =", f"{(advantages - shifted).abs().max().item():.2e}")
    print("  → GRPO 只关心组内谁更好，不关心这一组整体的绝对水平。")

    same = group_advantages(torch.full((1, 6), -2.0)).squeeze(0)
    one = group_advantages(torch.tensor([[-2.0]])).squeeze(0)
    print("\n③ 没有组内差异：")
    print("  所有 reward = -2 ->", same.tolist())
    print("  G=1            ->", one.tolist())
    print("  → 没有‘谁相对更好’，GRPO 就没有更新方向。")

    print("\n  记住：GRPO 用更多采样换掉 critic；真实配置的 n_generation = 12。")
    print("  下一课 rl1c：为什么组内归一化通常能降低梯度估计方差。")
