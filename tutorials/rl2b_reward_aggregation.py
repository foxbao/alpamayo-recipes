"""rl2b: ADE、comfort 如何聚合，以及 gate 带来的跳变。

本课假设 comfort 已经被压成 [-1, 0]，专注于权重、归一化和 ADE 阈值。
"""

ADE_THRESHOLD = 3.0
TRAJ_L2_WEIGHT = 0.4
COMFORT_WEIGHT = 0.1


def reward(l2_dist, comfort, use_gate=True):
    """对应真实 motion-only reward 的核心聚合逻辑。"""
    if use_gate and not (l2_dist < ADE_THRESHOLD):
        return -1.0
    return -TRAJ_L2_WEIGHT * (l2_dist / ADE_THRESHOLD) + COMFORT_WEIGHT * comfort


if __name__ == "__main__":
    print("=" * 72)
    print("① 先归一化，再加权")
    print("=" * 72)
    print(f"  reward = -{TRAJ_L2_WEIGHT}·(ADE/{ADE_THRESHOLD}) + {COMFORT_WEIGHT}·comfort")
    print("  其中 comfort 已经是 [-1, 0]。\n")
    print(f"  {'ADE':>7}{'comfort=0':>15}{'comfort=-0.5':>17}{'comfort=-1':>15}")
    print("  " + "-" * 57)
    for ade in [0.0, 1.0, 2.5, 2.99, 3.0, 3.5]:
        values = [reward(ade, comfort) for comfort in (0.0, -0.5, -1.0)]
        print(f"  {ade:>7.2f}{values[0]:>15.3f}{values[1]:>17.3f}{values[2]:>15.3f}")

    print("\n② ADE gate")
    print("  ADE=2.99 ->", f"{reward(2.99, 0.0):.3f}")
    print("  ADE=3.00 ->", f"{reward(3.00, 0.0):.3f}")
    print("  代码使用严格的 l2_dist < 3.0，所以 3.0 处直接变成 -1。")
    print("  这可能帮助优化先区分‘可接受/不可接受’，但不是平滑的距离函数。")

    print("\n③ 不使用 gate 时的对照")
    print("  ADE=3.5, gate=True  ->", f"{reward(3.5, 0.0):.3f}")
    print("  ADE=3.5, gate=False ->", f"{reward(3.5, 0.0, use_gate=False):.3f}")
    print("  下一课 rl2c：为什么 reasoning reward 需要额外的 CoC grader 和 gate。")
