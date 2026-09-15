"""rl2c: reasoning reward —— rollout 的 CoC 如何进入最终 reward。

真实 Alpamayo 使用外部 grader（示例是 Lingo-Judge）。为了让本课不依赖模型下载，
这里用预先给定的 toy grader 分数，只演示文本抽取、分数变换和 gate。
"""

ADE_THRESHOLD = 3.0
REASONING_THRESHOLD = -0.4
TRAJ_L2_WEIGHT = 0.2
COMFORT_WEIGHT = 0.0
REASONING_WEIGHT = 0.3


def extract_cot(completion):
    """toy 版 CoC 抽取；真实代码使用 extract_between_special_tokens。"""
    start = "<|cot_start|>"
    end = "<|cot_end|>"
    if start not in completion or end not in completion:
        return ""
    text = completion.split(start, 1)[1].split(end, 1)[0]
    return text.strip()


def normalize_reasoning_score(reasoning_score):
    """Map the accepted interval [-0.4, 0] to [0, 1] (higher is better)."""
    return (reasoning_score - REASONING_THRESHOLD) / (0.0 - REASONING_THRESHOLD)


def joint_reward(completion, ade, comfort, raw_score):
    cot = extract_cot(completion)
    reasoning_score = raw_score - 1.0
    passed = bool(cot) and reasoning_score > REASONING_THRESHOLD and ade < ADE_THRESHOLD
    if not passed:
        return -1.0, reasoning_score, False
    value = (
        -TRAJ_L2_WEIGHT * (ade / ADE_THRESHOLD)
        + COMFORT_WEIGHT * comfort
        + REASONING_WEIGHT * normalize_reasoning_score(reasoning_score)
    )
    return value, reasoning_score, True


if __name__ == "__main__":
    print("=" * 72)
    print("① CoC 是 rollout 完成后的整段评分，不是 token-level SFT 标签")
    print("=" * 72)
    print("  rollout completion -> 抽取 CoC -> grader -> reasoning_score -> gate\n")
    examples = [
        ("缺少 CoC", "answer only", 0.8, 0.9),
        ("分数太低", "<|cot_start|>unclear<|cot_end|>trajectory", 0.8, 0.4),
        ("刚好等于阈值", "<|cot_start|>borderline<|cot_end|>trajectory", 0.8, 0.6),
        ("通过", "<|cot_start|>clear scene explanation<|cot_end|>trajectory", 0.8, 0.8),
    ]
    print(f"  {'案例':<12}{'raw':>8}{'score':>10}{'通过 gate':>12}{'final reward':>15}")
    print("  " + "-" * 61)
    for name, completion, ade, raw_score in examples:
        final, score, passed = joint_reward(completion, ade, 0.0, raw_score)
        print(f"  {name:<12}{raw_score:>8.1f}{score:>10.1f}{str(passed):>12}{final:>15.3f}")

    print("\n② 为什么先减 1？")
    print("  grader 的 toy 原始分数假设在 [0, 1]；减 1 后变成约 [-1, 0]，")
    print("  可以和负向阈值 -0.4 直接比较。raw=0.6 时 score=-0.4，")
    print("  但代码使用严格的 >，所以等号仍然不通过。")
    print("  通过 gate 后再把 [-0.4, 0] 映射到 [0, 1]：阈值得 0，完美分数得 1。")
    print("\n  真实 joint reward 的入口：")
    print("    rewards/aggregated_reward_with_reasoning.py")
    print("    models/reasoning_vla/alpamayo_cosmos_rl_post_training_reasoning_entry.py")
    print("  下一课 rl3a：先理解 policy、rollout、controller 的系统分工。")
