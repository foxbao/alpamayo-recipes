"""rl3b: 沿着 rl1/rl2 的数据流阅读真实 RL 代码。

这是代码导航课，不要求一次读完 ``recipes/alpamayo1_x_rl``。先看最短路径，
再按需要展开 rollout、trainer、FSDP 和权重交接等工程细节。
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.abspath(os.path.join(HERE, "..", "recipes", "alpamayo1_x_rl"))

READING_PATH = [
    ("toml/alpamayo_rvla_rl_local_test.toml", "先看 G=12、clip、KL、同步频率"),
    ("rewards/traj_reward.py", "rollout 如何变成 ADE"),
    ("rewards/aggregated_reward.py", "comfort、权重和 ADE gate"),
    ("utils/trajectory_decode.py", "离散轨迹 token 如何还原为 xyz"),
    ("models/reasoning_vla/rollout.py", "只读 rollout_generation：采样和 log-prob 前的数据"),
    ("models/reasoning_vla/trainer.py", "只读 step_training：current/old/ref log-prob 和 GRPO loss"),
    ("models/reasoning_vla/alpamayo_cosmos_rl_post_training_entry.py",
     "最后看 Cosmos-RL 如何注册各组件"),
]

CONCEPTS = [
    ("组大小 G", "toml/alpamayo_rvla_rl_local_test.toml", "[rollout] n_generation = 12"),
    ("advantage", "models/reasoning_vla/trainer.py", "rollout.advantage"),
    ("GRPO loss", "models/reasoning_vla/trainer.py", "cosmos_rl...grpo_trainer.compute_loss"),
    ("ADE", "rewards/traj_reward.py", "calculate_ade"),
    ("comfort", "rewards/comfort_reward.py", "COMFORT_METRIC_CONFIG_DICT / _within_bound"),
    ("reasoning reward", "rewards/aggregated_reward_with_reasoning.py", "reasoning_score + CoT gate"),
]


def line_count(relative_path):
    path = os.path.join(RL, relative_path)
    return sum(1 for _ in open(path, encoding="utf-8")) if os.path.exists(path) else 0


if __name__ == "__main__":
    print("=" * 72)
    print("① 最短阅读路径")
    print("=" * 72)
    for index, (path, reason) in enumerate(READING_PATH, 1):
        mark = "✅" if os.path.exists(os.path.join(RL, path)) else "❌"
        print(f"  {index}. {mark} {path:<58} {reason}")

    print("\n② rl1 / rl2 概念落点")
    print("=" * 72)
    for concept, path, symbol in CONCEPTS:
        print(f"  {concept:<16} {path:<52} {symbol}")

    print("\n③ 第一遍可以跳过")
    print("=" * 72)
    print("  · prefetch/server.py：性能工程，不改变 reward 或 policy loss")
    print("  · utils/fsdp.py：分布式参数切分")
    print("  · weight_mapper.py / weight_loading.py：checkpoint 交接")
    print("\n  如果要读 reasoning 分支，再看：")
    print("  · rewards/aggregated_reward_with_reasoning.py")
    print("  · models/reasoning_vla/alpamayo_cosmos_rl_post_training_reasoning_entry.py")
    print("\n  原来的 rl3_real_code.py 仍保留，适合最后做完整代码地图。")
