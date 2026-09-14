"""玩具版离散轨迹 tokenizer —— 镜像真实的 DiscreteTrajectoryTokenizer

用法：`from toy_traj_tokens import encode, decode, NUM_BINS, N_WAYPOINTS`

真实代码 `src/alpamayo1_5/action_space/discrete_action_space.py:47` 的 encode 就四步：

    1. action = self.action_space.traj_to_action(hist_xyz, hist_rot, fut_xyz, fut_rot)
       → 把轨迹转成动作 (加速度, 曲率)
    2. action = (action - dims_min) / (dims_max - dims_min)      归一化到 [0,1]
    3. action = (action * (num_bins - 1)).round().long()         量化到 bin
    4. action = action.clamp(0, num_bins - 1)
       → reshape(B, -1)   形状 (B, tokens_per_trajectory)

decode 是反过来两步（`discrete_action_space.py:80`）：

    action = tokens / (num_bins - 1)
    action = action * (dims_max - dims_min) + dims_min
    → action_to_traj(action, ...)  再积分回轨迹

★ 关键：一个 token 就是一个【bin 刻度】，单独看没有语义。
  一条轨迹由 tokens_per_trajectory 个 token 组成。

  真实：tokens_per_trajectory = 128（64 waypoint × 2 维），num_bins = 3000
        release config 里 dims = [[-10, -10], [10, 10]]（注意：动作是【归一化后】的量，
        所以两个维度的范围都写成 ±10）
  toy：tokens_per_trajectory = 16（8 waypoint × 2 维），num_bins = 16

【toy 的两处简化】
  - 跳过第 1 步：toy 直接拿动作（不做 traj_to_action 那套积分/最小二乘）。
    真实必须先有轨迹，因为数据集给的是 xyz，不是 (加速度, 曲率)。
  - dims 用【动作空间自己的界】(±9.8, ±0.33)，而不是真实那种「归一化后的 ±10」。
    两者都是「先定一个范围再等分」，机制相同；toy 用物理量纲更直观。
"""

import torch

N_WAYPOINTS = 8          # 真实 64
DIMS = 2                 # (加速度, 曲率)
NUM_BINS = 64            # 真实 3000。★ 这个数直接决定量化误差，见 train3b
DIMS_MIN = (-9.8, -0.33)
DIMS_MAX = (+9.8, +0.33)

TOKENS_PER_TRAJ = N_WAYPOINTS * DIMS          # 16，真实是 128


def bin_names(num_bins=NUM_BINS):
    """造出 `<i0>` … `<i{n-1}>` —— 真实就是这么造的：
    `discrete_tokens = [f"<i{v}>" for v in range(self.traj_vocab_size)]`
    """
    return [f"<i{v}>" for v in range(num_bins)]


def encode(action, num_bins=NUM_BINS):
    """动作 → 离散 token。action: (B, N_WAYPOINTS, 2) → tokens: (B, N_WAYPOINTS*2)

    和真实完全一致的三步（归一化 → 量化 → clamp）。
    """
    dims_min = torch.tensor(DIMS_MIN, dtype=action.dtype, device=action.device)
    dims_max = torch.tensor(DIMS_MAX, dtype=action.dtype, device=action.device)
    norm = (action - dims_min) / (dims_max - dims_min)        # → [0,1]
    bins = (norm * (num_bins - 1)).round().long()             # → 整数 bin
    bins = bins.clamp(0, num_bins - 1)
    return bins.reshape(action.shape[0], -1)                  # (B, 16)


def decode(tokens, num_bins=NUM_BINS):
    """离散 token → 动作。tokens: (B, N_WAYPOINTS*2) → (B, N_WAYPOINTS, 2)

    注意要先 reshape 再算 —— 否则 (B,16) 和 (2,) 广播会报错。
    """
    dims_min = torch.tensor(DIMS_MIN, dtype=torch.float32, device=tokens.device)
    dims_max = torch.tensor(DIMS_MAX, dtype=torch.float32, device=tokens.device)
    norm = tokens.float().reshape(-1, N_WAYPOINTS, DIMS) / (num_bins - 1)
    return norm * (dims_max - dims_min) + dims_min


def resolution(num_bins=NUM_BINS):
    """每个 bin 对应多大的动作间隔（物理单位）——真实教程 real4 打印的就是这个。"""
    return tuple((hi - lo) / (num_bins - 1) for lo, hi in zip(DIMS_MIN, DIMS_MAX))


if __name__ == "__main__":
    print(f"  waypoint 数    : {N_WAYPOINTS}（真实 64）")
    print(f"  num_bins       : {NUM_BINS}（真实 3000）")
    print(f"  token/轨迹     : {TOKENS_PER_TRAJ}（真实 128）")
    print(f"  dims           : [{list(DIMS_MIN)}, {list(DIMS_MAX)}]")
    r = resolution()
    print(f"  量化分辨率     : 加速度 {r[0]:.5f} m/s²  |  曲率 {r[1]:.6f} 1/m")
    print(f"  token 名       : {bin_names()[:4]} ... {bin_names()[-1]}")
