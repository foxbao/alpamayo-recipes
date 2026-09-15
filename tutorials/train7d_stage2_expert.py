"""train7d: 阶段二 expert 的完整、可运行 toy 训练。

现在把前面几课串起来：VLM 得到 cache，crop 掉真值轨迹，expert 学连续动作的速度场，
最后从噪声积分并检查不同场景的平均曲率。真实代码更大，但训练信号就是这几步。
"""

import torch
import torch.nn.functional as F

if __package__:
    from .tutorial_support.stage2_model import (
        ACTION_DIM, BATCH_PER_SCENE, DATA, N_STEPS, N_WAYPOINTS, Stage2Model, V, VOCAB, make_sample,
        train_vlm,
    )
else:
    from tutorial_support.stage2_model import (
        ACTION_DIM, BATCH_PER_SCENE, DATA, N_STEPS, N_WAYPOINTS, Stage2Model, V, VOCAB, make_sample,
        train_vlm,
    )


def train_expert_toy(model, input_ids, images, targets, crop_at, steps=180):
    """只更新 expert；复制每个场景以降低 toy 数据的梯度噪声。"""
    model.vlm.eval()
    for parameter in model.vlm.parameters():
        parameter.requires_grad_(False)
    ids = input_ids.repeat(BATCH_PER_SCENE, 1)
    imgs = images.repeat(BATCH_PER_SCENE, 1)
    target = targets.repeat(BATCH_PER_SCENE, 1, 1)
    optimizer = torch.optim.Adam(model.expert.parameters(), lr=5e-3)
    history = []
    for step in range(steps):
        with torch.no_grad():
            caches, _ = model.cache_for_expert(ids, imgs, crop_at)
        x0 = torch.randn_like(target)
        t = torch.rand(target.shape[0])
        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * target
        velocity = model.expert(xt, t, caches)
        loss = F.mse_loss(velocity, target - x0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        if step in (0, 59, steps - 1):
            print(f"  expert step {step:3d}: flow-matching loss={loss.item():.4f}")
    return sum(history[-20:]) / 20


@torch.no_grad()
def sample_mean_kappa(expert, caches, samples=12):
    values = []
    for _ in range(samples):
        action = torch.randn(1, N_WAYPOINTS, ACTION_DIM)
        for step in range(N_STEPS):
            time = torch.full((1,), step / N_STEPS)
            action = action + expert(action, time, caches) / N_STEPS
        values.append(action[0, :, 1].mean().item())
    return sum(values) / len(values)


if __name__ == "__main__":
    torch.manual_seed(0)
    samples = [make_sample(*item) for item in DATA]
    input_ids = torch.stack([item[0] for item in samples])
    images = torch.stack([item[1] for item in samples])
    targets = torch.stack([item[2] for item in samples])
    start = int((input_ids[0] == V["<traj_future_start>"]).nonzero()[0, 0])
    crop_at = start + 2

    print("① 阶段一风格序列（teacher forcing 时答案仍在序列中）")
    print("  ", " ".join(VOCAB[token.item()] for token in input_ids[0]))
    print(f"  crop_at={crop_at}: 保留到 <traj_future_start>，裁掉后面的真值轨迹")

    print("\n② 先得到阶段一 VLM 条件")
    model = Stage2Model(cotrain_vlm=False, stop_grad_from_vlm=True)
    vlm_loss = train_vlm(model, input_ids, images, n_iters=180)
    print(f"  VLM next-token loss={vlm_loss:.4f}")

    print("\n③ 只训练 action expert")
    tail_loss = train_expert_toy(model, input_ids, images, targets, crop_at)
    print(f"  最后 20 步平均 flow-matching loss={tail_loss:.4f}")

    print("\n④ 从噪声积分，观察条件是否影响连续动作")
    for index in range(len(samples)):
        with torch.no_grad():
            _, cache = model.vlm(input_ids[index : index + 1], images[index : index + 1])
            cache = [(key[:, :, :crop_at].detach(), value[:, :, :crop_at].detach()) for key, value in cache]
        prediction = sample_mean_kappa(model.expert, cache)
        target = targets[index, 0, 1].item()
        print(f"  scene {index}: target kappa={target:+.3f}, sampled mean={prediction:+.3f}")
    print("阶段二的重点：VLM 提供条件，expert 学连续动作的速度场，而不是 token CE。")
