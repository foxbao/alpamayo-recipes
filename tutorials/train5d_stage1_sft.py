"""train5d: 最小 Stage-1 SFT 的完整组装。

5a/5b/5c 分别讲 attention、next-token 对齐、图片前缀。本课接上 train2~4 的
label mask、轨迹 token 和两段 loss。详细实现仍在 train5_sft_stage1.py。
"""

import torch

if __package__:
    from .train5_sft_stage1 import (
        DATA,
        V,
        VOCAB,
        TinyVLM,
        compute_loss,
        generate,
        make_sample,
        mask_between,
    )
else:
    from train5_sft_stage1 import (
        DATA,
        V,
        VOCAB,
        TinyVLM,
        compute_loss,
        generate,
        make_sample,
        mask_between,
    )


if __name__ == "__main__":
    # 固定 seed，让每次运行的 loss/生成结果都可比较；限制 toy attention 的线程开销。
    torch.manual_seed(0)
    torch.set_num_threads(4)
    samples = [make_sample(*item) for item in DATA]
    input_ids = torch.stack([item[0] for item in samples])
    images = torch.stack([item[1] for item in samples])
    sequences = [item[2] for item in samples]
    model = TinyVLM()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    print("① 输入样本（图片 + CoC + 轨迹 token）")
    for i, sequence in enumerate(sequences):
        print(f"  image[{i}]: {' '.join(sequence)}")
    print(f"  input_ids shape={tuple(input_ids.shape)}, image shape={tuple(images.shape)}")

    cot_mask = mask_between(input_ids[:1], V["<cot_start>"], V["<cot_end>"])[0]
    traj_mask = mask_between(input_ids[:1], V["<traj_start>"], V["<traj_end>"])[0]
    print("  scene 0 的监督区间:")
    print("    CoC:", [VOCAB[token.item()] for token, keep in zip(input_ids[0], cot_mask) if keep])
    print("    traj:", [VOCAB[token.item()] for token, keep in zip(input_ids[0], traj_mask) if keep])
    print("    （真正的 next-token label 还要再向右错一位；train5b 已单独演示。）")

    print("\n② 一次前向后的损失拆分")
    for step in range(801):
        total, trajectory, others = compute_loss(model, input_ids, images)
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        if step in (0, 100, 300, 600, 800):
            print(f"  step {step:3d}: total={total.item():.4f}  "
                  f"trajectory={trajectory.item():.4f}  others={others.item():.4f}")

    print("\n③ 训练后自回归生成")
    for i, target in enumerate(sequences):
        generated = generate(model, images[i : i + 1], n_new=len(target) - 1)
        print(f"  scene {i}: {' '.join(generated)}")
        print(f"           target: {' '.join(target)}")
    print("\n完整模型看不懂时，回到对应的 5a/5b/5c，而不是同时调试所有概念。")
