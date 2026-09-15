"""train7b: cache crop 如何阻止 teacher-forcing 标签泄漏。

阶段二训练的 input_ids 中，`<traj_future_start>` 后仍有真值轨迹 token。若把整条
序列的 cache 交给 expert，它就能读答案；而推理时这些 token 不存在。
本课只替换答案部分，比较完整 cache 与 crop 后 cache 对 expert 输出的影响。
"""

import torch

if __package__:
    from .tutorial_support.stage2_model import (
        DATA, V, VOCAB, TinyExpert, TinyVLM, crop_caches, make_sample
    )
else:
    from tutorial_support.stage2_model import (
        DATA, V, VOCAB, TinyExpert, TinyVLM, crop_caches, make_sample
    )


if __name__ == "__main__":
    samples = [make_sample(*item) for item in DATA]
    first_ids, first_image, _ = samples[0]
    second_ids, _, _ = samples[1]
    start = int((first_ids == V["<traj_future_start>"]).nonzero()[0, 0])
    crop_at = start + 2  # 图片 token + 保留 traj_future_start
    changed_ids = first_ids.clone()
    changed_ids[start + 1 :] = second_ids[start + 1 :]

    vlm = TinyVLM()
    expert = TinyExpert()
    action = torch.randn(1, 8, 2)
    time = torch.tensor([0.5])
    _, full_a = vlm(first_ids[None], first_image[None])
    _, full_b = vlm(changed_ids[None], first_image[None])
    crop_a = crop_caches(full_a, crop_at)
    crop_b = crop_caches(full_b, crop_at)
    out_full_a = expert(action, time, full_a)
    out_full_b = expert(action, time, full_b)
    out_crop_a = expert(action, time, crop_a)
    out_crop_b = expert(action, time, crop_b)

    print("完整序列:", " ".join(VOCAB[token.item()] for token in first_ids))
    print("替换答案:", " ".join(VOCAB[token.item()] for token in changed_ids))
    print(f"<traj_future_start> 下标={start}; cache.crop({crop_at})")
    print("保留前缀:", " ".join(VOCAB[token.item()] for token in first_ids[: start + 1]))
    print("\n替换答案后 expert 输出的最大变化:")
    print(f"  不 crop: {(out_full_a - out_full_b).abs().max().item():.6f}")
    print(f"  crop 后: {(out_crop_a - out_crop_b).abs().max().item():.6f}")
    print("不 crop 时输出会读到答案；crop 后只保留 prompt/CoC/起始标记，训练条件才和推理一致。")
