from pathlib import Path

import numpy as np
import torch


mixed_missing_ps = [0.2, 0.4, 0.6]
seeds = [3407, 3408, 3409, 3410, 3411]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def calibrate_prob(prob: torch.Tensor, target_p: float) -> torch.Tensor:
    mean_p = prob.mean()
    if mean_p > 0:
        prob = prob * (target_p / mean_p)
    return prob.clamp(0.0, 0.95)


def miss_prob_mnar_low(data: torch.Tensor) -> torch.Tensor:
    # 速度过低 -> 易缺失，用负向自依赖模拟：值越小，-data 越大，sigmoid 越大
    logits = -data
    return torch.sigmoid(logits)


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    gz_path = data_dir / "Guangzhou_norm.csv"

    data_np = np.loadtxt(gz_path, delimiter=",")
    data = torch.tensor(data_np, dtype=torch.float32)
    # 计算缺失概率时，原有缺失标记 -200 视为 0，避免传播
    data_for_prob = torch.where(torch.isnan(data), torch.zeros_like(data), data)
    data_for_prob = torch.where(data_for_prob == -200, torch.zeros_like(data_for_prob), data_for_prob)

    n, d = data.shape
    print(f"Loaded Guangzhou_norm: shape={data.shape}")

    for p in mixed_missing_ps:
        for seed in seeds:
            set_seed(seed)

            prob_mnar = miss_prob_mnar_low(data_for_prob)
            prob_mnar = calibrate_prob(prob_mnar, p)
            mask = (torch.rand_like(data) > prob_mnar).float()

            masked = data.clone()
            masked = torch.nan_to_num(masked, nan=-200.0)
            masked[mask == 0] = -200

            mask_path = data_dir / f"Guangzhou_mask_mnar_p{int(p*100)}_seed{seed}.npy"
            masked_path = data_dir / f"Guangzhou_mnar_p{int(p*100)}_seed{seed}.csv"

            np.save(mask_path, mask.numpy())
            np.savetxt(masked_path, masked.numpy(), delimiter=",", fmt="%.6f")

            miss_ratio = 1 - mask.mean().item()
            print(
                f"MNAR p={p} seed={seed} -> mask {mask.shape}, "
                f"missing~{miss_ratio:.3f}, saved to {mask_path.name}"
            )


if __name__ == "__main__":
    main()
