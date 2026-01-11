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


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    physio_path = data_dir / "Physio_norm.csv"

    data_np = np.loadtxt(physio_path, delimiter=",")
    data = torch.tensor(data_np, dtype=torch.float32)

    # 用于概率计算：NaN 或 -200 视为 0，避免 NaN 传播/极端值放大
    data_for_prob = torch.where(torch.isnan(data), torch.zeros_like(data), data)
    data_for_prob = torch.where(data_for_prob == -200, torch.zeros_like(data_for_prob), data_for_prob)

    n, d = data.shape
    # 列索引（去掉 RecordID, Time, In-hospital_death 后的次序）
    HR_IDX = 14       # HR
    LACTATE_IDX = 16  # Lactate

    print(f"Loaded Physio_norm: shape={data.shape}, HR idx={HR_IDX}, Lactate idx={LACTATE_IDX}")

    for p in mixed_missing_ps:
        for seed in seeds:
            set_seed(seed)

            hr = data_for_prob[:, HR_IDX : HR_IDX + 1]
            lact = data_for_prob[:, LACTATE_IDX : LACTATE_IDX + 1]

            # MAR：HR 越高，所有变量都更可能缺失（整体相关性）
            prob_mar = torch.sigmoid(hr).expand(-1, d)

            # MNAR：乳酸值自身越高，越可能缺失
            prob_mnar = torch.zeros_like(data)
            prob_mnar[:, LACTATE_IDX : LACTATE_IDX + 1] = torch.sigmoid(lact)

            prob_mix = calibrate_prob((prob_mar + prob_mnar) / 2, p)
            mask = (torch.rand_like(data) > prob_mix).float()

            masked = data.clone()
            masked = torch.nan_to_num(masked, nan=-200.0)
            masked[mask == 0] = -200

            mask_path = data_dir / f"Physio_mask_mix_p{int(p*100)}_seed{seed}.npy"
            masked_path = data_dir / f"Physio_mix_p{int(p*100)}_seed{seed}.csv"

            np.save(mask_path, mask.numpy())
            np.savetxt(masked_path, masked.numpy(), delimiter=",", fmt="%.6f")

            miss_ratio = 1 - mask.mean().item()
            print(
                f"MIX p={p} seed={seed} -> mask {mask.shape}, "
                f"missing~{miss_ratio:.3f}, saved to {mask_path.name}"
            )


if __name__ == "__main__":
    main()
