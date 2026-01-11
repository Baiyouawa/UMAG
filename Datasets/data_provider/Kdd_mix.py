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


def miss_prob(data: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    sum_all = data @ alpha
    cross = sum_all.unsqueeze(1) - data * alpha.unsqueeze(0)
    logits = cross + data * beta.unsqueeze(0) + gamma.unsqueeze(0)
    return torch.sigmoid(logits)


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    kdd_path = data_dir / "KDD_norm.csv"

    data_np = np.loadtxt(kdd_path, delimiter=",")
    data = torch.tensor(data_np, dtype=torch.float32)
    # 用于计算缺失概率时不让 NaN 传播
    data_for_prob = torch.nan_to_num(data, nan=0.0)
    n, d = data.shape
    print(f"Loaded KDD_norm: shape={data.shape}")

    # humidity features are the 9th feature in each 11-feature block.
    humidity_idx = [8 + 11 * k for k in range(9)]

    for p in mixed_missing_ps:
        for seed in seeds:
            set_seed(seed)

            # MAR: humidity drives other features missing (alpha on humidity only, beta=0).
            alpha_mar = torch.zeros(d)
            alpha_mar[humidity_idx] = 1.0
            beta_mar = torch.zeros(d)
            gamma_mar = torch.zeros(d)
            prob_mar = miss_prob(data_for_prob, alpha_mar, beta_mar, gamma_mar)

            # MNAR: feature vanishes when its own value is high (beta>0).
            alpha_mnar = torch.zeros(d)
            beta_mnar = torch.ones(d)
            gamma_mnar = torch.zeros(d)
            prob_mnar = miss_prob(data_for_prob, alpha_mnar, beta_mnar, gamma_mnar)

            # Mixed MAR+MNAR, calibrated to the target overall missing rate.
            prob_mix = calibrate_prob((prob_mar + prob_mnar) / 2, p)
            mask_mix = (torch.rand_like(data) > prob_mix).float()

            masked = data.clone()
            masked = torch.nan_to_num(masked, nan=-200.0)
            masked[mask_mix == 0] = -200

            mask_path = data_dir / f"KDD_mask_mix_p{int(p*100)}_seed{seed}.npy"
            masked_path = data_dir / f"KDD_mix_p{int(p*100)}_seed{seed}.csv"

            np.save(mask_path, mask_mix.numpy())
            np.savetxt(masked_path, masked.numpy(), delimiter=",", fmt="%.6f")

            miss_ratio = 1 - mask_mix.mean().item()
            print(
                f"MIX p={p} seed={seed} -> mask {mask_mix.shape}, "
                f"missing~{miss_ratio:.3f}, saved to {mask_path.name}"
            )


if __name__ == "__main__":
    main()