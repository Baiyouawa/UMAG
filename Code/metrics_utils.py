from pathlib import Path
import numpy as np
import torch


def load_mean_std(dataset: str, data_root: str, device: str = "cpu"):
    """
    Load per-feature mean/std (computed on raw data) for a dataset.
    Returns tensors shaped as (1, 1, K) for broadcast over (B, L, K).
    """
    root = Path(data_root)
    mean_path = root / f"{dataset}_mean.npy"
    std_path = root / f"{dataset}_std.npy"
    if not mean_path.exists() or not std_path.exists():
        raise FileNotFoundError(
            f"Missing stats files for dataset '{dataset}'. "
            f"Expected: {mean_path.name}, {std_path.name} under {root}"
        )
    mean = torch.from_numpy(np.load(mean_path)).float().to(device)
    std = torch.from_numpy(np.load(std_path)).float().to(device)
    return mean.view(1, 1, -1), std.view(1, 1, -1)


def calc_mape_denorm(
    target: torch.Tensor,
    forecast: torch.Tensor,
    eval_points: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    eps: float = 1e-6,
):
    """
    MAPE on original scale: denormalize with mean/std, then compute |err|/|true|.
    """
    eval_p = torch.where(eval_points == 1)
    target_denorm = target * std + mean
    forecast_denorm = forecast * std + mean
    denom = torch.clamp(torch.abs(target_denorm), min=eps)
    return torch.mean(torch.abs((target_denorm - forecast_denorm) / denom))
