"""
Compute per-feature mean/std on raw data for each dataset and save to .npy.
Outputs (in Datasets/data):
  KDD_mean.npy / KDD_std.npy
  guangzhou_mean.npy / guangzhou_std.npy
  physio_mean.npy / physio_std.npy
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio


def save_stats(mean: np.ndarray, std: np.ndarray, prefix: Path):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(prefix.parent / f"{prefix.name}_mean.npy", mean)
    np.save(prefix.parent / f"{prefix.name}_std.npy", std)
    print(f"Saved: {prefix.name}_mean.npy, {prefix.name}_std.npy; shape={mean.shape}")


def stats_kdd(data_dir: Path):
    # 强制为数值，无法解析的转为 NaN，避免 object 类型导致 isnan 报错
    raw_df = pd.read_csv(data_dir / "KDD.csv", delimiter=",", header=0)
    raw_df = raw_df.apply(pd.to_numeric, errors="coerce")
    raw = raw_df.to_numpy()

    cols = []
    for i in range(9):
        cols.append(raw[:, i * 13 + 2 : (i + 1) * 13])
    data = np.stack(cols, axis=1).reshape(raw.shape[0], -1).astype(float)  # (N, 99)
    # data 已是 float，可直接 nan 统计
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    std[std == 0] = 1.0
    save_stats(mean, std, data_dir / "kdd")


def stats_guangzhou(data_dir: Path):
    mat = sio.loadmat(data_dir / "tensor.mat")
    tensor = mat["tensor"]
    data = tensor.reshape(tensor.shape[0], -1).T  # (8784, 214)
    data = data.astype(float)
    data[data == 0] = np.nan  # original missing
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    std[std == 0] = 1.0
    save_stats(mean, std, data_dir / "guangzhou")


def stats_physio(data_dir: Path):
    df = pd.read_csv(data_dir / "physio_set_ab_raw.csv")
    feats = df.drop(columns=["RecordID", "Time", "In-hospital_death"])
    feats = feats.replace(-1, np.nan).astype(float)
    mean = feats.mean(skipna=True).to_numpy()
    std = feats.std(skipna=True, ddof=0).to_numpy()
    std[std == 0] = 1.0
    save_stats(mean, std, data_dir / "physio")


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    stats_kdd(data_dir)
    stats_guangzhou(data_dir)
    stats_physio(data_dir)


if __name__ == "__main__":
    main()
