from pathlib import Path
from typing import Tuple
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split
import numpy as np
import torch
import torchcde

# 默认数据根目录，可由 configs.data_root 覆盖
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "Datasets" / "data"


def _resolve_paths(configs, data_name: str, mask_name: str) -> Tuple[Path, Path]:
    data_root = Path(getattr(configs, "data_root", DEFAULT_DATA_ROOT))
    data_path = data_root / data_name
    mask_path = data_root / mask_name
    return data_path, mask_path


def _load_data_and_mask(data_path: Path, mask_path: Path, seq_len: int, enc_in: int, missing_rate: float, seed: int):
    data = np.loadtxt(data_path, delimiter=",")
    if data.size == 0:
        raise ValueError(f"Data file is empty: {data_path}")
    total_len = (data.shape[0] // seq_len) * seq_len
    data = data[:total_len].reshape(-1, seq_len, enc_in)

    if missing_rate == 0:
        mask = np.ones_like(data)
        mask[(data == -200) | np.isnan(data)] = 0
    else:
        if mask_path.suffix == ".npy":
            try:
                mask = np.load(mask_path)
            except Exception:
                # 兼容意外命名为 .npy 但实际是文本的情况
                mask = np.loadtxt(mask_path, delimiter=",")
        else:
            mask = np.loadtxt(mask_path, delimiter=",")
        if mask.size == 0:
            raise ValueError(f"Mask file is empty: {mask_path}")
        mask = mask[:total_len].reshape(-1, seq_len, enc_in)
    return data, mask


def _prep_tensors(data: np.ndarray, mask: np.ndarray, configs):
    mask_gt = np.ones_like(data)
    mask_gt[(data == -200) | np.isnan(data)] = 0

    # 保证模拟掩码不把原始缺失当作可见：与 mask_gt 交叉
    mask = np.where(mask_gt == 1, mask, 0)

    data = np.where((data == -200) | np.isnan(data), 0.0, data)

    dataf = np.array(data)
    dataf[np.where(mask == 0)] = np.nan
    dataf = torch.from_numpy(dataf).float()
    dataf = torchcde.linear_interpolation_coeffs(dataf)

    maxdataf = dataf.clone().detach()

    if hasattr(configs, "flimit"):
        temp_list = []
        for j in range(dataf.shape[2]):
            x = dataf[:, :, j]
            xf = torch.fft.rfft(x)
            pass_f = torch.abs(torch.fft.rfftfreq(x.shape[1])) > configs.flimit
            rx = torch.fft.irfft(xf * pass_f, n=x.shape[1])
            temp_list.append(rx)
        dataf = torch.stack(temp_list, dim=2)

        temp_list = []
        for j in range(maxdataf.shape[2]):
            x_j = maxdataf[:, :, j]
            xf = torch.fft.rfft(x_j)
            freq = abs(xf)
            __, toplist = torch.topk(freq, configs.topf)
            for x in range(xf.shape[0]):
                for y in range(xf.shape[1]):
                    if y not in toplist[x]:
                        xf[x, y] = 0
            rx = torch.fft.irfft(xf, n=x_j.shape[1])
            temp_list.append(rx)
        maxdataf = torch.stack(temp_list, dim=2)

        dataf = torch.stack([dataf, maxdataf], dim=-1).reshape(-1, configs.seq_len, configs.enc_in * 2)

    data = torch.from_numpy(data).float()
    mask = torch.from_numpy(mask).float()
    observed_tp = torch.from_numpy(np.arange(configs.seq_len)).float()
    mask_gt = torch.from_numpy(mask_gt).float()

    return data, dataf, mask, observed_tp, mask_gt


class KDD_DATASET(Dataset):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        data_path, mask_path = _resolve_paths(
            configs,
            "KDD_norm.csv",
            f"KDD_mask_mix_p{int(configs.missing_rate*100)}_seed{configs.seed}.npy",
        )
        data, mask = _load_data_and_mask(data_path, mask_path, configs.seq_len, configs.enc_in, configs.missing_rate, configs.seed)
        self.data, self.dataf, self.mask, self.observed_tp, self.mask_gt = _prep_tensors(data, mask, configs)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return (
            self.data[index],
            self.dataf[index],
            self.mask[index],
            self.observed_tp,
            self.mask_gt[index],
        )


class GUANGZHOU_DATASET(Dataset):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        data_path, mask_path = _resolve_paths(
            configs,
            "Guangzhou_norm.csv",
            f"Guangzhou_mask_mnar_p{int(configs.missing_rate*100)}_seed{configs.seed}.npy",
        )
        data, mask = _load_data_and_mask(data_path, mask_path, configs.seq_len, configs.enc_in, configs.missing_rate, configs.seed)
        self.data, self.dataf, self.mask, self.observed_tp, self.mask_gt = _prep_tensors(data, mask, configs)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return (
            self.data[index],
            self.dataf[index],
            self.mask[index],
            self.observed_tp,
            self.mask_gt[index],
        )


class PHYSIO_DATASET(Dataset):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        data_path, mask_path = _resolve_paths(
            configs,
            "Physio_norm.csv",
            f"Physio_mask_mix_p{int(configs.missing_rate*100)}_seed{configs.seed}.npy",
        )
        data, mask = _load_data_and_mask(data_path, mask_path, configs.seq_len, configs.enc_in, configs.missing_rate, configs.seed)
        self.data, self.dataf, self.mask, self.observed_tp, self.mask_gt = _prep_tensors(data, mask, configs)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return (
            self.data[index],
            self.dataf[index],
            self.mask[index],
            self.observed_tp,
            self.mask_gt[index],
        )

def get_physio_dataset(configs):
    dataset = PHYSIO_DATASET(configs)
    train_size = int(len(dataset) * getattr(configs, "train_ratio", 0.8))
    val_size = len(dataset) - train_size
    g = torch.Generator().manual_seed(configs.seed)
    train_ds, test_ds = random_split(dataset, [train_size, val_size], generator=g)
    train_loader = DataLoader(train_ds, batch_size=configs.batch, num_workers=0, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=configs.batch, num_workers=0, shuffle=False)
    return train_loader, test_loader

def get_kdd_dataset(configs):
    dataset = KDD_DATASET(configs)
    train_size = int(len(dataset) * getattr(configs, "train_ratio", 0.8))
    val_size = len(dataset) - train_size
    g = torch.Generator().manual_seed(configs.seed)
    train_ds, test_ds = random_split(dataset, [train_size, val_size], generator=g)
    train_loader = DataLoader(train_ds, batch_size=configs.batch, num_workers=0, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=configs.batch, num_workers=0, shuffle=False)
    return train_loader, test_loader

def get_guangzhou_dataset(configs):
    dataset = GUANGZHOU_DATASET(configs)
    train_size = int(len(dataset) * getattr(configs, "train_ratio", 0.8))
    val_size = len(dataset) - train_size
    g = torch.Generator().manual_seed(configs.seed)
    train_ds, test_ds = random_split(dataset, [train_size, val_size], generator=g)
    train_loader = DataLoader(train_ds, batch_size=configs.batch, num_workers=0, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=configs.batch, num_workers=0, shuffle=False)
    return train_loader, test_loader


def get_dataset(configs):
    if configs.dataset == "kdd":
        return get_kdd_dataset(configs)
    if configs.dataset == "physio":
        return get_physio_dataset(configs)
    if configs.dataset == "guangzhou":
        return get_guangzhou_dataset(configs)