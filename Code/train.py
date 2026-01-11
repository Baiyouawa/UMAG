import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import timesnet_train
import fgti_train
import brits_train
import csdi_train
import mtsci_train
import grin_train
import spin_train
import imputeformer_train
import pristi_train


DATA_DEFAULTS = {
    "kdd": {"seq_len": 48, "enc_in": 99, "c_out": 99},
    "guangzhou": {"seq_len": 48, "enc_in": 214, "c_out": 214},
    "physio": {"seq_len": 48, "enc_in": 40, "c_out": 40},
}

MODEL_REGISTRY = {
    "timesnet": (timesnet_train.diffusion_train, timesnet_train.diffusion_test),
    "fgti": (fgti_train.diffusion_train, fgti_train.diffusion_test),
    "brits": (brits_train.diffusion_train, brits_train.diffusion_test),
    "csdi": (csdi_train.diffusion_train, csdi_train.diffusion_test),
    "mtsci": (mtsci_train.diffusion_train, mtsci_train.diffusion_test),
    "grin": (grin_train.diffusion_train, grin_train.diffusion_test),
    "spin": (spin_train.diffusion_train, spin_train.diffusion_test),
    "imputeformer": (imputeformer_train.diffusion_train, imputeformer_train.diffusion_test),
    "pristi": (pristi_train.diffusion_train, pristi_train.diffusion_test),
}

# 按模型划分默认训练轮次与学习率
MODEL_DEFAULTS = {
    "timesnet": {"epoch_diff": 120, "learning_rate_diff": 5e-3},
    "fgti": {"epoch_diff": 250, "learning_rate_diff": 1e-3},
    "brits": {"epoch_diff": 200, "learning_rate_diff": 1e-3},
    "csdi": {"epoch_diff": 200, "learning_rate_diff": 1e-3},
    "mtsci": {"epoch_diff": 200, "learning_rate_diff": 1e-3},
    "grin": {"epoch_diff": 200, "learning_rate_diff": 1e-3},
    "spin": {"epoch_diff": 200, "learning_rate_diff": 1e-3},
    "imputeformer": {"epoch_diff": 200, "learning_rate_diff": 1e-3},
    "pristi": {"epoch_diff": 200, "learning_rate_diff": 1e-3},
}


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch: int = 32
    dataset: str = "kdd"
    model: str = "timesnet"
    missing_rate: float = 0.2
    seed: int = 3407
    seq_len: int = 48
    enc_in: int = 99
    c_out: int = 99
    epoch_diff: Optional[int] = None
    learning_rate_diff: Optional[float] = None
    task_name: str = "imputation"
    d_model: int = 128
    e_layers: int = 2
    d_ff: int = 2048
    n_heads: int = 8
    d_layers: int = 1
    top_k: int = 5
    num_kernels: int = 6
    embed: str = "timeF"
    freq: str = "h"
    dropout: float = 0.2
    pred_len: int = 0
    label_len: int = 0
    num_class: int = 0
    data_root: str = str(Path(__file__).resolve().parents[1] / "Datasets" / "data")
    results_root: str = str(Path(__file__).resolve().parents[1] / "Results")
    train_ratio: float = 0.8
    # FGTI 专用超参，增加默认值以避免缺字段
    timeemb: int = 128           # 可设为与 d_model 相同
    featureemb: int = 32
    diffusion_step_num: int = 50
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = "quad"       # 仅支持 quad / linear
    channel: int = 64
    residual_layers: int = 4
    proj_t: int = 16
    nheads: int = 8              # FGTI 用的小写 nheads
    # BRITS 专用超参
    rnn_hid_size: int = 100
    impute_weight: float = 3.0
    consistency_weight: float = 0.1
    # CSDI 专用超参
    diffusion_embedding_dim: int = 128
    csdi_layers: int = 4
    is_unconditional: bool = False
    # MTSCI 专用超参
    mtsci_layers: int = 4
    # GRIN 专用超参
    grin_d_hidden: int = 64
    grin_d_ff: int = 64
    grin_ff_dropout: float = 0.0
    grin_layers: int = 1
    grin_kernel_size: int = 2
    grin_decoder_order: int = 1
    grin_d_emb: int = 8
    grin_layer_norm: bool = False
    grin_global_att: bool = False
    # SPIN 专用超参
    spin_hidden: int = 64
    spin_layers: int = 4
    spin_heads: int = 4
    # ImputeFormer 专用超参
    imputeformer_d_model: int = 64
    imputeformer_heads: int = 4
    imputeformer_layers: int = 3
    imputeformer_dropout: float = 0.1
    # PriSTI 专用超参
    pristi_layers: int = 4
    pristi_dropout: float = 0.0
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified training entry")
    p.add_argument("--device", type=str, default=Config.device)
    p.add_argument("--batch", type=int, default=Config.batch)
    p.add_argument("--dataset", type=str, default=Config.dataset, choices=list(DATA_DEFAULTS.keys()))
    p.add_argument("--model", type=str, default=Config.model, choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--missing_rate", type=float, default=Config.missing_rate, help="e.g. 0.2/0.4/0.6")
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--enc_in", type=int, default=None)
    p.add_argument("--c_out", type=int, default=None)
    p.add_argument("--epoch_diff", type=int, default=None)
    p.add_argument("--learning_rate_diff", type=float, default=None)
    p.add_argument("--task_name", type=str, default=Config.task_name)
    p.add_argument("--d_model", type=int, default=Config.d_model)
    p.add_argument("--e_layers", type=int, default=Config.e_layers)
    p.add_argument("--d_ff", type=int, default=Config.d_ff)
    p.add_argument("--n_heads", type=int, default=Config.n_heads)
    p.add_argument("--d_layers", type=int, default=Config.d_layers)
    p.add_argument("--top_k", type=int, default=Config.top_k)
    p.add_argument("--num_kernels", type=int, default=Config.num_kernels)
    p.add_argument("--embed", type=str, default=Config.embed)
    p.add_argument("--freq", type=str, default=Config.freq)
    p.add_argument("--dropout", type=float, default=Config.dropout)
    p.add_argument("--pred_len", type=int, default=Config.pred_len)
    p.add_argument("--label_len", type=int, default=Config.label_len)
    p.add_argument("--num_class", type=int, default=Config.num_class)
    p.add_argument("--data_root", type=str, default=Config.data_root)
    p.add_argument("--results_root", type=str, default=Config.results_root)
    p.add_argument("--train_ratio", type=float, default=Config.train_ratio, help="split ratio for train set")
    # 新增 FGTI 相关参数，可按需覆盖
    p.add_argument("--timeemb", type=int, default=Config.timeemb)
    p.add_argument("--featureemb", type=int, default=Config.featureemb)
    p.add_argument("--diffusion_step_num", type=int, default=Config.diffusion_step_num)
    p.add_argument("--beta_start", type=float, default=Config.beta_start)
    p.add_argument("--beta_end", type=float, default=Config.beta_end)
    p.add_argument("--schedule", type=str, default=Config.schedule, choices=["quad", "linear"])
    p.add_argument("--channel", type=int, default=Config.channel)
    p.add_argument("--residual_layers", type=int, default=Config.residual_layers)
    p.add_argument("--proj_t", type=int, default=Config.proj_t)
    p.add_argument("--nheads", type=int, default=Config.nheads)
    # BRITS
    p.add_argument("--rnn_hid_size", type=int, default=Config.rnn_hid_size)
    p.add_argument("--impute_weight", type=float, default=Config.impute_weight)
    p.add_argument("--consistency_weight", type=float, default=Config.consistency_weight)
    # CSDI
    p.add_argument("--diffusion_embedding_dim", type=int, default=Config.diffusion_embedding_dim)
    p.add_argument("--csdi_layers", type=int, default=Config.csdi_layers)
    p.add_argument("--is_unconditional", action="store_true", help="CSDI unconditional mode")
    # MTSCI
    p.add_argument("--mtsci_layers", type=int, default=Config.mtsci_layers)
    # GRIN
    p.add_argument("--grin_d_hidden", type=int, default=Config.grin_d_hidden)
    p.add_argument("--grin_d_ff", type=int, default=Config.grin_d_ff)
    p.add_argument("--grin_ff_dropout", type=float, default=Config.grin_ff_dropout)
    p.add_argument("--grin_layers", type=int, default=Config.grin_layers)
    p.add_argument("--grin_kernel_size", type=int, default=Config.grin_kernel_size)
    p.add_argument("--grin_decoder_order", type=int, default=Config.grin_decoder_order)
    p.add_argument("--grin_d_emb", type=int, default=Config.grin_d_emb)
    p.add_argument("--grin_layer_norm", action="store_true")
    p.add_argument("--grin_global_att", action="store_true")
    # SPIN
    p.add_argument("--spin_hidden", type=int, default=Config.spin_hidden)
    p.add_argument("--spin_layers", type=int, default=Config.spin_layers)
    p.add_argument("--spin_heads", type=int, default=Config.spin_heads)
    # ImputeFormer
    p.add_argument("--imputeformer_d_model", type=int, default=Config.imputeformer_d_model)
    p.add_argument("--imputeformer_heads", type=int, default=Config.imputeformer_heads)
    p.add_argument("--imputeformer_layers", type=int, default=Config.imputeformer_layers)
    p.add_argument("--imputeformer_dropout", type=float, default=Config.imputeformer_dropout)
    # PriSTI
    p.add_argument("--pristi_layers", type=int, default=Config.pristi_layers)
    p.add_argument("--pristi_dropout", type=float, default=Config.pristi_dropout)
    return p

def apply_dataset_defaults(cfg):
    defaults = DATA_DEFAULTS[cfg.dataset]
    if cfg.seq_len is None:
        cfg.seq_len = defaults["seq_len"]
    if cfg.enc_in is None:
        cfg.enc_in = defaults["enc_in"]
    if cfg.c_out is None:
        cfg.c_out = defaults["c_out"]


def apply_model_defaults(cfg):
    defaults = MODEL_DEFAULTS[cfg.model]
    if cfg.epoch_diff is None:
        cfg.epoch_diff = defaults["epoch_diff"]
    if cfg.learning_rate_diff is None:
        cfg.learning_rate_diff = defaults["learning_rate_diff"]


def main():
    parser = build_parser()
    cfg = parser.parse_args()
    apply_dataset_defaults(cfg)
    apply_model_defaults(cfg)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    train_fn, test_fn = MODEL_REGISTRY[cfg.model]

    print(f"[Run] model={cfg.model} dataset={cfg.dataset} miss={cfg.missing_rate} seed={cfg.seed}")
    model = train_fn(cfg)
    test_fn(cfg, model)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
