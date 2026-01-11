
from pathlib import Path

from models import TimesNet
import metrics_utils
import A_dataset
from torch import optim
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

def diffusion_train(configs):
    #从kdd.norm中读取，划分出train，test
    train_loader, test_loader = A_dataset.get_dataset(configs)
    model = TimesNet.Model(configs).to(configs.device)
    model_optim = optim.Adam(model.parameters(), lr=configs.learning_rate_diff, weight_decay=1e-6)
    p1 = int(0.75 * configs.epoch_diff)
    p2 = int(0.9 * configs.epoch_diff)
    #当炼丹跑到75%和90%时，把学习率降低10倍
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        model_optim, milestones=[p1, p2], gamma=0.1
    )
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(configs.epoch_diff):
        iter_count = 0
        train_loss = []
        epoch_time = time.time()
        model_optim.zero_grad()

        for observed_data_cpu, observed_dataf_cpu, observed_mask_cpu, observed_tp_cpu, gt_mask_cpu in tqdm(
            train_loader, desc=f"Train {epoch+1}/{configs.epoch_diff}", leave=False
        ):
            # 在这里把所有需要用到的张量从 CPU 移动到 GPU
            observed_data = observed_data_cpu.to(configs.device)
            observed_mask = observed_mask_cpu.to(configs.device)
            gt_mask = gt_mask_cpu.to(configs.device)
            observed_dataf = observed_dataf_cpu.to(configs.device)

            iter_count += 1
            model_optim.zero_grad()
            x_enc = observed_data * observed_mask
            imputed_output = model(x_enc, None, None, None, mask=observed_mask)
            eval_mask = gt_mask - observed_mask
            loss = loss_fn(imputed_output[eval_mask == 1], observed_data[eval_mask == 1])
            loss.backward()
            model_optim.step()
            train_loss.append(loss.item())
        lr_scheduler.step()
        
        if epoch % 50 == 0 or epoch == configs.epoch_diff-1:
            train_loss = np.average(train_loss)
            print("Epoch: {}. Cost time: {}. Train_loss: {}.".format(epoch + 1, time.time() - epoch_time, train_loss))
    return model

def diffusion_test(configs, model):
    train_loader, test_loader = A_dataset.get_dataset(configs)
    model.eval()

    target_2d = []
    forecast_2d = []
    eval_p_2d = []
    generate_data2d = [] # <--- 我们把它加回来了，用于保存 .csv

    print("Testset sum: ", len(test_loader.dataset) // configs.batch + 1)

    for observed_data_cpu, observed_dataf_cpu, observed_mask_cpu, observed_tp_cpu, gt_mask_cpu in tqdm(
        test_loader, desc="Test", leave=False
    ):
        # 同样，在这里把所有需要用到的张量从 CPU 移动到 GPU
        observed_data = observed_data_cpu.to(configs.device)
        observed_mask = observed_mask_cpu.to(configs.device)
        gt_mask = gt_mask_cpu.to(configs.device)
        observed_dataf = observed_dataf_cpu.to(configs.device)

        # --- 核心 ---
        x_enc = observed_data * observed_mask
        with torch.no_grad():
            imputed_output = model(x_enc, None, None, None, mask=observed_mask)
        eval_mask = gt_mask - observed_mask
        
        imputed_sample = imputed_output.detach().to("cpu")
        observed_data = observed_data.detach().to("cpu")
        observed_mask = observed_mask.detach().to("cpu")
        gt_mask = gt_mask.detach().to("cpu")
        #for CRPS
        imputed_data = observed_mask * observed_data + (1 - observed_mask) * imputed_sample
        evalmask = gt_mask - observed_mask

        target_2d.append(observed_data)
        forecast_2d.append(imputed_data)
        eval_p_2d.append(evalmask) 

        B, L, K = imputed_data.shape
        temp = imputed_data.reshape(B*L, K).numpy()
        generate_data2d.append(temp)

    generate_data2d = np.vstack(generate_data2d)
    results_root = Path(getattr(configs, "results_root", Path(".")))
    results_root.mkdir(parents=True, exist_ok=True)
    out_path = results_root / f"TimeNet_{configs.dataset}_p{int(configs.missing_rate*100)}_seed{configs.seed}.csv"
    np.savetxt(out_path, generate_data2d, delimiter=",")
    print(f"TimeNet imputation results saved to {out_path}") # 打印提示

    target_2d = torch.cat(target_2d, dim=0)
    forecast_2d = torch.cat(forecast_2d, dim=0)
    eval_p_2d = torch.cat(eval_p_2d, dim=0)

    mean, std = metrics_utils.load_mean_std(
        dataset=configs.dataset,
        data_root=getattr(configs, "data_root", Path(__file__).resolve().parents[1] / "Datasets" / "data"),
        device=target_2d.device,
    )


    RMSE = calc_RMSE(target_2d, forecast_2d, eval_p_2d)
    MAE = calc_MAE(target_2d, forecast_2d, eval_p_2d)
    MAPE = calc_MAPE(target_2d, forecast_2d, eval_p_2d, mean, std)
    WASS = calc_Wasserstein(target_2d, forecast_2d, eval_p_2d)
    WASS_PER_DIM = calc_Wasserstein_per_dim(target_2d, forecast_2d, eval_p_2d)
    WASS_TAIL = calc_Wasserstein_tail(target_2d, forecast_2d, eval_p_2d, q=0.9)

    print("RMSE: ", RMSE)
    print("MAE: ", MAE)
    print("MAPE: ", MAPE)
    print("Wasserstein: ", WASS)
    print("Wasserstein_per_dim: ", WASS_PER_DIM)
    print("Wasserstein_tail_q90: ", WASS_TAIL)

def calc_RMSE(target, forecast, eval_points):
    eval_p = torch.where(eval_points == 1)
    error_mean = torch.mean((target[eval_p] - forecast[eval_p])**2)
    return torch.sqrt(error_mean)

def calc_MAE(target, forecast, eval_points):
    eval_p = torch.where(eval_points == 1)
    error_mean = torch.mean(torch.abs(target[eval_p] - forecast[eval_p]))
    return error_mean


def calc_MAPE(target, forecast, eval_points, mean, std, eps: float = 1e-6):
    eval_p = torch.where(eval_points == 1)
    target_denorm = target * std + mean
    forecast_denorm = forecast * std + mean
    denom = torch.clamp(torch.abs(target_denorm), min=eps)
    return torch.mean(torch.abs((target_denorm - forecast_denorm) / denom))


def calc_Wasserstein(target, forecast, eval_points):
    eval_p = torch.where(eval_points == 1)
    if eval_p[0].numel() == 0:
        return torch.tensor(float("nan"), device=target.device)
    t = target[eval_p].reshape(-1)
    f = forecast[eval_p].reshape(-1)
    t_sorted, _ = torch.sort(t)
    f_sorted, _ = torch.sort(f)
    # 对齐长度（理论上长度一致），并取平均绝对差作为一维Wasserstein距离
    m = min(t_sorted.shape[0], f_sorted.shape[0])
    return torch.mean(torch.abs(t_sorted[:m] - f_sorted[:m]))


def _wasserstein_1d(t: torch.Tensor, f: torch.Tensor):
    t_sorted, _ = torch.sort(t)
    f_sorted, _ = torch.sort(f)
    m = min(t_sorted.shape[0], f_sorted.shape[0])
    return torch.mean(torch.abs(t_sorted[:m] - f_sorted[:m]))


def calc_Wasserstein_per_dim(target, forecast, eval_points):
    # 对每个特征维分别计算 1D Wasserstein，再取平均，捕捉各维分布差异
    B, L, K = target.shape
    ws = []
    for k in range(K):
        mask_k = eval_points[:, :, k] == 1
        if mask_k.sum() == 0:
            continue
        t = target[:, :, k][mask_k]
        f = forecast[:, :, k][mask_k]
        ws.append(_wasserstein_1d(t, f))
    if len(ws) == 0:
        return torch.tensor(float("nan"), device=target.device)
    return torch.mean(torch.stack(ws))


def calc_Wasserstein_tail(target, forecast, eval_points, q: float = 0.9):
    """
    两侧尾部 Wasserstein：同时考虑低分位与高分位，兼顾“值低易缺失”的场景。
    """
    B, L, K = target.shape
    ws_tail = []
    q_low = 1 - q
    for k in range(K):
        mask_k = eval_points[:, :, k] == 1
        if mask_k.sum() < 2:
            continue
        t_full = target[:, :, k][mask_k]
        f_full = forecast[:, :, k][mask_k]
        if t_full.numel() < 2:
            continue
        thr_low = torch.quantile(t_full, q_low)
        thr_high = torch.quantile(t_full, q)
        tail_mask = (t_full <= thr_low) | (t_full >= thr_high)
        if tail_mask.sum() == 0:
            continue
        t_tail = t_full[tail_mask]
        f_tail = f_full[tail_mask]
        ws_tail.append(_wasserstein_1d(t_tail, f_tail))
    if len(ws_tail) == 0:
        return torch.tensor(float("nan"), device=target.device)
    return torch.mean(torch.stack(ws_tail))
