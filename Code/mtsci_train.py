from pathlib import Path

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

import A_dataset
import metrics_utils
from models.mtsci import MTSCIModel


def diffusion_train(configs):
    train_loader, _ = A_dataset.get_dataset(configs)
    model = MTSCIModel(configs).to(configs.device)

    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate_diff, weight_decay=1e-6)
    p1 = int(0.75 * configs.epoch_diff)
    p2 = int(0.9 * configs.epoch_diff)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1)

    for epoch in range(configs.epoch_diff):
        model.train()
        train_loss = []
        for observed_data, _, observed_mask, observed_tp, gt_mask in tqdm(
            train_loader, desc=f"Train {epoch+1}/{configs.epoch_diff}", leave=False
        ):
            batch = (observed_data, observed_mask, observed_tp, gt_mask)
            loss = model(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        lr_scheduler.step()
        if epoch % 50 == 0 or epoch == configs.epoch_diff - 1:
            print(
                f"Epoch: {epoch+1}. Train_loss: {np.average(train_loss) if train_loss else 0:.6f}"
            )
    return model


def diffusion_test(configs, model):
    _, test_loader = A_dataset.get_dataset(configs)
    model.eval()

    target_2d = []
    forecast_2d = []
    eval_p_2d = []
    generate_data2d = []

    print("Testset sum: ", len(test_loader.dataset) // configs.batch + 1)

    with torch.no_grad():
        for observed_data, _, observed_mask, observed_tp, gt_mask in tqdm(test_loader, desc="Test", leave=False):
            batch = (observed_data, observed_mask, observed_tp, gt_mask)
            samples, target, eval_mask, observed_mask = _run_eval(model, batch, configs)

            imputed_sample = samples.median(dim=1).values
            imputed_data = observed_mask * target + (1 - observed_mask) * imputed_sample

            target_2d.append(target.cpu())
            forecast_2d.append(imputed_data.cpu())
            eval_p_2d.append(eval_mask.cpu())

            B, L, K = imputed_data.shape
            temp = imputed_data.reshape(B * L, K).cpu().numpy()
            generate_data2d.append(temp)

    generate_data2d = np.vstack(generate_data2d)
    results_root = Path(getattr(configs, "results_root", Path(".")))
    results_root.mkdir(parents=True, exist_ok=True)
    out_path = results_root / f"MTSCI_{configs.dataset}_p{int(configs.missing_rate*100)}_seed{configs.seed}.csv"
    np.savetxt(out_path, generate_data2d, delimiter=",")
    print(f"MTSCI imputation results saved to {out_path}")

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


def _run_eval(model, batch, configs, n_samples: int = 50):
    observed_data, observed_mask, observed_tp, gt_mask = batch
    observed_data = observed_data.to(configs.device)
    observed_mask = observed_mask.to(configs.device)
    gt_mask = gt_mask.to(configs.device)
    observed_tp = observed_tp.to(configs.device)
    out = model.evaluate((observed_data, observed_mask, observed_tp, gt_mask), n_samples=n_samples)
    samples, target, eval_mask, observed_mask, _ = out
    return samples, target, eval_mask, observed_mask


def calc_RMSE(target, forecast, eval_points):
    eval_p = torch.where(eval_points == 1)
    error_mean = torch.mean((target[eval_p] - forecast[eval_p]) ** 2)
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
    m = min(t_sorted.shape[0], f_sorted.shape[0])
    return torch.mean(torch.abs(t_sorted[:m] - f_sorted[:m]))


def _wasserstein_1d(t: torch.Tensor, f: torch.Tensor):
    t_sorted, _ = torch.sort(t)
    f_sorted, _ = torch.sort(f)
    m = min(t_sorted.shape[0], f_sorted.shape[0])
    return torch.mean(torch.abs(t_sorted[:m] - f_sorted[:m]))


def calc_Wasserstein_per_dim(target, forecast, eval_points):
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
