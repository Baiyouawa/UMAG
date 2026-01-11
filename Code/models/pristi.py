import numpy as np
import torch
import torch.nn as nn

from models.pristi_diff_models import Guide_diff


class PriSTIWrapper(nn.Module):
    """
    统一框架适配：
    - 输入: observed_data (B,L,K), observed_mask (B,L,K), gt_mask (B,L,K), observed_tp (B,L)
    - 前向: compute_loss(...) 返回标量 loss
    - impute: 返回插补结果 (B,L,K)
    """

    def __init__(self, configs):
        super().__init__()
        self.device = configs.device
        self.target_dim = configs.enc_in
        self.seq_len = configs.seq_len

        # 组装 PriSTI 所需配置（取自当前统一 Config）
        model_cfg = {
            "timeemb": getattr(configs, "timeemb", configs.d_model),
            "featureemb": getattr(configs, "featureemb", 32),
            "is_unconditional": getattr(configs, "is_unconditional", False),
            "target_strategy": "random",
            "use_guide": False,  # 我们默认不启用 guide/itp
        }
        diff_cfg = {
            "num_steps": getattr(configs, "diffusion_step_num", 50),
            "beta_start": getattr(configs, "beta_start", 1e-4),
            "beta_end": getattr(configs, "beta_end", 0.02),
            "schedule": getattr(configs, "schedule", "quad"),
            "channels": getattr(configs, "channel", 64),
            "layers": getattr(configs, "residual_layers", 4),
            "nheads": getattr(configs, "nheads", 8),
            "proj_t": getattr(configs, "proj_t", 16),
            "is_adp": False,
            "adj_file": None,  # 无图时使用单位矩阵
            "is_cross_t": False,
            "is_cross_s": True,
            # side_dim 在 init 中补充
            # device 在 init 中补充
        }

        self.model = PriSTICore(
            target_dim=self.target_dim,
            seq_len=self.seq_len,
            model_cfg=model_cfg,
            diff_cfg=diff_cfg,
            device=self.device,
        ).to(self.device)

    def compute_loss(self, observed_data, observed_mask, observed_tp, gt_mask):
        """
        训练用：返回标量 loss
        """
        loss = self.model.calc_loss_wrapper(observed_data, observed_mask, observed_tp, gt_mask, is_train=True)
        return loss

    def impute(self, observed_data, observed_mask, observed_tp, gt_mask, n_samples=50):
        """
        推理用：返回插补结果 (B,L,K)，使用多样本中位数。
        """
        samples = self.model.impute_wrapper(observed_data, observed_mask, observed_tp, gt_mask, n_samples=n_samples)
        imputed = samples.median(dim=1).values  # (B,L,K)
        return imputed


class PriSTICore(nn.Module):
    """
    精简适配版 PriSTI：输入为统一框架张量，cond_mask=observed_mask，评估掩码为 gt_mask - observed_mask。
    """

    def __init__(self, target_dim, seq_len, model_cfg, diff_cfg, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim
        self.seq_len = seq_len

        self.emb_time_dim = model_cfg["timeemb"]
        self.emb_feature_dim = model_cfg["featureemb"]
        self.is_unconditional = model_cfg["is_unconditional"]
        self.target_strategy = model_cfg["target_strategy"]
        self.use_guide = model_cfg["use_guide"]

        self.cde_output_channels = diff_cfg["channels"]
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        self.embed_layer = nn.Embedding(num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim)

        diff_cfg = diff_cfg.copy()
        diff_cfg["side_dim"] = self.emb_total_dim
        diff_cfg["device"] = device
        input_dim = 2  # cond + target
        self.diffmodel = Guide_diff(diff_cfg, input_dim, target_dim, self.use_guide)

        # parameters for diffusion models
        self.num_steps = diff_cfg["num_steps"]
        if diff_cfg["schedule"] == "quad":
            self.beta = np.linspace(diff_cfg["beta_start"] ** 0.5, diff_cfg["beta_end"] ** 0.5, self.num_steps) ** 2
        elif diff_cfg["schedule"] == "linear":
            self.beta = np.linspace(diff_cfg["beta_start"], diff_cfg["beta_end"], self.num_steps)
        else:
            raise ValueError("schedule must be quad or linear")

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha, dtype=torch.float32, device=self.device).unsqueeze(1).unsqueeze(1)

    # ---- helper ----
    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2, device=self.device) / d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(torch.arange(self.target_dim, device=self.device))  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)
        return side_info

    # ---- core losses ----
    def calc_loss_core(self, observed_data, cond_mask, observed_mask_full, side_info, itp_info, is_train, set_t=-1):
        B, K, L = observed_data.shape
        if not is_train:
            t = (torch.ones(B, device=self.device) * set_t).long()
        else:
            t = torch.randint(0, self.num_steps, [B], device=self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        if not self.use_guide:
            itp_info = cond_mask * observed_data
        predicted = self.diffmodel(total_input, side_info, t, itp_info, cond_mask)

        target_mask = observed_mask_full - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def calc_loss_wrapper(self, observed_data_b, observed_mask_b, observed_tp_b, gt_mask_b, is_train=True):
        """
        observed_data_b: (B,L,K)
        observed_mask_b: (B,L,K) 可见
        gt_mask_b:       (B,L,K) 原始可评估位置
        """
        # 转换 shape: B,K,L
        observed_data = observed_data_b.permute(0, 2, 1)
        cond_mask = observed_mask_b.permute(0, 2, 1)
        observed_mask_full = gt_mask_b.permute(0, 2, 1)
        observed_tp = observed_tp_b  # (B,L)

        side_info = self.get_side_info(observed_tp, cond_mask)
        itp_info = None  # use_guide=False

        loss_sum = 0
        if is_train:
            loss = self.calc_loss_core(observed_data, cond_mask, observed_mask_full, side_info, itp_info, is_train=True)
            return loss
        else:
            for t in range(self.num_steps):
                loss = self.calc_loss_core(
                    observed_data, cond_mask, observed_mask_full, side_info, itp_info, is_train=False, set_t=t
                )
                loss_sum += loss.detach()
            return loss_sum / self.num_steps

    # ---- sampling ----
    def impute(self, observed_data, cond_mask, side_info, n_samples, itp_info):
        B, K, L = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, K, L, device=self.device)

        for i in range(n_samples):
            if self.is_unconditional:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    if not self.use_guide:
                        cond_obs = (cond_mask * observed_data).unsqueeze(1)
                        noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                        diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                    else:
                        diff_input = ((1 - cond_mask) * current_sample).unsqueeze(1)  # (B,1,K,L)
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t], device=self.device), itp_info, cond_mask)

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def impute_wrapper(self, observed_data_b, observed_mask_b, observed_tp_b, gt_mask_b, n_samples=50):
        observed_data = observed_data_b.permute(0, 2, 1)
        cond_mask = observed_mask_b.permute(0, 2, 1)
        observed_tp = observed_tp_b

        side_info = self.get_side_info(observed_tp, cond_mask)
        itp_info = None
        samples = self.impute(observed_data, cond_mask, side_info, n_samples, itp_info)
        # 返回 shape (B,L,K)
        return samples.permute(0, 1, 3, 2)

    # ---- util ----
    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional:
            total_input = noisy_data.unsqueeze(1)
        else:
            if not self.use_guide:
                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
                total_input = torch.cat([cond_obs, noisy_target], dim=1)
            else:
                total_input = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        return total_input
