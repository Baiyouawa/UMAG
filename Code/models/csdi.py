import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps: int, embedding_dim: int):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(num_steps, embedding_dim // 2), persistent=False
        )
        self.proj1 = nn.Linear(embedding_dim, embedding_dim)
        self.proj2 = nn.Linear(embedding_dim, embedding_dim)

    def _build_embedding(self, num_steps: int, dim: int):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

    def forward(self, diffusion_step: torch.Tensor) -> torch.Tensor:
        x = self.embedding[diffusion_step]
        x = F.silu(self.proj1(x))
        x = F.silu(self.proj2(x))
        return x


def conv1d_init(in_channels: int, out_channels: int, kernel_size: int):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class ResidualBlock(nn.Module):
    def __init__(self, side_dim: int, channels: int, diffusion_dim: int, nheads: int):
        super().__init__()
        self.diff_proj = nn.Linear(diffusion_dim, channels)
        self.cond_proj = conv1d_init(side_dim, 2 * channels, 1)
        self.mid_proj = conv1d_init(channels, 2 * channels, 1)
        self.out_proj = conv1d_init(channels, 2 * channels, 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=nheads, dim_feedforward=4 * channels, activation="gelu"
        )
        self.time_layer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.feat_layer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def _attend_time(self, y: torch.Tensor, base_shape: Tuple[int, ...]) -> torch.Tensor:
        B, C, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, C, K, L).permute(0, 2, 1, 3).reshape(B * K, C, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, C, L).permute(0, 2, 1, 3).reshape(B, C, K * L)
        return y

    def _attend_feat(self, y: torch.Tensor, base_shape: Tuple[int, ...]) -> torch.Tensor:
        B, C, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, C, K, L).permute(0, 3, 1, 2).reshape(B * L, C, K)
        y = self.feat_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, C, K).permute(0, 2, 3, 1).reshape(B, C, K * L)
        return y

    def forward(self, x: torch.Tensor, cond_info: torch.Tensor, diffusion_emb: torch.Tensor):
        B, C, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, C, K * L)

        diffusion_emb = self.diff_proj(diffusion_emb).unsqueeze(-1)  # (B,C,1)
        y = x + diffusion_emb
        y = self._attend_time(y, base_shape)
        y = self._attend_feat(y, base_shape)
        y = self.mid_proj(y)  # (B,2C,KL)

        cond = cond_info.reshape(B, cond_info.shape[1], K * L)
        cond = self.cond_proj(cond)
        y = y + cond

        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        y = self.out_proj(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


class DiffCSDI(nn.Module):
    def __init__(self, config, inputdim: int):
        super().__init__()
        self.channels = config["channels"]
        self.diff_emb = DiffusionEmbedding(config["num_steps"], config["diffusion_embedding_dim"])
        self.input_proj = conv1d_init(inputdim, self.channels, 1)
        self.out_proj1 = conv1d_init(self.channels, self.channels, 1)
        self.out_proj2 = conv1d_init(self.channels, 1, 1)
        nn.init.zeros_(self.out_proj2.weight)

        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x: torch.Tensor, cond_info: torch.Tensor, diffusion_step: torch.Tensor):
        B, inputdim, K, L = x.shape
        x = x.reshape(B, inputdim, K * L)
        x = F.relu(self.input_proj(x))
        x = x.reshape(B, self.channels, K, L)

        diff_emb = self.diff_emb(diffusion_step)

        skips = []
        for block in self.blocks:
            x, skip = block(x, cond_info, diff_emb)
            skips.append(skip)

        x = torch.sum(torch.stack(skips), dim=0) / math.sqrt(len(self.blocks))
        x = x.reshape(B, self.channels, K * L)
        x = F.relu(self.out_proj1(x))
        x = self.out_proj2(x)
        x = x.reshape(B, K, L)
        return x


class CSDIModel(nn.Module):
    """
    Simplified CSDI tailored to existing A_dataset outputs.
    Uses provided observed_mask as condition; no extra random masking.
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.device = configs.device
        self.target_dim = configs.enc_in
        self.timeemb = configs.timeemb
        self.featureemb = configs.featureemb
        self.is_unconditional = getattr(configs, "is_unconditional", False)

        side_dim = self.timeemb + self.featureemb + (0 if self.is_unconditional else 1)
        diff_cfg = {
            "side_dim": side_dim,
            "channels": configs.channel,
            "diffusion_embedding_dim": getattr(configs, "diffusion_embedding_dim", configs.d_model),
            "nheads": configs.nheads,
            "layers": getattr(configs, "csdi_layers", 4),
            "num_steps": configs.diffusion_step_num,
            "beta_start": configs.beta_start,
            "beta_end": configs.beta_end,
        }
        diff_cfg["schedule"] = configs.schedule

        self.embed_layer = nn.Embedding(num_embeddings=self.target_dim, embedding_dim=self.featureemb)
        self.diffmodel = DiffCSDI(diff_cfg, inputdim=1 if self.is_unconditional else 2)

        # diffusion schedule
        if configs.schedule == "quad":
            beta = (
                torch.linspace(configs.beta_start ** 0.5, configs.beta_end ** 0.5, configs.diffusion_step_num) ** 2
            )
        else:
            beta = torch.linspace(configs.beta_start, configs.beta_end, configs.diffusion_step_num)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha_hat", 1 - beta)
        self.register_buffer("alpha", torch.cumprod(1 - beta, dim=0))
        self.register_buffer("alpha_torch", self.alpha.unsqueeze(1).unsqueeze(1))

    def time_embedding(self, pos: torch.Tensor, d_model: int) -> torch.Tensor:
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2, device=self.device) / d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp: torch.Tensor, cond_mask: torch.Tensor) -> torch.Tensor:
        B, K, L = cond_mask.shape
        time_embed = self.time_embedding(observed_tp, self.timeemb)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(torch.arange(self.target_dim, device=self.device))
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,emb)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)
        if not self.is_unconditional:
            side_info = torch.cat([side_info, cond_mask.unsqueeze(1)], dim=1)
        return side_info

    def set_input(self, noisy_data: torch.Tensor, observed_data: torch.Tensor, cond_mask: torch.Tensor) -> torch.Tensor:
        if self.is_unconditional:
            return noisy_data.unsqueeze(1)
        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        return torch.cat([cond_obs, noisy_target], dim=1)

    def calc_loss(self, observed_data: torch.Tensor, cond_mask: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
        B = observed_data.shape[0]
        t = torch.randint(0, self.configs.diffusion_step_num, [B], device=self.device)
        current_alpha = self.alpha_torch[t]
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input(noisy_data, observed_data, cond_mask)
        side_info = self.get_side_info(self.cached_tp, cond_mask)
        predicted = self.diffmodel(total_input, side_info, t)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def impute(self, observed_data: torch.Tensor, cond_mask: torch.Tensor, n_samples: int):
        B, K, L = observed_data.shape
        imputed = torch.zeros(B, n_samples, K, L, device=self.device)
        for i in range(n_samples):
            current_sample = torch.randn_like(observed_data)
            for t in range(self.configs.diffusion_step_num - 1, -1, -1):
                diff_input = self.set_input(current_sample, observed_data, cond_mask)
                side_info = self.get_side_info(self.cached_tp, cond_mask)
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t], device=self.device))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)
                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                    current_sample += sigma * noise
            imputed[:, i] = current_sample.detach()
        return imputed

    def forward(self, batch) -> torch.Tensor:
        observed_data, observed_mask, observed_tp, gt_mask = batch
        observed_data = observed_data.to(self.device)
        observed_mask = observed_mask.to(self.device)
        gt_mask = gt_mask.to(self.device)
        # expand tp to (B,L)
        if observed_tp.dim() == 1:
            observed_tp = observed_tp.unsqueeze(0).expand(observed_data.shape[0], -1)
        observed_tp = observed_tp.to(self.device)
        self.cached_tp = observed_tp

        cond_mask = observed_mask
        return self.calc_loss(observed_data, cond_mask, gt_mask)

    def evaluate(self, batch, n_samples: int = 50):
        observed_data, observed_mask, observed_tp, gt_mask = batch
        observed_data = observed_data.to(self.device)
        observed_mask = observed_mask.to(self.device)
        gt_mask = gt_mask.to(self.device)
        if observed_tp.dim() == 1:
            observed_tp = observed_tp.unsqueeze(0).expand(observed_data.shape[0], -1)
        observed_tp = observed_tp.to(self.device)
        self.cached_tp = observed_tp

        cond_mask = observed_mask
        target_mask = gt_mask - cond_mask
        side_info = self.get_side_info(self.cached_tp, cond_mask)
        samples = self.impute(observed_data, cond_mask, n_samples)
        return samples, observed_data, target_mask, observed_mask, observed_tp
