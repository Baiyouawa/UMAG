import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.pristi_components import (
    Conv1d_with_init,
    GuidanceConstruct,
    DiffusionEmbedding,
    TemporalLearning,
    SpatialLearning,
)
from layers.pristi_components import compute_support_identity


class Guide_diff(nn.Module):
    """
    轻量移植自 PriSTI 原仓库，用于统一框架。
    支持 adj_file=None 时退化为单位邻接。
    """

    def __init__(self, config, inputdim=1, target_dim=36, is_itp=False):
        super().__init__()
        self.channels = config["channels"]
        self.is_itp = is_itp
        self.device = config["device"]

        # 插值分支
        if self.is_itp:
            self.itp_channels = config["channels"]
            self.itp_projection = Conv1d_with_init(inputdim - 1, self.itp_channels, 1)

            self.itp_modeling = GuidanceConstruct(
                channels=self.itp_channels,
                nheads=config["nheads"],
                target_dim=target_dim,
                order=2,
                include_self=True,
                device=self.device,
                is_adp=config["is_adp"],
                adj_file=config["adj_file"],
                proj_t=config["proj_t"],
            )
            self.cond_projection = Conv1d_with_init(config["side_dim"], self.itp_channels, 1)
            self.itp_projection2 = Conv1d_with_init(self.itp_channels, 1, 1)

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        # 改为纯单位邻接，适配 KDD/Guangzhou/Physio（无外部图）
        adj = torch.eye(target_dim, device=self.device)
        self.support = compute_support_identity(target_dim, device=self.device)
        self.is_adp = False

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                NoiseProject(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    target_dim=target_dim,
                    proj_t=config["proj_t"],
                    is_adp=config.get("is_adp", False),
                    device=self.device,
                    adj_file=adj_file,
                    is_cross_t=config.get("is_cross_t", False),
                    is_cross_s=config.get("is_cross_s", True),
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, side_info, diffusion_step, itp_x, cond_mask):
        if self.is_itp:
            x = torch.cat([x, itp_x], dim=1)
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        if self.is_itp:
            itp_x = itp_x.reshape(B, inputdim - 1, K * L)
            itp_x = self.itp_projection(itp_x)
            itp_cond_info = side_info.reshape(B, -1, K * L)
            itp_cond_info = self.cond_projection(itp_cond_info)
            itp_x = itp_x + itp_cond_info
            itp_x = self.itp_modeling(itp_x, [B, self.itp_channels, K, L], self.support)
            itp_x = F.relu(itp_x)
            itp_x = itp_x.reshape(B, self.itp_channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, side_info, diffusion_emb, itp_x, self.support)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class NoiseProject(nn.Module):
    def __init__(
        self,
        side_dim,
        channels,
        diffusion_embedding_dim,
        nheads,
        target_dim,
        proj_t,
        order=2,
        include_self=True,
        device=None,
        is_adp=False,
        adj_file=None,
        is_cross_t=False,
        is_cross_s=True,
    ):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.forward_time = TemporalLearning(channels=channels, nheads=nheads, is_cross=is_cross_t)
        self.forward_feature = SpatialLearning(
            channels=channels,
            nheads=nheads,
            target_dim=target_dim,
            order=order,
            include_self=include_self,
            device=device,
            is_adp=is_adp,
            adj_file=adj_file,
            proj_t=proj_t,
            is_cross=is_cross_s,
        )

    def forward(self, x, side_info, diffusion_emb, itp_info, support):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb
        cond_info = side_info.reshape(B, -1, K * L)
        cond_info = self.cond_projection(cond_info)

        mid = self.mid_projection(x)
        mid = mid + cond_info
        mid = mid.reshape(B, 2, channel, K, L)

        x = x.reshape(B, channel, K, L)
        x = self.forward_time(mid[:, 0].reshape(B, channel, K * L), base_shape, itp_info)
        x = self.forward_feature(x, base_shape, support, itp_info)
        x = x.reshape(B, channel, K * L)
        x = x + mid[:, 1].reshape(B, channel, K * L)

        x = self.output_projection(x)
        x = x.reshape(B, 2, channel, K, L)
        residual, skip = x[:, 0], x[:, 1]
        return residual, skip
