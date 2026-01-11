import torch
import torch.nn as nn
from layers.grin_components import GRINet


class GRINWrapper(nn.Module):
    """
    适配统一框架的 GRIN。
    - 输入: observed_data (B,L,K)，observed_mask (B,L,K)
    - 输出: imputed (B,L,K)
    """

    def __init__(self, configs):
        super().__init__()
        self.device = configs.device
        self.target_dim = configs.enc_in

        adj = torch.eye(self.target_dim).numpy()  # 无外部图时用单位矩阵
        self.model = GRINet(
            adj=adj,
            d_in=1,
            d_hidden=getattr(configs, "grin_d_hidden", configs.d_model),
            d_ff=getattr(configs, "grin_d_ff", configs.d_ff),
            ff_dropout=getattr(configs, "grin_ff_dropout", 0.0),
            n_layers=getattr(configs, "grin_layers", 1),
            kernel_size=getattr(configs, "grin_kernel_size", 2),
            decoder_order=getattr(configs, "grin_decoder_order", 1),
            global_att=getattr(configs, "grin_global_att", False),
            d_u=0,
            d_emb=getattr(configs, "grin_d_emb", 8),
            layer_norm=getattr(configs, "grin_layer_norm", False),
            merge="mlp",
            impute_only_holes=False,  # 我们自行选择评估掩码
        ).to(self.device)

    def forward(self, observed_data: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
        # GRIN 期望 shape: B,S,N,C
        x = observed_data.unsqueeze(-1)  # B,L,K,1
        m = observed_mask.unsqueeze(-1)
        out = self.model(x, mask=m)
        if isinstance(out, tuple):
            out = out[0]
        return out.squeeze(-1)
