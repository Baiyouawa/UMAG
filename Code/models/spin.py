import torch
import torch.nn as nn
from layers.spin_components import SPINModel


class SPINWrapper(nn.Module):
    """
    统一框架下的 SPIN：
    - 输入: observed_data (B,L,K)，observed_mask (B,L,K)，observed_tp (B,L)
    - 输出: imputed (B,L,K)
    """

    def __init__(self, configs):
        super().__init__()
        self.device = configs.device
        self.target_dim = configs.enc_in
        self.model = SPINModel(
            input_size=1,
            hidden_size=getattr(configs, "spin_hidden", configs.d_model),
            n_nodes=self.target_dim,
            n_layers=getattr(configs, "spin_layers", 4),
            nheads=getattr(configs, "spin_heads", 4),
        ).to(self.device)

    def forward(self, observed_data: torch.Tensor, observed_mask: torch.Tensor, observed_tp: torch.Tensor) -> torch.Tensor:
        x = observed_data.unsqueeze(-1)  # (B,L,K,1) but SPINModel expects channels merged
        x = x.squeeze(-1)  # (B,L,K)
        m = observed_mask
        if observed_tp.dim() == 1:
            observed_tp = observed_tp.unsqueeze(0).expand(x.shape[0], -1)
        out = self.model(x, m, observed_tp)
        return out
