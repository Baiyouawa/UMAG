import torch
import torch.nn as nn
from layers.imputeformer_components import ImputeFormerModel


class ImputeFormerWrapper(nn.Module):
    """
    统一框架的 ImputeFormer：
    - 输入: observed_data (B,L,K), observed_mask (B,L,K), observed_tp (B,L)
    - 输出: imputed (B,L,K)
    """

    def __init__(self, configs):
        super().__init__()
        self.device = configs.device
        self.target_dim = configs.enc_in
        self.model = ImputeFormerModel(
            num_nodes=self.target_dim,
            d_model=getattr(configs, "imputeformer_d_model", configs.d_model),
            n_heads=getattr(configs, "imputeformer_heads", 4),
            num_layers=getattr(configs, "imputeformer_layers", 3),
            dropout=getattr(configs, "imputeformer_dropout", 0.1),
        ).to(self.device)

    def forward(self, observed_data: torch.Tensor, observed_mask: torch.Tensor, observed_tp: torch.Tensor) -> torch.Tensor:
        if observed_tp.dim() == 1:
            observed_tp = observed_tp.unsqueeze(0).expand(observed_data.shape[0], -1)
        return self.model(observed_data, observed_mask, observed_tp)
