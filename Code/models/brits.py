import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureRegression(nn.Module):
    """Linear regression on other features (diagonal masked)."""

    def __init__(self, input_size: int):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(input_size, input_size))
        self.b = nn.Parameter(torch.Tensor(input_size))
        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer("m", m)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(0))
        nn.init.uniform_(self.W, -stdv, stdv)
        nn.init.uniform_(self.b, -stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)
        return F.linear(x, self.W * self.m, self.b)


class TemporalDecay(nn.Module):
    """Gamma = exp(-relu(W d + b))"""

    def __init__(self, input_size: int, output_size: int, diag: bool = False):
        super().__init__()
        self.diag = diag
        self.W = nn.Parameter(torch.Tensor(output_size, input_size))
        self.b = nn.Parameter(torch.Tensor(output_size))
        if diag:
            assert input_size == output_size
            self.register_buffer("m", torch.eye(input_size, input_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(0))
        nn.init.uniform_(self.W, -stdv, stdv)
        nn.init.uniform_(self.b, -stdv, stdv)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        if self.diag:
            gamma = F.relu(F.linear(d, self.W * self.m, self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        return torch.exp(-gamma)


def compute_deltas(mask: torch.Tensor) -> torch.Tensor:
    """
    Compute time gaps given observation mask.
    Args:
        mask: (B, L, D) with 1 for observed.
    Returns:
        deltas: (B, L, D)
    """
    B, L, D = mask.shape
    deltas = torch.zeros_like(mask)
    deltas[:, 0, :] = 1.0
    for t in range(1, L):
        deltas[:, t, :] = 1.0 + (1.0 - mask[:, t - 1, :]) * deltas[:, t - 1, :]
    return deltas


class RITS(nn.Module):
    """
    Single-direction RITS (imputation only, classification removed).
    """

    def __init__(self, input_size: int, rnn_hid_size: int, impute_weight: float):
        super().__init__()
        self.input_size = input_size
        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight

        self.rnn_cell = nn.LSTMCell(input_size * 2, rnn_hid_size)
        self.temp_decay_h = TemporalDecay(input_size=input_size, output_size=rnn_hid_size, diag=False)
        self.temp_decay_x = TemporalDecay(input_size=input_size, output_size=input_size, diag=True)

        self.hist_reg = nn.Linear(rnn_hid_size, input_size)
        self.feat_reg = FeatureRegression(input_size)
        self.weight_combine = nn.Linear(input_size * 2, input_size)

    def forward(
        self, values: torch.Tensor, masks: torch.Tensor, deltas: torch.Tensor, eval_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            values: (B, L, D) ground-truth values (already filled 0 for original NaN).
            masks: (B, L, D) 1 for observed (visible to model), 0 for missing.
            deltas: (B, L, D) time gaps.
            eval_masks: (B, L, D) 1 where we compute loss (simulated missing positions).
        Returns:
            imputations: (B, L, D)
            loss: scalar tensor
        """
        B, L, D = values.shape
        h = values.new_zeros((B, self.rnn_hid_size))
        c = values.new_zeros((B, self.rnn_hid_size))

        imputations = []
        loss = values.new_tensor(0.0)

        for t in range(L):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            h = h * gamma_h

            x_h = self.hist_reg(h)
            x_c = m * x + (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            alpha = self.weight_combine(torch.cat([gamma_x, m], dim=1))
            c_h = alpha * z_h + (1 - alpha) * x_h
            c_c = m * x + (1 - m) * c_h

            # imputation loss only on missing positions we want to fill
            m_eval = eval_masks[:, t, :]
            loss += (c_c - x).abs().mul(m_eval).sum() / (m_eval.sum() + 1e-5)

            inputs = torch.cat([c_c, m], dim=1)
            h, c = self.rnn_cell(inputs, (h, c))
            imputations.append(c_c.unsqueeze(1))

        imputations = torch.cat(imputations, dim=1)
        loss = loss * self.impute_weight
        return imputations, loss


class BRITS(nn.Module):
    """
    Bidirectional RITS for pure imputation.
    """

    def __init__(self, input_size: int, rnn_hid_size: int, impute_weight: float, consistency_weight: float):
        super().__init__()
        self.fwd = RITS(input_size, rnn_hid_size, impute_weight)
        self.bwd = RITS(input_size, rnn_hid_size, impute_weight)
        self.consistency_weight = consistency_weight

    def forward(self, values: torch.Tensor, masks: torch.Tensor, eval_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            values: (B, L, D)
            masks: (B, L, D)
            eval_masks: (B, L, D)
        Returns:
            imputations: (B, L, D)
            loss: scalar tensor
        """
        deltas_f = compute_deltas(masks)
        imp_f, loss_f = self.fwd(values, masks, deltas_f, eval_masks)

        values_b = torch.flip(values, dims=[1])
        masks_b = torch.flip(masks, dims=[1])
        eval_b = torch.flip(eval_masks, dims=[1])
        deltas_b = compute_deltas(masks_b)
        imp_b_rev, loss_b = self.bwd(values_b, masks_b, deltas_b, eval_b)
        imp_b = torch.flip(imp_b_rev, dims=[1])

        imps = (imp_f + imp_b) / 2
        consistency = (imp_f - imp_b).abs().mean() * self.consistency_weight
        loss = loss_f + loss_b + consistency
        return imps, loss
