import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: Optional[int] = None, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        d_out = d_out if d_out is not None else d_hidden
        layers = []
        dims = [d_in] + [d_hidden] * (n_layers - 1) + [d_out]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StaticGraphEmbedding(nn.Module):
    def __init__(self, n_nodes: int, emb_dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_nodes, emb_dim)
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, token_index=None):
        if token_index is None:
            return self.emb.weight
        return self.emb(token_index)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        # pos: (B, L)
        return self.pe[:, pos.long(), :]


class PositionalEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers: int = 1, n_nodes: Optional[int] = None):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.activation = nn.LeakyReLU()
        self.mlp = MLP(out_channels, out_channels, out_channels, n_layers=n_layers, dropout=0.0)
        self.positional = PositionalEncoding(out_channels)
        if n_nodes is not None:
            self.node_emb = StaticGraphEmbedding(n_nodes, out_channels)
        else:
            self.register_parameter("node_emb", None)

    def forward(self, x, node_emb=None, node_index=None):
        if node_emb is None and self.node_emb is not None:
            node_emb = self.node_emb(token_index=node_index)
        # x: [B, L, C], node_emb: [N, C] -> [B, L, N, C]
        x = self.lin(x)
        x = self.activation(x.unsqueeze(-2) + node_emb)
        out = self.mlp(x)
        out = self.positional(torch.arange(x.shape[1], device=x.device)).unsqueeze(2) + out
        return out


class TemporalGraphAdditiveAttention(nn.Module):
    """
    简化版时序-图注意力：仅做时间维自注意力（每个节点独立），忽略空间邻接以保持统一框架。
    """

    def __init__(self, input_size: int, output_size: int, nheads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=input_size, num_heads=nheads, dropout=dropout, batch_first=True)
        self.out = nn.Linear(input_size, output_size)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        # h: (B, L, N, H), mask: (B, L, N)
        B, L, N, H = h.shape
        h_flat = h.reshape(B * N, L, H)
        key_padding = ~(mask.reshape(B * N, L).bool())  # True for pad
        attn_out, _ = self.mha(h_flat, h_flat, h_flat, key_padding_mask=key_padding)
        attn_out = self.out(attn_out)
        attn_out = self.norm(attn_out)
        return attn_out.reshape(B, L, N, -1)


class SPINModel(nn.Module):
    """
    适配统一框架的简化 SPIN：
    - 使用时间自注意力（每节点独立），空间邻接使用单位阵。
    - 输入/输出 shape: (B, L, K)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_nodes: int,
        n_layers: int = 4,
        nheads: int = 4,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.hidden = hidden_size
        self.n_nodes = n_nodes

        self.u_enc = PositionalEncoder(in_channels=input_size, out_channels=hidden_size, n_layers=2, n_nodes=n_nodes)
        self.h_enc = MLP(input_size, hidden_size, n_layers=2)
        self.h_norm = nn.LayerNorm(hidden_size)

        self.x_skip = nn.ModuleList()
        self.encoder = nn.ModuleList()
        self.readout = nn.ModuleList()
        for _ in range(n_layers):
            self.x_skip.append(nn.Linear(input_size, hidden_size))
            self.encoder.append(TemporalGraphAdditiveAttention(hidden_size, hidden_size, nheads=nheads))
            self.readout.append(MLP(hidden_size, hidden_size, input_size, n_layers=2))

    def forward(self, x: torch.Tensor, mask: torch.Tensor, time_pos: torch.Tensor):
        # x: (B,L,K), mask: (B,L,K), time_pos: (B,L)
        x = x * mask
        # positional encoding
        q = self.u_enc(time_pos, node_index=torch.arange(self.n_nodes, device=x.device))
        h = self.h_enc(x) + q
        h = torch.where(mask.bool().unsqueeze(-1), h, q)
        h = self.h_norm(h)

        for l in range(self.n_layers):
            h = h + self.x_skip[l](x) * mask.unsqueeze(-1)
            h = self.encoder[l](h, mask)
            h = self.readout[l](h)

        return h  # (B,L,K)
