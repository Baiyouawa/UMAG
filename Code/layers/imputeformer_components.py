import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        # pos: (B, L)
        # pos: (B, L) -> (B, L, d_model)
        return self.pe[0, pos.long(), :]


class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalBlock(nn.Module):
    """Self-attention over time for each node independently."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = MLP(d_model, 4 * d_model, d_model, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x: (B, L, N, C), mask: (B, L, N)
        B, L, N, C = x.shape
        x_flat = x.reshape(B * N, L, C)
        key_padding = ~(mask.reshape(B * N, L).bool())
        attn_out, _ = self.mha(x_flat, x_flat, x_flat, key_padding_mask=key_padding)
        x_flat = self.norm1(x_flat + self.dropout(attn_out))
        ff_out = self.ff(x_flat)
        x_flat = self.norm2(x_flat + self.dropout(ff_out))
        return x_flat.reshape(B, L, N, C)


class SpatialBlock(nn.Module):
    """Self-attention over nodes for each time step independently."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = MLP(d_model, 4 * d_model, d_model, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x: (B, L, N, C), mask: (B, L, N)
        B, L, N, C = x.shape
        x = x.transpose(1, 2)  # (B, N, L, C) -> attention per time step -> reshape
        x_flat = x.reshape(B * N, L, C)
        key_padding = ~(mask.transpose(1, 2).reshape(B * N, L).bool())
        attn_out, _ = self.mha(x_flat, x_flat, x_flat, key_padding_mask=key_padding)
        x_flat = self.norm1(x_flat + self.dropout(attn_out))
        ff_out = self.ff(x_flat)
        x_flat = self.norm2(x_flat + self.dropout(ff_out))
        x = x_flat.reshape(B, N, L, C).transpose(1, 2)  # back to (B, L, N, C)
        return x


class ImputeFormerModel(nn.Module):
    """
    适配统一框架的简化 ImputeFormer：
    - 时间注意力 + 节点注意力堆叠
    - 输入/输出: (B, L, K)
    """

    def __init__(
        self,
        num_nodes: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model

        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.node_emb = nn.Embedding(num_nodes, d_model)

        self.temporal_blocks = nn.ModuleList(
            [TemporalBlock(d_model, n_heads, dropout) for _ in range(num_layers)]
        )
        self.spatial_blocks = nn.ModuleList(
            [SpatialBlock(d_model, n_heads, dropout) for _ in range(num_layers)]
        )

        self.readout = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, time_pos: torch.Tensor):
        # x: (B, L, K), mask: (B, L, K), time_pos: (B, L)
        B, L, K = x.shape
        x = x.unsqueeze(-1)  # (B,L,K,1)
        x = self.input_proj(x)

        # add positional + node embedding
        pe = self.pos_enc(time_pos)  # (B,L,d)
        pe = pe.unsqueeze(2).expand(-1, -1, K, -1)  # (B,L,K,d)
        node_idx = torch.arange(K, device=x.device).unsqueeze(0).expand(B, -1)
        node_emb = self.node_emb(node_idx).unsqueeze(1).expand(-1, L, -1, -1)
        x = x + pe + node_emb

        for t_block, s_block in zip(self.temporal_blocks, self.spatial_blocks):
            x = t_block(x, mask)
            x = s_block(x, mask)

        out = self.readout(x).squeeze(-1)  # (B,L,K)
        return out
