import math
import torch
import torch.nn as nn
from einops import rearrange

epsilon = 1e-8


def reverse_tensor(tensor=None, axis=-1):
    if tensor is None or tensor.dim() <= 1:
        return tensor
    indices = range(tensor.size()[axis])[::-1]
    indices = torch.tensor(indices, device=tensor.device)
    return tensor.index_select(axis, indices)


class SpatialAttention(nn.Module):
    def __init__(self, d_in, d_model, nheads, dropout=0.0):
        super().__init__()
        self.lin_in = nn.Linear(d_in, d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nheads, dropout=dropout)

    def forward(self, x, att_mask=None, **kwargs):
        b, s, n, f = x.size()
        x = rearrange(x, "b s n f -> n (b s) f")
        x = self.lin_in(x)
        x = self.self_attn(x, x, x, attn_mask=att_mask)[0]
        x = rearrange(x, "n (b s) f -> b s n f", b=b, s=s)
        return x


class SpatialConvOrderK(nn.Module):
    def __init__(self, c_in, c_out, support_len=3, order=2, include_self=True):
        super().__init__()
        self.include_self = include_self
        c_in = (order * support_len + (1 if include_self else 0)) * c_in
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.order = order

    @staticmethod
    def compute_support(adj, device=None):
        if device is not None:
            adj = adj.to(device)
        adj_bwd = adj.T
        adj_fwd = adj / (adj.sum(1, keepdims=True) + epsilon)
        adj_bwd = adj_bwd / (adj_bwd.sum(1, keepdims=True) + epsilon)
        return [adj_fwd, adj_bwd]

    @staticmethod
    def compute_support_orderK(adj, k, include_self=False, device=None):
        support = adj if isinstance(adj, (list, tuple)) else SpatialConvOrderK.compute_support(adj, device)
        supp_k = []
        for a in support:
            ak = a
            for _ in range(k - 1):
                ak = torch.matmul(ak, a.T)
                if not include_self:
                    ak.fill_diagonal_(0.0)
                supp_k.append(ak)
        return support + supp_k

    def forward(self, x, support):
        if x.dim() < 4:
            squeeze = True
            x = x.unsqueeze(-1)
        else:
            squeeze = False
        out = [x] if self.include_self else []
        if not isinstance(support, list):
            support = [support]
        for a in support:
            x1 = torch.einsum("ncvl,wv->ncwl", (x, a)).contiguous()
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = torch.einsum("ncvl,wv->ncwl", (x1, a)).contiguous()
                out.append(x2)
                x1 = x2

        out = torch.cat(out, dim=1)
        out = self.mlp(out)
        if squeeze:
            out = out.squeeze(-1)
        return out


class GCGRUCell(nn.Module):
    def __init__(self, d_in, num_units, support_len, order, activation="tanh"):
        super().__init__()
        self.activation_fn = getattr(torch, activation)
        self.forget_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len, order=order)
        self.update_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len, order=order)
        self.c_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len, order=order)

    def forward(self, x, h, adj):
        x_gates = torch.cat([x, h], dim=1)
        r = torch.sigmoid(self.forget_gate(x_gates, adj))
        u = torch.sigmoid(self.update_gate(x_gates, adj))
        x_c = torch.cat([x, r * h], dim=1)
        c = self.activation_fn(self.c_gate(x_c, adj))
        return u * h + (1.0 - u) * c


class GRIL(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        u_size=None,
        n_layers=1,
        dropout=0.0,
        kernel_size=2,
        decoder_order=1,
        global_att=False,
        support_len=2,
        n_nodes=None,
        layer_norm=False,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.u_size = int(u_size) if u_size is not None else 0
        self.n_layers = int(n_layers)
        rnn_input_size = 2 * self.input_size + self.u_size

        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(self.n_layers):
            self.cells.append(
                GCGRUCell(
                    d_in=rnn_input_size if i == 0 else self.hidden_size,
                    num_units=self.hidden_size,
                    support_len=support_len,
                    order=kernel_size,
                )
            )
            if layer_norm:
                self.norms.append(nn.GroupNorm(num_groups=1, num_channels=self.hidden_size))
            else:
                self.norms.append(nn.Identity())
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self.first_stage = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.input_size, kernel_size=1)
        self.spatial_decoder = SpatialDecoder(
            d_in=rnn_input_size + self.hidden_size,
            d_model=self.hidden_size,
            d_out=self.input_size,
            support_len=2,
            order=decoder_order,
            attention_block=global_att,
        )

        if n_nodes is not None:
            self.h0 = self.init_hidden_states(n_nodes)
        else:
            self.register_parameter("h0", None)

    def init_hidden_states(self, n_nodes):
        h0 = []
        for _ in range(self.n_layers):
            std = 1.0 / math.sqrt(self.hidden_size)
            vals = torch.distributions.Normal(0, std).sample((self.hidden_size, n_nodes))
            h0.append(nn.Parameter(vals))
        return nn.ParameterList(h0)

    def get_h0(self, x):
        if self.h0 is not None:
            return [h.expand(x.shape[0], -1, -1) for h in self.h0]
        return [torch.zeros(size=(x.shape[0], self.hidden_size, x.shape[2]), device=x.device)] * self.n_layers

    def update_state(self, x, h, adj):
        rnn_in = x
        for layer, (cell, norm) in enumerate(zip(self.cells, self.norms)):
            rnn_in = h[layer] = norm(cell(rnn_in, h[layer], adj))
            if self.dropout is not None and layer < (self.n_layers - 1):
                rnn_in = self.dropout(rnn_in)
        return h

    def forward(self, x, adj, mask=None, u=None, h=None, cached_support=False):
        *_, steps = x.size()
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.uint8)
        if h is None:
            h = self.get_h0(x)
        elif not isinstance(h, list):
            h = [*h]

        predictions, imputations, representations = [], [], []
        for step in range(steps):
            x_s = x[..., step]
            m_s = mask[..., step]
            h_s = h[-1]
            u_s = u[..., step] if u is not None else None
            xs_hat_1 = self.first_stage(h_s)
            x_s = torch.where(m_s, x_s, xs_hat_1)
            xs_hat_2, repr_s = self.spatial_decoder(x=x_s, m=m_s, h=h_s, u=u_s, adj=adj, cached_support=cached_support)
            x_s = torch.where(m_s, x_s, xs_hat_2)
            inputs = [x_s, m_s]
            if u_s is not None:
                inputs.append(u_s)
            inputs = torch.cat(inputs, dim=1)
            h = self.update_state(inputs, h, adj)
            imputations.append(xs_hat_2)
            predictions.append(xs_hat_1)
            representations.append(repr_s)

        imputations = torch.stack(imputations, dim=-1)
        predictions = torch.stack(predictions, dim=-1)
        representations = torch.stack(representations, dim=-1)
        return imputations, predictions, representations, None


class SpatialDecoder(nn.Module):
    def __init__(self, d_in, d_model, d_out, support_len, order=1, attention_block=False, nheads=2, dropout=0.0):
        super().__init__()
        self.order = order
        self.lin_in = nn.Conv1d(d_in, d_model, kernel_size=1)
        self.graph_conv = SpatialConvOrderK(
            c_in=d_model, c_out=d_model, support_len=support_len * order, order=1, include_self=False
        )
        if attention_block:
            self.spatial_att = SpatialAttention(d_in=d_model, d_model=d_model, nheads=nheads, dropout=dropout)
            self.lin_out = nn.Conv1d(3 * d_model, d_model, kernel_size=1)
        else:
            self.register_parameter("spatial_att", None)
            self.lin_out = nn.Conv1d(2 * d_model, d_model, kernel_size=1)
        self.read_out = nn.Conv1d(2 * d_model, d_out, kernel_size=1)
        self.activation = nn.PReLU()
        self.adj = None

    def forward(self, x, m, h, u, adj, cached_support=False):
        x_in = [x, m, h] if u is None else [x, m, u, h]
        x_in = torch.cat(x_in, 1)
        if self.order > 1:
            if cached_support and (self.adj is not None):
                adj = self.adj
            else:
                adj = SpatialConvOrderK.compute_support_orderK(adj, self.order, include_self=False, device=x_in.device)
                self.adj = adj if cached_support else None

        x_in = self.lin_in(x_in)
        out = self.graph_conv(x_in, adj)
        if self.spatial_att is not None:
            x_in_att = rearrange(x_in, "b f n -> b 1 n f")
            out_att = self.spatial_att(x_in_att, torch.eye(x_in_att.size(2), dtype=torch.bool, device=x_in_att.device))
            out_att = rearrange(out_att, "b s n f -> b f (n s)")
            out = torch.cat([out, out_att], 1)
        out = torch.cat([out, h], 1)
        out = self.activation(self.lin_out(out))
        out = torch.cat([out, h], 1)
        return self.read_out(out), out


class BiGRIL(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        ff_size,
        ff_dropout,
        n_layers=1,
        dropout=0.0,
        n_nodes=None,
        support_len=2,
        kernel_size=2,
        decoder_order=1,
        global_att=False,
        u_size=0,
        embedding_size=0,
        layer_norm=False,
        merge="mlp",
    ):
        super().__init__()
        self.fwd_rnn = GRIL(
            input_size=input_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=dropout,
            n_nodes=n_nodes,
            support_len=support_len,
            kernel_size=kernel_size,
            decoder_order=decoder_order,
            global_att=global_att,
            u_size=u_size,
            layer_norm=layer_norm,
        )
        self.bwd_rnn = GRIL(
            input_size=input_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=dropout,
            n_nodes=n_nodes,
            support_len=support_len,
            kernel_size=kernel_size,
            decoder_order=decoder_order,
            global_att=global_att,
            u_size=u_size,
            layer_norm=layer_norm,
        )

        if n_nodes is None:
            embedding_size = 0
        if embedding_size > 0:
            self.emb = nn.Parameter(torch.empty(embedding_size, n_nodes))
            nn.init.kaiming_normal_(self.emb, nonlinearity="relu")
        else:
            self.register_parameter("emb", None)

        if merge == "mlp":
            self._impute_from_states = True
            self.out = nn.Sequential(
                nn.Conv2d(in_channels=4 * hidden_size + input_size + embedding_size, out_channels=ff_size, kernel_size=1),
                nn.ReLU(),
                nn.Dropout(ff_dropout),
                nn.Conv2d(in_channels=ff_size, out_channels=input_size, kernel_size=1),
            )
        elif merge in ["mean", "sum", "min", "max"]:
            self._impute_from_states = False
            self.out = getattr(torch, merge)
        else:
            raise ValueError(f"Merge option {merge} not allowed.")
        self.supp = None

    def forward(self, x, adj, mask=None, u=None, cached_support=False):
        if cached_support and (self.supp is not None):
            supp = self.supp
        else:
            supp = SpatialConvOrderK.compute_support(adj, x.device)
            self.supp = supp if cached_support else None
        fwd_out, fwd_pred, fwd_repr, _ = self.fwd_rnn(x, supp, mask=mask, u=u, cached_support=cached_support)
        rev_x, rev_mask, rev_u = [reverse_tensor(tens) for tens in (x, mask, u)]
        *bwd_res, _ = self.bwd_rnn(rev_x, supp, mask=rev_mask, u=rev_u, cached_support=cached_support)
        bwd_out, bwd_pred, bwd_repr = [reverse_tensor(res) for res in bwd_res]

        if self._impute_from_states:
            inputs = [fwd_repr, bwd_repr, mask]
            if self.emb is not None:
                b, *_, s = fwd_repr.shape
                inputs += [self.emb.view(1, *self.emb.shape, 1).expand(b, -1, -1, s)]
            imputation = torch.cat(inputs, dim=1)
            imputation = self.out(imputation)
        else:
            imputation = torch.stack([fwd_out, bwd_out], dim=1)
            imputation = self.out(imputation, dim=1)

        predictions = torch.stack([fwd_out, bwd_out, fwd_pred, bwd_pred], dim=0)
        return imputation, predictions


class GRINet(nn.Module):
    def __init__(
        self,
        adj,
        d_in,
        d_hidden,
        d_ff,
        ff_dropout,
        n_layers=1,
        kernel_size=2,
        decoder_order=1,
        global_att=False,
        d_u=0,
        d_emb=0,
        layer_norm=False,
        merge="mlp",
        impute_only_holes=True,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_u = int(d_u) if d_u is not None else 0
        self.d_emb = int(d_emb) if d_emb is not None else 0
        self.register_buffer("adj", torch.tensor(adj).float())
        self.impute_only_holes = impute_only_holes

        self.bigrill = BiGRIL(
            input_size=self.d_in,
            ff_size=d_ff,
            ff_dropout=ff_dropout,
            hidden_size=self.d_hidden,
            embedding_size=self.d_emb,
            n_nodes=self.adj.shape[0],
            n_layers=n_layers,
            kernel_size=kernel_size,
            decoder_order=decoder_order,
            global_att=global_att,
            u_size=self.d_u,
            layer_norm=layer_norm,
            merge=merge,
        )

    def forward(self, x, mask=None, u=None, **kwargs):
        x = rearrange(x, "b s n c -> b c n s")
        if mask is not None:
            mask = rearrange(mask, "b s n c -> b c n s")
        if u is not None:
            u = rearrange(u, "b s n c -> b c n s")

        imputation, prediction = self.bigrill(x, self.adj, mask=mask, u=u, cached_support=self.training)
        if self.impute_only_holes and not self.training and mask is not None:
            imputation = torch.where(mask, x, imputation)
        imputation = torch.transpose(imputation, -3, -1)
        prediction = torch.transpose(prediction, -3, -1)
        if self.training:
            return imputation, prediction
        return imputation
