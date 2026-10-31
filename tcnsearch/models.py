"""All model families for the search, with one shared building block for time.

Every temporal operation is a `CausalConv1d`. It has two modes:
  * full mode   (default): x [B, C, T] is left-padded with (k-1)*d zeros, so row t only sees rows <= t.
  * stream mode (row by row): x [B, C, 1] plus `taps` [B, C, k-1] = the conv's own inputs at
    t-(k-1)d, ..., t-d. The conv records `now` = its input at t so the caller can store it.
Everything else (Linear / 1x1 conv / LayerNorm / activations) acts on one row at a time.
So ANY model built from these parts runs row by row with a generic ring buffer (streaming.py),
and full mode == stream mode is checked automatically for every configuration (checks.py).

Families (config key "family"):
  tcn         residual TCN, two causal convs per block (Bai et al. 2018)
  multiscale  several residual-TCN branches with different dilations, concatenated before the head
  modern      causal ModernTCN: large-kernel depthwise conv (+ small-kernel branch) + ConvFFN, residual
  modern_vi   ModernTCN keeping the 112 input variables separate (grouped convs), as in the paper
  gru         the GRU baseline (same layout as my_baseline/train.py: gru1, gru2, head)
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_IN = 112
ACTS = ("relu", "silu", "gelu", "gated")
NORMS = ("weight", "layer", "none")


# ----------------------------------------------------------------------------- building blocks
class CausalConv1d(nn.Module):
    def __init__(self, cin, cout, k, d=1, groups=1, weight_norm=False):
        super().__init__()
        self.cin, self.cout, self.k, self.d, self.groups = cin, cout, k, d, groups
        conv = nn.Conv1d(cin, cout, k, dilation=d, groups=groups)
        if weight_norm:
            conv = nn.utils.parametrizations.weight_norm(conv)
        self.conv = conv
        self.stream = False
        self.taps = None      # stream mode input: [B, cin, k-1]
        self.taps_flat = None  # fast stream mode input (batch 1): [(k-1)*cin], tap-major, oldest first
        self.now = None       # stream mode output: [B, cin, 1]
        self.fast = None      # set by prepare_fast_stream(): matrix form of the weights

    @property
    def lookback(self):
        return (self.k - 1) * self.d

    def forward(self, x):
        if not self.stream:
            return self.conv(F.pad(x, (self.lookback, 0)))
        self.now = x
        if self.fast is not None:                                            # one row, matrix form (fast in ONNX)
            v = x if self.k == 1 else torch.cat([self.taps_flat, x], 1)      # [1, k*cin, 1], tap-major
            if self.fast == "dense":
                return self.W2 @ v + self.b2                                 # [cout, k*cin] @ [1, k*cin, 1]
            return (v.reshape(1, self.k, self.cin) * self.W2).sum(1).reshape(1, self.cout, 1) + self.b2
        inp = x if self.k == 1 else torch.cat([self.taps, x], 2)            # oldest ... newest
        return F.conv1d(inp, self.conv.weight, self.conv.bias, groups=self.groups)  # dilation already applied

    def prepare_fast_stream(self):
        """Precompute the weight in matrix form for batch-1 streaming. Requires plain (folded) weights."""
        w = self.conv.weight.detach()                                        # [cout, cin/groups, k]
        self.register_buffer("b2", self.conv.bias.detach().reshape(-1, 1).clone())
        if self.groups == 1:                                                 # W2[o, j*cin + c] = w[o, c, j]
            self.register_buffer("W2", w.permute(0, 2, 1).reshape(self.cout, self.k * self.cin).contiguous())
            self.fast = "dense"
        elif self.groups == self.cin == self.cout:
            self.register_buffer("W2", w[:, 0, :].transpose(0, 1).unsqueeze(0).contiguous())   # [1, k, C]
            self.fast = "depthwise"


class StreamPointwise(nn.Module):
    """Batch-1, one-row replacement for a 1x1 Conv1d (groups 1): a single matrix multiply."""

    def __init__(self, conv: nn.Conv1d):
        super().__init__()
        self.register_buffer("W", conv.weight.detach()[:, :, 0].contiguous())     # [cout, cin]
        self.register_buffer("b", conv.bias.detach().reshape(-1, 1).clone())

    def forward(self, x):                       # [1, C, 1] -> [1, cout, 1], no reshapes
        return self.W @ x + self.b


def to_fast_stream(model):
    """In place: matrix-form causal convs and 1x1 convs for batch-1 row-by-row inference."""
    for c in causal_convs(model):
        c.prepare_fast_stream()
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Conv1d) and child.kernel_size == (1,) and child.groups == 1 \
                    and not isinstance(parent, CausalConv1d):
                setattr(parent, name, StreamPointwise(child))
    return model


class ChannelNorm(nn.Module):
    """LayerNorm over channels at each row (causal: never mixes rows)."""

    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):                      # [B, C, T]
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


def act_fn(name):
    if name == "relu":
        return F.relu
    if name == "silu":
        return F.silu
    if name == "gelu":
        return F.gelu
    if name == "gated":                        # WaveNet gate: input has 2C channels -> C
        def gated(x):
            a, b = x.chunk(2, dim=1)
            return torch.tanh(a) * torch.sigmoid(b)
        return gated
    raise ValueError(f"unknown activation {name}")


def maybe_norm(norm, c):
    return ChannelNorm(c) if norm == "layer" else nn.Identity()


# ----------------------------------------------------------------------------- residual TCN
class TCNBlock(nn.Module):
    """u = act(norm(conv1(h))); u = act(norm(conv2(u))); h = h + u; h = act(h) (Bai post-activation, not for gated)."""

    def __init__(self, C, k, d, act="relu", norm="weight", dropout=0.0):
        super().__init__()
        m = 2 if act == "gated" else 1
        wn = norm == "weight"
        self.c1 = CausalConv1d(C, m * C, k, d, weight_norm=wn)
        self.c2 = CausalConv1d(C, m * C, k, d, weight_norm=wn)
        self.n1, self.n2 = maybe_norm(norm, m * C), maybe_norm(norm, m * C)
        self.act, self.post = act_fn(act), act != "gated"
        self.drop = nn.Dropout(dropout)

    def forward(self, h):
        u = self.drop(self.act(self.n1(self.c1(h))))
        u = self.drop(self.act(self.n2(self.c2(u))))
        h = h + u
        return self.act(h) if self.post else h


class TCNTrunk(nn.Module):
    def __init__(self, n_in, C, k, dilations, act, norm, dropout):
        super().__init__()
        self.inp = nn.Conv1d(n_in, C, 1)
        self.blocks = nn.ModuleList([TCNBlock(C, k, d, act, norm, dropout) for d in dilations])
        self.out_channels = C
        self.receptive_field = 1 + 2 * (k - 1) * sum(dilations)

    def forward(self, x):                      # [B, n_in, T] -> [B, C, T]
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return h


class ResidualTCN(nn.Module):
    def __init__(self, channels=64, kernel=3, dilations=(1, 2, 4, 8, 16, 32, 64, 128), act="relu",
                 norm="weight", dropout=0.0, n_in=N_IN):
        super().__init__()
        self.trunk = TCNTrunk(n_in, channels, kernel, list(dilations), act, norm, dropout)
        self.head = nn.Conv1d(channels, 2, 1)
        self.receptive_field = self.trunk.receptive_field

    def forward(self, x):                      # x [B, T, 112] -> [B, T, 2]
        return self.head(self.trunk(x.transpose(1, 2))).transpose(1, 2)


class MultiScaleTCN(nn.Module):
    """branches: list of {"channels", "kernel", "dilations"}; outputs concatenated -> 1x1 head."""

    def __init__(self, branches, act="relu", norm="weight", dropout=0.0, n_in=N_IN):
        super().__init__()
        self.branches = nn.ModuleList([TCNTrunk(n_in, b["channels"], b["kernel"], b["dilations"], act, norm, dropout)
                                       for b in branches])
        tot = sum(b.out_channels for b in self.branches)
        self.head = nn.Conv1d(tot, 2, 1)
        self.receptive_field = max(b.receptive_field for b in self.branches)

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.head(torch.cat([b(x) for b in self.branches], 1)).transpose(1, 2)


# ----------------------------------------------------------------------------- ModernTCN (causal)
class ConvFFN(nn.Module):
    """1x1 C -> r*C -> act -> 1x1 back to C (grouped when groups > 1). Gated act doubles the hidden layer."""

    def __init__(self, C, ratio, act, groups=1, dropout=0.0):
        super().__init__()
        hid = int(C * ratio)
        m = 2 if act == "gated" else 1
        self.f1 = nn.Conv1d(C, m * hid, 1, groups=groups)
        self.f2 = nn.Conv1d(hid, C, 1, groups=groups)
        self.act, self.drop = act_fn(act), nn.Dropout(dropout)

    def forward(self, x):
        return self.f2(self.drop(self.act(self.f1(x))))


class ModernBlock(nn.Module):
    """h = h + FFN(norm(DWConv_K,d(h) + DWConv_small(h)))."""

    def __init__(self, D, K, d, ratio, act, norm, small_k=5, dropout=0.0):
        super().__init__()
        wn = norm == "weight"
        self.dw = CausalConv1d(D, D, K, d, groups=D, weight_norm=wn)
        self.dw_small = CausalConv1d(D, D, small_k, 1, groups=D, weight_norm=wn) if small_k > 1 else None
        self.norm = maybe_norm(norm, D)
        self.ffn = ConvFFN(D, ratio, act, dropout=dropout)

    def forward(self, h):
        u = self.dw(h)
        if self.dw_small is not None:
            u = u + self.dw_small(h)
        return h + self.ffn(self.norm(u))


class ModernTCN(nn.Module):
    def __init__(self, width=64, kernel=51, dilations=(1, 4, 15), ffn_ratio=2.0, act="gelu", norm="layer",
                 small_kernel=5, dropout=0.0, n_in=N_IN):
        super().__init__()
        self.inp = nn.Conv1d(n_in, width, 1)
        self.blocks = nn.ModuleList([ModernBlock(width, kernel, d, ffn_ratio, act, norm, small_kernel, dropout)
                                     for d in dilations])
        self.head = nn.Conv1d(width, 2, 1)
        self.receptive_field = 1 + (kernel - 1) * sum(dilations)

    def forward(self, x):
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = b(h)
        return self.head(h).transpose(1, 2)


class _Permute(nn.Module):
    """Channel layout [var, dv] <-> [dv, var] (per row, no time mixing)."""

    def __init__(self, a, b):
        super().__init__()
        self.a, self.b = a, b

    def forward(self, x):                      # [B, a*b, T] -> [B, b*a, T]
        B, _, T = x.shape
        return x.reshape(B, self.a, self.b, T).transpose(1, 2).reshape(B, self.a * self.b, T)


class ModernVIBlock(nn.Module):
    """Variable-independent ModernTCN block: depthwise conv per (variable, dim); FFN1 within each variable;
    FFN2 across variables within each embedding dim."""

    def __init__(self, V, Dv, K, d, ratio, act, norm, small_k=5, dropout=0.0):
        super().__init__()
        C = V * Dv
        wn = norm == "weight"
        self.dw = CausalConv1d(C, C, K, d, groups=C, weight_norm=wn)
        self.dw_small = CausalConv1d(C, C, small_k, 1, groups=C, weight_norm=wn) if small_k > 1 else None
        self.norm = maybe_norm(norm, C)
        self.ffn1 = ConvFFN(C, ratio, act, groups=V, dropout=dropout)            # inside each variable
        self.to_dim, self.to_var = _Permute(V, Dv), _Permute(Dv, V)
        self.ffn2 = ConvFFN(C, ratio, act, groups=Dv, dropout=dropout)           # across variables

    def forward(self, h):
        u = self.dw(h)
        if self.dw_small is not None:
            u = u + self.dw_small(h)
        u = self.norm(u)
        u = u + self.ffn1(u)
        u = self.to_var(self.ffn2(self.to_dim(u)) + self.to_dim(u))
        return h + u


class ModernTCNVI(nn.Module):
    def __init__(self, var_dim=8, kernel=51, dilations=(1, 4, 15), ffn_ratio=1.0, act="gelu", norm="layer",
                 small_kernel=5, dropout=0.0, n_in=N_IN):
        super().__init__()
        self.V, self.Dv = n_in, var_dim
        self.inp = nn.Conv1d(n_in, n_in * var_dim, 1, groups=n_in)               # each variable -> var_dim
        self.blocks = nn.ModuleList([ModernVIBlock(n_in, var_dim, kernel, d, ffn_ratio, act, norm, small_kernel, dropout)
                                     for d in dilations])
        self.head = nn.Conv1d(n_in * var_dim, 2, 1)
        self.receptive_field = 1 + (kernel - 1) * sum(dilations)

    def forward(self, x):
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = b(h)
        return self.head(h).transpose(1, 2)


# ----------------------------------------------------------------------------- GRU baseline
class GRUModel(nn.Module):
    """Same layout as my_baseline/train.py GRUModel (state_dict keys gru1.*, gru2.*, head.*)."""

    is_recurrent = True

    def __init__(self, hidden=64, n_in=N_IN):
        super().__init__()
        self.hidden = hidden
        self.gru1 = nn.GRU(n_in, hidden, batch_first=True)
        self.gru2 = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)
        self.receptive_field = None            # unbounded memory

    def forward(self, x, state=None):
        h1, h2 = state if state is not None else (None, None)
        y1, h1 = self.gru1(x, h1)
        y2, h2 = self.gru2(y1, h2)
        return self.head(y2), (h1, h2)


# ----------------------------------------------------------------------------- factory
def build_model(cfg: dict) -> nn.Module:
    cfg = dict(cfg)
    fam = cfg.pop("family")
    if fam == "tcn":
        return ResidualTCN(**cfg)
    if fam == "multiscale":
        return MultiScaleTCN(**cfg)
    if fam == "modern":
        return ModernTCN(**cfg)
    if fam == "modern_vi":
        return ModernTCNVI(**cfg)
    if fam == "gru":
        return GRUModel(**cfg)
    raise ValueError(f"unknown family {fam}")


def is_recurrent(model) -> bool:
    return getattr(model, "is_recurrent", False)


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def causal_convs(model):
    return [m for m in model.modules() if isinstance(m, CausalConv1d)]


def inference_copy(model):
    """Deep copy in eval mode with weight-norm folded into plain weights (same maths, faster, exportable)."""
    m = copy.deepcopy(model).cpu().eval()
    for c in causal_convs(m):
        if nn.utils.parametrize.is_parametrized(c.conv):
            # replace by a plain conv holding the computed weight (never mutate the shared parametrized class)
            old = c.conv
            new = nn.Conv1d(c.cin, c.cout, c.k, dilation=c.d, groups=c.groups)
            with torch.no_grad():
                new.weight.copy_(old.weight); new.bias.copy_(old.bias)
            c.conv = new
    return m


def pow2_dilations(L):
    return [2 ** i for i in range(L)]


def rf_tcn(kernel, dilations):
    return 1 + 2 * (kernel - 1) * sum(dilations)
