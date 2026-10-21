"""Surrogates G_phi: (tau(.), nu) -> w(.) on the operator grid (Step 4.1 / 4.3).

Inputs (b, 4, ny+1, nx+1): (tau - 1)/0.15, x/L, y/W, (nu - 0.30)/0.03.
Outputs (b, ny+1, nx+1, 2): w divided by its training-set standard deviation.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from ..grid import Grid


def make_inputs(tau, nu, grid: Grid):
    """tau: (b, ny+1, nx+1), nu: (b,) -> (b, 4, ny+1, nx+1)."""
    b = tau.shape[0]
    Y, X = grid.mesh()
    X = torch.as_tensor(X / grid.L, dtype=tau.dtype, device=tau.device).expand(b, -1, -1)
    Y = torch.as_tensor(Y / grid.W, dtype=tau.dtype, device=tau.device).expand(b, -1, -1)
    n = ((nu - 0.30) / 0.03).view(b, 1, 1).expand_as(tau)
    return torch.stack([(tau - 1) / 0.15, X, Y, n], 1)


class DeepONet(nn.Module):
    """Branch: 4 stride-2 conv layers -> 256 -> 2p. Trunk: Fourier features of (x, y) ->
    MLP -> 2p. w_c(x) = sum_k b_ck t_ck(x) + b0_c."""

    def __init__(self, grid: Grid, p=128, n_fourier=64, trunk_width=128, trunk_layers=3,
                 branch_width=256, in_channels=4):
        super().__init__()
        self.grid, self.p = grid, p
        ch = [in_channels, 32, 64, 96, 128]
        convs = []
        for a, b_ in zip(ch[:-1], ch[1:]):
            convs += [nn.Conv2d(a, b_, 3, stride=2, padding=1), nn.GELU()]
        self.conv = nn.Sequential(*convs, nn.AdaptiveAvgPool2d((4, 5)), nn.Flatten())
        self.branch = nn.Sequential(nn.Linear(128 * 20, branch_width), nn.GELU(),
                                    nn.Linear(branch_width, 2 * p))
        g = torch.Generator().manual_seed(0)
        self.register_buffer("Bf", torch.randn(2, n_fourier // 2, generator=g) * 4.0)
        layers, d = [], n_fourier
        for _ in range(trunk_layers):
            layers += [nn.Linear(d, trunk_width), nn.GELU()]
            d = trunk_width
        layers.append(nn.Linear(d, 2 * p))
        self.trunk = nn.Sequential(*layers)
        self.bias = nn.Parameter(torch.zeros(2))
        Y, X = grid.mesh()
        xy = torch.tensor(np.stack([X.ravel() / grid.L, Y.ravel() / grid.W], 1), dtype=torch.float32)
        self.register_buffer("xy", xy)

    def forward(self, x):
        b = x.shape[0]
        coef = self.branch(self.conv(x)).view(b, 2, self.p)
        z = 2 * math.pi * self.xy @ self.Bf
        t = self.trunk(torch.cat([torch.sin(z), torch.cos(z)], -1)).view(-1, 2, self.p)
        w = torch.einsum("bcp,ncp->bnc", coef, t) / math.sqrt(self.p) + self.bias
        return w.view(b, *self.grid.shape, 2)


class FNO(nn.Module):
    """Thin wrapper around neuralop.models.FNO that returns (b, ny+1, nx+1, 2)."""

    def __init__(self, grid: Grid, n_modes=(32, 32), hidden_channels=64, n_layers=4,
                 domain_padding=0.125, in_channels=4):
        super().__init__()
        from neuralop.models import FNO as _FNO
        self.net = _FNO(n_modes=tuple(n_modes), hidden_channels=hidden_channels,
                        in_channels=in_channels, out_channels=2, n_layers=n_layers,
                        domain_padding=domain_padding, positional_embedding=None)

    def forward(self, x):
        return self.net(x).permute(0, 2, 3, 1)


def build(name: str, cfg, grid: Grid) -> nn.Module:
    m = cfg["model"]
    if name == "deeponet":
        return DeepONet(grid, **m["deeponet"])
    if name in ("fno", "pino"):
        return FNO(grid, **m["fno"])
    raise ValueError(name)


class Surrogate(nn.Module):
    """Trained network + normalization: (tau, nu) -> w in nondimensional units."""

    def __init__(self, net: nn.Module, grid: Grid, w_std):
        super().__init__()
        self.net, self.grid = net, grid
        self.register_buffer("w_std", torch.as_tensor(w_std, dtype=torch.float32).view(2))

    def forward(self, tau, nu):
        return self.net(make_inputs(tau, nu, self.grid)) * self.w_std

    def u_hat(self, tau, nu):
        return self.grid.u0_hat_torch(nu) + self(tau, nu)


def load_surrogate(path, cfg, grid: Grid, device="cpu") -> Surrogate:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = build(ck["model"], cfg, grid)
    s = Surrogate(net, grid, ck["w_std"])
    s.load_state_dict(clean_state(ck["state"]))
    return s.to(device).eval()


def clean_state(sd):
    """neuraloperator adds a non-tensor '_metadata' entry to its state dict; drop it."""
    return {k: v for k, v in sd.items() if not k.endswith("_metadata")}
