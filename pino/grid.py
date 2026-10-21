"""The fixed operator grid shared by the FE data, the surrogates and the residual.

Arrays are (ny+1, nx+1[, c]) with rows = y and columns = x.  Nondimensional
displacement u_hat = E t0 u / P; physical displacement u = P/(E t0) * u_hat.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Grid:
    L: float = 300.0
    W: float = 250.0
    nx: int = 150
    ny: int = 125
    t0: float = 3.2

    @classmethod
    def from_cfg(cls, cfg):
        g = cfg["geometry"]
        return cls(L=g["L"], W=g["W"], nx=g["nx"], ny=g["ny"], t0=g["t0"])

    @property
    def h(self) -> float:
        hx, hy = self.L / self.nx, self.W / self.ny
        assert abs(hx - hy) < 1e-12, "the residual assumes square elements"
        return hx

    @property
    def hxy(self):
        return (self.L / self.nx, self.W / self.ny)

    @property
    def shape(self):
        return (self.ny + 1, self.nx + 1)

    @property
    def x(self):
        return np.linspace(0.0, self.L, self.nx + 1)

    @property
    def y(self):
        return np.linspace(0.0, self.W, self.ny + 1)

    def mesh(self):
        """(Y, X) node coordinates in mm, each (ny+1, nx+1)."""
        return np.meshgrid(self.y, self.x, indexing="ij")

    @property
    def xc(self):  # element centres
        return 0.5 * (self.x[:-1] + self.x[1:])

    @property
    def yc(self):
        return 0.5 * (self.y[:-1] + self.y[1:])

    # ------------------------------------------------------------------ intact solution
    def u0_hat(self, nu):
        """Intact nondimensional solution (x/W, -nu y/W). nu scalar -> (ny+1,nx+1,2)."""
        Y, X = self.mesh()
        return np.stack([X / self.W, -nu * Y / self.W], axis=-1)

    def u0_hat_torch(self, nu: torch.Tensor):
        """nu: (b,) -> (b, ny+1, nx+1, 2)."""
        dev, dt = nu.device, nu.dtype
        X = torch.tensor(self.mesh()[1], device=dev, dtype=dt)
        Y = torch.tensor(self.mesh()[0], device=dev, dtype=dt)
        ux = (X / self.W).expand(nu.shape[0], -1, -1)
        uy = -nu.view(-1, 1, 1) * Y / self.W
        return torch.stack([ux, uy], dim=-1)

    def loads_and_mask(self, dtype=torch.float64, device="cpu", bc="roller"):
        """f_hat: right-edge nodal loads h/W in x (half at corners) -> total force 1.
        free: 1 except u1 on the left edge and u2 at the pinned corner (0, 0)."""
        f = torch.zeros(*self.shape, 2, dtype=dtype, device=device)
        f[:, -1, 0] = self.h / self.W
        f[0, -1, 0] *= 0.5
        f[-1, -1, 0] *= 0.5
        free = torch.ones_like(f)
        free[:, 0, 0] = 0.0
        if bc == "clamped":
            free[:, 0, 1] = 0.0
        else:
            free[0, 0, 1] = 0.0
        return f, free


def q1_strain(u, h):
    """Element-centre small strains from nodal displacement with the Q1 gradient.

    u: (..., ny+1, nx+1, 2) torch or numpy. Returns (..., ny, nx, 3) = (e11, e22, e12)
    with the tensor shear e12 = gamma/2.  This is the one strain function of the project.
    """
    bl, br = u[..., :-1, :-1, :], u[..., :-1, 1:, :]
    tl, tr = u[..., 1:, :-1, :], u[..., 1:, 1:, :]
    hx, hy = (h, h) if np.isscalar(h) else h
    dx = (br - bl + tr - tl) / (2.0 * hx)  # d/dx of (u1, u2)
    dy = (tl - bl + tr - br) / (2.0 * hy)  # d/dy of (u1, u2)
    e11, e22 = dx[..., 0], dy[..., 1]
    e12 = 0.5 * (dy[..., 0] + dx[..., 1])
    stack = torch.stack if isinstance(u, torch.Tensor) else np.stack
    return stack([e11, e22, e12], -1)
