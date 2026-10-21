"""Discrete Q1 FE residual ||K(tau, nu) u - f||^2 / ||f||^2 on the operator grid (Step 4.2).

K_e = t_e / (1 - nu^2) (A + nu B) with two fixed 8x8 matrices, so K(t)u is a gather, two
small matrix products and a scatter-add: fast on a GPU and fully differentiable.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..grid import Grid


def q1_element_mats(dtype=torch.float32, device="cpu"):
    """Unit-thickness, E = 1 stiffness of a square Q1 element: K_e = (A + nu*B)/(1 - nu^2).
    For a square element the result does not depend on its size."""
    g = np.array([-1.0, 1.0]) / np.sqrt(3.0)
    D0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0.5]])
    D1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -0.5]])
    xi_n, eta_n = np.array([-1, 1, 1, -1]), np.array([-1, -1, 1, 1])  # BL, BR, TR, TL
    A, B = np.zeros((8, 8)), np.zeros((8, 8))
    for xi in g:
        for eta in g:
            dxi, deta = 0.25 * xi_n * (1 + eta * eta_n), 0.25 * eta_n * (1 + xi * xi_n)
            Bm = np.zeros((3, 8))
            Bm[0, 0::2], Bm[1, 1::2] = dxi, deta
            Bm[2, 0::2], Bm[2, 1::2] = deta, dxi
            A += Bm.T @ D0 @ Bm
            B += Bm.T @ D1 @ Bm
    return (torch.tensor(A, dtype=dtype, device=device),
            torch.tensor(B, dtype=dtype, device=device))


def apply_K(u_hat, tau, nu, A, B):
    """K(tau, nu) u_hat as nodal forces, (b, ny+1, nx+1, 2)."""
    b = u_hat.shape[0]
    te = 0.25 * (tau[:, :-1, :-1] + tau[:, :-1, 1:] + tau[:, 1:, 1:] + tau[:, 1:, :-1])
    ue = torch.cat([u_hat[:, :-1, :-1], u_hat[:, :-1, 1:],
                    u_hat[:, 1:, 1:], u_hat[:, 1:, :-1]], dim=-1)          # (b, ny, nx, 8)
    n = nu.view(b, 1, 1, 1)
    fe = te.unsqueeze(-1) / (1 - n ** 2) * (ue @ A.T + n * (ue @ B.T))
    return (F.pad(fe[..., 0:2], (0, 0, 0, 1, 0, 1)) + F.pad(fe[..., 2:4], (0, 0, 1, 0, 0, 1))
            + F.pad(fe[..., 4:6], (0, 0, 1, 0, 1, 0)) + F.pad(fe[..., 6:8], (0, 0, 0, 1, 1, 0)))


def residual(u_hat, tau, nu, A, B, f_hat, free):
    """u_hat: (b, ny+1, nx+1, 2) full nondimensional displacement; tau: (b, ny+1, nx+1);
    nu: (b,); f_hat, free: (ny+1, nx+1, 2). Returns ||K u - f||^2 / ||f||^2 per sample."""
    r = (apply_K(u_hat, tau, nu, A, B) - f_hat) * free
    return r.pow(2).sum((1, 2, 3)) / (f_hat * free).pow(2).sum()


class Residual(torch.nn.Module):
    """Holds A, B, f_hat and free as buffers so .to(device) moves them."""

    def __init__(self, grid: Grid, dtype=torch.float32, bc="roller"):
        super().__init__()
        A, B = q1_element_mats(dtype)
        f, free = grid.loads_and_mask(dtype, bc=bc)
        self.register_buffer("A", A)
        self.register_buffer("B", B)
        self.register_buffer("f_hat", f)
        self.register_buffer("free", free)

    def forward(self, u_hat, tau, nu):
        return residual(u_hat, tau, nu, self.A, self.B, self.f_hat, self.free)
