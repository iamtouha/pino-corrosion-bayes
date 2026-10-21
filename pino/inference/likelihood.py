"""Gaussian likelihoods with structured covariance (Theory 4 / Step 6.2).

Sigma = B + U diag(lam) U^T, with B one of
  * s2 * I                           (the spec's loglik, Woodbury + determinant lemma)
  * block-diagonal Kronecker SE + nugget, one block per (load, component) on a rectilinear
    DIC grid (exact, never forms n x n)
  * dense (small designs such as the 56-point grid)
U stacks the surrogate-error basis (and optionally a DIC-noise basis).
"""
from __future__ import annotations

import math

import torch

from ..vdic.noise import KronSE, NoiseModel


def loglik(r, s2, U, lam):
    """r: residual (n,); Sigma = s2*I + U diag(lam) U^T, U: (n, k). Woodbury + determinant lemma."""
    n, k = U.shape
    Ut_r = U.T @ r
    C = torch.diag(1.0 / lam) + (U.T @ U) / s2               # (k, k)
    L = torch.linalg.cholesky(C)
    z = torch.cholesky_solve((Ut_r / s2).unsqueeze(-1), L).squeeze(-1)
    quad = (r @ r) / s2 - (Ut_r / s2) @ z
    logdet = n * math.log(s2) + torch.log(lam).sum() + 2 * torch.log(torch.diagonal(L)).sum()
    return -0.5 * (quad + logdet + n * math.log(2 * math.pi))


class KronBlocks:
    """B = blockdiag over blocks b of (sigma_b^2 Kx(x)Ky + s_b^2 I) on the same point grid."""

    def __init__(self, xs, ys, sig, ell, nug, device, dtype=torch.float64):
        self.py, self.px = len(ys), len(xs)
        self.kron = {}
        self.blocks = []
        for s_, l_, n_ in zip(sig, ell, nug):
            key = round(float(l_), 9)
            if key not in self.kron:
                self.kron[key] = KronSE(xs, ys, float(l_), device, dtype)
            k = self.kron[key]
            self.blocks.append((k, s_ ** 2 * k.lam + n_ ** 2))     # eigenvalues of the block
        self.nb = len(self.blocks)
        self.m = self.py * self.px
        self.logdet = sum(torch.log(ev).sum() for _, ev in self.blocks)

    def solve(self, R):
        """R: (..., n) with n = nb * m -> B^{-1} R."""
        sh = R.shape
        R = R.reshape(*sh[:-1], self.nb, self.py, self.px)
        out = torch.empty_like(R)
        for b, (k, ev) in enumerate(self.blocks):
            out[..., b, :, :] = k.unrotate(k.rotate(R[..., b, :, :]) / ev)
        return out.reshape(sh)


class GaussianLik(torch.nn.Module):
    """log N(d | mean, B + U diag(lam) U^T), batched over residual rows."""

    def __init__(self, base: str, n: int, U=None, lam=None, s2=None, kron: KronBlocks | None = None,
                 dense=None, device="cpu", dtype=torch.float64):
        super().__init__()
        self.base, self.n = base, n
        self.kron = kron
        if base == "dense":
            self.Lb = torch.linalg.cholesky(dense.to(device, dtype))
            self.logdet_b = 2 * torch.log(torch.diagonal(self.Lb)).sum()
        elif base == "iso":
            self.s2 = float(s2)
            self.logdet_b = n * math.log(self.s2)
        elif base == "kron":
            self.logdet_b = kron.logdet
        self.U = None if U is None or U.shape[1] == 0 else U.to(device, dtype)
        if self.U is not None:
            self.lam = lam.to(device, dtype)
            BiU = self.solve_b(self.U.T).T                    # (n, k)
            C = torch.diag(1.0 / self.lam) + self.U.T @ BiU
            self.Lc = torch.linalg.cholesky(C)
            self.BiU = BiU
            self.logdet = self.logdet_b + torch.log(self.lam).sum() + 2 * torch.log(torch.diagonal(self.Lc)).sum()
        else:
            self.logdet = self.logdet_b

    def solve_b(self, R):
        if self.base == "iso":
            return R / self.s2
        if self.base == "dense":
            return torch.cholesky_solve(R.T, self.Lb).T
        return self.kron.solve(R)

    def forward(self, r):
        """r: (b, n) residuals -> (b,) log-likelihoods."""
        Bi_r = self.solve_b(r)
        quad = (r * Bi_r).sum(-1)
        if self.U is not None:
            y = r @ self.BiU                                  # (b, k) = U^T B^{-1} r
            z = torch.cholesky_solve(y.T, self.Lc).T
            quad = quad - (y * z).sum(-1)
        return -0.5 * (quad + self.logdet + self.n * math.log(2 * math.pi))


def dense_noise_cov(noise: NoiseModel, xs, ys, n_loads: int, dtype=torch.float64):
    """Dense Sigma_DIC for a small rectilinear design: blockdiag over loads and components."""
    import numpy as np
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel()], 1)
    d2 = ((P[:, None] - P[None]) ** 2).sum(-1)
    blocks = []
    for _ in range(n_loads):
        for c in range(3):
            Kc = noise.sigma[c] ** 2 * np.exp(-d2 / (2 * noise.ell[c] ** 2)) + noise.nugget[c] ** 2 * np.eye(len(P))
            blocks.append(torch.tensor(Kc, dtype=dtype))
    return torch.block_diag(*blocks)
