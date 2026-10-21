"""Damage parameterizations -> nodal thickness ratio tau = t/t0 on the operator grid.

All builders are torch (differentiable, batched) so the same code serves dataset
generation, unlabelled physics fields and the observation function used by the samplers.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from scipy.stats import norm, qmc

from .grid import Grid

TAU_MIN = 1e-3


def _coords(grid: Grid, device, dtype):
    Y, X = grid.mesh()
    return (torch.tensor(X, device=device, dtype=dtype),
            torch.tensor(Y, device=device, dtype=dtype))


# ----------------------------------------------------------------------------- pit
def pit_tau(theta: torch.Tensor, grid: Grid) -> torch.Tensor:
    """theta: (b, 4) = (xc, yc, r, delta) in mm, mm, mm, fraction of t0 -> (b, ny+1, nx+1)."""
    X, Y = _coords(grid, theta.device, theta.dtype)
    xc, yc, r, d = (theta[:, i].view(-1, 1, 1) for i in range(4))
    d2 = (X - xc) ** 2 + (Y - yc) ** 2
    return 1.0 - d * torch.exp(-d2 / (2.0 * r ** 2))


# ----------------------------------------------------------------------------- KL field
class KLBasis:
    """Truncated KL expansion of a unit-variance squared-exponential field on the grid nodes.

    The SE kernel is separable, so the 2D eigenpairs are products of 1D ones."""

    def __init__(self, grid: Grid, ell: float, var_fraction=0.95, K_max=40):
        lx, vx = self._eig1d(grid.x, ell)
        ly, vy = self._eig1d(grid.y, ell)
        lam = np.outer(ly, lx)                         # (ny+1, nx+1) eigenvalue products
        order = np.argsort(lam.ravel())[::-1]
        total = len(grid.x) * len(grid.y)              # trace of the unit-diagonal kernel
        cum = np.cumsum(lam.ravel()[order]) / total
        K = int(np.searchsorted(cum, var_fraction) + 1)
        self.K_needed = K
        K = min(K, K_max)
        self.K = K
        self.var_held = float(cum[K - 1])
        jy, ix = np.unravel_index(order[:K], lam.shape)
        self.lam = lam.ravel()[order[:K]]
        # phi_k(y, x) = vy[:, j] vx[:, i]; orthonormal on the nodes
        self.phi = np.einsum("yk,xk->kyx", vy[:, jy], vx[:, ix])
        self.ell = ell

    @staticmethod
    def _eig1d(z, ell):
        C = np.exp(-(z[:, None] - z[None, :]) ** 2 / (2 * ell ** 2))
        w, v = np.linalg.eigh(C)
        w = np.clip(w, 0, None)
        return w[::-1], v[:, ::-1]

    def field(self, xi: torch.Tensor) -> torch.Tensor:
        """xi: (b, >=K) standard normals -> g (b, ny+1, nx+1)."""
        phi = torch.as_tensor(self.phi, device=xi.device, dtype=xi.dtype)
        s = torch.as_tensor(np.sqrt(self.lam), device=xi.device, dtype=xi.dtype)
        return torch.einsum("bk,kyx->byx", xi[:, : self.K] * s, phi)


@lru_cache(maxsize=8)
def kl_basis(grid: Grid, ell: float, var_fraction: float, K_max: int) -> KLBasis:
    return KLBasis(grid, ell, var_fraction, K_max)


def kl_tau(xi: torch.Tensor, c: torch.Tensor, basis: KLBasis, dt_max: float) -> torch.Tensor:
    g = basis.field(xi)
    return 1.0 - dt_max * torch.sigmoid(g - c.view(-1, 1, 1))


# ----------------------------------------------------------------------------- triangle
def triangle_tau(theta: torch.Tensor, grid: Grid) -> torch.Tensor:
    """theta: (b, 7) = (x1, y1, x2, y2, x3, y3, delta): uniform loss inside the triangle."""
    X, Y = _coords(grid, theta.device, theta.dtype)
    x1, y1, x2, y2, x3, y3, d = (theta[:, i].view(-1, 1, 1) for i in range(7))

    def side(ax, ay, bx, by):
        return (X - bx) * (ay - by) - (ax - bx) * (Y - by)

    s1, s2, s3 = side(x1, y1, x2, y2), side(x2, y2, x3, y3), side(x3, y3, x1, y1)
    neg = (s1 < 0) | (s2 < 0) | (s3 < 0)
    pos = (s1 > 0) | (s2 > 0) | (s3 > 0)
    inside = ~(neg & pos)
    return 1.0 - d * inside.to(theta.dtype)


# ----------------------------------------------------------------------------- sampling
def sobol(n: int, d: int, seed: int) -> np.ndarray:
    m = int(np.ceil(np.log2(max(n, 2))))
    return qmc.Sobol(d, scramble=True, seed=seed).random_base2(m)[:n]


def _scale(u, lo, hi):
    return lo + (hi - lo) * u


def sample_pit(n, cfg, seed):
    """Returns theta (n, 4) and nu (n,)."""
    p = cfg["priors"]["pit"]
    u = sobol(n, 5, seed)
    th = np.stack([_scale(u[:, 0], *p["xc"]), _scale(u[:, 1], *p["yc"]),
                   _scale(u[:, 2], *p["r"]), _scale(u[:, 3], *p["delta"])], 1)
    nu = _scale(u[:, 4], *cfg["priors"]["nu"])
    return th, nu


def sample_kl(n, cfg, seed):
    """Returns theta (n, 2 + K_max) = (ell, c, xi...) and nu (n,). ell alternates."""
    k = cfg["priors"]["kl"]
    K = k["K_max"]
    u = sobol(n, K + 2, seed)
    xi = norm.ppf(np.clip(u[:, :K], 1e-6, 1 - 1e-6))
    c = _scale(u[:, K], *k["c"])
    nu = _scale(u[:, K + 1], *cfg["priors"]["nu"])
    ells = np.asarray(k["ells"])[np.arange(n) % len(k["ells"])]
    return np.concatenate([ells[:, None], c[:, None], xi], 1), nu


def sample_triangle(n, cfg, grid: Grid, seed):
    t = cfg["priors"]["triangle"]
    rng = np.random.default_rng(seed)
    th = np.zeros((n, 7))
    for i in range(n):
        cx = rng.uniform(0.25 * grid.L, 0.75 * grid.L)
        cy = rng.uniform(0.25 * grid.W, 0.75 * grid.W)
        s = rng.uniform(*t["size"])
        ang = rng.uniform(0, 2 * np.pi) + np.array([0, 2.1, 4.2]) + rng.normal(0, 0.25, 3)
        rad = s / np.sqrt(3) * rng.uniform(0.7, 1.3, 3)
        th[i, 0:6:2] = cx + rad * np.cos(ang)
        th[i, 1:6:2] = cy + rad * np.sin(ang)
        th[i, 6] = rng.uniform(*t["delta"])
    nu = rng.uniform(*cfg["priors"]["nu"], n)
    return th, nu


def tau_from_theta(family: str, theta, cfg, grid: Grid, device="cpu", dtype=torch.float64):
    """Dispatch on damage family; theta numpy or torch (b, d) -> torch tau (b, ny+1, nx+1)."""
    th = torch.as_tensor(theta, device=device, dtype=dtype)
    if family == "pit":
        tau = pit_tau(th, grid)
    elif family == "kl":
        k = cfg["priors"]["kl"]
        tau = torch.empty(th.shape[0], *grid.shape, device=device, dtype=dtype)
        for ell in torch.unique(th[:, 0]).tolist():
            m = th[:, 0] == ell
            basis = kl_basis(grid, float(ell), k["var_fraction"], k["K_max"])
            tau[m] = kl_tau(th[m, 2:], th[m, 1], basis, k["dt_max"])
    elif family == "triangle":
        tau = triangle_tau(th, grid)
    else:
        raise ValueError(family)
    return tau.clamp_min(TAU_MIN)
