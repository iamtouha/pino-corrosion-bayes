"""Inverse problem assembly (Step 6): priors, the observation function, Sigma_G, likelihoods.

Parameter vectors put the damage parameters first and (E, nu) last:
  pit: (xc, yc, r, delta, E, nu)          kl: (c, xi_1..xi_K, E, nu)
Observations are in microstrain, stacked [load][component][point].
"""
from __future__ import annotations

import dataclasses

import math

import numpy as np
import torch

from .. import thickness as th
from ..grid import Grid, q1_strain
from ..vdic.noise import NoiseModel
from .likelihood import GaussianLik, KronBlocks, dense_noise_cov

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- priors
class Family:
    def __init__(self, name, cfg, grid: Grid):
        self.name, self.cfg, self.grid = name, cfg, grid
        pr = cfg["priors"]
        if name == "pit":
            p = pr["pit"]
            self.names = ["xc", "yc", "r", "delta", "E", "nu"]
            self.lo = np.array([p["xc"][0], p["yc"][0], p["r"][0], p["delta"][0], pr["E"][0], pr["nu"][0]])
            self.hi = np.array([p["xc"][1], p["yc"][1], p["r"][1], p["delta"][1], pr["E"][1], pr["nu"][1]])
            self.gauss = np.zeros(6, bool)
        elif name == "kl":
            k = pr["kl"]
            self.basis = th.kl_basis(grid, float(k["infer_ell"]), k["var_fraction"], k["K_max"])
            K = self.basis.K
            self.names = ["c"] + [f"xi{i + 1}" for i in range(K)] + ["E", "nu"]
            self.lo = np.r_[k["c"][0], -np.inf * np.ones(K), pr["E"][0], pr["nu"][0]]
            self.hi = np.r_[k["c"][1], np.inf * np.ones(K), pr["E"][1], pr["nu"][1]]
            self.gauss = np.r_[False, np.ones(K, bool), False, False]
        else:
            raise ValueError(name)
        self.dim = len(self.names)

    @property
    def width(self):
        w = self.hi - self.lo
        w[self.gauss] = 1.0 * 2 * 1.645      # 90% width of N(0,1) for normalization
        return w

    def sample(self, n, rng):
        u = rng.random((n, self.dim))
        x = self.lo + (self.hi - self.lo) * np.where(self.gauss, 0.5, u)
        x[:, self.gauss] = rng.standard_normal((n, int(self.gauss.sum())))
        return x

    def log_prior(self, x):
        lp = np.zeros(len(x))
        u = ~self.gauss
        inside = np.all((x[:, u] >= self.lo[u]) & (x[:, u] <= self.hi[u]), 1)
        lp[~inside] = -np.inf
        if self.gauss.any():
            lp += -0.5 * (x[:, self.gauss] ** 2).sum(1)
        return lp

    def tau(self, x: torch.Tensor) -> torch.Tensor:
        if self.name == "pit":
            return th.pit_tau(x[:, :4], self.grid).clamp_min(th.TAU_MIN)
        k = self.cfg["priors"]["kl"]
        return th.kl_tau(x[:, 1:-2], x[:, 0], self.basis, k["dt_max"]).clamp_min(th.TAU_MIN)

    def derived(self, x: np.ndarray) -> dict:
        """Better-identified summaries: lost volume (and for pits delta r^2 itself)."""
        if self.name == "pit":
            return {"delta_r2": x[:, 3] * x[:, 2] ** 2}
        tau = self.tau(torch.as_tensor(x, dtype=torch.float64)).numpy()
        h = self.grid.h
        return {"lost_volume_mm3": self.grid.t0 * h * h * (1 - tau).sum((1, 2)),
                "max_loss": (1 - tau).max((1, 2))}


# ----------------------------------------------------------------------------- observer
class Observer:
    """observe(theta) -> predicted data (b, n_obs) in microstrain, differentiable in theta."""

    def __init__(self, fam: Family, H, loads_kN, surrogate=None, fe_pool=None, dtype=torch.float32):
        self.fam, self.H, self.grid = fam, H, fam.grid
        self.loads = torch.tensor(loads_kN, dtype=torch.float64) * 1e3
        self.sur, self.fe = surrogate, fe_pool
        self.dtype = dtype

    @property
    def n_obs(self):
        return len(self.loads) * 3 * self.H.n_points

    def unit_obs(self, x: torch.Tensor) -> torch.Tensor:
        """H(grad u_hat) for each theta row (nondimensional, per unit P/(E t0))."""
        if self.sur is not None:
            x = x.to(DEV)
            tau = self.fam.tau(x).to(self.dtype)
            nu = x[:, -1].to(self.dtype)
            u = self.sur.u_hat(tau, nu)
        else:
            tau = self.fam.tau(x.detach().cpu().double()).numpy()
            u = torch.as_tensor(self.fe.solve(tau, x[:, -1].detach().cpu().numpy()), device=DEV)
        return self.H(q1_strain(u.to(self.H.k.dtype), self.grid.h))

    def scale(self, E):
        """(b, n_loads) microstrain factor P_l / (E t0) * 1e6."""
        return (self.loads.to(E.device, E.dtype)[None] / (E[:, None] * self.grid.t0)) * 1e6

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        h = self.unit_obs(x).double()                            # (b, m)
        s = self.scale(x[:, -2].to(h.device, torch.float64))     # (b, L)
        return (s[:, :, None] * h[:, None, :]).reshape(len(x), -1)


class FEPool:
    """Persistent pool of 2D FE workers for FE-in-the-loop reference posteriors."""

    def __init__(self, grid: Grid, workers: int):
        import multiprocessing as mp
        from ..fe import plate2d
        plate2d.PlateSolver2D(grid).solve(np.ones(grid.shape), 0.3)
        self.pool = mp.get_context("fork").Pool(workers, initializer=plate2d._init_worker, initargs=(grid,))
        self._solve = plate2d._solve_one

    def solve(self, taus, nus):
        return np.stack(self.pool.map(self._solve, list(zip(taus, nus)), chunksize=8))

    def close(self):
        self.pool.close(); self.pool.join()


# ----------------------------------------------------------------------------- Sigma_G
def surrogate_error(sur, H, tau_D, w_D, nu_D, rank, bs=16):
    """R = H(eps_FE) - H(G_phi) on held-out set D (nondimensional, unit load).
    Returns mean mu (m,), basis U (m, k), eigenvalues lam (k,), leftover diag variance (m,)."""
    rs = []
    with torch.no_grad():
        for s in range(0, len(nu_D), bs):
            t, w, n = tau_D[s:s + bs].to(DEV), w_D[s:s + bs].to(DEV), nu_D[s:s + bs].to(DEV)
            wp = sur(t, n)
            rs.append((H(q1_strain(w, sur.grid.h)) - H(q1_strain(wp, sur.grid.h))).double().cpu())
    R = torch.cat(rs)                                            # (N_D, m)
    mu = R.mean(0)
    Rc = R - mu
    Uf, S, _ = torch.linalg.svd(Rc.T, full_matrices=False)       # (m, N_D)
    k = min(rank, len(S))
    lam = S[:k] ** 2 / max(1, len(R) - 1)
    U = Uf[:, :k]
    left = (Rc - (Rc @ U) @ U.T).pow(2).sum(0) / max(1, len(R) - 1)
    return {"mu": mu, "U": U, "lam": lam, "left": left, "n": len(R)}


def build_likelihood(obs: Observer, noise: NoiseModel, design_axes, sigma_g=None, E_nom=206e3,
                     device=DEV, nugget_floor_frac=0.0):
    """Returns (GaussianLik, mu_vector). Sigma = Sigma_DIC (+ Sigma_G stacked over loads).

    nugget_floor_frac: the white-noise std used in the likelihood is at least this fraction
    of the correlated std. The fitted nugget of the virtual DIC is ~1e-6 of sigma (VSG output
    is smooth), which makes the full-field SE covariance numerically singular (cond ~1e12):
    float32 round-off in the surrogate then dominates the log-likelihood."""
    if nugget_floor_frac:
        noise = dataclasses.replace(noise, nugget=[max(n_, nugget_floor_frac * s_) for n_, s_ in
                                                   zip(noise.nugget, noise.sigma)])
    xs, ys = design_axes
    L = len(obs.loads)
    m = obs.H.n_points
    n = obs.n_obs
    sig = np.array(noise.sigma) * 1e6; ell = np.array(noise.ell); nug = np.array(noise.nugget) * 1e6
    c = (obs.loads / (E_nom * obs.grid.t0) * 1e6).numpy()      # (L,)
    extra = np.zeros((L, 3))
    U = lam = None
    mu = torch.zeros(n, dtype=torch.float64)
    if sigma_g is not None:
        Ug = sigma_g["U"]
        U = torch.cat([float(cl) * Ug for cl in c], 0)           # fully correlated across loads
        U = U / U.norm(dim=0, keepdim=True)
        lam = sigma_g["lam"] * float((c ** 2).sum())
        mu = torch.cat([float(cl) * sigma_g["mu"] for cl in c])
        left = sigma_g["left"].reshape(3, m)
        extra = np.stack([float(cl) ** 2 * left.mean(1).numpy() for cl in c])   # (L, 3)
    if m <= 400:
        Sd = dense_noise_cov(noise, xs, ys, L) * 1e12
        Sd = Sd + torch.diag(torch.tensor(np.repeat(extra.ravel(), m)))
        lik = GaussianLik("dense", n, U, lam, dense=Sd, device=device)
    else:
        kb = KronBlocks(xs, ys, np.tile(sig, L), np.tile(ell, L),
                        np.sqrt(np.tile(nug ** 2, L) + extra.ravel()), device)
        lik = GaussianLik("kron", n, U, lam, kron=kb, device=device)
    return lik, mu.to(device)


def make_loglik_fn(obs: Observer, lik: GaussianLik, mu, d_obs, batch=16):
    """numpy (n, d) -> numpy (n,) log-likelihood, batched on the GPU (for TMCMC)."""
    d = d_obs.to(DEV, torch.float64)

    def f(x):
        out = []
        with torch.no_grad():
            for s in range(0, len(x), batch):
                xt = torch.as_tensor(x[s:s + batch], dtype=torch.float64)
                pred = obs(xt).to(DEV)
                out.append(lik(d[None] - pred - mu[None]).cpu())
        return torch.cat(out).numpy()

    return f
