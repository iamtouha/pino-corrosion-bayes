"""DIC noise model Sigma_DIC (Step 3.4) and the fast noise emulator (Step 3.6).

Per strain component c: n_c = sigma_c * (separable SE field, length ell_c) + s_c * white.
On a rectilinear point grid the SE part is Kx (x) Ky, so sampling and the likelihood use
1D eigendecompositions only; nothing of size n x n is ever formed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares


@dataclass
class NoiseModel:
    sigma: list = field(default_factory=lambda: [30e-6, 30e-6, 15e-6])   # strain units
    ell: list = field(default_factory=lambda: [4.0, 4.0, 4.0])           # mm
    nugget: list = field(default_factory=lambda: [5e-6, 5e-6, 3e-6])
    bias_strain: list = field(default_factory=list)   # mean residual vs strain level (table)
    bias_value: list = field(default_factory=list)
    source: str = "default"

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def load_noise(cfg) -> NoiseModel:
    p = cfg.data_dir / "vdic" / "noise_model.json"
    return NoiseModel.load(p) if p.exists() else NoiseModel()


def se_1d(z, ell):
    return np.exp(-(z[:, None] - z[None, :]) ** 2 / (2 * ell ** 2))


class KronSE:
    """Eigendecomposition of an SE kernel on a (py, px) point grid."""

    def __init__(self, xs, ys, ell, device="cpu", dtype=torch.float64):
        wx, Qx = np.linalg.eigh(se_1d(np.asarray(xs), ell))
        wy, Qy = np.linalg.eigh(se_1d(np.asarray(ys), ell))
        t = lambda a: torch.tensor(a, device=device, dtype=dtype)
        self.Qx, self.Qy = t(Qx), t(Qy)
        self.lam = t(np.outer(np.clip(wy, 0, None), np.clip(wx, 0, None)))   # (py, px)

    def sample(self, n, gen=None):
        z = torch.randn(n, *self.lam.shape, generator=gen, device=self.lam.device, dtype=self.lam.dtype)
        return torch.einsum("yi,nij,xj->nyx", self.Qy, z * self.lam.sqrt(), self.Qx)

    def rotate(self, R):          # Q^T R  for R (..., py, px)
        return torch.einsum("yi,...yx,xj->...ij", self.Qy, R, self.Qx)

    def unrotate(self, R):
        return torch.einsum("yi,...ij,xj->...yx", self.Qy, R, self.Qx)


class NoiseEmulator:
    """Draws virtual-DIC noise on a rectilinear design from a fitted NoiseModel."""

    def __init__(self, model: NoiseModel, xs, ys, device="cpu", dtype=torch.float64):
        self.m = model
        self.kron = [KronSE(xs, ys, model.ell[c], device, dtype) for c in range(3)]
        self.shape = (len(ys), len(xs))

    def sample(self, n, gen=None):
        """(n, 3 * py * px) noise in strain units, component-major."""
        out = []
        for c in range(3):
            k = self.kron[c]
            e = self.m.sigma[c] * k.sample(n, gen)
            e = e + self.m.nugget[c] * torch.randn(e.shape, generator=gen, device=e.device, dtype=e.dtype)
            out.append(e.reshape(n, -1))
        return torch.cat(out, 1)

    def bias(self, strain_obs: torch.Tensor):
        """Mean residual as a function of the local strain magnitude (|e11|), from the table."""
        if not self.m.bias_strain:
            return torch.zeros_like(strain_obs)
        s = np.asarray(self.m.bias_strain); v = np.asarray(self.m.bias_value)
        x = strain_obs.detach().abs().cpu().numpy()
        return torch.as_tensor(np.interp(x, s, v), dtype=strain_obs.dtype, device=strain_obs.device)


# ----------------------------------------------------------------------------- fitting
def empirical_cov_profile(res: np.ndarray, spacing: float, max_lag: int):
    """res: (n_real, py, px) zero-mean residual fields. Returns lags (mm) and covariance along
    x and y axes (averaged), lag 0 included."""
    res = res - res.mean(0, keepdims=True)  # remove systematic part (bias), keep noise only
    cov = []
    for L in range(max_lag + 1):
        cx = (res[:, :, L:] * res[:, :, :res.shape[2] - L]).mean()
        cy = (res[:, L:, :] * res[:, :res.shape[1] - L, :]).mean()
        cov.append(0.5 * (cx + cy))
    return np.arange(max_lag + 1) * spacing, np.asarray(cov)


def fit_se_nugget(lags, cov):
    """cov(0) = sigma^2 + s^2, cov(r>0) = sigma^2 exp(-r^2 / 2 ell^2)."""
    c0 = max(cov[0], 1e-30)

    def f(p):
        s2, ell, n2 = np.exp(p)
        model = s2 * np.exp(-lags ** 2 / (2 * ell ** 2))
        model[0] += n2
        return (model - cov) / c0

    p0 = np.log([0.8 * c0, max(lags[1], 1e-3) * 3, 0.2 * c0])
    r = least_squares(f, p0, bounds=(np.log([1e-12 * c0, 0.1, 1e-12 * c0]), np.log([10 * c0, 100, 10 * c0])))
    s2, ell, n2 = np.exp(r.x)
    return float(np.sqrt(s2)), float(ell), float(np.sqrt(n2))
