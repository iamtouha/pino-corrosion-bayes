"""Surrogate-error and model-discrepancy likelihood terms (follow-up study T2, README).

* woodbury_ll: log N(r | 0, diag(D) + U diag(lam) U^T), batched over rows, with a different
  diagonal per row (inferred noise scales) - the Sigma_G low-rank term via Woodbury and the
  matrix determinant lemma.
* Kennedy-O'Hagan discrepancy: zero-mean GP over the point coordinates per field component,
  SE kernel + white noise; hyperparameters by type-II maximum likelihood on a residual.
"""
from __future__ import annotations

import math

import torch


def woodbury_ll(r, Dvar, U, lam):
    """r, Dvar: (b, m); U: (m, k); lam: (k,). Returns (b,) log-density without the 2 pi term."""
    Di = 1.0 / Dvar
    UtDr = (r * Di) @ U                                                   # (b, k)
    C = torch.diag_embed(1.0 / lam).expand(len(r), -1, -1) + torch.einsum("mk,bm,ml->bkl", U, Di, U)
    L = torch.linalg.cholesky(C)
    z = torch.cholesky_solve(UtDr.unsqueeze(-1), L).squeeze(-1)
    quad = (r * r * Di).sum(1) - (UtDr * z).sum(1)
    logdet = torch.log(Dvar).sum(1) + torch.log(lam).sum() + 2 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)
    return -0.5 * (quad + logdet)


def se_cov(xy, a, ell):
    """SE kernel a^2 exp(-|x - x'|^2 / (2 ell^2)) on points xy (n, d)."""
    d2 = (xy[:, None, :] - xy[None, :, :]).pow(2).sum(-1)
    return a ** 2 * torch.exp(-0.5 * d2 / ell ** 2)


def gp_cov(xy, hp, n_comp, extra=None, n_se=1):
    """Covariance of n_comp stacked components on the same points:
    blockdiag_c(sum_k SE(a_ck, ell_ck) + s_c^2 I) (+ extra).
    hp = log of [a_c1, ell_c1, ..., a_ck, ell_ck, s_c] for c = 1..n_comp (n_se = k terms)."""
    n = len(xy)
    h = torch.exp(hp).view(n_comp, 2 * n_se + 1)
    K = torch.zeros(n_comp * n, n_comp * n, dtype=xy.dtype, device=xy.device)
    I = torch.eye(n, dtype=xy.dtype, device=xy.device)
    for c in range(n_comp):
        B = h[c, -1] ** 2 * I
        for k in range(n_se):
            B = B + se_cov(xy, h[c, 2 * k], h[c, 2 * k + 1])
        K[c * n:(c + 1) * n, c * n:(c + 1) * n] = B
    return K if extra is None else K + extra


def gp_nll(xy, r, hp, n_comp, extra=None, n_se=1):
    """Negative log marginal likelihood of residual r (n_comp * n,) (with the 2 pi term)."""
    K = gp_cov(xy, hp, n_comp, extra, n_se)
    L = torch.linalg.cholesky(K)
    a = torch.cholesky_solve(r[:, None], L)[:, 0]
    return 0.5 * (r @ a) + torch.log(torch.diagonal(L)).sum() + 0.5 * len(r) * math.log(2 * math.pi)


def fit_gp_hyper(xy, r, n_comp, extra=None, hp0=None, ell0=5.0, rounds=3, n_se=1, ell_long=None, ells=None):
    """Type-II ML of the per-component kernel hyperparameters (see gp_cov). Returns (log-hp, nll).
    Default start: amplitudes split over the SE terms, length scales `ells` (one per term), or
    ell0 (and ell_long for a second term)."""
    if hp0 is None:
        ells = list(ells) if ells else [ell0] + ([ell_long or 8 * ell0] if n_se > 1 else [])
        rows = []
        for sd in r.view(n_comp, -1).std(1).tolist():
            row = []
            for k in range(n_se):
                row += [0.7 * sd / n_se ** 0.5, ells[min(k, len(ells) - 1)]]
            rows.append(row + [0.5 * sd])
        hp0 = torch.log(torch.tensor(rows, dtype=r.dtype, device=r.device).reshape(-1))
    hp = hp0.clone().detach().requires_grad_(True)
    opt = torch.optim.LBFGS([hp], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        v = gp_nll(xy, r, hp, n_comp, extra, n_se)
        v.backward()
        return v

    for _ in range(rounds):
        opt.step(closure)
    with torch.no_grad():
        v = float(gp_nll(xy, r, hp, n_comp, extra, n_se))
    return hp.detach(), v
