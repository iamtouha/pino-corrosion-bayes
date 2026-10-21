"""Transitional MCMC (Ching & Chen 2007), batched: log_lik takes a (n, d) array (Step 6.3)."""
from __future__ import annotations

import numpy as np


def next_beta(ll, beta):
    cov = lambda b: (lambda w: w.std() / w.mean())(np.exp((b - beta) * (ll - ll.max())))
    if cov(1.0) <= 1.0:
        return 1.0
    lo, hi = beta, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if cov(mid) <= 1.0 else (lo, mid)
    return max(lo, beta + 1e-12)


def tmcmc(log_lik, sample_prior, log_prior, n=4000, scale=0.2, n_mh=3, seed=0, verbose=False,
          max_stages=200, target_acc=0.23, move_frac=0.6, max_mh=None, min_scale=1e-3):
    """Adaptive TMCMC: the (population-wide) proposal scale is tuned towards `target_acc`
    after every MH sweep, and each stage runs at least n_mh sweeps and continues (up to
    max_mh) until `move_frac` of the particles have moved at least once.  With a fixed
    chain length and scale, a sharp likelihood (e.g. the full-field design without Sigma_G) leaves the resampled
    duplicates in place, and the population collapses onto one particle over the stages."""
    rng = np.random.default_rng(seed)
    max_mh = max_mh or 5 * n_mh
    th = sample_prior(n, rng); ll = log_lik(th); beta, logZ = 0.0, 0.0
    stages, accs, steps, n_eval = 0, [], [], n
    while beta < 1.0 and stages < max_stages:
        nb = next_beta(ll, beta)
        a = (nb - beta) * ll; w = np.exp(a - a.max()); p = w / w.sum()
        logZ += np.log(w.mean()) + a.max()                   # model evidence, for free
        m = p @ th; C0 = ((th - m).T * p) @ (th - m)
        C0 = C0 + 1e-12 * np.eye(th.shape[1]) * np.trace(C0) / th.shape[1]
        idx = rng.choice(n, n, p=p); th, ll = th[idx].copy(), ll[idx].copy()
        lp = log_prior(th)
        moved = np.zeros(n, bool); acc_tot, k = 0.0, 0
        while k < n_mh or (moved.mean() < move_frac and k < max_mh):
            C = scale ** 2 * C0
            prop = th + rng.multivariate_normal(np.zeros(th.shape[1]), C, n)
            lp_p = log_prior(prop); ll_p = np.full(n, -np.inf); ok = np.isfinite(lp_p)
            if ok.any():
                ll_p[ok] = log_lik(prop[ok]); n_eval += int(ok.sum())
            acc = np.log(rng.random(n)) < (lp_p + nb * ll_p) - (lp + nb * ll)
            th[acc], ll[acc], lp[acc] = prop[acc], ll_p[acc], lp_p[acc]
            moved |= acc; acc_tot += acc.mean(); k += 1
            scale = float(np.clip(scale * np.exp(2.0 * (acc.mean() - target_acc)), min_scale, 1.0))
        beta = nb
        stages += 1
        accs.append(acc_tot / k); steps.append(k)
        if verbose:
            print(f"    stage {stages}: beta={beta:.3e} acc={accs[-1]:.2f} mh={k} "
                  f"moved={moved.mean():.2f} scale={scale:.3f}", flush=True)
    uniq = len(np.unique(th, axis=0))
    return th, logZ, {"stages": stages, "acceptance": accs, "mh_steps": steps, "n_eval": n_eval,
                      "unique_frac": uniq / n}
