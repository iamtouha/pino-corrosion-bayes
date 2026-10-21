"""Step 3: virtual DIC pipeline -> calibrated sigma_read, filter width ell_f, noise model.

Stages (run in this order; --stage all does all three):
  calibrate  3.5  uniform plate at 80 kN, subset radii 30/60/90 px; tune sigma_read so the
                  60 px strain scatter matches dic.calib_target_std_ue. Gate: within x1.5 and
                  a large drop 30 -> 60 px, small drop 60 -> 90 px.
  filter     3.3  noise-free deep pits; fit the Gaussian width ell_f of H.
  noise      3.4  noise realizations on fixed speckles; fit Sigma_DIC (SE + nugget per
                  component), tabulate the mean residual vs strain level (bias) for the
                  components and for the von Mises equivalent strain.
The fits use the crop window dic.crop_mm (cheaper); the model is stationary, so it
transfers to the full field.
"""
import json
import time

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from pino import config
from pino.fe.plate2d import PlateSolver2D
from pino.grid import Grid, q1_strain
from pino.thickness import pit_tau
from pino.vdic.engine import DEV, VirtualDIC
from pino.vdic.noise import NoiseModel, empirical_cov_profile, fit_se_nugget
from pino.vdic.observe import ObsOperator, dic_axes, dic_axes_px


def fe_field(solver, grid, cfg, theta, P_kN):
    m = cfg["material"]
    tau = pit_tau(torch.tensor([theta], dtype=torch.float64), grid)[0].numpy()
    uh = solver.solve(tau, m["nu"])
    scale = P_kN * 1e3 / (m["E"] * grid.t0)
    return scale * uh, scale * q1_strain(torch.tensor(uh)[None], grid.h)   # mm, strain


def vdic_dir(cfg):
    p = cfg.data_dir / "vdic"
    p.mkdir(parents=True, exist_ok=True)
    return p


def calibrate(cfg, grid, solver):
    d = cfg["dic"]
    xs, ys = dic_axes_px(cfg, "crop")
    U, _ = fe_field(solver, grid, cfg, [150.0, 125.0, 10.0, 0.0], d["calib_load_kN"])
    target = d["calib_target_std_ue"]

    def scatter(radius, sig, reps=2):
        vd = VirtualDIC(cfg, grid, xs, ys, radius_px=radius, seed=11)
        vd.set_field(U)
        s = [vd.measure(sig)[0][0].std().item() * 1e6 for _ in range(reps)]
        return float(np.mean(s))

    sig = d["sigma_read"]
    hist = []
    for it in range(6):                        # scatter ~ a + b * sigma_read: secant steps
        s60 = scatter(60, sig)
        hist.append((sig, s60))
        print(f"  sigma_read={sig:.3f}: scatter(60px)={s60:.2f} ue (target {target})", flush=True)
        if abs(s60 / target - 1) < 0.03:
            break
        if len(hist) >= 2:
            (s1, y1), (s2, y2) = hist[-2], hist[-1]
            slope = (y2 - y1) / (s2 - s1) if s2 != s1 else y2 / s2
            sig = max(0.05, s2 + (target - y2) / max(slope, 1e-6))
        else:
            sig = sig * target / max(s60, 1e-9)
    radii = {r: scatter(r, sig) for r in d["calib_radii_px"]}
    r30, r60, r90 = (radii[r] for r in (30, 60, 90))
    trend = (r30 - r60) > (r60 - r90) > 0
    ratio = r60 / target
    out = dict(sigma_read=sig, history=hist, scatter_by_radius_ue=radii, target_std_ue=target,
               ratio_to_target=ratio, trend_ok=bool(trend),
               passed=bool(1 / 1.5 <= ratio <= 1.5 and trend),
               note="target is a placeholder until read off the paper's Fig. 8 (configs/base.yaml)")
    config.save_json(out, vdic_dir(cfg) / "calibration.json", cfg)
    print("  3.5 calibration:", out)
    return sig


def sigma_read(cfg):
    p = vdic_dir(cfg) / "calibration.json"
    return json.loads(p.read_text())["sigma_read"] if p.exists() else cfg["dic"]["sigma_read"]


def fit_filter(cfg, grid, solver):
    """Fit ell_f from the DIC transfer function: sinusoidal strain fields of wavelength lam
    are measured (noise-free, averaged over speckle patterns); the amplitude ratio
    T(lam) = A_dic / A_true is fitted with the Gaussian kernel's exp(-2 pi^2 ell^2 / lam^2).
    (Pits with r >= 8 mm are too smooth to identify ell_f: the 2-3 mm smoothing changes
    their peak by ~2 ue, below the pattern-induced DIC error.) Deep pits then check the fit."""
    d = cfg["dic"]
    xs, ys = dic_axes_px(cfg, "crop")
    xm, ym = dic_axes(cfg, "crop")
    n_pat = d.get("filter_patterns", 3)
    vds = [VirtualDIC(cfg, grid, xs, ys, seed=21 + p) for p in range(n_pat)]
    hf = 0.25                                                     # fine grid for the probe field
    xf = np.linspace(0, grid.L, int(round(grid.L / hf)) + 1)
    yf = np.linspace(0, grid.W, int(round(grid.W / hf)) + 1)
    Yf, Xf = np.meshgrid(yf, xf, indexing="ij")
    A = d.get("probe_amplitude", 2e-3)
    lams = d.get("probe_wavelengths_mm", [6.0, 8.0, 12.0, 16.0, 24.0, 32.0])
    X, Y = np.meshgrid(xm, ym)
    T = []
    t0 = time.time()
    for lam in lams:
        k = 2 * np.pi / lam
        U = np.zeros((*Xf.shape, 2))
        U[..., 0] = A / k * np.sin(k * Xf)                         # e11 = A cos(k x)
        U[..., 1] = A / k * np.sin(k * Yf)                         # e22 = A cos(k y)
        e = 0
        for vd in vds:
            vd.set_field(U)
            e = e + vd.measure(0.0, noise=False)[0].cpu().numpy()
        e = e / n_pat
        amp = []
        for comp, Z in ((0, X), (1, Y)):                          # LS fit of a cos + b sin
            M = np.stack([np.cos(k * Z).ravel(), np.sin(k * Z).ravel(), np.ones(Z.size)], 1)
            c = np.linalg.lstsq(M, e[comp].ravel(), rcond=None)[0]
            amp.append(np.hypot(c[0], c[1]) / A)
        T.append(float(np.mean(amp)))
        print(f"  lambda={lam:5.1f} mm: transfer {T[-1]:.3f} ({time.time() - t0:.0f}s)", flush=True)
    lams_a, T_a = np.array(lams), np.array(T)
    r = minimize_scalar(lambda l: float(((np.exp(-2 * np.pi ** 2 * l ** 2 / lams_a ** 2) - T_a) ** 2).sum()),
                        bounds=(0.01, 10.0), method="bounded")
    ell = float(r.x)
    # validation on deep pits: misfit with the fitted H vs no smoothing
    rng = np.random.default_rng(5)
    x0, x1, y0, y1 = d["crop_mm"]
    pairs = []
    for kcase in range(d["fit_filter_cases"]):
        th = [rng.uniform(x0 + 30, x1 - 30), rng.uniform(y0 + 30, y1 - 30),
              rng.uniform(4, 15), rng.uniform(0.10, 0.15)]
        U, eps = fe_field(solver, grid, cfg, th, d["calib_load_kN"])
        acc = 0
        for vd in vds:
            vd.set_field(U)
            acc = acc + vd.measure(0.0, noise=False)[0].cpu()
        pairs.append((eps, acc / n_pat))

    def misfit(l):
        H = ObsOperator(grid, xm, ym, l, torch.float64)
        return float(sum(((H.fields(H(e_))[0] - ed) ** 2).sum() for e_, ed in pairs))

    out = dict(ell_f=ell, wavelengths_mm=lams, transfer=T,
               transfer_model=np.exp(-2 * np.pi ** 2 * ell ** 2 / lams_a ** 2).tolist(),
               pit_misfit_fitted=misfit(ell), pit_misfit_unfiltered=misfit(1e-6),
               n_patterns=n_pat, n_pit_cases=len(pairs))
    config.save_json(out, vdic_dir(cfg) / "filter_fit.json", cfg)
    print("  3.3 filter:", out)
    return ell


def fit_noise(cfg, grid, solver, ell_f):
    d = cfg["dic"]
    xs, ys = dic_axes_px(cfg, "crop")
    xm, ym = dic_axes(cfg, "crop")
    H = ObsOperator(grid, xm, ym, ell_f, torch.float64)
    sig = sigma_read(cfg)
    rng = np.random.default_rng(9)
    x0, x1, y0, y1 = d["crop_mm"]
    loads = np.linspace(10, 100, d["fit_noise_fields"])
    res, lev, eq_true, eq_noisy = [], [], [], []
    t0 = time.time()
    for k in range(d["fit_noise_fields"]):
        intact = k % 3 == 0
        th = [rng.uniform(x0 + 30, x1 - 30), rng.uniform(y0 + 30, y1 - 30), rng.uniform(8, 40),
              0.0 if intact else rng.uniform(0.02, 0.15)]
        U, eps = fe_field(solver, grid, cfg, th, float(loads[k]))
        ht = H.fields(H(eps))[0]                                   # (3, py, px)
        vd = VirtualDIC(cfg, grid, xs, ys, seed=100 + k)
        vd.set_field(U)
        for j in range(d["fit_noise_realizations"]):
            e, _ = vd.measure(sig)
            e = e.cpu()
            res.append((e - ht).numpy())
            lev.append(ht.numpy())
            eq_true.append(von_mises(ht).numpy()); eq_noisy.append(von_mises(e).numpy())
        print(f"  noise field {k + 1}/{d['fit_noise_fields']} ({loads[k]:.0f} kN): "
              f"{time.time() - t0:.0f}s", flush=True)
    res, lev = np.stack(res), np.stack(lev)                        # (n, 3, py, px)
    spacing = d["step_px"] * d["px_mm"]
    sig_c, ell_c, nug_c, prof = [], [], [], {}
    for c in range(3):
        # keep the per-field (speckle-pattern) bias: it is part of a real test's DIC error;
        # empirical_cov_profile removes only the mean over all fields
        rc = res[:, c]
        lags, cov = empirical_cov_profile(rc, spacing, max_lag=12)
        s_, l_, n_ = fit_se_nugget(lags, cov)
        sig_c.append(s_); ell_c.append(l_); nug_c.append(n_)
        prof[c] = dict(lags_mm=lags.tolist(), cov=cov.tolist())
    # bias vs strain level (component e11) and the equivalent-strain trap
    lv = np.abs(lev[:, 0]).ravel(); rv = res[:, 0].ravel()
    bins = np.quantile(lv, np.linspace(0, 1, 11))
    mid, bias = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (lv >= a) & (lv <= b)
        mid.append(float(lv[m].mean())); bias.append(float(rv[m].mean()))
    eqt, eqn = np.stack(eq_true).ravel(), np.stack(eq_noisy).ravel()
    qb = np.quantile(eqt, np.linspace(0, 1, 11))
    eq_tab = []
    for a, b in zip(qb[:-1], qb[1:]):
        m = (eqt >= a) & (eqt <= b)
        eq_tab.append((float(eqt[m].mean() * 1e6), float((eqn[m] - eqt[m]).mean() * 1e6)))
    nm = NoiseModel(sigma=sig_c, ell=ell_c, nugget=nug_c, bias_strain=mid, bias_value=bias,
                    source=f"virtual DIC, sigma_read={sig:.3f}, {len(res)} realizations")
    nm.save(vdic_dir(cfg) / "noise_model.json")
    out = dict(model=nm.__dict__, cov_profiles=prof,
               eq_strain_bias_ue=eq_tab,
               component_bias_ue=[(a * 1e6, b * 1e6) for a, b in zip(mid, bias)])
    config.save_json(out, vdic_dir(cfg) / "noise_fit.json", cfg)
    print("  3.4 noise model:", nm)
    print("  von Mises bias (true eq ue, mean bias ue):", [(round(a), round(b, 1)) for a, b in eq_tab])
    plot_noise(cfg, out)


def von_mises(e):
    """Equivalent (von Mises) strain from plane-stress components incl. e33 (Theory 1)."""
    nu = 0.3
    e11, e22, e12 = e[0], e[1], e[2]
    e33 = -nu / (1 - nu) * (e11 + e22)
    return (np.sqrt(2) / 3) * torch.sqrt((e11 - e22) ** 2 + (e22 - e33) ** 2 + (e33 - e11) ** 2 + 6 * e12 ** 2)


def plot_noise(cfg, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    m = out["model"]
    for c, name in enumerate(["e11", "e22", "e12"]):
        p = out["cov_profiles"][c]
        lags = np.array(p["lags_mm"])
        ax[0].plot(lags, np.array(p["cov"]) * 1e12, "o", label=f"{name} empirical")
        mod = m["sigma"][c] ** 2 * np.exp(-lags ** 2 / (2 * m["ell"][c] ** 2)) * 1e12
        mod[0] += m["nugget"][c] ** 2 * 1e12
        ax[0].plot(lags, mod, "-", color=ax[0].lines[-1].get_color())
    ax[0].set_xlabel("lag (mm)"); ax[0].set_ylabel("covariance (ue^2)"); ax[0].legend()
    eq = np.array(out["eq_strain_bias_ue"]); cb = np.array(out["component_bias_ue"])
    ax[1].plot(eq[:, 0], eq[:, 1], "o-", label="von Mises eq. strain")
    ax[1].plot(cb[:, 0], cb[:, 1], "s-", label="e11 component")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("true strain level (ue)"); ax[1].set_ylabel("mean DIC residual (ue)"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(cfg.results_dir("step3") / "noise_model.png", dpi=130)


def main():
    cfg, args = config.cli(__doc__, lambda ap: ap.add_argument(
        "--stage", default="all", choices=["all", "calibrate", "filter", "noise"]))
    grid = Grid.from_cfg(cfg)
    solver = PlateSolver2D(grid)
    print("device:", DEV)
    vd = vdic_dir(cfg)

    def want(stage, out):            # "all" resumes: skip parts whose output exists
        if args.stage == stage:
            return True
        if args.stage == "all" and (vd / out).exists():
            print(f"{stage}: {out} exists, skipping (use --stage {stage} to redo)", flush=True)
            return False
        return args.stage == "all"

    if want("calibrate", "calibration.json"):
        calibrate(cfg, grid, solver)
    if want("filter", "filter_fit.json"):
        fit_filter(cfg, grid, solver)
    if want("noise", "noise_model.json"):
        ell = json.loads((vdic_dir(cfg) / "filter_fit.json").read_text())["ell_f"]
        fit_noise(cfg, grid, solver, ell)


if __name__ == "__main__":
    main()
