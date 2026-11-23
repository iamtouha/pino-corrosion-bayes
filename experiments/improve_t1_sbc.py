"""T1 step 3 (follow-up studies, README): simulation-based calibration of the net-section capacity.

Exact-model check of the sampler and of equal-tailed intervals on a bounded derived quantity:
truth theta ~ prior, data = PINO N=2000 prediction + a draw from the likelihood's own noise
covariance (grid56, Sigma_DIC with the nugget floor, no Sigma_G), TMCMC with the same
likelihood. If TMCMC is correct, the PIT u = P(q < q_true | data) is Uniform(0, 1) for every
quantity q (parameters, delta r^2, capacity), whatever the data informativeness.
Reports PIT histograms, 90 % equal-tailed and one-sided (lower 5 %) coverage, overall and for
near-intact truths. Outputs: results/laptop/improve/t1/sbc/{rep_*.npz, sbc.json, sbc.md, sbc.png}
"""
import time

import numpy as np
import torch
from scipy.stats import kstest

from pino import config
from pino.grid import Grid

from experiments.improve_t1_particles import capacity_kN

QK = ["xc", "yc", "r", "delta", "delta_r2", "capacity"]


def run(cfg):
    from pino.inference.problem import DEV, Family, Observer, build_likelihood, make_loglik_fn
    from pino.inference.tmcmc import tmcmc
    from pino.surrogates.models import load_surrogate
    from pino.vdic.noise import load_noise
    from pino.vdic.observe import dic_axes, make_obs
    tc = cfg["improve"]["t1"]; sc = tc["sbc"]
    grid = Grid.from_cfg(cfg); fam = Family("pit", cfg, grid)
    od = cfg.results_dir("improve", "t1", "sbc")
    sur = load_surrogate(cfg.results_dir("step5") / "ckpt" / f"pino_A_N{tc['N']}_s0.pt", cfg, grid, DEV)
    H = make_obs(cfg, grid, "grid56").to(DEV)
    obs = Observer(fam, H, cfg["dic"]["loads_kN"], surrogate=sur)
    lik, mu = build_likelihood(obs, load_noise(cfg), dic_axes(cfg, "grid56"), None,
                               nugget_floor_frac=cfg["inference"].get("nugget_floor_frac", 0.0))
    rng = np.random.default_rng(sc["seed"])
    X = fam.sample(sc["n_rep"], rng)
    Z = rng.standard_normal((sc["n_rep"], lik.n))
    with torch.no_grad():
        clean = obs(torch.as_tensor(X)).cpu()
    D = clean + (lik.Lb.cpu() @ torch.as_tensor(Z).T).T
    for k in range(sc["n_rep"]):
        f = od / f"rep_{k:03d}.npz"
        if f.exists():
            continue
        t0 = time.time()
        ll = make_loglik_fn(obs, lik, mu, D[k])
        S, logZ, info = tmcmc(ll, fam.sample, fam.log_prior, n=sc["n"], scale=cfg["inference"]["tmcmc"]["scale"],
                              n_mh=sc["n_mh"], seed=10_000 + k)
        np.savez(f, samples=S, x_true=X[k], wall=time.time() - t0, n_eval=info["n_eval"], unique_frac=info["unique_frac"])
        print(f"  [sbc {k}] {time.time() - t0:.0f}s stages={info['stages']}", flush=True)


def report(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pino.inference.problem import Family
    tc = cfg["improve"]["t1"]; sc = tc["sbc"]
    grid = Grid.from_cfg(cfg); fam = Family("pit", cfg, grid)
    su = cfg["material"]["ultimate"]
    od = cfg.results_dir("improve", "t1", "sbc")
    files = sorted(od.glob("rep_*.npz"))
    if not files:
        print("no SBC replications yet"); return
    intact = capacity_kN(fam, grid, np.array([[150, 125, 20, 0.0, 206e3, 0.3]]), su)[0]
    U, cov, low, near = {k: [] for k in QK}, {k: [] for k in QK}, {k: [] for k in QK}, []
    for f in files:
        z = np.load(f); S, xt = z["samples"], z["x_true"]
        vals = dict(zip(QK, [S[:, 0], S[:, 1], S[:, 2], S[:, 3], S[:, 3] * S[:, 2] ** 2, capacity_kN(fam, grid, S, su)]))
        tv = dict(zip(QK, [xt[0], xt[1], xt[2], xt[3], xt[3] * xt[2] ** 2, capacity_kN(fam, grid, xt[None], su)[0]]))
        near.append(intact - tv["capacity"] < sc["near_intact_kN"])
        for k in QK:
            v = vals[k]
            U[k].append(float((v < tv[k]).mean() + 0.5 * (v == tv[k]).mean()))
            lo, hi = np.quantile(v, [0.05, 0.95])
            cov[k].append(bool(lo <= tv[k] <= hi)); low[k].append(bool(tv[k] >= lo))
    near = np.array(near)
    out = dict(n_rep=len(files), intact_capacity_kN=float(intact), n_near_intact=int(near.sum()),
               near_intact_kN=sc["near_intact_kN"], quantities={})
    for k in QK:
        c, u, l = np.array(cov[k]), np.array(U[k]), np.array(low[k])
        out["quantities"][k] = dict(cov90=float(c.mean()), cov90_near_intact=float(c[near].mean()) if near.any() else None,
                                    cov90_rest=float(c[~near].mean()) if (~near).any() else None,
                                    lower95_coverage=float(l.mean()), ks_uniform_p=float(kstest(u, "uniform").pvalue),
                                    frac_u_above_095=float((u > 0.95).mean()), frac_u_below_005=float((u < 0.05).mean()),
                                    pit=u.tolist())
    config.save_json(out, od / "sbc.json", cfg)
    md = [f"SBC: {len(files)} replications (truth ~ prior, exact model), {int(near.sum())} near-intact "
          f"(capacity loss < {sc['near_intact_kN']} kN)", "",
          "| quantity | cov90 | cov90 near-intact | cov90 rest | lower-5 % bound coverage | u > 0.95 | u < 0.05 | KS p (PIT uniform) |",
          "|---|---|---|---|---|---|---|---|"]
    f_ = lambda v: "" if v is None else f"{v:.2f}"
    for k, q in out["quantities"].items():
        md.append(f"| {k} | {q['cov90']:.2f} | {f_(q['cov90_near_intact'])} | {f_(q['cov90_rest'])} | "
                  f"{q['lower95_coverage']:.2f} | {q['frac_u_above_095']:.2f} | {q['frac_u_below_005']:.2f} | {q['ks_uniform_p']:.2f} |")
    (od / "sbc.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
    fig, axs = plt.subplots(1, len(QK), figsize=(3 * len(QK), 2.8))
    for a, k in zip(axs, QK):
        a.hist(U[k], bins=10, range=(0, 1)); a.axhline(len(files) / 10, color="k", ls="--", lw=0.8); a.set_title(k)
    fig.suptitle("SBC: PIT of the truth (uniform if calibrated)"); fig.tight_layout(); fig.savefig(od / "sbc.png", dpi=120)


def main():
    cfg, args = config.cli(__doc__, lambda ap: ap.add_argument("--stage", default="all", choices=["all", "run", "report"]))
    if args.stage in ("all", "run"):
        run(cfg)
    if args.stage in ("all", "report"):
        report(cfg)


if __name__ == "__main__":
    main()
