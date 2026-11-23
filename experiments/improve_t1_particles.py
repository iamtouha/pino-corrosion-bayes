"""T1 (follow-up studies, README): is the net-section capacity under-coverage a TMCMC sampling artefact?

Reruns the Step 7 surrogate posterior (PINO N=2000 with Sigma_G, grid56 design, the same
noisy data in results/laptop/step7/pit/data.npz) on the first `improve.t1.n_cases` damaged
cases, for each TMCMC size in `improve.t1.runs` (n = 1000 / n_mh 2 as in laptop.yaml, and
n = 4000 / n_mh 3 as in base.yaml) and each seed offset (repeatability). Every particle is
saved, one .npz per (n, seed, case), so a restart skips finished posteriors.

--stage report: 90 % coverage of the pit parameters, of the lost volume delta*r^2 and of the
net-section capacity (pino.capacity.capacity.net_section, as in step8c_capacity.py; on all
particles and on the step-8c 200-sample subsample), seed-to-seed repeatability, and the
direction of the capacity misses in the Step 7 baseline.
Outputs: results/laptop/improve/t1/{posteriors/*.npz, summary.json, summary.md, capacity_intervals.png}
"""
import time

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from pino import config
from pino.capacity.capacity import net_section
from pino.grid import Grid

KEYS = ["xc", "yc", "r", "delta"]
LEVELS = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])


def t1cfg(cfg):
    return cfg["improve"]["t1"]


def damaged_cases(x_true, n):
    return np.flatnonzero(x_true[:, 3] > 0)[:n]


def capacity_kN(fam, grid, S, sigma_u, bs=400):
    out = []
    for s in range(0, len(S), bs):
        tau = fam.tau(torch.as_tensor(S[s:s + bs], dtype=torch.float64)).numpy()
        out.append(net_section(tau, grid, sigma_u) / 1e3)
    return np.concatenate(out)


def run(cfg):
    from pino.inference.problem import (DEV, Family, Observer, build_likelihood, make_loglik_fn,
                                        surrogate_error)
    from pino.inference.tmcmc import tmcmc
    from pino.surrogates.models import load_surrogate
    from pino.surrogates.train import load_all
    from pino.vdic.noise import load_noise
    from pino.vdic.observe import dic_axes, make_obs

    tc = t1cfg(cfg)
    grid = Grid.from_cfg(cfg)
    fam = Family("pit", cfg, grid)
    inf = cfg["inference"]
    des = tc["design"]
    loads = cfg["dic"]["loads_kN"]
    z = np.load(cfg.results_dir("step7", "pit") / "data.npz")
    x_true, D = z["x_true"], z[f"d_{des}"]
    idx = damaged_cases(x_true, tc["n_cases"])
    od = cfg.results_dir("improve", "t1", "posteriors")
    ck = cfg.results_dir("step5") / "ckpt" / f"{tc['model']}_A_N{tc['N']}_s0.pt"
    sur = load_surrogate(ck, cfg, grid, DEV)
    H = make_obs(cfg, grid, des).to(DEV)
    obs = Observer(fam, H, loads, surrogate=sur)
    sg = None
    if tc["sigma_g"]:
        tau_D, w_D, nu_D = load_all(cfg, "D_pit")
        sg = surrogate_error(sur, H, tau_D, w_D, nu_D, inf["sigma_g_rank"])
    lik, mu = build_likelihood(obs, load_noise(cfg), dic_axes(cfg, des), sg,
                               nugget_floor_frac=inf.get("nugget_floor_frac", 0.0))
    print(f"T1: cases {idx.tolist()}, runs {tc['runs']}, seed offsets {tc['seed_offsets']}", flush=True)
    for r in tc["runs"]:
        for off in tc["seed_offsets"]:
            for i in idx:
                f = od / f"n{r['n']}_s{off}_{i:03d}.npz"
                if f.exists():
                    continue
                t0 = time.time()
                ll = make_loglik_fn(obs, lik, mu, torch.as_tensor(D[i]), batch=tc.get("loglik_batch", 16))
                S, logZ, info = tmcmc(ll, fam.sample, fam.log_prior, n=r["n"], scale=inf["tmcmc"]["scale"],
                                      n_mh=r["n_mh"], seed=int(i) + off)
                wall = time.time() - t0
                np.savez(f, samples=S, x_true=x_true[i], wall=wall, logZ=logZ, stages=info["stages"],
                         n_eval=info["n_eval"], unique_frac=info["unique_frac"],
                         acceptance=np.array(info["acceptance"]), mh_steps=np.array(info["mh_steps"]))
                print(f"  [n={r['n']} seed+{off} case {i}] {wall:.0f}s stages={info['stages']} "
                      f"evals={info['n_eval']} unique={info['unique_frac']:.2f} "
                      f"mean={np.round(S.mean(0)[:4], 3)} true={np.round(x_true[i][:4], 3)}", flush=True)


def _interval(v, lv):
    return np.quantile(v, (1 - lv) / 2), np.quantile(v, (1 + lv) / 2)


def report(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pino.inference.problem import Family

    tc = t1cfg(cfg)
    grid = Grid.from_cfg(cfg)
    fam = Family("pit", cfg, grid)
    su = cfg["material"]["ultimate"]
    width = fam.width
    od = cfg.results_dir("improve", "t1")
    rng = np.random.default_rng(0)
    groups = {}
    for f in sorted((od / "posteriors").glob("*.npz")):
        n_, s_, i_ = f.stem.split("_")
        groups.setdefault((int(n_[1:]), int(s_[1:])), {})[int(i_)] = f
    if not groups:
        print("no T1 posteriors yet"); return
    caches = {}
    rows = []
    for (n, off), files in sorted(groups.items()):
        rec = dict(n=n, seed_offset=off, n_cases=len(files))
        cov = {k: [] for k in KEYS + ["delta_r2", "capacity", "capacity_sub200"]}
        cap_lv = []
        pit_u, wall, neval, uniq, corr, capw = [], [], [], [], [], []
        for i, f in sorted(files.items()):
            z = np.load(f)
            S, xt = z["samples"], z["x_true"]
            cap = capacity_kN(fam, grid, S, su)
            ct = capacity_kN(fam, grid, xt[None], su)[0]
            caches[(n, off, i)] = (S, cap, ct, xt)
            for j, k in enumerate(KEYS):
                lo, hi = _interval(S[:, j], 0.9); cov[k].append(lo <= xt[j] <= hi)
            dr = S[:, 3] * S[:, 2] ** 2
            lo, hi = _interval(dr, 0.9); cov["delta_r2"].append(lo <= xt[3] * xt[2] ** 2 <= hi)
            lo, hi = _interval(cap, 0.9); cov["capacity"].append(lo <= ct <= hi); capw.append(hi - lo)
            sub = cap[rng.choice(len(cap), min(tc["capacity_subsample"], len(cap)), replace=False)]
            lo, hi = _interval(sub, 0.9); cov["capacity_sub200"].append(lo <= ct <= hi)
            cap_lv.append([(lambda a: a[0] <= ct <= a[1])(_interval(cap, lv)) for lv in LEVELS])
            pit_u.append(float((cap < ct).mean()))
            wall.append(float(z["wall"])); neval.append(int(z["n_eval"])); uniq.append(float(z["unique_frac"]))
            corr.append(float(np.corrcoef(S[:, 2], S[:, 3])[0, 1]))
        rec.update({f"cov90_{k}": float(np.mean(v)) for k, v in cov.items()})
        rec["capacity_coverage_by_level"] = dict(zip(LEVELS.tolist(), np.mean(cap_lv, 0).tolist()))
        rec["capacity_pit_u"] = pit_u
        rec["capacity_width90_kN_mean"] = float(np.mean(capw))
        rec["corr_r_delta_mean"] = float(np.mean(corr))
        rec["unique_frac_mean"] = float(np.mean(uniq))
        rec["wall_s_mean"] = float(np.mean(wall))
        rec["n_eval_mean"] = float(np.mean(neval))
        rec["cases"] = sorted(files)
        rows.append(rec)
    # seed-to-seed repeatability at each n (cases present for both seeds)
    rep = []
    offs = sorted({o for _, o in groups})
    for n in sorted({n for n, _ in groups}):
        if len(offs) < 2 or (n, offs[1]) not in groups:
            continue
        common = sorted(set(groups[(n, offs[0])]) & set(groups[(n, offs[1])]))
        w1, dq, flips = [], [], 0
        for i in common:
            Sa, ca, ct, _ = caches[(n, offs[0], i)]
            Sb, cb, _, _ = caches[(n, offs[1], i)]
            w1.append([wasserstein_distance(Sa[:, j], Sb[:, j]) / width[j] for j in range(4)])
            la, ha = _interval(ca, 0.9); lb, hb = _interval(cb, 0.9)
            wd = 0.5 * ((ha - la) + (hb - lb))
            dq.append([abs(la - lb) / wd, abs(ha - hb) / wd])
            flips += int((la <= ct <= ha) != (lb <= ct <= hb))
        w1, dq = np.array(w1), np.array(dq)
        rep.append(dict(n=n, n_common=len(common),
                        W1_over_width_mean={k: float(w1[:, j].mean()) for j, k in enumerate(KEYS)},
                        W1_over_width_max={k: float(w1[:, j].max()) for j, k in enumerate(KEYS)},
                        cap_q05_shift_over_width_mean=float(dq[:, 0].mean()),
                        cap_q95_shift_over_width_mean=float(dq[:, 1].mean()),
                        cap_coverage_flips=flips))
    # n = 1000 vs n = 4000, same seed: distance between the posteriors
    vs = []
    ns = sorted({n for n, _ in groups})
    if len(ns) >= 2:
        for off in offs:
            if (ns[0], off) in groups and (ns[-1], off) in groups:
                common = sorted(set(groups[(ns[0], off)]) & set(groups[(ns[-1], off)]))
                w1 = np.array([[wasserstein_distance(caches[(ns[0], off, i)][0][:, j],
                                                     caches[(ns[-1], off, i)][0][:, j]) / width[j]
                                for j in range(4)] for i in common])
                vs.append(dict(seed_offset=off, n_common=len(common),
                               W1_over_width_mean={k: float(w1[:, j].mean()) for j, k in enumerate(KEYS)}))
    # direction of the capacity misses in the step-7 baseline (n = 1000, all 27 damaged cases)
    base = {}
    pdir = cfg.results_dir("step7", "pit", "posteriors")
    for cond in ["pino_N100_SG", "pino_N2000_SG", "pino_N100_noSG", "pino_N2000_noSG"]:
        lo_miss = hi_miss = n_ = 0
        u = []
        for f in sorted(pdir.glob(f"{cond}_grid56_*.npz")):
            z = np.load(f)
            if z["x_true"][3] == 0:
                continue
            cap = capacity_kN(fam, grid, z["samples"], su)
            ct = capacity_kN(fam, grid, z["x_true"][None], su)[0]
            lo, hi = _interval(cap, 0.9)
            lo_miss += int(ct < lo); hi_miss += int(ct > hi); n_ += 1
            u.append(float((cap < ct).mean()))
        if n_:
            base[cond] = dict(n_damaged=n_, cov90_capacity=1 - (lo_miss + hi_miss) / n_,
                              truth_below_q05=lo_miss, truth_above_q95=hi_miss, pit_u=u)
    out = dict(runs=rows, repeatability=rep, n_small_vs_large=vs, step7_baseline=base)
    config.save_json(out, od / "summary.json", cfg)
    # markdown table
    md = ["| n | seed offset | cases | cov90 xc | yc | r | delta | delta r^2 | capacity | capacity (200 sub) | "
          "cap. 90% width kN | unique frac | evals | wall s |", "|" + "---|" * 14]
    for r in rows:
        md.append(f"| {r['n']} | {r['seed_offset']} | {r['n_cases']} | {r['cov90_xc']:.2f} | {r['cov90_yc']:.2f} | "
                  f"{r['cov90_r']:.2f} | {r['cov90_delta']:.2f} | {r['cov90_delta_r2']:.2f} | "
                  f"{r['cov90_capacity']:.2f} | {r['cov90_capacity_sub200']:.2f} | "
                  f"{r['capacity_width90_kN_mean']:.1f} | {r['unique_frac_mean']:.2f} | "
                  f"{r['n_eval_mean']:.0f} | {r['wall_s_mean']:.0f} |")
    md += ["", "Repeatability (seed vs seed):", ""]
    for r in rep:
        md.append(f"- n={r['n']} ({r['n_common']} cases): W1/width mean "
                  + ", ".join(f"{k} {v:.3f}" for k, v in r["W1_over_width_mean"].items())
                  + f"; capacity q05/q95 shift {r['cap_q05_shift_over_width_mean']:.2f}/"
                    f"{r['cap_q95_shift_over_width_mean']:.2f} of the 90% width; coverage flips {r['cap_coverage_flips']}")
    for r in vs:
        md.append(f"- n={ns[0]} vs n={ns[-1]} (seed offset {r['seed_offset']}, {r['n_common']} cases): W1/width mean "
                  + ", ".join(f"{k} {v:.3f}" for k, v in r["W1_over_width_mean"].items()))
    md += ["", "Step 7 baseline capacity misses (n = 1000, all damaged grid56 cases):", ""]
    for c, b in base.items():
        md.append(f"- {c}: cov90 {b['cov90_capacity']:.2f} ({b['n_damaged']} cases), truth below q05: "
                  f"{b['truth_below_q05']}, above q95: {b['truth_above_q95']}")
    (od / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    # plot: capacity 90 % intervals per case and run vs the truth
    keys = sorted(groups)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    allc = sorted({i for g in groups.values() for i in g})
    for gi, key in enumerate(keys):
        for ci, i in enumerate(allc):
            if (key[0], key[1], i) not in caches:
                continue
            _, cap, ct, _ = caches[(key[0], key[1], i)]
            lo, hi = _interval(cap, 0.9)
            xpos = ci + (gi - (len(keys) - 1) / 2) * 0.15
            ax.plot([xpos, xpos], [lo, hi], color=f"C{gi}", lw=2, label=f"n={key[0]} seed+{key[1]}" if ci == 0 else None)
            ax.plot(xpos, np.median(cap), "o", color=f"C{gi}", ms=3)
    for ci, i in enumerate(allc):
        ct = next(caches[(k[0], k[1], i)][2] for k in keys if (k[0], k[1], i) in caches)
        ax.plot([ci - 0.35, ci + 0.35], [ct, ct], "k-", lw=1)
    ax.set_xticks(range(len(allc))); ax.set_xticklabels([str(i) for i in allc])
    ax.set_xlabel("case"); ax.set_ylabel("net-section capacity (kN)")
    ax.set_title("T1: 90 % capacity intervals (bars) vs truth (black)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(od / "capacity_intervals.png", dpi=130)


def main():
    cfg, args = config.cli(__doc__, lambda ap: ap.add_argument(
        "--stage", default="all", choices=["all", "run", "report"]))
    if args.stage in ("all", "run"):
        run(cfg)
    if args.stage in ("all", "report"):
        report(cfg)


if __name__ == "__main__":
    main()
