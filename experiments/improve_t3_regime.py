"""T3 (follow-up studies, README): when does Sigma_G matter? Coverage / W1 vs (surrogate error / DIC noise).

Blocks (improve.t3.blocks), each a set of surrogate checkpoints and DIC noise scales s:
  * cross-family: B-trained PINO (KL fields) used for pit inference, s = 1. Larger surrogate
    error at the same parameterisation (pit-set error 0.263 / 0.067 for N = 100 / 2000).
  * noise sweep: A-trained PINO N = 100 / 2000, s in {1, 1/3, 1/10}. Data and likelihood are
    scaled consistently: d_s = clean + s (d - clean) from results/laptop/step7/pit/data.npz
    (same emulator draws), likelihood sigma and nugget times s (the nugget floor is a fraction
    of the scaled sigma).
Each (block, N, s, design) runs with and without Sigma_G (Sigma_G estimated with that
surrogate on D_pit) on improve.t3.n_damaged damaged cases, plus an FE-in-the-loop reference
on the first improve.t3.fe_ref_cases damaged cases per (s, design) for W1.

--surrogate-only: no FE pool, no dolfinx import (for Kaggle); FE references are skipped.
--fe-only: only the FE-in-the-loop references (CPU; run with GPU_MODE=none while the GPU is busy).
--shard i/k: run every k-th surrogate posterior starting at i (two GPUs / two machines).
--deadline-hours h: start no new posterior after h hours (Kaggle's 12 h session limit).
--stage report: results/laptop/improve/t3/regime.{json,md,png}.
One .npz per posterior in results/laptop/improve/t3/posteriors/; finished ones are skipped.
"""
import shutil
import time

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from pino import config
from pino.grid import Grid

KEYS = ["xc", "yc", "r", "delta"]


def stag(s):
    return f"{s:.3g}"


def scaled_noise(noise, s):
    import dataclasses
    return dataclasses.replace(noise, sigma=[v * s for v in noise.sigma], nugget=[v * s for v in noise.nugget])


def case_list(x_true, tc):
    return np.flatnonzero(x_true[:, 3] > 0)[: tc["n_damaged"]]


def sg_ratio(lik_sg, noise_s, obs, sg, E_nom=206e3):
    """RMS surrogate-error std / RMS DIC-noise std in the observation space (microstrain)."""
    c = (obs.loads / (E_nom * obs.grid.t0) * 1e6).numpy()
    var_g = ((sg["U"] ** 2) @ sg["lam"] + sg["left"]).numpy()             # unit load
    rms_g = np.sqrt(np.mean([cl ** 2 * var_g.mean() for cl in c]))
    sig = np.array(noise_s.sigma) * 1e6
    rms_n = np.sqrt(np.mean(sig ** 2))
    bias = np.sqrt(np.mean([cl ** 2 * (sg["mu"] ** 2).mean().item() for cl in c]))
    return float(rms_g / rms_n), float(rms_g), float(rms_n), float(bias)


def run(cfg, args):
    from experiments.step7_benchmark import summarize_posterior
    from pino.inference.problem import DEV, Family, Observer, build_likelihood, make_loglik_fn, surrogate_error
    from pino.inference.tmcmc import tmcmc
    from pino.surrogates.models import load_surrogate
    from pino.surrogates.train import load_all
    from pino.vdic.noise import load_noise
    from pino.vdic.observe import dic_axes, make_obs

    tc = cfg["improve"]["t3"]
    grid = Grid.from_cfg(cfg)
    fam = Family("pit", cfg, grid)
    inf = cfg["inference"]
    loads = cfg["dic"]["loads_kN"]
    noise = load_noise(cfg)
    nf = inf.get("nugget_floor_frac", 0.0)
    z = np.load(cfg.results_dir("step7", "pit") / "data.npz")
    x_true = z["x_true"]
    idx = case_list(x_true, tc)
    od = cfg.results_dir("improve", tc.get("out_name", "t3"), "posteriors")
    tm = tc["tmcmc"]
    tau_D, w_D, nu_D = load_all(cfg, "D_pit")
    shard_i, shard_k = (int(v) for v in args.shard.split("/"))
    job = -1
    meta = {}

    t_start = time.time()

    def post(f, obs, lik, mu, d, seed):
        nonlocal job
        job += 1
        if f.exists() or job % shard_k != shard_i:
            return
        if args.deadline_hours and time.time() - t_start > args.deadline_hours * 3600:
            return
        t0 = time.time()
        ll = make_loglik_fn(obs, lik, mu, torch.as_tensor(d))
        S, logZ, info = tmcmc(ll, fam.sample, fam.log_prior, n=tm["n"], scale=tm["scale"], n_mh=tm["n_mh"], seed=seed)
        np.savez(f, wall=time.time() - t0, logZ=logZ, stages=info["stages"], n_eval=info["n_eval"],
                 unique_frac=info["unique_frac"], x_true=x_true[seed], **summarize_posterior(S, fam))
        print(f"  [{f.stem}] {time.time() - t0:.0f}s stages={info['stages']} evals={info['n_eval']}", flush=True)

    designs = [d for d in tc["designs"] if not args.design or d == args.design]
    for blk in ([] if args.fe_only else tc["blocks"]):
        for N in blk["N"]:
            ck = cfg.results_dir("step5") / "ckpt" / f"pino_{blk['train']}_N{N}_s0.pt"
            sur = load_surrogate(ck, cfg, grid, DEV)
            for des in designs:
                H = make_obs(cfg, grid, des).to(DEV)
                obs = Observer(fam, H, loads, surrogate=sur)
                sg = surrogate_error(sur, H, tau_D, w_D, nu_D, inf["sigma_g_rank"])
                axes = dic_axes(cfg, des)
                clean, d1 = z[f"clean_{des}"], z[f"d_{des}"]
                for s in blk["scales"]:
                    ns = scaled_noise(noise, s)
                    liks = {"noSG": build_likelihood(obs, ns, axes, None, nugget_floor_frac=nf),
                            "SG": build_likelihood(obs, ns, axes, sg, nugget_floor_frac=nf)}
                    key = f"{blk['train']}{N}_s{stag(s)}_{des}"
                    meta[key] = dict(zip(["ratio", "rms_sg_ue", "rms_noise_ue", "rms_sg_bias_ue"],
                                         sg_ratio(None, ns, obs, {k: v for k, v in sg.items() if k != "n"})))
                    D = clean + s * (d1 - clean)
                    for i in idx:
                        for cond, (lik, mu) in liks.items():
                            post(od / f"{blk['train']}{N}_s{stag(s)}_{cond}_{des}_{i:03d}.npz",
                                 obs, lik, mu, D[i], int(i))
            del sur
            torch.cuda.empty_cache()
    if not args.fe_only:
        config.save_json(meta, cfg.results_dir("improve", tc.get("out_name", "t3")) / f"ratios_{args.shard.replace('/', 'of')}.json", cfg)
    if args.surrogate_only:
        return
    # --- FE-in-the-loop references (noise-only likelihood), per noise scale and design
    from pino.inference.problem import FEPool
    fe = FEPool(grid, cfg["dataset"]["workers"])
    ftm = tc.get("fe_tmcmc", tm)
    scales = sorted({s for blk in tc["blocks"] for s in blk["scales"]}, reverse=True)
    s7 = cfg.results_dir("step7", "pit", "posteriors")
    same_as_step7 = (ftm["n"] == inf["fe_tmcmc"]["n"] and ftm["n_mh"] == inf["fe_tmcmc"]["n_mh"])
    for des in designs:
        H = make_obs(cfg, grid, des, dtype=torch.float64).to(DEV)
        obs = Observer(fam, H, loads, fe_pool=fe)
        clean, d1 = z[f"clean_{des}"], z[f"d_{des}"]
        for s in scales:
            lik, mu = build_likelihood(obs, scaled_noise(noise, s), dic_axes(cfg, des), None, nugget_floor_frac=nf)
            D = clean + s * (d1 - clean)
            for i in idx[: tc["fe_ref_cases"]]:
                f = od / f"FE_s{stag(s)}_{des}_{i:03d}.npz"
                if f.exists():
                    continue
                if s == 1.0 and same_as_step7 and (s7 / f"FE_{des}_{i:03d}.npz").exists():
                    shutil.copy(s7 / f"FE_{des}_{i:03d}.npz", f)
                    print(f"  [{f.stem}] reused step-7 FE reference", flush=True)
                    continue
                t0 = time.time()
                ll = make_loglik_fn(obs, lik, mu, torch.as_tensor(D[i]), batch=200)
                S, logZ, info = tmcmc(ll, fam.sample, fam.log_prior, n=ftm["n"], scale=ftm["scale"],
                                      n_mh=ftm["n_mh"], seed=int(i))
                np.savez(f, wall=time.time() - t0, logZ=logZ, stages=info["stages"], n_eval=info["n_eval"],
                         unique_frac=info["unique_frac"], x_true=x_true[i], **summarize_posterior(S, fam))
                print(f"  [{f.stem}] {time.time() - t0:.0f}s, {info['n_eval']} FE solves", flush=True)
    fe.close()


def report(cfg, args):
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pino.inference.problem import Family
    grid = Grid.from_cfg(cfg)
    fam = Family("pit", cfg, grid)
    width = fam.width[:4]
    root = cfg.results_dir("improve", cfg["improve"]["t3"].get("out_name", "t3"))
    meta = {}
    for f in sorted(root.glob("ratios_*.json")):
        meta.update({k: v for k, v in json.loads(f.read_text()).items() if k != "config_hash"})
    recs = {}
    for f in sorted((root / "posteriors").glob("*.npz")):
        p = f.stem.split("_")
        i, des = int(p[-1]), p[-2]
        recs.setdefault("_".join(p[:-2]), {}).setdefault(des, {})[i] = dict(np.load(f))
    rows = []
    for cond, by_des in sorted(recs.items()):
        if cond.startswith("FE"):
            continue
        sur, s, sgc = cond.split("_")
        for des, cases in sorted(by_des.items()):
            idx = sorted(cases)
            lv = cases[idx[0]]["levels"]; nl = len(lv); l90 = int(np.argmin(np.abs(lv - 0.9)))
            xt = np.stack([cases[i]["x_true"] for i in idx])
            cov = np.array([[cases[i]["q"][l90, j] <= xt[n, j] <= cases[i]["q"][nl + l90, j] for j in range(4)]
                            for n, i in enumerate(idx)])
            ref = recs.get(f"FE_{s}", {}).get(des, {})
            w1 = [[wasserstein_distance(cases[i]["samples"][:, j], ref[i]["samples"][:, j]) / width[j]
                   for j in range(4)] for i in idx if i in ref]
            m = meta.get(f"{sur}_{s}_{des}", {})
            rows.append(dict(surrogate=sur, scale=float(s[1:]), sigma_g=sgc, design=des, n_cases=len(idx),
                             **{f"cov90_{k}": float(cov[:, j].mean()) for j, k in enumerate(KEYS)},
                             cov90_mean=float(cov.mean()),
                             W1_mean=float(np.mean(w1)) if w1 else None, W1_max=float(np.max(w1)) if w1 else None,
                             n_ref=len(w1), wall_s=float(np.mean([cases[i]["wall"] for i in idx])), **m))
    config.save_json({"rows": rows}, root / "regime.json", cfg)
    cols = ["surrogate", "scale", "design", "sigma_g", "n_cases", "ratio", "cov90_xc", "cov90_yc", "cov90_r",
            "cov90_delta", "cov90_mean", "W1_mean", "W1_max", "n_ref", "wall_s"]
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else ("" if v is None else str(v))
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in sorted(rows, key=lambda r: (r["design"], r.get("ratio", 0), r["sigma_g"])):
        md.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    (root / "regime.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    des_list = sorted({r["design"] for r in rows})
    fig, axs = plt.subplots(2, len(des_list), figsize=(5.5 * len(des_list), 8), squeeze=False)
    for c, des in enumerate(des_list):
        for sgc, mk in (("SG", "o"), ("noSG", "x")):
            rr = [r for r in rows if r["design"] == des and r["sigma_g"] == sgc and "ratio" in r]
            for r in rr:
                col = "C0" if r["surrogate"].startswith("A") else "C3"
                axs[0, c].plot(r["ratio"], r["cov90_mean"], mk, color=col, ms=7)
                axs[0, c].annotate(f"{r['surrogate']} s={r['scale']:.2g}", (r["ratio"], r["cov90_mean"]), fontsize=6)
                if r["W1_mean"] is not None:
                    axs[1, c].plot(r["ratio"], r["W1_mean"], mk, color=col, ms=7)
        for a in axs[:, c]:
            a.set_xscale("log"); a.set_xlabel("RMS surrogate-error std / RMS DIC-noise std")
        axs[0, c].axhline(0.9, color="k", ls="--", lw=0.8); axs[0, c].axhline(0.8, color="grey", ls=":", lw=0.8)
        axs[0, c].set_ylabel("90 % coverage (mean of xc, yc, r, delta)"); axs[0, c].set_title(f"{des}: o with Sigma_G, x without")
        axs[1, c].set_ylabel("W1 to FE reference / prior width"); axs[1, c].axhline(0.1, color="k", ls="--", lw=0.8)
    fig.tight_layout(); fig.savefig(root / "regime.png", dpi=130)


def main():
    cfg, args = config.cli(__doc__, lambda ap: (
        ap.add_argument("--stage", default="all", choices=["all", "run", "report"]),
        ap.add_argument("--surrogate-only", action="store_true"),
        ap.add_argument("--fe-only", action="store_true", help="only the FE-in-the-loop references"),
        ap.add_argument("--design", default=None),
        ap.add_argument("--shard", default="0/1"),
        ap.add_argument("--deadline-hours", type=float, default=0.0,
                        help="stop starting new posteriors after this many hours (Kaggle)")))
    if args.stage in ("all", "run"):
        run(cfg, args)
    if args.stage in ("all", "report"):
        report(cfg, args)


if __name__ == "__main__":
    main()
