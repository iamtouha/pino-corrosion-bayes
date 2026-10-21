"""Step 5: data-efficiency study. models x N x seeds x families, resumable.

For the PINO, lam_r is picked per (family, N) from train.lam_r_grid by short runs on the
validation set (train.lam_r_select_steps; 0 -> first grid value).
Writes results/<tag>/step5/runs/*.json, summary.csv and summary.md.

--deadline-hours H: do not start a run (or a lam_r selection) that is not expected to finish
within H hours of the start, judged from the longest run of the same model so far in this
process (for sessions with a hard time limit, e.g. Kaggle). Unstarted runs are left for the
next session; finished runs are skipped on restart.
"""
import json
import time

import numpy as np
import pandas as pd
import torch

from pino import config
from pino.grid import Grid
from pino.surrogates.train import (DEV, evaluate, load_all, load_split, time_inference,
                                   train_run)
from pino.surrogates.models import load_surrogate
from pino.vdic.observe import make_obs


class Budget:
    """Expected run length per model: the longest measured run (train_seconds + 15 % for
    evaluation, from finished runs and from this process), else --est-minutes."""

    def __init__(self, hours, out=None, est_minutes=None):
        self.t0, self.limit = time.time(), (hours or 0) * 3600
        self.est = {m: 60.0 * float(v) for m, v in (est_minutes or {}).items()}
        self.longest = {}                 # measured; replaces the estimate once available
        for f in (out / "runs").glob("*.json") if out is not None else []:
            r = json.loads(f.read_text())
            self.record(r["model"], r.get("train_seconds", 0.0) * 1.15)

    def allows(self, kind, factor=1.0):
        if not self.limit:
            return True
        need = self.longest.get(kind, self.est.get(kind, 0.0)) * factor
        return time.time() - self.t0 + need < self.limit

    def record(self, kind, seconds):
        self.longest[kind] = max(self.longest.get(kind, 0.0), seconds)


class OutOfTime(Exception):
    pass


def pick_lam(cfg, fam, N, out, budget=None):
    t = cfg["train"]
    if t.get("lam_r_select_N"):
        N = int(t["lam_r_select_N"])
    f = out / f"lam_select_{fam}_N{N}.json"
    if f.exists():
        return json.loads(f.read_text())["lam_r"]
    grid_vals = t["lam_r_grid"]
    if t["lam_r_select_steps"] <= 0 or len(grid_vals) == 1:
        lam = grid_vals[0]; scores = {}
    else:
        scores = {}
        for lam in grid_vals:
            # a selection run is lam_r_select_steps / steps of a full PINO run
            frac = t["lam_r_select_steps"] / t["steps"]
            if budget and not budget.allows("pino", frac * len(grid_vals)):
                raise OutOfTime
            _, info = train_run(cfg, "pino", fam, N, 0, lam, t["lam_r_select_steps"],
                                out / "ckpt" / f"select_{fam}_N{N}_lam{lam}.pt")
            scores[lam] = info["val_rel_l2"]
            print(f"  lam_r={lam}: val {info['val_rel_l2']:.4f}", flush=True)
        lam = min(scores, key=scores.get)
    config.save_json({"lam_r": lam, "scores": scores}, f, cfg)
    return lam


def main():
    cfg, args = config.cli(__doc__, lambda ap: (
        ap.add_argument("--models", nargs="*"), ap.add_argument("--N", nargs="*", type=int),
        ap.add_argument("--families", nargs="*"), ap.add_argument("--seeds", nargs="*", type=int),
        ap.add_argument("--deadline-hours", type=float, default=None),
        ap.add_argument("--est-minutes", nargs="*", default=[],
                        help="first-run estimates, e.g. deeponet=10 fno=45 pino=90"),
        ap.add_argument("--summary-only", action="store_true")))
    s = cfg["study"]
    models, Ns = args.models or s["models"], args.N or s["N"]
    fams, seeds = args.families or s["families"], args.seeds or s["seeds"]
    out = cfg.results_dir("step5")
    if args.summary_only:
        return summarize(cfg, out)
    budget = Budget(args.deadline_hours, out,
                    est_minutes=dict(e.split("=") for e in args.est_minutes))
    grid = Grid.from_cfg(cfg)
    H = make_obs(cfg, grid, "full").to(DEV)
    tests = {"A": load_split(cfg, "A", "test"), "B": load_split(cfg, "B", "test"), "C": load_all(cfg, "C")}
    try:
        run_all(cfg, args, out, models, Ns, fams, seeds, H, tests, budget)
    except OutOfTime:
        print(f"deadline: stopping after {(time.time() - budget.t0) / 3600:.2f} h; "
              "rerun to continue with the unfinished runs", flush=True)
    summarize(cfg, out)


def run_all(cfg, args, out, models, Ns, fams, seeds, H, tests, budget):
    for fam in fams:
        for N in Ns:
            lam = pick_lam(cfg, fam, N, out, budget) if "pino" in models else None
            for m in models:
                for seed in seeds:
                    tag = f"{m}_{fam}_N{N}_s{seed}"
                    rf = out / "runs" / f"{tag}.json"
                    if rf.exists():
                        continue
                    if not budget.allows(m):
                        raise OutOfTime
                    t_run = time.time()
                    print(f"[{tag}] training (lam_r={lam if m == 'pino' else 0})", flush=True)
                    ck = out / "ckpt" / f"{tag}.pt"
                    model, info = train_run(cfg, m, fam, N, seed, lam if m == "pino" else 0.0,
                                            cfg["train"]["steps"], ck)
                    res = {k: evaluate(model, *v, H) for k, v in tests.items()}
                    ms = time_inference(model, tests["A"][0], tests["A"][2])
                    rec = dict(model=m, family=fam, N=N, seed=seed,
                               lam_r=lam if m == "pino" else 0.0, ms_per_1000=ms,
                               **{f"test_{k}": v["rel_l2"] for k, v in res.items()},
                               **{f"test_{k}_median": v["median_rel_l2"] for k, v in res.items()},
                               **info)
                    config.save_json(rec, rf, cfg)
                    print(f"[{tag}] A {res['A']['rel_l2']:.4f} B {res['B']['rel_l2']:.4f} "
                          f"C {res['C']['rel_l2']:.4f}  {ms:.1f} ms/1000", flush=True)
                    del model; torch.cuda.empty_cache()
                    budget.record(m, time.time() - t_run)


def summarize(cfg, out):
    recs = [json.loads(p.read_text()) for p in sorted((out / "runs").glob("*.json"))]
    if not recs:
        return
    df = pd.DataFrame(recs)
    cols = ["test_A", "test_B", "test_C", "ms_per_1000"]
    g = df.groupby(["family", "model", "N"])[cols].agg(["mean", "std"]).reset_index()
    g.columns = ["_".join(c).strip("_") for c in g.columns]
    g.to_csv(out / "summary.csv", index=False)
    lines = []
    for fam, sub in g.groupby("family"):
        lines += [f"\n### Trained on set {fam} (perturbation-strain relative L2 after H, mean ± sd over seeds)\n",
                  "| Model | N | Pit test (A) | KL test (B) | Triangle OOD (C) | ms per 1000 fields |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for _, r in sub.sort_values(["model", "N"]).iterrows():
            f = lambda k: f"{r[k + '_mean']:.3f} ± {0 if np.isnan(r[k + '_std']) else r[k + '_std']:.3f}"
            lines.append(f"| {r['model']} | {r['N']} | {f('test_A')} | {f('test_B')} | {f('test_C')} | "
                         f"{r['ms_per_1000_mean']:.0f} |")
    # gate
    gate = []
    for fam, tcol in (("A", "test_A"), ("B", "test_B")):
        sub = g[(g.family == fam) & (g.model == "pino")]
        if len(sub):
            Nmax = sub.N.max()
            v = float(sub[sub.N == Nmax][tcol + "_mean"].iloc[0])
            gate.append((fam, Nmax, v, v < 0.10))
    lines.append("\n**Gate** (PINO error < 10 % at the largest N on its own family): " +
                 ", ".join(f"set {f} N={n}: {v:.3f} {'PASS' if ok else 'FAIL'}" for f, n, v, ok in gate))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
