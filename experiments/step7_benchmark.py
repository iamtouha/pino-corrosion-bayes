"""Step 6-7: Bayesian identification benchmark on synthetic cases (pit family by default).

Truth: FE solves at prior draws (some with delta = 0). Data: fast-emulator virtual DIC at
each load (independent noise per load), for the 56-point and the full-field designs.
Conditions: FE in the loop (TMCMC, few cases), surrogate without Sigma_G, surrogate with
Sigma_G, for each surrogate type and N in inference.models x inference.surrogate_N.
Realism check: a few cases with data from the full virtual-DIC image pipeline.

Stage --run computes posteriors (resumable, one .npz per posterior); --report computes
the metrics table, calibration plot and detection-limit map from whatever is finished.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from pino import config
from pino.grid import Grid, q1_strain
from pino.inference.problem import (DEV, Family, FEPool, Observer, build_likelihood,
                                    make_loglik_fn, surrogate_error)
from pino.inference.tmcmc import tmcmc
from pino.surrogates.models import load_surrogate
from pino.surrogates.train import load_all
from pino.vdic.noise import NoiseEmulator, load_noise
from pino.vdic.observe import dic_axes, dic_axes_px, make_obs

PIT_KEYS = ["xc", "yc", "r", "delta"]


def out_dir(cfg, fam):
    return cfg.results_dir("step7", fam)


def truth_cases(cfg, fam: Family):
    inf = cfg["inference"]
    x = fam.sample(inf["n_cases"], np.random.default_rng(777))
    if fam.name == "pit":
        x[: inf["n_zero_delta"], 3] = 0.0
    return x


def make_data(cfg, fam, grid, x_true, designs, fe):
    """Clean FE observations + emulator noise, per design. Cached to data.npz."""
    p = out_dir(cfg, fam.name) / "data.npz"
    if p.exists():
        z = np.load(p)
        return {k: z[k] for k in z.files}
    noise = load_noise(cfg)
    loads = cfg["dic"]["loads_kN"]
    gen = torch.Generator(device="cpu").manual_seed(4242)
    out = {"x_true": x_true}
    for des in designs:
        H = make_obs(cfg, grid, des, dtype=torch.float64).to(DEV)
        obs = Observer(fam, H, loads, fe_pool=fe)
        clean = obs(torch.as_tensor(x_true)).cpu()
        xs, ys = dic_axes(cfg, des)
        em = NoiseEmulator(noise, xs, ys)
        n = torch.stack([em.sample(len(loads), gen).reshape(-1) for _ in range(len(x_true))]) * 1e6
        out[f"clean_{des}"] = clean.numpy()
        out[f"d_{des}"] = (clean + n).numpy()
    np.savez(p, **out)
    return out


def realism_data(cfg, fam, grid, x_true, k, designs):
    """Data from the full virtual-DIC image pipeline (render + DIC + VSG) for k cases."""
    from pino.fe.plate2d import PlateSolver2D
    from pino.vdic.engine import VirtualDIC
    import torch.nn.functional as F
    p = out_dir(cfg, fam.name) / "data_fulldic.npz"
    if p.exists():
        z = np.load(p)
        return {k_: z[k_] for k_ in z.files}
    sig = json.loads((cfg.data_dir / "vdic" / "calibration.json").read_text())["sigma_read"] \
        if (cfg.data_dir / "vdic" / "calibration.json").exists() else cfg["dic"]["sigma_read"]
    s = PlateSolver2D(grid)
    xs_px, ys_px = dic_axes_px(cfg, "full")
    xs, ys = dic_axes(cfg, "full")
    vd = VirtualDIC(cfg, grid, xs_px, ys_px, seed=31337)
    out = {f"d_{des}": [] for des in designs}
    for i in range(k):
        x = x_true[i]
        tau = fam.tau(torch.as_tensor(x[None]))[0].numpy()
        uh = s.solve(tau, x[-1])
        per_des = {des: [] for des in designs}
        for P in cfg["dic"]["loads_kN"]:
            vd.set_field(P * 1e3 / (x[-2] * grid.t0) * uh)
            e, _ = vd.measure(sig)
            e = e.cpu() * 1e6                                  # (3, py, px) microstrain
            for des in designs:
                if des == "full":
                    per_des[des].append(e.reshape(-1))
                else:
                    qx, qy = dic_axes(cfg, des)
                    gx = 2 * (np.asarray(qx) - xs[0]) / (xs[-1] - xs[0]) - 1
                    gy = 2 * (np.asarray(qy) - ys[0]) / (ys[-1] - ys[0]) - 1
                    GY, GX = np.meshgrid(gy, gx, indexing="ij")
                    g = torch.tensor(np.stack([GX, GY], -1))[None]
                    v = F.grid_sample(e[None], g, mode="bilinear", align_corners=True)[0]
                    per_des[des].append(v.reshape(-1))
        for des in designs:
            out[f"d_{des}"].append(torch.cat(per_des[des]).numpy())
        print(f"  full virtual DIC case {i + 1}/{k}", flush=True)
    out = {k_: np.stack(v) for k_, v in out.items()}
    np.savez(p, **out)
    return out


def summarize_posterior(S, fam):
    lv = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    qs = np.concatenate([(1 - lv) / 2, (1 + lv) / 2])
    der = fam.derived(S)
    rng = np.random.default_rng(0)
    sub = S[rng.choice(len(S), min(1000, len(S)), replace=False)]
    out = {"mean": S.mean(0), "q": np.quantile(S, qs, axis=0), "levels": lv, "samples": sub,
           "p_detect": float((S[:, 3] > 0.01).mean()) if fam.name == "pit" else np.nan}
    for k, v in der.items():
        out[f"der_{k}_q"] = np.quantile(v, qs)
        out[f"der_{k}_mean"] = v.mean()
    return out


def run(cfg, args):
    grid = Grid.from_cfg(cfg)
    inf = cfg["inference"]
    fam = Family(args.family, cfg, grid)
    designs = inf["designs"]
    loads = cfg["dic"]["loads_kN"]
    noise = load_noise(cfg)
    x_true = truth_cases(cfg, fam)
    fe = FEPool(grid, cfg["dataset"]["workers"])
    data = make_data(cfg, fam, grid, x_true, designs, fe)
    od = out_dir(cfg, fam.name) / "posteriors"
    od.mkdir(parents=True, exist_ok=True)
    tm = inf["tmcmc"]
    nf = inf.get("nugget_floor_frac", 0.0)
    n_full = inf.get("full_design_cases", inf["n_cases"])
    D_name = "D_pit" if fam.name == "pit" else "D_kl"
    tau_D, w_D, nu_D = load_all(cfg, D_name)

    def post(tag, des, i, obs, lik, mu, d, tcfg, seed):
        f = od / f"{tag}_{des}_{i:03d}.npz"
        if f.exists():
            return
        t0 = time.time()
        ll = make_loglik_fn(obs, lik, mu, torch.as_tensor(d))
        S, logZ, info = tmcmc(ll, fam.sample, fam.log_prior, n=tcfg["n"], scale=tcfg["scale"],
                              n_mh=tcfg["n_mh"], seed=seed)
        sm = summarize_posterior(S, fam)
        np.savez(f, wall=time.time() - t0, logZ=logZ, stages=info["stages"], n_eval=info["n_eval"],
                 x_true=x_true[i], **sm)
        print(f"  [{tag} {des} case {i}] {time.time() - t0:.0f}s stages={info['stages']} "
              f"mean={np.round(sm['mean'][:4], 3)} true={np.round(x_true[i][:4], 3)}", flush=True)

    # --- surrogate conditions
    for m in inf.get("models", ["pino"]):
        for N in inf["surrogate_N"]:
            ck = cfg.results_dir("step5") / "ckpt" / f"{m}_A_N{N}_s0.pt" if fam.name == "pit" \
                else cfg.results_dir("step5") / "ckpt" / f"{m}_B_N{N}_s0.pt"
            if not ck.exists():
                print(f"  missing checkpoint {ck}, skipping", flush=True)
                continue
            sur = load_surrogate(ck, cfg, grid, DEV)
            for des in designs:
                H = make_obs(cfg, grid, des).to(DEV)
                obs = Observer(fam, H, loads, surrogate=sur)
                sg = surrogate_error(sur, H, tau_D, w_D, nu_D, inf["sigma_g_rank"])
                axes = dic_axes(cfg, des)
                liks = {"noSG": build_likelihood(obs, noise, axes, None, nugget_floor_frac=nf),
                        "SG": build_likelihood(obs, noise, axes, sg, nugget_floor_frac=nf)}
                ncase = len(x_true) if des != "full" else min(n_full, len(x_true))
                for i in range(ncase):
                    for cond, (lik, mu) in liks.items():
                        post(f"{m}_N{N}_{cond}", des, i, obs, lik, mu, data[f"d_{des}"][i], tm, i)
                # realism check: full virtual-DIC data, best condition only
                if args.realism and m == inf.get("models", ["pino"])[-1] and N == max(inf["surrogate_N"]):
                    rd = realism_data(cfg, fam, grid, x_true, inf["full_dice_cases"], designs)
                    for i in range(inf["full_dice_cases"]):
                        lik, mu = liks["SG"]
                        post(f"{m}_N{N}_SG_fulldic", des, i, obs, lik, mu, rd[f"d_{des}"][i], tm, i)
            del sur
            torch.cuda.empty_cache()

    # --- FE in the loop reference (noise-only likelihood: the FE model is the truth model)
    ftm = inf.get("fe_tmcmc", tm)
    for des in designs:
        H = make_obs(cfg, grid, des, dtype=torch.float64).to(DEV)
        obs = Observer(fam, H, loads, fe_pool=fe)
        lik, mu = build_likelihood(obs, noise, dic_axes(cfg, des), None, nugget_floor_frac=nf)
        for i in range(inf["fe_reference_cases"]):
            if fam.name == "pit" and x_true[i, 3] == 0 and inf["n_cases"] > inf["n_zero_delta"]:
                j = inf["n_zero_delta"] + i                     # reference on damaged cases
            else:
                j = i
            if j >= len(x_true):
                continue
            f = od / f"FE_{des}_{j:03d}.npz"
            if f.exists():
                continue
            t0 = time.time()
            ll = make_loglik_fn(obs, lik, mu, torch.as_tensor(data[f"d_{des}"][j]), batch=200)
            S, logZ, info = tmcmc(ll, fam.sample, fam.log_prior, n=ftm["n"], scale=ftm["scale"],
                                  n_mh=ftm["n_mh"], seed=j)
            np.savez(f, wall=time.time() - t0, logZ=logZ, stages=info["stages"], n_eval=info["n_eval"],
                     x_true=x_true[j], **summarize_posterior(S, fam))
            print(f"  [FE {des} case {j}] {time.time() - t0:.0f}s, {info['n_eval']} FE solves", flush=True)
    fe.close()


# ----------------------------------------------------------------------------- report
def report(cfg, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = Grid.from_cfg(cfg)
    fam = Family(args.family, cfg, grid)
    od = out_dir(cfg, fam.name)
    files = sorted((od / "posteriors").glob("*.npz"))
    if not files:
        print("no posteriors yet"); return
    recs = {}
    for f in files:
        parts = f.stem.split("_")
        des, i = parts[-2], int(parts[-1])
        cond = "_".join(parts[:-2])
        z = np.load(f)
        recs.setdefault((cond, des), {})[i] = {k: z[k] for k in z.files}
    width = fam.width
    keys = PIT_KEYS if fam.name == "pit" else fam.names[:3]
    kidx = [fam.names.index(k) for k in keys]
    table, cal = [], {}
    for (cond, des), cases in sorted(recs.items()):
        idx = sorted(cases)
        xt = np.stack([cases[i]["x_true"] for i in idx])
        dmg = xt[:, 3] > 0 if fam.name == "pit" else np.ones(len(idx), bool)
        lv = cases[idx[0]]["levels"]
        nl = len(lv)
        cov = np.zeros((nl, len(keys)))
        for li in range(nl):
            for kk, j in enumerate(kidx):
                inside = [(cases[i]["q"][li, j] <= xt[n, j] <= cases[i]["q"][nl + li, j])
                          for n, i in enumerate(idx) if dmg[n]]
                cov[li, kk] = np.mean(inside) if inside else np.nan
        # derived lost volume coverage at 90%
        l90 = int(np.argmin(np.abs(lv - 0.9)))
        dk = [k for k in cases[idx[0]] if k.startswith("der_") and k.endswith("_q")][0]
        dtrue = list(fam.derived(xt).values())[0]
        dcov = np.mean([cases[i][dk][l90] <= dtrue[n] <= cases[i][dk][nl + l90]
                        for n, i in enumerate(idx) if dmg[n]]) if dmg.any() else np.nan
        err = np.stack([np.abs(cases[i]["mean"][kidx] - xt[n, kidx]) / width[kidx]
                        for n, i in enumerate(idx) if dmg[n]]) if dmg.any() else np.full((1, len(keys)), np.nan)
        # Wasserstein-1 to the FE reference for the same design and case
        w1 = []
        ref = recs.get(("FE", des), {})
        if cond != "FE":
            for i in idx:
                if i in ref:
                    a, b = cases[i]["samples"], ref[i]["samples"]
                    w1.append([wasserstein_distance(a[:, j], b[:, j]) / width[j] for j in kidx])
        row = dict(condition=cond, design=des, n_cases=len(idx), n_damaged=int(dmg.sum()),
                   **{f"cov90_{k}": cov[l90, kk] for kk, k in enumerate(keys)},
                   cov90_derived=dcov,
                   **{f"err_{k}": float(np.nanmean(err[:, kk])) for kk, k in enumerate(keys)},
                   **({f"W1_{k}": float(np.mean(np.array(w1)[:, kk])) for kk, k in enumerate(keys)} if w1 else {}),
                   wall_s=float(np.mean([cases[i]["wall"] for i in idx])))
        if fam.name == "pit":
            row["false_alarm_p_detect"] = float(np.mean([cases[i]["p_detect"] for n, i in enumerate(idx) if not dmg[n]])) if (~dmg).any() else np.nan
            gate = all(0.85 <= cov[l90, kk] <= 0.95 for kk in range(len(keys)))
            row["gate_cov90"] = bool(gate)
            if w1:
                row["gate_W1"] = bool(np.max(np.array(w1)) < 0.1)
        table.append(row)
        cal[(cond, des)] = (lv, np.nanmean(cov, 1))
    config.save_json({"rows": table}, od / "metrics.json", cfg)
    # markdown
    cols = list(dict.fromkeys(k for r in table for k in r))
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in table:
        md.append("| " + " | ".join(f"{r.get(c):.3f}" if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols) + " |")
    (od / "metrics.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    # calibration plot
    des_list = sorted({d for _, d in cal})
    fig, axs = plt.subplots(1, len(des_list), figsize=(5 * len(des_list), 4.5), squeeze=False)
    for a, des in zip(axs[0], des_list):
        a.plot([0.4, 1], [0.4, 1], "k--", lw=0.8)
        for (cond, d), (lv, c) in cal.items():
            if d == des:
                a.plot(lv, c, "o-", label=cond)
        a.set_title(f"design: {des}"); a.set_xlabel("nominal coverage"); a.set_ylabel("empirical coverage")
        a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(od / "calibration.png", dpi=130)
    # detection-limit map
    if fam.name == "pit":
        cands = [k for k in recs if k[0].endswith("_SG") and k[1] == "grid56"] or list(recs)
        key = sorted(cands)[-1]
        cases = recs[key]
        xt = np.stack([cases[i]["x_true"] for i in sorted(cases)])
        pd_ = np.array([cases[i]["p_detect"] for i in sorted(cases)])
        fig, a = plt.subplots(figsize=(5.5, 4.5))
        sc = a.scatter(xt[:, 3] * grid.t0, xt[:, 2], c=pd_, cmap="viridis", vmin=0, vmax=1, s=40)
        plt.colorbar(sc, label="P(delta > 0.01 | data)")
        a.set_xlabel("true depth (mm)"); a.set_ylabel("true radius r (mm)")
        a.set_title(f"detection map: {key[0]} ({key[1]})")
        fig.tight_layout(); fig.savefig(od / "detection_map.png", dpi=130)


def main():
    cfg, args = config.cli(__doc__, lambda ap: (
        ap.add_argument("--family", default="pit", choices=["pit", "kl"]),
        ap.add_argument("--stage", default="all", choices=["all", "run", "report"]),
        ap.add_argument("--realism", action="store_true", default=True),
        ap.add_argument("--no-realism", dest="realism", action="store_false")))
    if args.stage in ("all", "run"):
        run(cfg, args)
    if args.stage in ("all", "report"):
        report(cfg, args)


if __name__ == "__main__":
    main()
