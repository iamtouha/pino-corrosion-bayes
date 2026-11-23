"""T5 steps 5-7 (follow-up studies, README): inference with the 3D forward model, model-form error of the
membrane assumption, FE reference by importance reweighting, timing.

Truth: the step-7 pit truth set (results/laptop/step7/pit/data.npz: x_true), first
improve.t5.infer.n_damaged damaged cases + the intact ones. Data: 3D solid FE (forward.nz,
forward.rtol) at the truth, observed face(s) forward.observed_face, after H, 3 loads, plus
emulator DIC noise (fixed seed). Cached: results/laptop/improve/t5/data3d.npz.

Conditions per face x design (improve.t5.infer.designs), TMCMC n = inference.tmcmc (T1: 1000):
  fno3d_SG     3D-face FNO (improve_t5_train.py) with Sigma_G from D3_pit
  membrane_SG  the 2D membrane PINO N=2000 with its own Sigma_G (D_pit): model-form error
FE reference (improve.t5.infer.ref_cases damaged cases, grid56): importance reweighting of the
fno3d_SG posterior by the 3D-FE likelihood, w_i ~ exp(ll_FE - ll_sur) over
improve.t5.infer.ref_samples unique particles; ESS reported. W1 of every condition to the
reweighted posterior. Timing: 3D FE solve vs surrogate evaluation, wall time per posterior.
Outputs: results/laptop/improve/t5/{data3d.npz, posteriors/*.npz, ref/*.npz, infer.{json,md}}
"""
import multiprocessing as mp
import time

import h5py
import numpy as np
import torch
from scipy.stats import wasserstein_distance

from pino import config
from pino.grid import Grid, q1_strain

KEYS = ["xc", "yc", "r", "delta"]


def faces_of(cfg):
    f = cfg["forward"]["observed_face"]
    return ["flat", "corroded"] if f == "both" else [f]


def _solve3d(args):
    from pino.fe.plate3d import solve_plate3d
    grid, tau, nu, nz, rtol = args
    t = time.time()
    uf, uc, _ = solve_plate3d(grid, tau, nu, nz=nz, rtol=rtol)
    return uf, uc, time.time() - t


class FE3DPool:
    """solve(taus, nus) -> face u_hat (b, ny+1, nx+1, 2) for one face, 3D solid, parallel."""

    def __init__(self, cfg, grid, face, workers):
        self.cfg, self.grid, self.face = cfg, grid, face
        self.pool = mp.get_context("fork").Pool(workers, maxtasksperchild=20)
        self.times = []

    def solve(self, taus, nus):
        fw = self.cfg["forward"]
        res = self.pool.map(_solve3d, [(self.grid, t, float(n), fw["nz"], fw["rtol"]) for t, n in zip(taus, nus)],
                            chunksize=1)
        self.times += [r[2] for r in res]
        return np.stack([r[0] if self.face == "flat" else r[1] for r in res])

    def close(self):
        self.pool.close(); self.pool.join()


def load_d3(cfg, face):
    with h5py.File(cfg.data_dir / "D3_pit.h5", "r") as f:
        done = np.flatnonzero(f["done"][:] == 1)
        return (torch.from_numpy(f["t_tilde"][done]), torch.from_numpy(f[f"w_{face}"][done]),
                torch.from_numpy(f["nu"][done]))


def make_data3d(cfg, grid, fam, x_true, idx, designs, workers):
    from pino.inference.problem import DEV
    from pino.vdic.noise import NoiseEmulator, load_noise
    from pino.vdic.observe import dic_axes, make_obs
    p = cfg.results_dir("improve", "t5") / "data3d.npz"
    if p.exists():
        z = np.load(p)
        return {k: z[k] for k in z.files}
    fw = cfg["forward"]
    taus = fam.tau(torch.as_tensor(x_true[idx])).numpy()
    t0 = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_solve3d, [(grid, taus[k], float(x_true[i, -1]), fw["nz"], fw["rtol"])
                                  for k, i in enumerate(idx)], chunksize=1)
    print(f"  {len(idx)} truth 3D solves in {time.time() - t0:.0f}s", flush=True)
    loads = torch.tensor(cfg["dic"]["loads_kN"], dtype=torch.float64) * 1e3
    noise = load_noise(cfg)
    gen = torch.Generator().manual_seed(5353)
    out = {"idx": np.asarray(idx), "x_true": x_true[idx], "solve_s": np.array([r[2] for r in res])}
    for des in designs:
        H = make_obs(cfg, grid, des, dtype=torch.float64).to(DEV)
        xs, ys = dic_axes(cfg, des)
        em = NoiseEmulator(noise, xs, ys)
        nz_ = torch.stack([em.sample(len(loads), gen).reshape(-1) for _ in idx]) * 1e6
        for fi, face in enumerate(("flat", "corroded")):
            u = torch.as_tensor(np.stack([r[fi] for r in res])).to(DEV)
            h = H(q1_strain(u, grid.h)).cpu()                                     # (b, m) unit load
            E = torch.as_tensor(x_true[idx, -2])
            sc = loads[None] / (E[:, None] * grid.t0) * 1e6                      # (b, L)
            clean = (sc[:, :, None] * h[:, None, :]).reshape(len(idx), -1)
            out[f"clean_{face}_{des}"] = clean.numpy()
            out[f"d_{face}_{des}"] = (clean + nz_).numpy()                      # same noise draw per face
    np.savez(p, **out)
    return out


def run(cfg, args):
    from experiments.step7_benchmark import summarize_posterior
    from pino.inference.problem import (DEV, Family, Observer, build_likelihood, make_loglik_fn,
                                        surrogate_error)
    from pino.inference.tmcmc import tmcmc
    from pino.surrogates.models import load_surrogate
    from pino.surrogates.train import load_all
    from pino.vdic.noise import load_noise
    from pino.vdic.observe import dic_axes, make_obs
    tc = cfg["improve"]["t5"]; ic = tc["infer"]
    grid = Grid.from_cfg(cfg)
    fam = Family("pit", cfg, grid)
    inf = cfg["inference"]; tm = inf["tmcmc"]; nf = inf.get("nugget_floor_frac", 0.0)
    loads = cfg["dic"]["loads_kN"]; noise = load_noise(cfg)
    x_all = np.load(cfg.results_dir("step7", "pit") / "data.npz")["x_true"]
    dmg = np.flatnonzero(x_all[:, 3] > 0)[: ic["n_damaged"]]
    idx = np.r_[np.flatnonzero(x_all[:, 3] == 0), dmg]
    D = make_data3d(cfg, grid, fam, x_all, idx, ic["designs"], tc["workers"])
    od = cfg.results_dir("improve", "t5", "posteriors")
    rd = cfg.results_dir("improve", "t5", "ref")
    tau_Dm, w_Dm, nu_Dm = load_all(cfg, "D_pit")
    for face in faces_of(cfg):
        tau_D3, w_D3, nu_D3 = load_d3(cfg, face)
        surs = {"fno3d": (cfg.results_dir("improve", "t5") / "ckpt" / f"fno_A3_{face}_s0.pt", (tau_D3, w_D3, nu_D3)),
                "membrane": (cfg.results_dir("step5") / "ckpt" / "pino_A_N2000_s0.pt", (tau_Dm, w_Dm, nu_Dm))}
        for sname, (ck, Dset) in surs.items():
            sur = load_surrogate(ck, cfg, grid, DEV)
            for des in ic["designs"]:
                H = make_obs(cfg, grid, des).to(DEV)
                obs = Observer(fam, H, loads, surrogate=sur)
                sg = surrogate_error(sur, H, *Dset, inf["sigma_g_rank"])
                lik, mu = build_likelihood(obs, noise, dic_axes(cfg, des), sg, nugget_floor_frac=nf)
                for k, i in enumerate(idx):
                    f = od / f"{sname}_SG_{face}_{des}_{i:03d}.npz"
                    if f.exists():
                        continue
                    t0 = time.time()
                    ll = make_loglik_fn(obs, lik, mu, torch.as_tensor(D[f"d_{face}_{des}"][k]))
                    S, logZ, info = tmcmc(ll, fam.sample, fam.log_prior, n=tm["n"], scale=tm["scale"],
                                          n_mh=tm["n_mh"], seed=int(i))
                    np.savez(f, wall=time.time() - t0, n_eval=info["n_eval"], stages=info["stages"],
                             x_true=x_all[i], all_samples=S, ll_sur=ll(S), **summarize_posterior(S, fam))
                    print(f"  [{f.stem}] {time.time() - t0:.0f}s evals={info['n_eval']}", flush=True)
            del sur
            torch.cuda.empty_cache()
    # importance-reweighting reference with the 3D FE likelihood (grid56, fno3d posterior)
    if args.no_ref:
        return
    for face in faces_of(cfg):
        fe = None
        for i in dmg[: ic["ref_cases"]]:
            f = rd / f"ref_{face}_grid56_{i:03d}.npz"
            if f.exists():
                continue
            fe = fe or FE3DPool(cfg, grid, face, tc["workers"])
            k = int(np.flatnonzero(idx == i)[0])
            z = np.load(od / f"fno3d_SG_{face}_grid56_{i:03d}.npz")
            S, ll_s = z["all_samples"], z["ll_sur"]
            uq, first = np.unique(S, axis=0, return_index=True)
            rng = np.random.default_rng(int(i))
            pick = first[rng.choice(len(first), min(ic["ref_samples"], len(first)), replace=False)]
            # multiplicity of each picked particle in the posterior population (TMCMC duplicates)
            mult = np.array([np.all(S == S[j], 1).sum() for j in pick])
            H = make_obs(cfg, grid, "grid56", dtype=torch.float64).to(DEV)
            obs = Observer(fam, H, loads, fe_pool=fe)
            lik, mu = build_likelihood(obs, noise, dic_axes(cfg, "grid56"), None, nugget_floor_frac=nf)
            t0 = time.time()
            ll_fe = make_loglik_fn(obs, lik, mu, torch.as_tensor(D[f"d_{face}_grid56"][k]), batch=60)(S[pick])
            lw = np.log(mult) + ll_fe - ll_s[pick]
            w = np.exp(lw - lw.max()); w /= w.sum()
            ess = 1.0 / (w ** 2).sum()
            np.savez(f, samples=S[pick], weights=w, ess=ess, n=len(pick), wall=time.time() - t0,
                     x_true=x_all[i], fe_solve_s=np.array(fe.times[-len(pick):]))
            print(f"  [ref {face} case {i}] {len(pick)} 3D solves in {time.time() - t0:.0f}s, ESS {ess:.0f}", flush=True)
        if fe is not None:
            fe.close()


def wq(v, w, q):
    o = np.argsort(v); c = np.cumsum(w[o]); return np.interp(q, c / c[-1], v[o])


def report(cfg, args):
    from pino.inference.problem import Family
    tc = cfg["improve"]["t5"]
    grid = Grid.from_cfg(cfg)
    fam = Family("pit", cfg, grid)
    width = fam.width[:4]
    root = cfg.results_dir("improve", "t5")
    recs = {}
    for f in sorted((root / "posteriors").glob("*.npz")):
        p = f.stem.split("_")
        recs.setdefault("_".join(p[:-1]), {})[int(p[-1])] = np.load(f)
    refs = {}
    for f in sorted((root / "ref").glob("*.npz")):
        _, face, des, i = f.stem.split("_")
        refs[(face, des, int(i))] = np.load(f)
    rows = []
    for key, cases in recs.items():
        sname, _, face, des = key.split("_")
        idx = sorted(cases)
        xt = np.stack([cases[i]["x_true"] for i in idx])
        dm = xt[:, 3] > 0
        lv = cases[idx[0]]["levels"]; nl = len(lv); l90 = int(np.argmin(np.abs(lv - 0.9)))
        cov = np.array([[cases[i]["q"][l90, j] <= xt[n, j] <= cases[i]["q"][nl + l90, j] for j in range(4)]
                        for n, i in enumerate(idx) if dm[n]])
        bias = np.array([(cases[i]["mean"][:4] - xt[n, :4]) / width for n, i in enumerate(idx) if dm[n]])
        w1, cov_ref = [], []
        for i in idx:
            r = refs.get((face, des, i))
            if r is None:
                continue
            S = cases[i]["all_samples"]
            w1.append([wasserstein_distance(S[:, j], r["samples"][:, j], None, r["weights"]) / width[j]
                       for j in range(4)])
        rows.append(dict(surrogate=sname, face=face, design=des, n_damaged=int(dm.sum()),
                         **{f"cov90_{k}": float(cov[:, j].mean()) for j, k in enumerate(KEYS)},
                         **{f"bias_{k}": float(bias[:, j].mean()) for j, k in enumerate(KEYS)},
                         false_alarm=float(np.mean([cases[i]["p_detect"] for n, i in enumerate(idx) if not dm[n]]))
                         if (~dm).any() else None,
                         W1_to_3dFE_mean=float(np.mean(w1)) if w1 else None, n_ref=len(w1),
                         wall_s=float(np.mean([cases[i]["wall"] for i in idx])),
                         n_eval=float(np.mean([cases[i]["n_eval"] for i in idx]))))
    ref_rows = [dict(face=k[0], design=k[1], case=k[2], ess=float(v["ess"]), n=int(v["n"]),
                     fe_solve_s_mean=float(np.mean(v["fe_solve_s"])), wall_s=float(v["wall"])) for k, v in refs.items()]
    tr = root / "train.json"
    out = dict(rows=rows, reference=ref_rows, train=(__import__("json").loads(tr.read_text()) if tr.exists() else None))
    d3 = root / "data3d.npz"
    if d3.exists():
        out["fe3d_solve_s_mean"] = float(np.load(d3)["solve_s"].mean())
    config.save_json(out, root / "infer.json", cfg)
    cols = ["surrogate", "face", "design", "n_damaged", "cov90_xc", "cov90_yc", "cov90_r", "cov90_delta",
            "bias_r", "bias_delta", "false_alarm", "W1_to_3dFE_mean", "n_ref", "wall_s"]
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else ("" if v is None else str(v))
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        md.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    md += ["", "Importance-reweighting references:", ""] + [
        f"- {r['face']} {r['design']} case {r['case']}: ESS {r['ess']:.0f} of {r['n']}, 3D solve {r['fe_solve_s_mean']:.0f}s, wall {r['wall_s'] / 60:.0f} min"
        for r in ref_rows]
    (root / "infer.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    cfg, args = config.cli(__doc__, lambda ap: (
        ap.add_argument("--stage", default="all", choices=["all", "run", "report"]),
        ap.add_argument("--no-ref", action="store_true")))
    if args.stage in ("all", "run"):
        run(cfg, args)
    if args.stage in ("all", "report"):
        report(cfg, args)


if __name__ == "__main__":
    main()
