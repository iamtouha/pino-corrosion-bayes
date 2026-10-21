"""Positive control for the identification pipeline: one large, deep pit.

Truth theta = (150, 125, 40 mm, delta = 0.15), E, nu at the prior centre. Two data sources:
  self : the surrogate's own prediction + emulator noise (isolates likelihood/sampler bugs)
  fe   : FE truth + emulator noise (adds surrogate error; Sigma_G likelihood)
for each design. If 'self' does not recover the pit, the problem is in the likelihood or the
sampler, not in the surrogate.
"""
import time

import numpy as np
import torch

from pino import config
from pino.grid import Grid
from pino.inference.problem import (DEV, Family, FEPool, Observer, build_likelihood,
                                    make_loglik_fn, surrogate_error)
from pino.inference.tmcmc import tmcmc
from pino.surrogates.models import load_surrogate
from pino.surrogates.train import load_all
from pino.vdic.noise import NoiseEmulator, load_noise
from pino.vdic.observe import dic_axes, make_obs


def main():
    cfg, args = config.cli(__doc__, lambda ap: (
        ap.add_argument("--N", type=int, default=None),
        ap.add_argument("--delta", type=float, default=0.15),
        ap.add_argument("--r", type=float, default=40.0),
        ap.add_argument("--designs", nargs="*", default=None),
        ap.add_argument("--sources", nargs="*", default=["self", "fe"]),
        ap.add_argument("--out", default="positive_control.json"),
        ap.add_argument("--nugget-floor", type=float, default=None)))
    grid = Grid.from_cfg(cfg)
    inf = cfg["inference"]
    fam = Family("pit", cfg, grid)
    loads = cfg["dic"]["loads_kN"]
    noise = load_noise(cfg)
    m, N = inf.get("models", ["pino"])[-1], args.N or max(inf["surrogate_N"])
    sur = load_surrogate(cfg.results_dir("step5") / "ckpt" / f"{m}_A_N{N}_s0.pt", cfg, grid, DEV)
    x = 0.5 * (fam.lo + fam.hi)
    x[:4] = [150.0, 125.0, args.r, args.delta]
    xt = torch.as_tensor(x[None])
    fe = FEPool(grid, 2)
    tau_D, w_D, nu_D = load_all(cfg, "D_pit")
    gen = torch.Generator(device="cpu").manual_seed(99)
    tm = inf["tmcmc"]
    NF = args.nugget_floor if args.nugget_floor is not None else inf.get("nugget_floor_frac", 0.0)
    rows = []
    for des in args.designs or inf["designs"]:
        H = make_obs(cfg, grid, des).to(DEV)
        obs_s = Observer(fam, H, loads, surrogate=sur)
        obs_f = Observer(fam, make_obs(cfg, grid, des, dtype=torch.float64).to(DEV), loads, fe_pool=fe)
        em = NoiseEmulator(noise, *dic_axes(cfg, des))
        n = em.sample(len(loads), gen).reshape(-1) * 1e6
        clean = {"self": obs_s(xt).cpu().double()[0], "fe": obs_f(xt).cpu().double()[0]}
        sg = surrogate_error(sur, H, tau_D, w_D, nu_D, inf["sigma_g_rank"])
        for src, c in clean.items():
            if src not in args.sources:
                continue
            lik, mu = build_likelihood(obs_s, noise, dic_axes(cfg, des), None if src == "self" else sg,
                                       nugget_floor_frac=NF)
            d = c + n
            ll = make_loglik_fn(obs_s, lik, mu, d)
            t0 = time.time()
            S, logZ, info = tmcmc(ll, fam.sample, fam.log_prior, n=tm["n"], scale=tm["scale"],
                                  n_mh=tm["n_mh"], seed=0, verbose=True)
            q = np.quantile(S[:, :4], [0.05, 0.5, 0.95], 0)
            row = dict(design=des, source=src, nugget_floor_frac=NF, wall_s=time.time() - t0, stages=info["stages"],
                       unique_frac=info["unique_frac"], n_eval=info["n_eval"],
                       truth=x[:4].tolist(), q05=q[0].tolist(), q50=q[1].tolist(), q95=q[2].tolist(),
                       covered=[bool(q[0, j] <= x[j] <= q[2, j]) for j in range(4)],
                       snr=float((c - obs_s(torch.as_tensor(np.r_[x[:3], 0.0, x[4:]][None])).cpu()[0]).norm()
                                 / n.norm()))
            rows.append(row)
            print(row, flush=True)
    fe.close()
    config.save_json({"rows": rows}, cfg.results_dir("step7") / args.out, cfg)


if __name__ == "__main__":
    main()
