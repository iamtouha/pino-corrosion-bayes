"""Step 6.4: NUTS (Pyro) on Step 7 benchmark cases with the surrogate, compared with TMCMC.

Acceptance: R-hat < 1.01 and ESS > 400 for every parameter. Reports the marginal
Wasserstein-1 distance (normalized by prior width) between NUTS and TMCMC posteriors.
"""
import time

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from pino import config
from pino.grid import Grid
from pino.inference.nuts import run_nuts
from pino.inference.problem import DEV, Family, Observer, build_likelihood, surrogate_error
from pino.surrogates.models import load_surrogate
from pino.surrogates.train import load_all
from pino.vdic.noise import load_noise
from pino.vdic.observe import dic_axes, make_obs


def main():
    cfg, args = config.cli(__doc__, lambda ap: (
        ap.add_argument("--cases", type=int, nargs="*", default=None),
        ap.add_argument("--design", default="grid56")))
    inf = cfg["inference"]
    grid = Grid.from_cfg(cfg)
    fam = Family("pit", cfg, grid)
    m, N = inf.get("models", ["pino"])[-1], max(inf["surrogate_N"])
    sur = load_surrogate(cfg.results_dir("step5") / "ckpt" / f"{m}_A_N{N}_s0.pt", cfg, grid, DEV)
    H = make_obs(cfg, grid, args.design).to(DEV)
    obs = Observer(fam, H, cfg["dic"]["loads_kN"], surrogate=sur)
    sg = surrogate_error(sur, H, *load_all(cfg, "D_pit"), inf["sigma_g_rank"])
    lik, mu = build_likelihood(obs, load_noise(cfg), dic_axes(cfg, args.design), sg,
                               nugget_floor_frac=inf.get("nugget_floor_frac", 0.0))
    bench = cfg.results_dir("step7", "pit")
    data = np.load(bench / "data.npz")
    cases = args.cases or [inf["n_zero_delta"], inf["n_zero_delta"] + 1]
    nc = inf["nuts"]
    rows = []
    for i in cases:
        t0 = time.time()
        S, diag = run_nuts(fam, obs, lik, mu, torch.as_tensor(data[f"d_{args.design}"][i]),
                           num_samples=nc["num_samples"], warmup_steps=nc["warmup_steps"],
                           chains=nc["chains"], target_accept=nc["target_accept"], seed=i,
                           max_tree_depth=nc.get("max_tree_depth", 10))
        row = dict(case=i, wall_s=time.time() - t0, x_true=data["x_true"][i].tolist(),
                   mean=S.mean(0).tolist(), **diag)
        tf = bench / "posteriors" / f"{m}_N{N}_SG_{args.design}_{i:03d}.npz"
        if tf.exists():
            T = np.load(tf)["samples"]
            row["W1_vs_tmcmc"] = [wasserstein_distance(S[:, j], T[:, j]) / fam.width[j] for j in range(4)]
        rows.append(row)
        print(row, flush=True)
    config.save_json({"rows": rows}, cfg.results_dir("step6") / "nuts.json", cfg)


if __name__ == "__main__":
    main()
