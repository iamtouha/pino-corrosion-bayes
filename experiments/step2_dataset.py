"""Step 2: generate the FE datasets A-D and the unlabelled set U.

Each case is solved at unit load and unit modulus; we store w = u_hat - u0_hat(nu).
Resumable: finished rows are flagged in 'done'.
"""
import time

import h5py
import numpy as np
import torch

from pino import config, data, thickness as th
from pino.fe.plate2d import solve_many
from pino.grid import Grid


def params(cfg, grid, name, n):
    seed = data.SEEDS[name] + 100 * cfg["seed"]
    fam = data.FAMILY[name]
    if fam == "pit":
        return th.sample_pit(n, cfg, seed)
    if fam == "kl":
        return th.sample_kl(n, cfg, seed)
    if fam == "triangle":
        return th.sample_triangle(n, cfg, grid, seed)
    raise ValueError(name)


def gen_labelled(cfg, grid, name, chunk=200):
    n = cfg["dataset"]["sizes"][name]
    theta, nu = params(cfg, grid, name, n)
    fam = data.FAMILY[name]
    p = data.path(cfg, name)
    if p.exists():
        f = h5py.File(p, "a")
        if f["nu"].shape[0] != n:
            f.close(); f = data.create(cfg, name, n, theta.shape[1], grid)
    else:
        f = data.create(cfg, name, n, theta.shape[1], grid)
    f["theta"][:] = theta.astype("f4"); f["nu"][:] = nu.astype("f4")
    done = f["done"][:]
    todo = np.flatnonzero(done == 0)
    print(f"[{name}] {fam}: {n} cases, {len(todo)} to solve", flush=True)
    t0 = time.time()
    for s in range(0, len(todo), chunk):
        idx = todo[s:s + chunk]
        tau = th.tau_from_theta(fam, theta[idx], cfg, grid).numpy()
        u = solve_many(grid, tau, nu[idx], workers=cfg["dataset"]["workers"])
        w = np.stack([u[i] - grid.u0_hat(nu[k]) for i, k in enumerate(idx)])
        for i, k in enumerate(idx):                 # h5py needs increasing indices; idx is sorted
            f["t_tilde"][k] = tau[i]; f["w"][k] = w[i]
        f["done"][idx] = 1
        f.flush()
        el = time.time() - t0
        print(f"  {s + len(idx)}/{len(todo)}  {el / (s + len(idx)):.2f} s/case wall", flush=True)
    f.close()


def gen_unlabelled(cfg, grid):
    n = cfg["dataset"]["sizes"]["U"]
    n_pit = n // 2
    tp, nup = th.sample_pit(n_pit, cfg, data.SEEDS["U"] + 100 * cfg["seed"])
    tk, nuk = th.sample_kl(n - n_pit, cfg, data.SEEDS["U"] + 1 + 100 * cfg["seed"])
    D = tk.shape[1]
    theta = np.zeros((n, D)); theta[:n_pit, :4] = tp; theta[n_pit:] = tk
    nu = np.concatenate([nup, nuk]); code = np.r_[np.zeros(n_pit), np.ones(n - n_pit)].astype("u1")
    f = data.create(cfg, "U", n, D, grid, with_w=False)
    f["theta"][:] = theta.astype("f4"); f["nu"][:] = nu.astype("f4")
    f.create_dataset("family_code", data=code)
    for s in range(0, n, 1000):
        e = min(n, s + 1000)
        tau = np.empty((e - s, *grid.shape))
        m = code[s:e] == 0
        if m.any():
            tau[m] = th.tau_from_theta("pit", theta[s:e][m, :4], cfg, grid).numpy()
        if (~m).any():
            tau[~m] = th.tau_from_theta("kl", theta[s:e][~m], cfg, grid).numpy()
        f["t_tilde"][s:e] = tau.astype("f4")
    f["done"][:] = 1
    f.close()
    print(f"[U] {n} unlabelled thickness fields", flush=True)


def main():
    cfg, args = config.cli(__doc__, lambda ap: ap.add_argument(
        "--sets", nargs="*", default=["A", "B", "C", "D_pit", "D_kl", "U"]))
    torch.set_num_threads(4)
    grid = Grid.from_cfg(cfg)
    k = cfg["priors"]["kl"]
    for ell in k["ells"]:
        b = th.kl_basis(grid, float(ell), k["var_fraction"], k["K_max"])
        print(f"KL ell={ell}: K={b.K} (needed {b.K_needed} for {k['var_fraction']:.0%}), "
              f"variance held {b.var_held:.3f}", flush=True)
    for name in args.sets:
        if name == "U":
            gen_unlabelled(cfg, grid)
        else:
            gen_labelled(cfg, grid, name)
    config.save_json({"sets": args.sets, "sizes": cfg["dataset"]["sizes"]},
                     cfg.data_dir / "dataset_manifest.json", cfg)


if __name__ == "__main__":
    main()
