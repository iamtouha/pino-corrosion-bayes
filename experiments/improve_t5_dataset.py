"""T5 step 3 (follow-up studies, README): 3D solid datasets for one-sided corrosion (pit family).

Sets improve.t5.sets (A3: train pool + test, D3_pit: held out for Sigma_G), pit thickness from
the same prior as set A (Sobol, own seeds). Each case is one solve_plate3d (forward.nz layers,
CG rtol forward.rtol) at unit load and modulus; BOTH faces are stored, as perturbations of the
intact in-plane solution on the operator grid (DIC on both faces):
  w_flat     (N, ny+1, nx+1, 2)   z = 0 face (intact back face)
  w_corroded (N, ny+1, nx+1, 2)   z = t0 tau face
plus t_tilde, theta, nu, done, solve_s, cg_its. data/<name>.h5 are new files (the 2D sets
are untouched). Resumable: finished rows are flagged in 'done'; a crash loses one chunk.
"""
import multiprocessing as mp
import time

import h5py
import numpy as np

from pino import config, thickness as th
from pino.grid import Grid


def _solve(args):
    from pino.fe.plate3d import solve_plate3d
    grid, tau, nu, nz, rtol = args
    t = time.time()
    try:
        uf, uc, its = solve_plate3d(grid, tau, nu, nz=nz, rtol=rtol)
    except RuntimeError as e:                       # diverged: retry once with a tighter solve
        print("  retry:", e, flush=True)
        uf, uc, its = solve_plate3d(grid, tau, nu, nz=nz, rtol=rtol * 0.01)
    return uf, uc, its, time.time() - t


def create(cfg, path, n, grid):
    f = h5py.File(path, "w")
    sh = grid.shape
    f.create_dataset("t_tilde", (n, *sh), "f4", chunks=(1, *sh))
    for k in ("w_flat", "w_corroded"):
        f.create_dataset(k, (n, *sh, 2), "f4", chunks=(1, *sh, 2))
    f.create_dataset("theta", (n, 4), "f4")
    f.create_dataset("nu", (n,), "f4")
    f.create_dataset("done", (n,), "u1")
    f.create_dataset("solve_s", (n,), "f4")
    f.create_dataset("cg_its", (n,), "i4")
    f.attrs["family"] = "pit"
    f.attrs["model"] = "solid3d"
    f.attrs["nz"] = cfg["forward"]["nz"]
    f.attrs["rtol"] = cfg["forward"]["rtol"]
    f.attrs["config_hash"] = cfg.hash
    return f


def generate(cfg, grid, name, n, seed):
    tc = cfg["improve"]["t5"]
    fw = cfg["forward"]
    p = cfg.data_dir / f"{name}.h5"
    theta, nu = th.sample_pit(n, cfg, seed)
    if p.exists():
        f = h5py.File(p, "a")
        assert f["nu"].shape[0] == n and f.attrs["nz"] == fw["nz"], f"{p} exists with other settings"
    else:
        f = create(cfg, p, n, grid)
        f["theta"][:] = theta.astype("f4"); f["nu"][:] = nu.astype("f4")
    todo = np.flatnonzero(f["done"][:] == 0)
    print(f"[{name}] {n} cases, {len(todo)} to solve, nz={fw['nz']} rtol={fw['rtol']:g}, "
          f"{tc['workers']} workers", flush=True)
    ctx = mp.get_context("fork")
    t0 = time.time()
    with ctx.Pool(tc["workers"], maxtasksperchild=20) as pool:
        for s in range(0, len(todo), tc["chunk"]):
            idx = todo[s:s + tc["chunk"]]
            tau = th.tau_from_theta("pit", theta[idx], cfg, grid).numpy()
            res = pool.map(_solve, [(grid, tau[i], float(nu[k]), fw["nz"], fw["rtol"])
                                    for i, k in enumerate(idx)], chunksize=1)
            for i, k in enumerate(idx):
                uf, uc, its, dt = res[i]
                u0 = grid.u0_hat(nu[k])
                f["t_tilde"][k] = tau[i]
                f["w_flat"][k] = uf - u0
                f["w_corroded"][k] = uc - u0
                f["solve_s"][k] = dt; f["cg_its"][k] = its
            f["done"][idx] = 1
            f.flush()
            el = time.time() - t0
            m = s + len(idx)
            print(f"  {m}/{len(todo)}  {el / m:.1f} s/case wall, eta {(len(todo) - m) * el / m / 3600:.1f} h", flush=True)
    f.close()


def main():
    cfg, args = config.cli(__doc__, lambda ap: ap.add_argument("--sets", nargs="*", default=None))
    grid = Grid.from_cfg(cfg)
    sets = cfg["improve"]["t5"]["sets"]
    for name in (args.sets or list(sets)):
        n, seed = sets[name]
        generate(cfg, grid, name, n, seed + 100 * cfg["seed"])


if __name__ == "__main__":
    main()
