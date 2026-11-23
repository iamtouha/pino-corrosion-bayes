"""T5 step 2 (follow-up studies, README): profile the 3D solid solver before generating datasets.

1. Accuracy vs cost, single process: nz in improve.t5.profile.nz and CG rtol in
   improve.t5.profile.rtol on a few pits (the deepest prior pits, where bending is largest).
   Reference: nz = max, rtol = min. Error = max |H(eps_face) - H(eps_face_ref)| over the full
   DIC design, as a fraction of the damage signal max |H(eps_2D) - H(eps_0)|, for both faces.
   Also wall time, CG iterations and peak RSS per solve.
2. Throughput with k parallel worker processes (improve.t5.profile.workers) at the chosen
   (nz, rtol): solves per hour and peak RSS per worker, to size the dataset run.
Output: results/laptop/improve/t5/profile.json (+ printed table).
"""
import multiprocessing as mp
import resource
import time

import numpy as np
import torch

from pino import config
from pino.grid import Grid, q1_strain
from pino.thickness import pit_tau


def _solve(args):
    from pino.fe.plate3d import solve_plate3d
    grid, tau, nu, nz, rtol = args
    t = time.time()
    uf, uc, its = solve_plate3d(grid, tau, nu, nz=nz, rtol=rtol)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024          # MB (Linux: KB)
    return uf, uc, its, time.time() - t, rss


def main():
    cfg, args = config.cli(__doc__)
    pc = cfg["improve"]["t5"]["profile"]
    grid = Grid.from_cfg(cfg)
    from pino.fe.plate2d import PlateSolver2D
    from pino.vdic.observe import make_obs
    H = make_obs(cfg, grid, "full").double()
    nu = cfg["material"]["nu"]
    cases = [tuple(c) for c in pc["cases"]]                                 # (delta_frac, r_mm)
    s2 = PlateSolver2D(grid)
    e0 = H(q1_strain(torch.tensor(grid.u0_hat(nu))[None], grid.h))
    out = {"accuracy": [], "throughput": []}
    ctx = mp.get_context("fork")
    nz_ref, rtol_ref = max(pc["nz"]), min(pc["rtol"])
    for d, r in cases:
        tau = pit_tau(torch.tensor([[150.0, 125.0, r, d]], dtype=torch.float64), grid)[0].numpy()
        e2 = H(q1_strain(torch.tensor(s2.solve(tau, nu))[None], grid.h))
        sig = (e2 - e0).abs().max().item()
        res = {}
        for nz in pc["nz"]:
            for rtol in pc["rtol"]:
                with ctx.Pool(1) as p:                                       # fresh process: clean RSS
                    uf, uc, its, dt, rss = p.map(_solve, [(grid, tau, nu, nz, rtol)])[0]
                res[(nz, rtol)] = (uf, uc)
                out["accuracy"].append(dict(delta_frac=d, r_mm=r, nz=nz, rtol=rtol, solve_s=dt, cg_its=its,
                                            peak_rss_mb=rss))
                print(f"  delta={d} r={r} nz={nz} rtol={rtol:g}: {dt:.1f}s, {its} its, {rss:.0f} MB", flush=True)
        ef_ref = [H(q1_strain(torch.tensor(u)[None], grid.h)) for u in res[(nz_ref, rtol_ref)]]
        for rec in out["accuracy"]:
            if rec["delta_frac"] != d or rec["r_mm"] != r:
                continue
            uf, uc = res[(rec["nz"], rec["rtol"])]
            for face, u, ref in (("flat", uf, ef_ref[0]), ("corroded", uc, ef_ref[1])):
                ef = H(q1_strain(torch.tensor(u)[None], grid.h))
                rec[f"{face}_err_over_signal"] = (ef - ref).abs().max().item() / sig
            rec["signal_max_ue_at_80kN"] = sig * 80e3 / (cfg["material"]["E"] * grid.t0) * 1e6
    # throughput at the chosen setting
    nz, rtol = pc["chosen"]["nz"], pc["chosen"]["rtol"]
    rng = np.random.default_rng(0)
    for k in pc["workers"]:
        n = 2 * k
        th = np.stack([rng.uniform(50, 250, n), rng.uniform(30, 220, n), rng.uniform(8, 50, n),
                       rng.uniform(0, 0.15, n)], 1)
        taus = pit_tau(torch.tensor(th), grid).numpy()
        t = time.time()
        with ctx.Pool(k) as p:
            rr = p.map(_solve, [(grid, taus[i], nu, nz, rtol) for i in range(n)], chunksize=1)
        wall = time.time() - t
        out["throughput"].append(dict(workers=k, n=n, wall_s=wall, solves_per_hour=3600 * n / wall,
                                      mean_solve_s=float(np.mean([x[3] for x in rr])),
                                      peak_rss_mb_per_worker=float(max(x[4] for x in rr))))
        print(f"  workers={k}: {3600 * n / wall:.0f} solves/h, solve {np.mean([x[3] for x in rr]):.0f}s, "
              f"RSS {max(x[4] for x in rr):.0f} MB/worker", flush=True)
    config.save_json(out, cfg.results_dir("improve", "t5") / "profile.json", cfg)
    print("accuracy (error / damage signal, after H, full design):")
    for rec in out["accuracy"]:
        print(f"  d={rec['delta_frac']} r={rec['r_mm']} nz={rec['nz']} rtol={rec['rtol']:g}: "
              f"flat {rec['flat_err_over_signal']:.4f}  corroded {rec['corroded_err_over_signal']:.4f}  "
              f"{rec['solve_s']:.0f}s {rec['peak_rss_mb']:.0f}MB")


if __name__ == "__main__":
    main()
