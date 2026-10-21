"""Step 1b: measure the one-sided bending effect with a 3D solid model.

~10 cases: delta in {0.05, 0.1, 0.2, 0.4} mm x r in {10, 30} mm + 2 intact plates.
Surface strain on the flat (z = 0) and corroded faces vs the 2D membrane strain, after H.
Decision: max difference < ~2 % of the damage signal -> keep the membrane model.
"""
import time

import numpy as np
import torch

from pino import config
from pino.fe.plate2d import PlateSolver2D
from pino.fe.plate3d import solve_plate3d
from pino.grid import Grid, q1_strain
from pino.thickness import pit_tau
from pino.vdic.observe import make_obs


def main():
    cfg, args = config.cli(__doc__, lambda ap: (
        ap.add_argument("--nz", type=int, default=4),
        ap.add_argument("--deg", type=int, default=1),
        ap.add_argument("--quick", action="store_true")))
    g = Grid.from_cfg(cfg)
    nu = cfg["material"]["nu"]
    H = make_obs(cfg, g, "full").double()
    s2 = PlateSolver2D(g)
    cases = [(0.0, 10.0), (0.0, 30.0)] + [(d, r) for d in (0.05, 0.1, 0.2, 0.4) for r in (10.0, 30.0)]
    if args.quick:
        cases = [(0.0, 10.0), (0.2, 30.0), (0.4, 10.0)]
    u0 = torch.tensor(g.u0_hat(nu))[None]
    e0 = H(q1_strain(u0, g.h))
    rows = []
    for dmm, r in cases:
        th = torch.tensor([[150.0, 125.0, r, dmm / g.t0]], dtype=torch.float64)
        tau = pit_tau(th, g)[0].numpy()
        u2 = torch.tensor(s2.solve(tau, nu))[None]
        t = time.time()
        uf, uc, its = solve_plate3d(g, tau, nu, nz=args.nz, deg=args.deg)
        dt = time.time() - t
        e2 = H(q1_strain(u2, g.h))
        sig = (e2 - e0).abs().max().item()
        rec = dict(delta_mm=dmm, r_mm=r, solve_s=dt, cg_its=its, signal_max=sig / (1 / g.W))
        for face, u in (("flat", uf), ("corroded", uc)):
            ef = H(q1_strain(torch.tensor(u)[None], g.h))
            diff = (ef - e2).abs().max().item()
            rec[f"{face}_maxdiff_over_signal"] = diff / sig if dmm > 0 else float("nan")
            rec[f"{face}_maxdiff_over_e0"] = diff / (1 / g.W)
            rec[f"{face}_rel_l2_vs_signal"] = ((ef - e2).norm() / (e2 - e0).norm()).item() if dmm > 0 else float("nan")
        rows.append(rec)
        print(rec, flush=True)
    dmg = [r_ for r_ in rows if r_["delta_mm"] > 0]
    worst = max(max(r_["flat_maxdiff_over_signal"], r_["corroded_maxdiff_over_signal"]) for r_ in dmg)
    intact = max(r_["flat_maxdiff_over_e0"] for r_ in rows if r_["delta_mm"] == 0)
    decision = "membrane" if worst < 0.02 else "shell"
    out = dict(cases=rows, worst_maxdiff_over_signal=worst, intact_maxdiff_over_e0=intact,
               decision=decision, nz=args.nz, deg=args.deg,
               note=("3D/2D difference measured after H on the full DIC design. "
                     "If 'shell', the membrane forward model misses the eccentricity bending; "
                     "record which face the camera sees (see README)."))
    config.save_json(out, cfg.results_dir("step1") / f"bending_nz{args.nz}_deg{args.deg}.json", cfg)
    print(f"worst |3D-2D| / damage signal = {worst:.3f} -> decision: {decision}")


if __name__ == "__main__":
    main()
