"""Step 1 gate: verify the 2D plate solver.

1. Uniform plate, P = 50 kN, E = 206 GPa: e11 = P/(E t0 W) = 303 ue, e22 = -nu e11 (< 0.1 %)
2. Scaling u(P, E) = (P/E) u(1, 1) (round-off)
3. Mesh convergence h = 4, 2, 1 mm (+ Q2 at 2 mm) for a deep pit: peak strain change 2 -> 1 mm < 1 %
4. Strain-gauge sanity: intact-plate strain at gauge-like points vs load (the paper's Fig. 7
   values are not reproduced here; compare by eye)
"""
import time

import numpy as np
import torch

from pino import config
from pino.fe.plate2d import PlateSolver2D
from pino.grid import Grid, q1_strain
from pino.thickness import pit_tau


def main():
    cfg, _ = config.cli(__doc__)
    g = Grid.from_cfg(cfg)
    m = cfg["material"]
    E, nu, t0, W = m["E"], m["nu"], g.t0, g.W
    out = {}

    # 1. uniform plate in physical units
    P = 50e3
    s = PlateSolver2D(g)
    t = time.time(); uh = s.solve(np.ones(g.shape), nu); dt = time.time() - t
    eps = q1_strain(P / (E * t0) * uh, g.h)
    e_ref = P / (E * t0 * W)
    err11 = np.abs(eps[..., 0] / e_ref - 1).max()
    err22 = np.abs(eps[..., 1] / (-nu * e_ref) - 1).max()
    err12 = np.abs(eps[..., 2]).max() / e_ref
    out["uniform"] = dict(e11_expected_ue=e_ref * 1e6, e11_max_rel_err=err11, e22_max_rel_err=err22,
                          e12_max_over_e11=err12, solve_seconds=dt,
                          passed=bool(max(err11, err22, err12) < 1e-3))

    # 2. scaling, solving with physical E and P (thickness ratio tau = 1)
    tau = pit_tau(torch.tensor([[150.0, 125.0, 20.0, 0.1]], dtype=torch.float64), g)[0].numpy()
    u11 = s.solve(tau, nu)
    sp = PlateSolver2D(g, E=E, P=P)
    uPE = sp.solve(tau, nu)
    sc = np.abs(uPE - (P / E) * u11).max() / np.abs((P / E) * u11).max()
    out["scaling"] = dict(max_rel_diff=sc, passed=bool(sc < 1e-10))

    # 3. mesh convergence, deep pit (delta = 0.15, r = 10 mm)
    th = torch.tensor([[150.0, 125.0, 10.0, 0.15]], dtype=torch.float64)
    conv = {}
    for label, h, deg in [("h4", 4.0, 1), ("h2", 2.0, 1), ("h1", 1.0, 1), ("h2_Q2", 2.0, 2)]:
        nx, ny = int(round(g.L / h)), int(round(g.W / h))
        gg = Grid(L=g.L, W=g.W, nx=nx, ny=ny, t0=g.t0)
        sv = PlateSolver2D(gg, nx=nx, ny=ny, deg=deg)
        u = sv.solve(pit_tau(th, gg)[0].numpy(), nu)
        e = q1_strain(u, gg.hxy)[..., 0] / (1 / W)      # e11 / e11_intact
        conv[label] = dict(peak_e11_ratio=float(e.max()), n_unknowns=int(2 * (nx + 1) * (ny + 1)))
    ch = abs(conv["h1"]["peak_e11_ratio"] / conv["h2"]["peak_e11_ratio"] - 1)
    out["convergence"] = dict(**conv, change_2_to_1mm=ch, passed=bool(ch < 0.01))

    # 4. strain-gauge sanity: intact plate, gauge-like points along the centreline
    loads = np.arange(5, 101, 5) * 1e3
    pts = [(75.0, 125.0), (150.0, 125.0), (225.0, 125.0)]
    e_int = q1_strain(uh, g.h)
    gauges = {}
    for (x, y) in pts:
        i, j = int(x / g.h), int(y / g.h)
        gauges[f"x{x:.0f}_y{y:.0f}"] = [(float(P_ / 1e3), float(P_ / (E * t0) * e_int[j, i, 0] * 1e6)) for P_ in loads]
    out["gauges_kN_ue"] = gauges
    out["note_gauges"] = "Compare by eye with the paper's Fig. 7 elastic range (bolts not modelled)."
    out["passed"] = all(out[k]["passed"] for k in ("uniform", "scaling", "convergence"))
    config.save_json(out, cfg.results_dir("step1") / "verify.json", cfg)
    for k in ("uniform", "scaling", "convergence"):
        print(k, out[k])
    print("STEP 1 GATE:", "PASS" if out["passed"] else "FAIL")


if __name__ == "__main__":
    main()
