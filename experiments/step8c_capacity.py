"""Step 8c: push posterior samples (from the Step 7 benchmark) through capacity analyses.

First pass: net-section capacity P_u = sigma_u * min_x integral t dy (capacity.n_samples
samples per case). Full analysis: CalculiX elasto-plastic shells (bilinear: yield 245 MPa,
hardening 1/100 of E) for capacity.n_ccx samples of one case; capacity = load at the 0.2 %
offset of the force-elongation curve. Compared with the same analyses at the true damage.
"""
import glob

import numpy as np
import torch

from pino import config
from pino.capacity.capacity import calculix_curve, net_section, offset_capacity
from pino.grid import Grid
from pino.inference.problem import Family


def main():
    cfg, args = config.cli(__doc__, lambda ap: ap.add_argument("--condition", default=None))
    grid = Grid.from_cfg(cfg)
    fam = Family("pit", cfg, grid)
    m = cfg["material"]
    cap = cfg["capacity"]
    pdir = cfg.results_dir("step7", "pit", "posteriors")
    files = sorted(glob.glob(str(pdir / "*_SG_grid56_*.npz")))
    if args.condition:
        files = [f for f in files if f"/{args.condition}_grid56_" in f]
    if not files:
        print("no Step 7 posteriors yet"); return
    rng = np.random.default_rng(0)
    rows = []
    for f in files:
        z = np.load(f)
        if z["x_true"][3] == 0:
            continue
        S = z["samples"][rng.choice(len(z["samples"]), min(cap["n_samples"], len(z["samples"])), replace=False)]
        tau = fam.tau(torch.as_tensor(S)).numpy()
        Pu = net_section(tau, grid, m["ultimate"]) / 1e3
        Pt = net_section(fam.tau(torch.as_tensor(z["x_true"][None])).numpy(), grid, m["ultimate"])[0] / 1e3
        q = np.quantile(Pu, [0.05, 0.5, 0.95])
        rows.append(dict(file=f.split("/")[-1], true_kN=Pt, median_kN=q[1], q05=q[0], q95=q[2],
                         covered=bool(q[0] <= Pt <= q[2])))
    out = dict(net_section=rows, coverage90=float(np.mean([r["covered"] for r in rows])) if rows else None)
    # CalculiX on the deepest case
    if rows and cap.get("n_ccx", 0) > 0:
        f = max(files, key=lambda f_: np.load(f_)["x_true"][3])
        z = np.load(f)
        disp = 0.005 * grid.L
        res = []
        for k, x in enumerate([z["x_true"]] + list(z["samples"][:cap["n_ccx"]])):
            tau = fam.tau(torch.as_tensor(x[None])).numpy()[0]
            e, F_ = calculix_curve(grid, tau, x[4], x[5], m["yield"], m["hardening_ratio"], disp,
                                   workdir=cfg.results_dir("step8c", "ccx", f"run{k}"))
            res.append(offset_capacity(e, F_, grid.L) / 1e3)
            print(f"  ccx {'truth' if k == 0 else f'sample {k}'}: {res[-1]:.1f} kN", flush=True)
        out["calculix"] = dict(case=f.split("/")[-1], truth_kN=res[0], samples_kN=res[1:],
                               q05_q50_q95=np.quantile(res[1:], [0.05, 0.5, 0.95]).tolist())
    config.save_json(out, cfg.results_dir("step8c") / "capacity.json", cfg)
    print({k: v for k, v in out.items() if k != "net_section"}, "net-section coverage90:", out["coverage90"])


if __name__ == "__main__":
    main()
