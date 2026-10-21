"""Step 8a: validation on the public plate-with-hole DIC dataset (Zenodo 11257192).

S235 plate, parallel region 80 x 20 mm, thickness 3.05 mm, hole R4 at its centre; traction
106.26 N/mm^2 at the end of the parallel region (Anton et al. 2025, arXiv:2405.18311, Sec. 6.1).
Authors' NLS-FEM reference: K = 128 085, G = 73 541 N/mm^2 -> E = 185.2 GPa, nu = 0.259.

1. The hole is a near-zero thickness region (tau = 1e-3 inside a circle, smooth edge) on the
   grid operator; checked against an FE mesh with a real hole (strain outside within 1 %).
2. A dataset for this geometry is generated and a PINO trained on it.
3. Displacements are observed directly, with three rigid-body parameters and a noise scale;
   (xc, yc, r, E, nu) are inferred with TMCMC. Gate: hole centre within its radius, E within 5 %.

The left edge is clamped (realdata.bc), as in the test and the authors' model; three
rigid-body parameters absorb the remaining DIC rigid motion, and u_x / u_y get separate
noise scales. Note: the measured axial strain differs by ~15 % between the two ends
(left ~520 ue, right ~430-460 ue), which no uniform-traction elastic model reproduces, so
the identified E depends on how u_x and u_y are weighted (a direct FE least-squares fit
gives 224 GPa with equal weights, 199 GPa with the authors' weights).
"""
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pino import config
from pino.fe.plate2d import PlateSolver2D, solve_many
from pino.grid import Grid, q1_strain
from pino.inference.tmcmc import tmcmc
from pino.surrogates.models import FNO, Surrogate, clean_state
from pino.surrogates.residual import Residual
from pino.surrogates.train import DEV, rel_l2

FILES = ["README.txt", "20231116_displacements_interpolated.csv", "20231116_displacements_raw.csv",
         "20231116_mcmc_samples_fem.csv", "20231116_geometry.png"]


def download(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if not (d / f).exists():
            urllib.request.urlretrieve(f"https://zenodo.org/api/records/11257192/files/{f}/content", d / f)


def hole_tau(theta, grid: Grid, edge=None):
    """theta (b, 3) = (xc, yc, r): tau = 1e-3 inside, 1 outside, smooth over ~h/2."""
    edge = edge or 0.25 * grid.h
    Y, X = grid.mesh()
    X = torch.as_tensor(X, dtype=theta.dtype, device=theta.device)
    Y = torch.as_tensor(Y, dtype=theta.dtype, device=theta.device)
    dist = torch.sqrt((X - theta[:, 0, None, None]) ** 2 + (Y - theta[:, 1, None, None]) ** 2)
    return 1e-3 + (1 - 1e-3) * torch.sigmoid((dist - theta[:, 2, None, None]) / edge)


def check_real_hole(rd, grid: Grid, nu=0.26):
    """FE with a meshed hole (gmsh, P2 triangles) vs the tau = 1e-3 representation."""
    import gmsh
    import ufl
    from dolfinx import fem, geometry, mesh as dmesh
    from dolfinx.fem.petsc import LinearProblem
    from dolfinx.io import gmshio
    from mpi4py import MPI
    xc, yc, R = rd["hole"]
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    occ = gmsh.model.occ
    rect = occ.addRectangle(0, 0, 0, grid.L, grid.W)
    disk = occ.addDisk(xc, yc, 0, R, R)
    occ.cut([(2, rect)], [(2, disk)]); occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [s[1] for s in gmsh.model.getEntities(2)], 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.4)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.1)
    gmsh.model.mesh.generate(2)
    msh, _, _ = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0, gdim=2)
    gmsh.finalize()
    V = fem.functionspace(msh, ("Lagrange", 2, (2,)))
    lam, mu = nu / (1 - nu ** 2), 1 / (2 * (1 + nu))
    eps = lambda w: ufl.sym(ufl.grad(w))
    sig = lambda w: lam * ufl.tr(eps(w)) * ufl.Identity(2) + 2 * mu * eps(w)
    V0, _ = V.sub(0).collapse(); V1, _ = V.sub(1).collapse()
    left = dmesh.locate_entities_boundary(msh, 1, lambda x: np.isclose(x[0], 0.0))
    right = dmesh.locate_entities_boundary(msh, 1, lambda x: np.isclose(x[0], grid.L))
    bcs = [fem.dirichletbc(fem.Function(V0), fem.locate_dofs_topological((V.sub(0), V0), 1, left), V.sub(0)),
           fem.dirichletbc(fem.Function(V1), fem.locate_dofs_geometrical(
               (V.sub(1), V1), lambda x: np.isclose(x[0], 0) & np.isclose(x[1], 0)), V.sub(1))]
    tags = dmesh.meshtags(msh, 1, right, np.ones(len(right), dtype=np.int32))
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    q = fem.Constant(msh, np.array([1.0 / grid.W, 0.0]))
    uh = LinearProblem(ufl.inner(sig(u), eps(v)) * ufl.dx, ufl.dot(q, v) * ds(1), bcs=bcs,
                       petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                      "pc_factor_mat_solver_type": "mumps"}).solve()
    # evaluate at grid nodes outside the hole
    Y, X = grid.mesh()
    pts = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
    outside = np.hypot(pts[:, 0] - xc, pts[:, 1] - yc) > R + 1e-9
    tree = geometry.bb_tree(msh, 2)
    cand = geometry.compute_collisions_points(tree, pts[outside])
    cells = geometry.compute_colliding_cells(msh, cand, pts[outside])
    cell = np.array([cells.links(i)[0] if len(cells.links(i)) else -1 for i in range(outside.sum())])
    ok = cell >= 0
    vals = np.full((outside.sum(), 2), np.nan)
    vals[ok] = uh.eval(pts[outside][ok], cell[ok])
    U_real = np.full((X.size, 2), np.nan); U_real[np.flatnonzero(outside)] = vals
    U_real = U_real.reshape(*grid.shape, 2)
    s = PlateSolver2D(grid)
    U_tau = s.solve(hole_tau(torch.tensor([rd["hole"]], dtype=torch.float64), grid)[0].numpy(), nu)
    e_r, e_t = q1_strain(U_real, grid.h), q1_strain(U_tau, grid.h)
    # elements whose centre is >= margin outside the hole
    cx, cy = np.meshgrid(grid.xc, grid.yc)
    m = (np.hypot(cx - xc, cy - yc) > R + rd["check_margin_mm"]) & np.all(np.isfinite(e_r), -1)
    rel = np.linalg.norm(e_r[m] - e_t[m]) / np.linalg.norm(e_r[m])
    peak = np.abs(e_r[m][:, 0] / e_t[m][:, 0] - 1).max()
    return {"rel_l2_strain_outside": float(rel), "max_rel_e11_outside": float(peak),
            "margin_mm": rd["check_margin_mm"], "passed": bool(rel < 0.01)}


def train_pino(rd, grid, tau, w, nu, steps, lam_r=0.1, seed=0):
    torch.manual_seed(seed)
    w_std = w.reshape(-1, 2).std(0)
    net = FNO(grid, n_modes=rd["fno_modes"], hidden_channels=64, n_layers=4,
              domain_padding=rd["fno_padding"])
    model = Surrogate(net, grid, w_std).to(DEV)
    resid = Residual(grid, torch.float64, bc=rd.get("bc", "roller")).to(DEV)
    n_tr = int(0.9 * len(nu))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, 1e-5)
    pr = rd["prior"]
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for it in range(1, steps + 1):
        i = torch.randint(0, n_tr, (16,), generator=gen)
        t, wt, n = tau[i].to(DEV), w[i].to(DEV), nu[i].to(DEV)
        wp = model(t, n)
        loss = rel_l2(wp, wt) + rel_l2(q1_strain(wp, grid.h), q1_strain(wt, grid.h))
        if it > steps // 10:
            th = torch.stack([torch.empty(16).uniform_(*pr["xc"], generator=gen),
                              torch.empty(16).uniform_(*pr["yc"], generator=gen),
                              torch.empty(16).uniform_(*pr["r"], generator=gen)], 1).double()
            tu = hole_tau(th, grid).to(DEV)
            nuu = torch.empty(16).uniform_(*pr["nu"], generator=gen).to(DEV)
            for tt, nn_, ww in ((t, n, wp), (tu.float(), nuu, model(tu.float(), nuu))):
                u = grid.u0_hat_torch(nn_.double()) + ww.double()
                loss = loss + lam_r * resid(u, tt.double(), nn_.double()).mean().float()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if it % 1000 == 0:
            print(f"    step {it}: loss {loss.item():.4f} ({(time.time() - t0) / it * 1e3:.0f} ms/step)", flush=True)
    model.eval()
    with torch.no_grad():
        te = slice(n_tr, None)
        num = den = 0.0
        for s in range(n_tr, len(nu), 16):
            sl = slice(s, s + 16)
            wp = model(tau[sl].to(DEV), nu[sl].to(DEV)).cpu()
            a, b = q1_strain(wp, grid.h), q1_strain(w[sl], grid.h)
            num += (a - b).pow(2).sum().item(); den += b.pow(2).sum().item()
    return model, float(np.sqrt(num / den))


def main():
    cfg, args = config.cli(__doc__)
    rd = cfg["realdata"]
    ddir = cfg.path("data", "zenodo_11257192")
    download(ddir)
    out = cfg.results_dir("step8a")
    grid = Grid(L=rd["L"], W=rd["W"], nx=rd["nx"], ny=rd["ny"], t0=rd["t0"])
    res = {}

    # 1. hole representation check
    chk = out / "hole_check.json"
    if not chk.exists():
        config.save_json(check_real_hole(rd, grid), chk, cfg)
    res["hole_check"] = json.loads(chk.read_text())
    print("hole check:", res["hole_check"], flush=True)

    # 2. dataset + PINO for this geometry
    ck = out / "pino_hole.pt"
    pr = rd["prior"]
    if not ck.exists():
        rng = np.random.default_rng(0)
        n = rd["n_train"]
        th = np.stack([rng.uniform(*pr["xc"], n), rng.uniform(*pr["yc"], n), rng.uniform(*pr["r"], n)], 1)
        nu = rng.uniform(*pr["nu"], n)
        tau = hole_tau(torch.tensor(th), grid).numpy()
        t0 = time.time()
        u = solve_many(grid, tau, nu, workers=cfg["dataset"]["workers"], bc=rd.get("bc", "roller"))
        print(f"  {n} FE solves in {time.time() - t0:.0f}s", flush=True)
        w = np.stack([u[i] - grid.u0_hat(nu[i]) for i in range(n)])
        model, err = train_pino(rd, grid, torch.tensor(tau).float(), torch.tensor(w).float(),
                                torch.tensor(nu).float(), rd["steps"])
        torch.save({"state": model.state_dict(), "w_std": model.w_std, "test_rel_l2_strain": err}, ck)
    ckd = torch.load(ck, map_location=DEV, weights_only=False)
    net = FNO(grid, n_modes=rd["fno_modes"], hidden_channels=64, n_layers=4, domain_padding=rd["fno_padding"])
    model = Surrogate(net, grid, ckd["w_std"]).to(DEV)
    model.load_state_dict(clean_state(ckd["state"])); model.eval()
    res["pino_test_rel_l2_strain"] = ckd["test_rel_l2_strain"]
    print("PINO held-out strain error:", ckd["test_rel_l2_strain"], flush=True)

    # 3. inference on the measured displacements
    D = np.genfromtxt(ddir / "20231116_displacements_interpolated.csv", delimiter=",", skip_header=1)
    keep = (D[:, 0] >= rd["edge_skip_mm"]) & (D[:, 0] <= rd["L"] - rd["edge_skip_mm"])
    D = D[keep]
    xy = torch.tensor(D[:, :2]); d_obs = torch.tensor(D[:, 2:]).T.reshape(-1).to(DEV)   # [ux..., uy...]
    gx = 2 * xy[:, 0] / grid.L - 1; gy = 2 * xy[:, 1] / grid.W - 1
    g = torch.stack([gx, gy], -1).view(1, 1, -1, 2).float().to(DEV)
    P = rd["traction"] * rd["W"] * rd["t0"]                      # N on the modelled end
    # separate noise scales for u_x and u_y: the data decide how much each component is trusted
    names = ["xc", "yc", "r", "E", "nu", "tx", "ty", "rot", "log_sigma_x", "log_sigma_y"]
    lo = np.array([pr["xc"][0], pr["yc"][0], pr["r"][0], pr["E"][0], pr["nu"][0], -0.2, -0.2, -0.01,
                   np.log(1e-5), np.log(1e-5)])
    hi = np.array([pr["xc"][1], pr["yc"][1], pr["r"][1], pr["E"][1], pr["nu"][1], 0.2, 0.2, 0.01,
                   np.log(1e-1), np.log(1e-1)])
    n_pt = len(D)

    def log_lik(x):
        out_ = []
        with torch.no_grad():
            for s in range(0, len(x), 16):
                xb = torch.tensor(x[s:s + 16], device=DEV)
                tau = hole_tau(xb[:, :3], grid).float()
                u = model.u_hat(tau, xb[:, 4].float())                     # (b, ny+1, nx+1, 2)
                up = F.grid_sample(u.permute(0, 3, 1, 2), g.expand(len(xb), -1, -1, -1),
                                   mode="bilinear", align_corners=True)[:, :, 0].double()   # (b, 2, n)
                up = up * (P / (xb[:, 3] * rd["t0"]))[:, None, None]
                X, Y = xy[:, 0].to(DEV), xy[:, 1].to(DEV)
                up[:, 0] += xb[:, 5, None] - xb[:, 7, None] * Y
                up[:, 1] += xb[:, 6, None] + xb[:, 7, None] * X
                r = (d_obs[None] - up.reshape(len(xb), -1)).view(len(xb), 2, -1)
                sx, sy = torch.exp(xb[:, 8]), torch.exp(xb[:, 9])
                ll = (-0.5 * r[:, 0].pow(2).sum(1) / sx ** 2 - n_pt * torch.log(sx)
                      - 0.5 * r[:, 1].pow(2).sum(1) / sy ** 2 - n_pt * torch.log(sy))
                out_.append(ll.cpu())
        return torch.cat(out_).numpy()

    sample = lambda n, rng: lo + (hi - lo) * rng.random((n, len(lo)))
    log_prior = lambda x: np.where(np.all((x >= lo) & (x <= hi), 1), 0.0, -np.inf)
    t0 = time.time()
    S, logZ, info = tmcmc(log_lik, sample, log_prior, n=rd["tmcmc_n"], n_mh=3, seed=1)
    wall = time.time() - t0
    q = np.quantile(S, [0.05, 0.5, 0.95], 0)
    post = {k: {"mean": float(S[:, i].mean()), "q05": float(q[0, i]), "q50": float(q[1, i]), "q95": float(q[2, i])}
            for i, k in enumerate(names)}
    xc, yc, R = rd["hole"]
    dc = float(np.hypot(post["xc"]["mean"] - xc, post["yc"]["mean"] - yc))
    E_ref = rd["E_ref"]
    eE = post["E"]["mean"] / E_ref - 1
    res.update(posterior=post, wall_s=wall, tmcmc=info, n_obs_points=int(len(D)),
               hole_centre_error_mm=dc, E_rel_error=float(eE), E_ref=E_ref, nu_ref=rd["nu_ref"],
               gate={"centre_within_radius": bool(dc < post["r"]["mean"]),
                     "E_within_5pct": bool(abs(eE) < 0.05)})
    res["passed"] = all(res["gate"].values())
    config.save_json(res, out / "result.json", cfg)
    print(json.dumps({k: res[k] for k in ("hole_centre_error_mm", "E_rel_error", "gate")}, indent=1))
    for k in names:
        print(f"  {k}: {post[k]['mean']:.5g} [{post[k]['q05']:.5g}, {post[k]['q95']:.5g}]")
    plot(cfg, rd, D, S, names, out)


def plot(cfg, rd, D, S, names, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    sc = ax[0].scatter(D[:, 0], D[:, 1], c=D[:, 2], s=6, cmap="viridis")
    plt.colorbar(sc, ax=ax[0], label="u_x (mm)")
    from matplotlib.patches import Circle
    for i in np.random.default_rng(0).choice(len(S), 40):
        ax[0].add_patch(Circle((S[i, 0], S[i, 1]), S[i, 2], fill=False, color="r", alpha=0.15))
    ax[0].add_patch(Circle(rd["hole"][:2], rd["hole"][2], fill=False, color="k", ls="--"))
    ax[0].set_aspect("equal"); ax[0].set_title("data + posterior hole samples (red), true hole (dashed)")
    ax[1].hist(S[:, 3] / 1e3, 40); ax[1].axvline(rd["E_ref"] / 1e3, color="k", ls="--")
    ax[1].set_xlabel("E (GPa)"); ax[1].set_title("E posterior vs authors' NLS-FEM")
    ax[2].hist(S[:, 4], 40); ax[2].axvline(rd["nu_ref"], color="k", ls="--"); ax[2].set_xlabel("nu")
    fig.tight_layout(); fig.savefig(out / "posterior.png", dpi=130)


if __name__ == "__main__":
    main()
