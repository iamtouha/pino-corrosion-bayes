"""T2 (follow-up studies, README): fix the over-confidence of Step 8a (Zenodo 11257192 plate with a hole).

Step 8a used independent Gaussian noise on u_x / u_y with inferred scales and no surrogate
error: r = 3.35 [3.34, 3.37] mm against the nominal 4 mm. Conditions here (same data,
same PINO checkpoint results/laptop/step8a/pino_hole.pt, same priors and rigid-body terms):

  iid       Step 8a's likelihood, rerun with this code (reference)
  iid_SG    + Sigma_G of the hole surrogate: residuals FE - PINO at the observation points
            on improve.t2.n_D fresh FE solves of the hole family (low rank + leftover
            diagonal, Woodbury), scaled with P / (E_ref t0), as pino.inference.problem
  GP_SG     Kennedy-O'Hagan discrepancy: zero-mean GP over (x, y) per displacement
            component, SE kernel + white noise; hyperparameters by type-II maximum
            likelihood on the residual of the current MAP (with Sigma_G in the covariance),
            then TMCMC with that fixed covariance; repeated improve.t2.gp_iters times
  taper_SG  iid_SG + one boundary-condition/geometry asymmetry parameter g: a linear axial
            strain gradient eps(x) = eps_bar (1 + g (x - L/2) / L) (phenomenological
            kinematic correction for the measured left/right strain difference)

Accept (follow-up study T2): (40, 10) mm and r = 4 mm inside the 90 % intervals, E within
5 % of 185 GPa. Outputs: results/laptop/improve/t2/{D_hole.npz, post_*.npz, summary.{json,md}, *.png}
"""
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from pino import config
from pino.grid import Grid
from pino.inference.discrepancy import fit_gp_hyper, gp_cov, woodbury_ll
from pino.inference.tmcmc import tmcmc
from pino.surrogates.models import FNO, Surrogate, clean_state
from pino.surrogates.train import DEV

from experiments.step8a_zenodo import download, hole_tau

BASE_NAMES = ["xc", "yc", "r", "E", "nu", "tx", "ty", "rot"]


class Problem:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rd = rd = cfg["realdata"]
        self.tc = cfg["improve"]["t2"]
        ddir = cfg.path("data", "zenodo_11257192")
        download(ddir)
        self.out = cfg.results_dir("improve", "t2")
        self.grid = grid = Grid(L=rd["L"], W=rd["W"], nx=rd["nx"], ny=rd["ny"], t0=rd["t0"])
        ckd = torch.load(cfg.results_dir("step8a") / "pino_hole.pt", map_location=DEV, weights_only=False)
        net = FNO(grid, n_modes=rd["fno_modes"], hidden_channels=64, n_layers=4, domain_padding=rd["fno_padding"])
        self.model = Surrogate(net, grid, ckd["w_std"]).to(DEV)
        self.model.load_state_dict(clean_state(ckd["state"])); self.model.eval()
        self.sur_err = float(ckd["test_rel_l2_strain"])
        D = np.genfromtxt(ddir / "20231116_displacements_interpolated.csv", delimiter=",", skip_header=1)
        keep = (D[:, 0] >= rd["edge_skip_mm"]) & (D[:, 0] <= rd["L"] - rd["edge_skip_mm"])
        self.D = D = D[keep]
        self.n = len(D)
        self.xy = torch.tensor(D[:, :2], device=DEV)
        self.d = torch.tensor(D[:, 2:]).T.reshape(-1).to(DEV)              # [ux..., uy...]
        gx = 2 * self.xy[:, 0] / grid.L - 1; gy = 2 * self.xy[:, 1] / grid.W - 1
        self.g = torch.stack([gx, gy], -1).view(1, 1, -1, 2).float()
        self.P = rd["traction"] * rd["W"] * rd["t0"]
        self.c_nom = self.P / (rd["E_ref"] * rd["t0"])                      # unit -> mm at E_ref
        pr = rd["prior"]
        self.lo0 = np.array([pr["xc"][0], pr["yc"][0], pr["r"][0], pr["E"][0], pr["nu"][0], -0.2, -0.2, -0.01])
        self.hi0 = np.array([pr["xc"][1], pr["yc"][1], pr["r"][1], pr["E"][1], pr["nu"][1], 0.2, 0.2, 0.01])

    def interp(self, u):
        """u: (b, ny+1, nx+1, 2) -> (b, 2n) at the observation points, [ux..., uy...]."""
        v = F.grid_sample(u.permute(0, 3, 1, 2).float(), self.g.expand(len(u), -1, -1, -1),
                          mode="bilinear", align_corners=True)[:, :, 0]
        return v.reshape(len(u), -1).double()

    def predict(self, xb, taper=False):
        """xb: (b, >= 8) torch on DEV -> (b, 2n) predicted displacement in mm."""
        tau = hole_tau(xb[:, :3], self.grid).float()
        up = self.interp(self.model.u_hat(tau, xb[:, 4].float())).view(len(xb), 2, -1)
        up = up * (self.P / (xb[:, 3] * self.rd["t0"]))[:, None, None]
        X, Y = self.xy[:, 0], self.xy[:, 1]
        up[:, 0] += xb[:, 5, None] - xb[:, 7, None] * Y
        up[:, 1] += xb[:, 6, None] + xb[:, 7, None] * X
        if taper:
            g = xb[:, 8, None]
            eb = (self.rd["traction"] / xb[:, 3])[:, None]
            L = self.grid.L
            s = (X - L / 2)[None] / L
            up[:, 0] += eb * g * L * s ** 2 / 2
            up[:, 1] += -xb[:, 4, None] * eb * g * s * Y[None]
        return up.reshape(len(xb), -1)


# ----------------------------------------------------------------------------- Sigma_G
def hole_sigma_g(pb: Problem):
    """FE - PINO residuals at the observation points on fresh hole-family draws (unit load)."""
    from pino.fe.plate2d import solve_many
    tc, rd, grid = pb.tc, pb.rd, pb.grid
    f = pb.out / "D_hole.npz"
    if not f.exists():
        pr = rd["prior"]
        rng = np.random.default_rng(tc["D_seed"])
        n = tc["n_D"]
        th = np.stack([rng.uniform(*pr["xc"], n), rng.uniform(*pr["yc"], n), rng.uniform(*pr["r"], n)], 1)
        nu = rng.uniform(*pr["nu"], n)
        tau = hole_tau(torch.tensor(th), grid).numpy()
        t0 = time.time()
        u = solve_many(grid, tau, nu, workers=pb.cfg["dataset"]["workers"], bc=rd.get("bc", "roller"))
        print(f"  D_hole: {n} FE solves in {time.time() - t0:.0f}s", flush=True)
        np.savez(f, theta=th, nu=nu, u=u.astype(np.float32))
    z = np.load(f)
    R = []
    with torch.no_grad():
        for s in range(0, len(z["nu"]), 32):
            th = torch.tensor(z["theta"][s:s + 32], device=DEV)
            nu = torch.tensor(z["nu"][s:s + 32], device=DEV).float()
            ufe = torch.tensor(z["u"][s:s + 32], device=DEV)
            usur = pb.model.u_hat(hole_tau(th, grid).float(), nu)
            R.append((pb.interp(ufe) - pb.interp(usur)).cpu())
    R = torch.cat(R).double() * pb.c_nom                                  # mm at E_ref
    mu = R.mean(0)
    Rc = R - mu
    Uf, S, _ = torch.linalg.svd(Rc.T, full_matrices=False)
    k = min(tc["sigma_g_rank"], len(S))
    lam = S[:k] ** 2 / (len(R) - 1)
    U = Uf[:, :k]
    left = (Rc - (Rc @ U) @ U.T).pow(2).sum(0) / (len(R) - 1)
    rms = Rc.pow(2).mean(0).sqrt().view(2, -1).mean(1)
    info = dict(n_D=len(R), rank=k, var_captured=float(lam.sum() / Rc.pow(2).sum(0).sum() * (len(R) - 1)),
                rms_mm_x=float(rms[0]), rms_mm_y=float(rms[1]),
                mean_abs_mu_mm_x=float(mu.view(2, -1)[0].abs().mean()), mean_abs_mu_mm_y=float(mu.view(2, -1)[1].abs().mean()))
    return dict(mu=mu.to(DEV), U=U.to(DEV), lam=lam.to(DEV), left=left.to(DEV), info=info)


# ----------------------------------------------------------------------------- likelihoods
def n_se_of(cond):
    return int(cond[2]) if cond[2:3].isdigit() else 1


def gp_total_cov(pb, sg, hp, n_se=1):
    """blockdiag(sum of SE + s^2 I per displacement component) + Sigma_G (2n, 2n)."""
    extra = None if sg is None else (sg["U"] * sg["lam"]) @ sg["U"].T + torch.diag(sg["left"])
    return gp_cov(pb.xy, hp, 2, extra, n_se)


def fit_gp(pb, sg, r, hp0=None, n_se=1):
    """Type-II ML of the kernel hyperparameters per displacement component on residual r (2n,)."""
    extra = None if sg is None else (sg["U"] * sg["lam"]) @ sg["U"].T + torch.diag(sg["left"])
    hp, nll = fit_gp_hyper(pb.xy, r.to(DEV), 2, extra, hp0, ell0=pb.tc.get("gp_ell0", 5.0), n_se=n_se,
                           ell_long=pb.tc.get("gp_ell_long"),
                           ells=pb.tc.get("gp_ells_3") if n_se == 3 else None)
    h = torch.exp(hp).view(2, -1).cpu().numpy()
    out = {}
    for c, comp in enumerate("xy"):
        for k in range(n_se):
            out[f"a{k + 1}_{comp}"], out[f"ell{k + 1}_{comp}"] = h[c, 2 * k], h[c, 2 * k + 1]
        out[f"s_{comp}"] = h[c, -1]
    out["nll"] = nll
    return hp, out


def make_condition(pb, cond, sg, hp=None):
    """Returns (names, lo, hi, log_lik numpy fn)."""
    names, lo, hi = list(BASE_NAMES), pb.lo0.copy(), pb.hi0.copy()
    taper = "taper" in cond
    if taper:
        names.append("g"); lo = np.r_[lo, -0.5]; hi = np.r_[hi, 0.5]
    iid = cond in ("iid", "iid_SG", "taper_SG")
    if iid:
        names += ["log_sigma_x", "log_sigma_y"]
        lo = np.r_[lo, np.log(1e-5), np.log(1e-5)]; hi = np.r_[hi, np.log(1e-1), np.log(1e-1)]
    n = pb.n
    use_sg = cond.endswith("SG")
    mu = sg["mu"] if use_sg else torch.zeros(2 * n, device=DEV, dtype=torch.float64)
    Lc = None
    if cond.startswith("GP"):
        Lc = torch.linalg.cholesky(gp_total_cov(pb, sg if use_sg else None, hp, n_se_of(cond)))
        logdet = 2 * torch.log(torch.diagonal(Lc)).sum()

    def ll(x, bs=64):
        out = []
        with torch.no_grad():
            for s in range(0, len(x), bs):
                xb = torch.tensor(x[s:s + bs], device=DEV)
                r = pb.d[None] - pb.predict(xb, taper) - mu[None]
                if Lc is not None:
                    z = torch.linalg.solve_triangular(Lc, r.T, upper=False)
                    out.append((-0.5 * (z * z).sum(0) - 0.5 * logdet).cpu())
                    continue
                sx, sy = torch.exp(xb[:, -2]), torch.exp(xb[:, -1])
                Dv = torch.cat([sx[:, None].expand(-1, n) ** 2, sy[:, None].expand(-1, n) ** 2], 1)
                if use_sg:
                    out.append(woodbury_ll(r, Dv + sg["left"][None], sg["U"], sg["lam"]).cpu())
                else:
                    out.append((-0.5 * (r * r / Dv).sum(1) - 0.5 * torch.log(Dv).sum(1)).cpu())
        return torch.cat(out).numpy()

    return names, lo, hi, ll


def run_condition(pb, cond, sg, tag, hp=None, seed=1):
    f = pb.out / f"post_{tag}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return {k: z[k] for k in z.files}
    names, lo, hi, ll = make_condition(pb, cond, sg, hp)
    sample = lambda n, rng: lo + (hi - lo) * rng.random((n, len(lo)))
    log_prior = lambda x: np.where(np.all((x >= lo) & (x <= hi), 1), 0.0, -np.inf)
    t0 = time.time()
    S, logZ, info = tmcmc(ll, sample, log_prior, n=pb.tc["tmcmc_n"], n_mh=pb.tc["tmcmc_n_mh"], seed=seed)
    llS = ll(S)
    res = dict(samples=S, names=np.array(names), ll=llS, logZ=logZ, wall=time.time() - t0,
               stages=info["stages"], n_eval=info["n_eval"], unique_frac=info["unique_frac"],
               hp=(hp.cpu().numpy() if hp is not None else np.zeros(0)))
    np.savez(f, **res)
    print(f"  [{tag}] {res['wall']:.0f}s stages={info['stages']} logZ={logZ:.1f}", flush=True)
    return res


def summarize(pb, res):
    S, names = res["samples"], [str(n) for n in res["names"]]
    q = np.quantile(S, [0.05, 0.5, 0.95], 0)
    post = {k: dict(mean=float(S[:, i].mean()), q05=float(q[0, i]), q50=float(q[1, i]), q95=float(q[2, i]),
                    width90=float(q[2, i] - q[0, i])) for i, k in enumerate(names)}
    xc, yc, R = pb.rd["hole"]
    inside = lambda k, v: bool(post[k]["q05"] <= v <= post[k]["q95"])
    eE = post["E"]["mean"] / pb.rd["E_ref"] - 1
    acc = dict(xc_in90=inside("xc", xc), yc_in90=inside("yc", yc), r_in90=inside("r", R),
               E_within_5pct=bool(abs(eE) < 0.05))
    return dict(posterior=post, E_rel_error=float(eE), accept=acc, accepted=all(acc.values()),
                logZ=float(res["logZ"]), wall_s=float(res["wall"]), stages=int(res["stages"]),
                centre_error_mm=float(np.hypot(post["xc"]["mean"] - xc, post["yc"]["mean"] - yc)))


def residual_stats(pb, r):
    """Left/right asymmetry of the MAP residual: mean d(u_x)/dx residual per x-third."""
    n = pb.n
    X = pb.D[:, 0]
    rx = r[:n]
    out = {"rms_mm_x": float(np.sqrt((r[:n] ** 2).mean())), "rms_mm_y": float(np.sqrt((r[n:] ** 2).mean()))}
    for name, m in (("left", X < pb.grid.L / 3), ("mid", (X >= pb.grid.L / 3) & (X <= 2 * pb.grid.L / 3)),
                    ("right", X > 2 * pb.grid.L / 3)):
        A = np.c_[np.ones(m.sum()), X[m]]
        out[f"resid_strain_xx_{name}_ue"] = float(np.linalg.lstsq(A, rx[m], rcond=None)[0][1] * 1e6)
    return out


def map_residual(pb, res, taper=False, sg=None):
    x = res["samples"][int(np.argmax(res["ll"]))]
    with torch.no_grad():
        p = pb.predict(torch.tensor(x[None], device=DEV), taper)[0]
    mu = sg["mu"] if sg is not None else 0.0
    return (pb.d - p - mu).cpu(), x


def plot(pb, results, resids):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ["xc", "yc", "r", "E", "nu"]
    truth = dict(xc=pb.rd["hole"][0], yc=pb.rd["hole"][1], r=pb.rd["hole"][2], E=pb.rd["E_ref"], nu=pb.rd["nu_ref"])
    fig, ax = plt.subplots(1, len(keys), figsize=(16, 3.4))
    tags = list(results)
    for j, k in enumerate(keys):
        for i, t in enumerate(tags):
            p = results[t]["posterior"][k]
            ax[j].plot([p["q05"], p["q95"]], [i, i], lw=3, color=f"C{i}")
            ax[j].plot(p["q50"], i, "ko", ms=3)
        ax[j].axvline(truth[k], color="k", ls="--", lw=0.8)
        ax[j].set_yticks(range(len(tags))); ax[j].set_yticklabels(tags if j == 0 else [""] * len(tags), fontsize=7)
        ax[j].set_title(k + (" (ref., not truth)" if k in ("E", "nu") else ""), fontsize=9)
    fig.tight_layout(); fig.savefig(pb.out / "intervals.png", dpi=130)
    n = pb.n
    fig, axs = plt.subplots(len(resids), 2, figsize=(11, 2.0 * len(resids)), squeeze=False)
    for i, (t, r) in enumerate(resids.items()):
        for c in range(2):
            sc = axs[i, c].scatter(pb.D[:, 0], pb.D[:, 1], c=r[c * n:(c + 1) * n] * 1e3, s=5, cmap="RdBu_r")
            plt.colorbar(sc, ax=axs[i, c], label="µm")
            axs[i, c].set_title(f"{t}: MAP residual u_{'xy'[c]}", fontsize=8); axs[i, c].set_aspect("equal")
    fig.tight_layout(); fig.savefig(pb.out / "map_residuals.png", dpi=110)


def main():
    cfg, args = config.cli(__doc__)
    pb = Problem(cfg)
    tc = pb.tc
    sg = hole_sigma_g(pb)
    print("Sigma_G:", sg["info"], flush=True)
    results, resids, extra = {}, {}, {}
    conds = tc["conditions"]
    if "iid" in conds:
        r = run_condition(pb, "iid", None, "iid")
        results["iid"] = summarize(pb, r)
        resids["iid"], _ = map_residual(pb, r)
    r_sg = None
    if "iid_SG" in conds or "GP_SG" in conds:
        r_sg = run_condition(pb, "iid_SG", sg, "iid_SG")
        results["iid_SG"] = summarize(pb, r_sg)
        resids["iid_SG"], _ = map_residual(pb, r_sg, sg=sg)
    r_tp = None
    if any("taper" in c for c in conds):
        r_tp = run_condition(pb, "taper_SG", sg, "taper_SG")
        results["taper_SG"] = summarize(pb, r_tp)
        resids["taper_SG"], _ = map_residual(pb, r_tp, taper=True, sg=sg)
    extra["gp_hyper"] = []
    for cond in [c for c in conds if c.startswith("GP")]:
        taper = "taper" in cond
        res_prev, hp = (r_tp if taper else r_sg), None
        for it in range(tc["gp_iters"]):
            tag = f"{cond}_it{it}"
            f = pb.out / f"post_{tag}.npz"
            if f.exists():                      # resume with the hyperparameters used then
                hp = torch.tensor(np.load(f)["hp"], device=DEV)
                h = {"stored": True}
            else:
                rr, _ = map_residual(pb, res_prev, taper=taper, sg=sg)
                hp, h = fit_gp(pb, sg, rr.double(), hp, n_se_of(cond))
            h.update(condition=cond, iter=it, map_from=("taper_SG" if taper else "iid_SG") if it == 0 else f"{cond}_it{it - 1}")
            extra["gp_hyper"].append({k: (float(v) if not isinstance(v, (str, bool)) else v) for k, v in h.items()}
                                     | {"hp_log": torch.as_tensor(hp).cpu().tolist()})
            print(f"  GP hyperparameters ({tag}):", extra["gp_hyper"][-1], flush=True)
            res_prev = run_condition(pb, cond, sg, tag, hp)
            results[tag] = summarize(pb, res_prev)
        resids[cond], _ = map_residual(pb, res_prev, taper=taper, sg=sg)
    rstats = {t: residual_stats(pb, r.numpy()) for t, r in resids.items()}
    # data-only strain asymmetry (fit of u_x along x per third)
    raw = residual_stats(pb, pb.d.cpu().numpy())
    out = dict(sigma_g=sg["info"], surrogate_test_rel_l2_strain=pb.sur_err, n_obs_points=pb.n,
               data_strain_xx_ue={k: v for k, v in raw.items() if k.startswith("resid_strain")},
               conditions=results, map_residuals=rstats, **extra)
    config.save_json(out, pb.out / "summary.json", pb.cfg)
    keys = ["xc", "yc", "r", "E", "nu"]
    md = ["| condition | " + " | ".join(f"{k} mean [90 %]" for k in keys) + " | E err | accept | logZ |",
          "|" + "---|" * (len(keys) + 4)]
    for t, s in results.items():
        p = s["posterior"]
        fmt = lambda k: (f"{p[k]['mean'] / 1e3:.1f} [{p[k]['q05'] / 1e3:.1f}, {p[k]['q95'] / 1e3:.1f}]" if k == "E"
                         else f"{p[k]['mean']:.3f} [{p[k]['q05']:.3f}, {p[k]['q95']:.3f}]")
        md.append(f"| {t} | " + " | ".join(fmt(k) for k in keys)
                  + f" | {100 * s['E_rel_error']:+.1f} % | {'yes' if s['accepted'] else 'no ' + str([k for k, v in s['accept'].items() if not v])} | {s['logZ']:.0f} |")
    md += ["", "90 % interval widths:", "", "| condition | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
    for t, s in results.items():
        md.append(f"| {t} | " + " | ".join(f"{s['posterior'][k]['width90']:.4g}" for k in keys) + " |")
    (pb.out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(json.dumps({"sigma_g": sg["info"], "map_residuals": rstats, "data": out["data_strain_xx_ue"]}, indent=1, default=float))
    plot(pb, results, {t: r.numpy() for t, r in resids.items()})


if __name__ == "__main__":
    main()
