"""Training of DeepONet / FNO / PINO (Step 4.4) and the evaluation metric (Step 4.1).

Loss = relL2(w) + alpha relL2(eps(w)) [labelled]  + lam_r * residual [labelled + unlabelled].
relL2 is pooled over the batch (||a-b|| / ||b|| over all samples), because single intact or
near-intact samples have w ~ 0 and would make a per-sample ratio meaningless.
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from .. import data
from .. import thickness as th
from ..grid import Grid, q1_strain
from ..vdic.observe import make_obs
from .models import Surrogate, build, clean_state
from .residual import Residual

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def rel_l2(a, b):
    return (a - b).norm() / b.norm().clamp_min(1e-30)


def load_split(cfg, family_set, part, N=None):
    idx = data.split_indices(cfg, family_set)[part]
    if N is not None:
        idx = idx[:N]
    d = data.load_arrays(cfg, family_set, idx)
    return (torch.from_numpy(d["t_tilde"]), torch.from_numpy(d["w"]), torch.from_numpy(d["nu"]))


def load_all(cfg, name):
    d = data.load_arrays(cfg, name)
    return (torch.from_numpy(d["t_tilde"]), torch.from_numpy(d["w"]), torch.from_numpy(d["nu"]))


class UnlabelledSampler:
    """Regenerates thickness fields of set U on the GPU from their stored parameters."""

    def __init__(self, cfg, grid: Grid, device=DEV):
        with h5py.File(data.path(cfg, "U"), "r") as f:
            self.theta = torch.from_numpy(f["theta"][:]).double()
            self.nu = torch.from_numpy(f["nu"][:]).float()
            self.code = torch.from_numpy(f["family_code"][:])
        self.cfg, self.grid, self.dev = cfg, grid, device
        self.n = len(self.nu)

    def batch(self, b, gen):
        i = torch.randint(0, self.n, (b,), generator=gen)
        th_, code = self.theta[i], self.code[i]
        tau = torch.empty(b, *self.grid.shape, dtype=torch.float64)
        m = code == 0
        if m.any():
            tau[m] = th.tau_from_theta("pit", th_[m, :4], self.cfg, self.grid)
        if (~m).any():
            tau[~m] = th.tau_from_theta("kl", th_[~m], self.cfg, self.grid)
        return tau.float().to(self.dev), self.nu[i].to(self.dev)


@torch.no_grad()
def evaluate(model: Surrogate, tau, w, nu, H, bs=16):
    """Perturbation-strain relative L2 after H: pooled and per-sample median."""
    grid = model.grid
    num = den = 0.0
    per = []
    for s in range(0, len(nu), bs):
        t, wt, n = tau[s:s + bs].to(DEV), w[s:s + bs].to(DEV), nu[s:s + bs].to(DEV)
        wp = model(t, n)
        a, b = H(q1_strain(wp, grid.h)), H(q1_strain(wt, grid.h))
        e2, b2 = (a - b).pow(2).sum(1), b.pow(2).sum(1)
        num += e2.sum().item(); den += b2.sum().item()
        per.append((e2.sqrt() / b2.sqrt().clamp_min(1e-30)).cpu())
    per = torch.cat(per)
    return {"rel_l2": math.sqrt(num / den), "median_rel_l2": per.median().item()}


@torch.no_grad()
def time_inference(model: Surrogate, tau, nu, n=1000, bs=16):
    t, nn_ = tau[:bs].to(DEV), nu[:bs].to(DEV)
    model(t, nn_)
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(max(1, n // bs)):
        model(t, nn_)
    if DEV == "cuda":
        torch.cuda.synchronize()
    return 1000.0 * (time.time() - t0) * 1000.0 / (max(1, n // bs) * bs)   # ms per 1000


def train_run(cfg, model_name: str, family_set: str, N: int, seed: int, lam_r: float,
              steps: int, out_path: Path, log_every=500, splits=None):
    """splits: optional {"train": (tau, w, nu), "val": (tau, w, nu)} instead of the HDF5 split
    of family_set (used by the 3D face datasets of T5)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    grid = Grid.from_cfg(cfg)
    t = cfg["train"]
    if splits is None:
        tau_tr, w_tr, nu_tr = load_split(cfg, family_set, "train", N)
        tau_va, w_va, nu_va = load_split(cfg, family_set, "val")
    else:
        tau_tr, w_tr, nu_tr = (a[:N] for a in splits["train"])
        tau_va, w_va, nu_va = splits["val"]
    w_std = w_tr.reshape(-1, 2).std(0).clamp_min(1e-12)
    net = build(model_name, cfg, grid)
    model = Surrogate(net, grid, w_std).to(DEV)
    resid = Residual(grid, torch.float64).to(DEV)
    H = make_obs(cfg, grid, "full").to(DEV)
    use_phys = model_name == "pino" and lam_r > 0
    unl = UnlabelledSampler(cfg, grid) if use_phys else None
    gen = torch.Generator().manual_seed(seed)

    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=t["lr_min"])
    warm = int(t["warmup_frac"] * steps)
    bs = min(t["batch"], N)
    best, best_state, hist = float("inf"), None, []
    t0 = time.time()
    for step in range(1, steps + 1):
        i = torch.randint(0, N, (bs,), generator=gen)
        tau, wt, nu = tau_tr[i].to(DEV), w_tr[i].to(DEV), nu_tr[i].to(DEV)
        wp = model(tau, nu)
        loss = rel_l2(wp, wt) + t["alpha"] * rel_l2(q1_strain(wp, grid.h), q1_strain(wt, grid.h))
        lr_val = 0.0
        opt.zero_grad(set_to_none=True)
        if use_phys and step > warm:
            # labelled and unlabelled terms are back-propagated one after the other, so only
            # one forward graph is alive at a time (the sum of the two keeps PINO within 4 GB;
            # the accumulated gradient is the same as for the summed loss)
            u = grid.u0_hat_torch(nu.double()) + wp.double()
            r = resid(u, tau.double(), nu.double()).mean()
            lr_val += r.item()
            loss = loss + lam_r * r.float()
            loss.backward()
            tu, nuu = unl.batch(bs, gen)
            u = grid.u0_hat_torch(nuu.double()) + model(tu, nuu).double()
            r = resid(u, tu.double(), nuu.double()).mean()
            lr_val += r.item()
            (lam_r * r.float()).backward()
            loss = loss.detach() + lam_r * r.detach().float()
        else:
            loss.backward()
        opt.step()
        sched.step()
        if step % log_every == 0:
            print(f"    step {step}/{steps} loss {loss.item():.4e} res {lr_val:.3e} "
                  f"{(time.time() - t0) / step * 1000:.1f} ms/step", flush=True)
        if step % t["eval_every"] == 0 or step == steps:
            model.eval()
            v = evaluate(model, tau_va, w_va, nu_va, H)
            model.train()
            hist.append({"step": step, **v})
            if v["rel_l2"] < best:
                best = v["rel_l2"]
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(clean_state(best_state))
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model_name, "state": clean_state(model.state_dict()), "w_std": w_std,
                "family_set": family_set, "N": N, "seed": seed, "lam_r": lam_r,
                "steps": steps, "val_rel_l2": best, "history": hist,
                "config_hash": cfg.hash}, out_path)
    return model, {"val_rel_l2": best, "train_seconds": time.time() - t0, "history": hist}
