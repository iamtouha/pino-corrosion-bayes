"""HDF5 dataset I/O (Step 2). One file per set:
t_tilde (N, ny+1, nx+1), w (N, ny+1, nx+1, 2), theta (N, d), nu (N,), all float32."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

FAMILY = {"A": "pit", "B": "kl", "C": "triangle", "D_pit": "pit", "D_kl": "kl", "U": "mixed"}
SEEDS = {"A": 1, "B": 2, "C": 3, "D_pit": 4, "D_kl": 5, "U": 6}


def path(cfg, name) -> Path:
    return cfg.data_dir / f"{name}.h5"


def create(cfg, name, n, theta_dim, grid, with_w=True):
    p = path(cfg, name)
    f = h5py.File(p, "w")
    sh = grid.shape
    f.create_dataset("t_tilde", (n, *sh), "f4", chunks=(1, *sh))
    if with_w:
        f.create_dataset("w", (n, *sh, 2), "f4", chunks=(1, *sh, 2))
    f.create_dataset("theta", (n, theta_dim), "f4")
    f.create_dataset("nu", (n,), "f4")
    f.create_dataset("done", (n,), "u1")
    f.attrs["family"] = FAMILY.get(name, name)
    f.attrs["config_hash"] = cfg.hash
    return f


def load_arrays(cfg, name, idx=None, keys=("t_tilde", "w", "theta", "nu")):
    with h5py.File(path(cfg, name), "r") as f:
        n = f["nu"].shape[0]
        sel = slice(None) if idx is None else np.sort(np.asarray(idx))
        out = {k: f[k][sel] for k in keys if k in f}
        out["family"] = f.attrs["family"]
        if "family_code" in f:
            out["family_code"] = f["family_code"][sel]
        out["n"] = n
    return out


def split_indices(cfg, name):
    """Fixed split of a set into train pool / val / test (Sobol order is already space filling,
    so a seeded permutation keeps each part representative)."""
    s = cfg["dataset"]["split"]
    with h5py.File(path(cfg, name), "r") as f:
        n = f["nu"].shape[0]
    perm = np.random.default_rng(1234).permutation(n)
    a, b = s["train"], s["train"] + s["val"]
    return {"train": perm[:a], "val": perm[a:b], "test": perm[b:b + s["test"]]}


class TensorSet(torch.utils.data.Dataset):
    def __init__(self, tau, w, nu):
        self.tau, self.w, self.nu = tau, w, nu

    def __len__(self):
        return len(self.nu)

    def __getitem__(self, i):
        return self.tau[i], self.w[i], self.nu[i]
