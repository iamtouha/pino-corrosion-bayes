"""The DIC observation operator H = S o (K_l * .) (Theory 2) and DIC point designs.

Strains live on element centres of the operator grid; H smooths each component with a
Gaussian of width ell_f (mm) and samples it bilinearly at the DIC points.
Observation vectors are component-major: [e11 at all points, e22 ..., e12 ...].
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from ..grid import Grid


def dic_axes(cfg, design: str = "full"):
    """1D point coordinates (x_mm, y_mm) of a rectilinear DIC design."""
    xs, ys = dic_axes_px(cfg, design)
    px = cfg["dic"]["px_mm"]
    return xs * px, ys * px


def dic_axes_px(cfg, design: str = "full"):
    """Same points as integer pixel indices (subset centres sit on whole pixels)."""
    d = cfg["dic"]
    px, step = d["px_mm"], d["step_px"]
    if design == "full":
        (x0, x1), (y0, y1) = d["roi"]["x"], d["roi"]["y"]
    elif design == "crop":
        x0, x1, y0, y1 = d["crop_mm"]
    elif design == "grid56":
        (x0, x1), (y0, y1) = d["roi"]["x"], d["roi"]["y"]
        g = d["grid_56"]
        # 7 rows x 8 columns spread over the ROI, inset by half a spacing
        xs = np.linspace(x0, x1, 2 * g["ncols"] + 1)[1::2]
        ys = np.linspace(y0, y1, 2 * g["nrows"] + 1)[1::2]
        return np.rint(xs / px).astype(int), np.rint(ys / px).astype(int)
    else:
        raise ValueError(design)
    c0, r0 = int(np.ceil(x0 / px)), int(np.ceil(y0 / px))
    xs = np.arange(c0, int(np.floor(x1 / px)) + 1, step)
    ys = np.arange(r0, int(np.floor(y1 / px)) + 1, step)
    return xs, ys


def gaussian_kernel1d(ell_mm: float, h: float, dtype=torch.float32, device="cpu"):
    if ell_mm <= 1e-9:
        return torch.ones(1, dtype=dtype, device=device)
    s = ell_mm / h
    n = max(1, int(math.ceil(3.5 * s)))
    z = torch.arange(-n, n + 1, dtype=dtype, device=device)
    k = torch.exp(-0.5 * (z / s) ** 2)
    return k / k.sum()


class ObsOperator(torch.nn.Module):
    """H for one design. forward(strain (b, ny, nx, 3)) -> (b, 3 * n_points)."""

    def __init__(self, grid: Grid, xs_mm, ys_mm, ell_f: float, dtype=torch.float32):
        super().__init__()
        self.grid, self.ell_f = grid, float(ell_f)
        self.xs, self.ys = np.asarray(xs_mm), np.asarray(ys_mm)
        self.shape = (len(self.ys), len(self.xs))
        k = gaussian_kernel1d(self.ell_f, grid.h, dtype)
        self.register_buffer("k", k)
        xc, yc = grid.xc, grid.yc
        Yp, Xp = np.meshgrid(self.ys, self.xs, indexing="ij")
        gx = 2 * (Xp - xc[0]) / (xc[-1] - xc[0]) - 1
        gy = 2 * (Yp - yc[0]) / (yc[-1] - yc[0]) - 1
        g = torch.tensor(np.stack([gx, gy], -1), dtype=dtype).unsqueeze(0)  # (1, py, px, 2)
        self.register_buffer("g", g)

    @property
    def n_points(self):
        return self.shape[0] * self.shape[1]

    def smooth(self, s):
        """s: (b, ny, nx, 3) -> smoothed (b, 3, ny, nx), replicate padding."""
        x = s.permute(0, 3, 1, 2)
        n = (self.k.numel() - 1) // 2
        if n == 0:
            return x
        c = x.shape[1]
        kx = self.k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
        ky = self.k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
        x = F.conv2d(F.pad(x, (n, n, 0, 0), mode="replicate"), kx, groups=c)
        x = F.conv2d(F.pad(x, (0, 0, n, n), mode="replicate"), ky, groups=c)
        return x

    def forward(self, strain):
        x = self.smooth(strain)
        g = self.g.expand(x.shape[0], -1, -1, -1).to(x.dtype)
        y = F.grid_sample(x, g, mode="bilinear", align_corners=True)   # (b, 3, py, px)
        return y.reshape(x.shape[0], -1)

    def fields(self, obs):
        """(b, 3 n) -> (b, 3, py, px)."""
        return obs.reshape(obs.shape[0], 3, *self.shape)


def make_obs(cfg, grid: Grid, design: str, ell_f: float | None = None, dtype=torch.float32):
    if ell_f is None:
        ell_f = load_filter_width(cfg)
    xs, ys = dic_axes(cfg, design)
    return ObsOperator(grid, xs, ys, ell_f, dtype)


def load_filter_width(cfg) -> float:
    import json
    p = cfg.data_dir / "vdic" / "filter_fit.json"
    if p.exists():
        return float(json.loads(p.read_text())["ell_f"])
    return 3.0  # until Step 3.3 has run
