"""Capacity propagation (Step 8c): net-section capacity and a CalculiX elasto-plastic analysis."""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..grid import Grid


def net_section(tau: np.ndarray, grid: Grid, sigma_u: float, roi_x=None) -> np.ndarray:
    """P_u = sigma_u * min over cross-sections x of integral t dy. tau: (b, ny+1, nx+1) -> N."""
    t = grid.t0 * np.asarray(tau)
    area = np.trapezoid(t, grid.y, axis=1)                      # (b, nx+1)
    if roi_x is not None:
        m = (grid.x >= roi_x[0]) & (grid.x <= roi_x[1])
        area = area[:, m]
    return sigma_u * area.min(1)


def _inp(grid: Grid, tau, E, nu, fy, h_ratio, disp, coarsen=2):
    """CalculiX deck: S4 shells with nodal thickness, bilinear plasticity, displacement control."""
    s = coarsen
    t = (grid.t0 * tau)[::s, ::s]
    xs, ys = grid.x[::s], grid.y[::s]
    ny, nx = t.shape
    nid = lambda j, i: j * nx + i + 1
    L = ["*HEADING", "PINO capacity", "*NODE, NSET=NALL"]
    for j in range(ny):
        for i in range(nx):
            L.append(f"{nid(j, i)}, {xs[i]:.6f}, {ys[j]:.6f}, 0.0")
    L.append("*ELEMENT, TYPE=S4, ELSET=EALL")
    e = 1
    for j in range(ny - 1):
        for i in range(nx - 1):
            L.append(f"{e}, {nid(j, i)}, {nid(j, i + 1)}, {nid(j + 1, i + 1)}, {nid(j + 1, i)}"); e += 1
    L.append("*NSET, NSET=LEFT"); L += [f"{nid(j, 0)}," for j in range(ny)]
    L.append("*NSET, NSET=RIGHT"); L += [f"{nid(j, nx - 1)}," for j in range(ny)]
    L.append("*NSET, NSET=CORNER"); L.append(f"{nid(0, 0)},")
    Et = h_ratio * E
    Hp = E * Et / (E - Et)
    L += ["*MATERIAL, NAME=STEEL", "*ELASTIC", f"{E}, {nu}", "*PLASTIC",
          f"{fy}, 0.0", f"{fy + Hp * 0.2}, 0.2",
          "*SHELL SECTION, ELSET=EALL, MATERIAL=STEEL, NODAL THICKNESS", f"{grid.t0}",
          "*NODAL THICKNESS"]
    L += [f"{nid(j, i)}, {t[j, i]:.6f}" for j in range(ny) for i in range(nx)]
    L += ["*BOUNDARY", "LEFT, 1, 1", "CORNER, 2, 2", "NALL, 3, 3",
          "*STEP, NLGEOM, INC=500", "*STATIC", "0.02, 1.0, 1e-5, 0.05",
          "*BOUNDARY", f"RIGHT, 1, 1, {disp}",
          "*NODE PRINT, NSET=RIGHT, TOTALS=ONLY", "RF", "*END STEP"]
    return "\n".join(L) + "\n"


def calculix_curve(grid: Grid, tau, E, nu, fy, h_ratio, disp, ccx="ccx", coarsen=2, workdir=None):
    """Returns (elongation mm, force N) arrays from a displacement-controlled ccx run."""
    if shutil.which(ccx) is None:
        raise RuntimeError("CalculiX (ccx) not found")
    wd = Path(workdir or tempfile.mkdtemp(prefix="ccx_"))
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "job.inp").write_text(_inp(grid, tau, E, nu, fy, h_ratio, disp, coarsen))
    subprocess.run([ccx, "-i", "job"], cwd=wd, capture_output=True, check=True,
                   env={"OMP_NUM_THREADS": "2", "PATH": "/usr/bin:/bin"})
    txt = (wd / "job.dat").read_text()
    times = [float(x) for x in re.findall(r"total force \(fx,fy,fz\) for set RIGHT and time\s+([0-9.E+-]+)", txt)]
    forces = [float(x) for x in re.findall(r"total force[^\n]*\n\s*\n\s*([-0-9.E+]+)", txt)]
    n = min(len(times), len(forces))
    return np.r_[0, np.array(times[:n]) * disp], np.r_[0, np.array(forces[:n])]


def offset_capacity(elong, force, L, offset=0.002):
    """Load where the force-elongation curve meets the 0.2 % offset line (general yield)."""
    k = force[1] / elong[1]
    line = k * (elong - offset * L)
    idx = np.flatnonzero(force <= line)
    if len(idx) == 0:
        return float("nan")
    i = idx[0]
    a = (force[i - 1] - line[i - 1]) / ((force[i - 1] - line[i - 1]) - (force[i] - line[i]))
    return float(force[i - 1] + a * (force[i] - force[i - 1]))
