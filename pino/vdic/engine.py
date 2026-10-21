"""Virtual DIC: speckle rendering, a subset-based 2D DIC engine and VSG strains (Step 3).

The DIC engine is a GPU implementation of the standard local method: first-order subset
shape function, ZNSSD criterion, inverse-compositional Gauss-Newton (IC-GN) with bicubic
interpolation, integer ZNCC search on sparse seeds for the initial guess.  Strain is the
gradient of a least-squares plane fit over a square window (virtual strain gauge).

Pixel (row, col) sits at (y, x) = (row, col) * px_mm. Displacements u (x) and v (y) in px.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- images
def render(centres: torch.Tensor, h: int, w: int, s=1.2, half=4) -> torch.Tensor:
    """centres: (n, 2) (row, col) in px of this window. Returns float image (h, w) in [20, 220]."""
    img = torch.zeros(h * w, device=centres.device, dtype=torch.float32)
    base = torch.floor(centres).long()
    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            rr, cc = base[:, 0] + dr, base[:, 1] + dc
            ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            d2 = (rr - centres[:, 0]) ** 2 + (cc - centres[:, 1]) ** 2
            img.index_add_(0, (rr * w + cc)[ok], torch.exp(-d2[ok] / (2 * s ** 2)).float())
    return 20 + 200 * img.clamp(0, 1).view(h, w)        # black paint, white speckles


def deform(centres_global: torch.Tensor, U_mm: torch.Tensor, grid, px_mm: float) -> torch.Tensor:
    """Move speckle centres (global px, (row, col)) by the FE displacement U_mm
    (ny+1, nx+1, 2) given on the operator grid nodes. Bilinear = the Q1 interpolant."""
    y = centres_global[:, 0] * px_mm
    x = centres_global[:, 1] * px_mm
    g = torch.stack([2 * x / grid.L - 1, 2 * y / grid.W - 1], -1).view(1, 1, -1, 2)
    Uf = U_mm.permute(2, 0, 1).unsqueeze(0).to(g.dtype)  # (1, 2, ny+1, nx+1)
    d = F.grid_sample(Uf, g, mode="bilinear", align_corners=True, padding_mode="border")[0, :, 0]
    return centres_global + torch.stack([d[1], d[0]], -1) / px_mm


def add_noise(img, sigma_read, gen: torch.Generator):
    noisy = img + sigma_read * torch.randn(img.shape, generator=gen, device=img.device)
    return noisy.round().clamp(0, 255)


# ----------------------------------------------------------------------------- DIC
def _grad5(img):
    """5-point central-difference gradients (d/drow, d/dcol)."""
    k = torch.tensor([1.0, -8.0, 0.0, 8.0, -1.0], device=img.device) / 12.0
    x = img[None, None]
    gc = F.conv2d(F.pad(x, (2, 2, 0, 0), mode="replicate"), k.view(1, 1, 1, 5))[0, 0]
    gr = F.conv2d(F.pad(x, (0, 0, 2, 2), mode="replicate"), k.view(1, 1, 5, 1))[0, 0]
    return gr, gc


def _bspline_w(t):
    """Cubic B-spline weights for neighbours -1, 0, +1, +2 at fractional offset t."""
    t2, t3 = t * t, t * t * t
    return ((1 - t) ** 3 / 6, (3 * t3 - 6 * t2 + 4) / 6, (-3 * t3 + 3 * t2 + 3 * t + 1) / 6, t3 / 6)


def bspline_prefilter(img):
    """Cubic B-spline interpolation coefficients (separable, FFT, mirror-extended)."""
    def along(x, dim):
        n = x.shape[dim]
        xe = torch.cat([x, x.flip(dim)], dim)                 # even extension, no wrap jump
        w = torch.fft.rfftfreq(2 * n, device=x.device) * 2 * math.pi
        den = (4 + 2 * torch.cos(w)) / 6
        X = torch.fft.rfft(xe, dim=dim)
        shape = [1] * x.ndim; shape[dim] = -1
        y = torch.fft.irfft(X / den.view(shape), n=2 * n, dim=dim)
        return y.narrow(dim, 0, n)
    return along(along(img.double(), 0), 1).float()


class SubsetDIC:
    def __init__(self, radius: int, max_iter=30, tol=1e-4, search=5, chunk_pts=8e6):
        self.R, self.max_iter, self.tol, self.search = radius, max_iter, tol, search
        d = torch.arange(-radius, radius + 1, device=DEV)
        dy, dx = torch.meshgrid(d, d, indexing="ij")
        self.dy, self.dx = dy.reshape(-1).float(), dx.reshape(-1).float()
        self.chunk = max(1, int(chunk_pts // self.dx.numel()))

    # -- helpers
    def _ref_subsets(self, ref, gr, gc, rows, cols):
        H, W = ref.shape
        rr = rows[:, None] + self.dy.long()[None]
        cc = cols[:, None] + self.dx.long()[None]
        idx = rr * W + cc
        f, fy, fx = ref.view(-1)[idx], gr.view(-1)[idx], gc.view(-1)[idx]
        fm = f.mean(1, keepdim=True)
        f0 = f - fm
        df = f0.pow(2).sum(1, keepdim=True).sqrt()
        dx, dy = self.dx[None], self.dy[None]
        J = torch.stack([fx, fx * dx, fx * dy, fy, fy * dx, fy * dy], -1)   # p=(u,ux,uy,v,vx,vy)
        Hm = torch.einsum("npi,npj->nij", J, J)
        return f0, df, J, torch.linalg.inv(Hm)

    def _sample(self, coef, rows, cols, p):
        """Cubic B-spline sample at warped subset points. coef: B-spline coefficient image
        (bspline_prefilter); p: (n, 6). Integer and fractional parts are kept apart so the
        sub-pixel position is exact in float32."""
        H, W = coef.shape
        dx, dy = self.dx[None], self.dy[None]
        ox = p[:, 0:1] + p[:, 1:2] * dx + p[:, 2:3] * dy
        oy = p[:, 3:4] + p[:, 4:5] * dx + p[:, 5:6] * dy
        fx, fy = torch.floor(ox), torch.floor(oy)
        tx, ty = ox - fx, oy - fy
        ix = cols[:, None] + self.dx.long()[None] + fx.long()
        iy = rows[:, None] + self.dy.long()[None] + fy.long()
        wx, wy = _bspline_w(tx), _bspline_w(ty)
        flat = coef.view(-1)
        out = torch.zeros_like(tx)
        for a in range(4):
            r = (iy + (a - 1)).clamp(0, H - 1) * W
            acc = torch.zeros_like(tx)
            for b in range(4):
                acc += wx[b] * flat[r + (ix + (b - 1)).clamp(0, W - 1)]
            out += wy[a] * acc
        return out

    @staticmethod
    def _mat(p):
        n = p.shape[0]
        M = torch.zeros(n, 3, 3, device=p.device, dtype=p.dtype)
        M[:, 0, 0], M[:, 0, 1], M[:, 0, 2] = 1 + p[:, 1], p[:, 2], p[:, 0]
        M[:, 1, 0], M[:, 1, 1], M[:, 1, 2] = p[:, 4], 1 + p[:, 5], p[:, 3]
        M[:, 2, 2] = 1
        return M

    @staticmethod
    def _vec(M):
        return torch.stack([M[:, 0, 2], M[:, 0, 0] - 1, M[:, 0, 1],
                            M[:, 1, 2], M[:, 1, 0], M[:, 1, 1] - 1], 1)

    def icgn(self, ref, gr, gc, dfm, rows, cols, p0):
        out_p, out_c = [], []
        for s in range(0, len(rows), self.chunk):
            r, c, p = rows[s:s + self.chunk], cols[s:s + self.chunk], p0[s:s + self.chunk].clone()
            f0, df, J, Hinv = self._ref_subsets(ref, gr, gc, r, c)
            for _ in range(self.max_iter):
                g = self._sample(dfm, r, c, p)
                g0 = g - g.mean(1, keepdim=True)
                dg = g0.pow(2).sum(1, keepdim=True).sqrt()
                e = f0 - (df / dg) * g0
                dp = -torch.einsum("nij,nj->ni", Hinv, torch.einsum("npi,np->ni", J, e))
                p = self._vec(self._mat(p) @ torch.linalg.inv(self._mat(dp)))
                step = dp[:, 0] ** 2 + dp[:, 3] ** 2 + (self.R * dp[:, [1, 2, 4, 5]]).pow(2).sum(1)
                if step.max().sqrt() < self.tol:
                    break
            g = self._sample(dfm, r, c, p)
            g0 = g - g.mean(1, keepdim=True)
            zncc = (f0 * g0).sum(1) / (df[:, 0] * g0.pow(2).sum(1).sqrt())
            out_p.append(p)
            out_c.append(zncc)
        return torch.cat(out_p), torch.cat(out_c)

    def integer_search(self, ref, dfm, rows, cols):
        H, W = ref.shape
        S = self.search
        rr = rows[:, None] + self.dy.long()[None]
        cc = cols[:, None] + self.dx.long()[None]
        f = ref.view(-1)[rr * W + cc]
        f0 = f - f.mean(1, keepdim=True)
        f0 = f0 / f0.norm(dim=1, keepdim=True)
        best = torch.full((len(rows),), -2.0, device=DEV)
        bu = torch.zeros(len(rows), device=DEV)
        bv = torch.zeros(len(rows), device=DEV)
        for sv in range(-S, S + 1):
            for su in range(-S, S + 1):
                r2 = (rr + sv).clamp(0, H - 1)
                c2 = (cc + su).clamp(0, W - 1)
                g = dfm.view(-1)[r2 * W + c2]
                g0 = g - g.mean(1, keepdim=True)
                z = (f0 * g0).sum(1) / g0.norm(dim=1)
                m = z > best
                best = torch.where(m, z, best)
                bu = torch.where(m, torch.full_like(bu, su), bu)
                bv = torch.where(m, torch.full_like(bv, sv), bv)
        return bu, bv

    @torch.no_grad()
    def run(self, ref, dfm, rows_1d, cols_1d, seed_every=8):
        """ref/dfm: (H, W) float images; rows_1d/cols_1d: point axes in this window's px.
        Returns u, v (px) and zncc, each (len(rows), len(cols))."""
        ref, dfm = ref.to(DEV).float(), dfm.to(DEV).float()
        cref = bspline_prefilter(ref)
        # gradient of the B-spline interpolant at the pixel centres: (c[+1] - c[-1]) / 2
        gr = 0.5 * (torch.roll(cref, -1, 0) - torch.roll(cref, 1, 0))
        gc = 0.5 * (torch.roll(cref, -1, 1) - torch.roll(cref, 1, 1))
        cdfm = bspline_prefilter(dfm)
        del cref
        R1 = torch.as_tensor(rows_1d, device=DEV).long()
        C1 = torch.as_tensor(cols_1d, device=DEV).long()
        # seeds on a sparse sub-grid that includes the last row/column
        si = torch.unique(torch.cat([torch.arange(0, len(R1), seed_every, device=DEV),
                                     torch.tensor([len(R1) - 1], device=DEV)]))
        sj = torch.unique(torch.cat([torch.arange(0, len(C1), seed_every, device=DEV),
                                     torch.tensor([len(C1) - 1], device=DEV)]))
        sr, sc = torch.meshgrid(R1[si], C1[sj], indexing="ij")
        sr, sc = sr.reshape(-1), sc.reshape(-1)
        bu, bv = self.integer_search(ref, dfm, sr, sc)
        p0 = torch.zeros(len(sr), 6, device=DEV)
        p0[:, 0], p0[:, 3] = bu, bv
        ps, _ = self.icgn(ref, gr, gc, cdfm, sr, sc, p0)
        # interpolate seed (u, v) to every point (bilinear on the seed sub-grid)
        seed_uv = ps[:, [0, 3]].T.reshape(1, 2, len(si), len(sj))
        Rr, Cc = R1.float(), C1.float()
        sy, sx = Rr[si], Cc[sj]
        gy = torch.from_numpy(np.interp(Rr.cpu().numpy(), sy.cpu().numpy(), np.linspace(-1, 1, len(si)))).to(DEV)
        gx = torch.from_numpy(np.interp(Cc.cpu().numpy(), sx.cpu().numpy(), np.linspace(-1, 1, len(sj)))).to(DEV)
        GY, GX = torch.meshgrid(gy.float(), gx.float(), indexing="ij")
        init = F.grid_sample(seed_uv, torch.stack([GX, GY], -1)[None], mode="bilinear",
                             align_corners=True)[0]
        rows, cols = torch.meshgrid(R1, C1, indexing="ij")
        p0 = torch.zeros(rows.numel(), 6, device=DEV)
        p0[:, 0], p0[:, 3] = init[0].reshape(-1), init[1].reshape(-1)
        p, zncc = self.icgn(ref, gr, gc, cdfm, rows.reshape(-1), cols.reshape(-1), p0)
        shp = (len(R1), len(C1))
        return p[:, 0].reshape(shp), p[:, 3].reshape(shp), zncc.reshape(shp)


# ----------------------------------------------------------------------------- VSG strain
def vsg_strain(U_mm, V_mm, spacing_mm: float, n: int):
    """Plane-fit gradients over an n x n window of DIC points ('valid' output).
    U_mm/V_mm: (py, px) displacement in x/y. Returns (3, py-n+1, px-n+1) = (e11, e22, e12)."""
    m = (n - 1) / 2
    z = (torch.arange(n, device=U_mm.device, dtype=U_mm.dtype) - m) * spacing_mm
    kd = z / (z.pow(2).sum())                                 # least-squares slope weights
    ka = torch.full((n,), 1.0 / n, device=U_mm.device, dtype=U_mm.dtype)
    x = torch.stack([U_mm, V_mm])[:, None]                    # (2, 1, py, px)
    ddx = F.conv2d(F.conv2d(x, kd.view(1, 1, 1, n)), ka.view(1, 1, n, 1))[:, 0]
    ddy = F.conv2d(F.conv2d(x, ka.view(1, 1, 1, n)), kd.view(1, 1, n, 1))[:, 0]
    return torch.stack([ddx[0], ddy[1], 0.5 * (ddy[0] + ddx[1])])


def vsg_points(cfg) -> int:
    d = cfg["dic"]
    n = int(round(d["vsg_window_mm"] / (d["step_px"] * d["px_mm"]))) + 1
    return n if n % 2 == 1 else n + 1


# ----------------------------------------------------------------------------- pipeline
class VirtualDIC:
    """Renders speckle images of a window around the requested DIC points, deforms them with
    an FE displacement field and runs DIC + VSG. Speckle centres are fixed per instance, so
    repeated calls with new noise are cheap noise realizations."""

    def __init__(self, cfg, grid, cols_px, rows_px, radius_px=None, seed=0):
        d = cfg["dic"]
        self.cfg, self.grid = cfg, grid
        self.px = d["px_mm"]
        self.R = int(radius_px or d["subset_radius_px"])
        self.n_vsg = vsg_points(cfg)
        m = (self.n_vsg - 1) // 2
        st = d["step_px"]
        cols_px, rows_px = np.asarray(cols_px), np.asarray(rows_px)
        self.cols_out, self.rows_out = cols_px, rows_px
        # extended axes so that the VSG 'valid' output lands exactly on the requested points
        self.cols_ext = np.concatenate([cols_px[0] - st * np.arange(m, 0, -1), cols_px,
                                        cols_px[-1] + st * np.arange(1, m + 1)])
        self.rows_ext = np.concatenate([rows_px[0] - st * np.arange(m, 0, -1), rows_px,
                                        rows_px[-1] + st * np.arange(1, m + 1)])
        marg = self.R + 12
        self.c0 = int(self.cols_ext.min() - marg)
        self.r0 = int(self.rows_ext.min() - marg)
        self.w = int(self.cols_ext.max() + marg - self.c0 + 1)
        self.h = int(self.rows_ext.max() + marg - self.r0 + 1)
        gen = torch.Generator(device=DEV).manual_seed(seed)
        n_sp = int(d["speckle_density"] * self.h * self.w)
        u = torch.rand(n_sp, 2, generator=gen, device=DEV)
        self.centres = torch.stack([self.r0 - 5 + u[:, 0] * (self.h + 10),
                                    self.c0 - 5 + u[:, 1] * (self.w + 10)], -1).double()
        self.s = d["speckle_sigma_px"]
        self.ref_clean = render(self.centres - self._off(), self.h, self.w, self.s)
        self.dic = SubsetDIC(self.R)
        self.gen = torch.Generator(device=DEV).manual_seed(seed + 7919)

    def _off(self):
        return torch.tensor([self.r0, self.c0], device=DEV, dtype=torch.float64)

    def set_field(self, U_mm: np.ndarray | torch.Tensor):
        U = torch.as_tensor(U_mm, device=DEV, dtype=torch.float64)
        moved = deform(self.centres, U, self.grid, self.px)
        self.def_clean = render(moved - self._off(), self.h, self.w, self.s)

    def measure(self, sigma_read: float, noise=True):
        ref, dfm = self.ref_clean, self.def_clean
        if noise and sigma_read > 0:
            ref, dfm = add_noise(ref, sigma_read, self.gen), add_noise(dfm, sigma_read, self.gen)
        u, v, zncc = self.dic.run(ref, dfm, self.rows_ext - self.r0, self.cols_ext - self.c0)
        st_mm = self.cfg["dic"]["step_px"] * self.px
        eps = vsg_strain(u.double() * self.px, v.double() * self.px, st_mm, self.n_vsg)
        return eps, (u, v, zncc)
