"""Pass/fail gates that need no trained model: Step 4.2 (residual) and Step 6.2 (likelihood),
plus unit checks of the DIC filter, thickness models and the DIC engine."""
import math

import numpy as np
import pytest
import torch

from pino import config
from pino.grid import Grid, q1_strain
from pino.surrogates.residual import Residual, q1_element_mats, residual
from pino.thickness import pit_tau, sample_kl, sample_pit, tau_from_theta

CFG = config.load()
G = Grid.from_cfg(CFG)


def test_residual_intact_below_1e12():
    R = Residual(G, torch.float64)
    nu = torch.tensor([0.3, 0.27, 0.33], dtype=torch.float64)
    u0 = G.u0_hat_torch(nu)
    r = R(u0, torch.ones(3, *G.shape, dtype=torch.float64), nu)
    assert r.max().item() < 1e-12, r


def test_residual_of_fe_solutions_below_1e10():
    pytest.importorskip("dolfinx")
    from pino.fe.plate2d import PlateSolver2D
    s = PlateSolver2D(G)
    R = Residual(G, torch.float64)
    th, nu = sample_pit(10, CFG, 99)
    thk, nuk = sample_kl(10, CFG, 98)
    taus = torch.cat([tau_from_theta("pit", th, CFG, G), tau_from_theta("kl", thk, CFG, G)])
    nus = np.r_[nu, nuk]
    u = torch.stack([torch.tensor(s.solve(t.numpy(), n)) for t, n in zip(taus, nus)])
    r = R(u, taus, torch.tensor(nus))
    assert len(r) == 20 and r.max().item() < 1e-10, r


def test_residual_gradcheck_6x6():
    g = Grid(L=10.0, W=10.0, nx=5, ny=5)
    A, B = q1_element_mats(torch.float64)
    f, free = g.loads_and_mask(torch.float64)
    u = torch.randn(2, 6, 6, 2, dtype=torch.float64, requires_grad=True)
    tau = (1 + 0.1 * torch.rand(2, 6, 6, dtype=torch.float64)).requires_grad_()
    nu = torch.tensor([0.3, 0.28], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda u_, t_, n_: residual(u_, t_, n_, A, B, f, free), (u, tau, nu))


def test_loglik_matches_dense_mvn():
    from pino.inference.likelihood import GaussianLik, loglik
    torch.manual_seed(0)
    n, k = 200, 12
    U = torch.randn(n, k, dtype=torch.float64)
    lam = torch.rand(k, dtype=torch.float64) + 0.1
    s2 = 0.7
    r = torch.randn(n, dtype=torch.float64)
    S = s2 * torch.eye(n, dtype=torch.float64) + U @ torch.diag(lam) @ U.T
    ref = torch.distributions.MultivariateNormal(torch.zeros(n, dtype=torch.float64), S).log_prob(r)
    assert abs(loglik(r, s2, U, lam) - ref).item() < 1e-6
    lik = GaussianLik("iso", n, U, lam, s2=s2)
    assert abs(lik(r[None])[0] - ref).item() < 1e-6
    dense = torch.diag(torch.rand(n, dtype=torch.float64) + 0.5) + 0.1
    S2 = dense + U @ torch.diag(lam) @ U.T
    ref2 = torch.distributions.MultivariateNormal(torch.zeros(n, dtype=torch.float64), S2).log_prob(r)
    assert abs(GaussianLik("dense", n, U, lam, dense=dense)(r[None])[0] - ref2).item() < 1e-6


def test_kron_likelihood_matches_dense():
    from pino.inference.likelihood import GaussianLik, KronBlocks
    from pino.vdic.noise import se_1d
    xs, ys = np.arange(0, 12.0, 1.2), np.arange(0, 8.0, 1.2)
    sig, ell, nug = [2.0, 1.0, 1.5], [3.0, 3.0, 2.0], [0.5, 0.4, 0.3]
    kb = KronBlocks(xs, ys, sig, ell, nug, "cpu")
    m = len(xs) * len(ys)
    blocks = []
    for s_, l_, n_ in zip(sig, ell, nug):
        K = np.kron(se_1d(ys, l_), se_1d(xs, l_))
        blocks.append(torch.tensor(s_ ** 2 * K + n_ ** 2 * np.eye(m)))
    D = torch.block_diag(*blocks)
    U = torch.randn(3 * m, 4, dtype=torch.float64); lam = torch.rand(4, dtype=torch.float64) + 0.2
    r = torch.randn(2, 3 * m, dtype=torch.float64)
    ref = torch.distributions.MultivariateNormal(torch.zeros(3 * m, dtype=torch.float64),
                                                 D + U @ torch.diag(lam) @ U.T).log_prob(r)
    got = GaussianLik("kron", 3 * m, U, lam, kron=kb)(r)
    assert torch.allclose(got, ref, atol=1e-6), (got, ref)


def test_obs_operator_uniform_field():
    from pino.vdic.observe import make_obs
    H = make_obs(CFG, G, "full", ell_f=3.0, dtype=torch.float64)
    e = q1_strain(torch.tensor(G.u0_hat(0.3))[None], G.h)
    d = H.fields(H(e))
    assert torch.allclose(d[0, 0], torch.full_like(d[0, 0], 1 / G.W), rtol=1e-10)
    assert torch.allclose(d[0, 1], torch.full_like(d[0, 1], -0.3 / G.W), rtol=1e-10)


def test_pit_tau():
    t = pit_tau(torch.tensor([[150.0, 126.0, 20.0, 0.1]], dtype=torch.float64), G)
    assert abs(t.min().item() - 0.9) < 1e-9 and abs(t[0, 0, 0].item() - 1) < 1e-6


def test_vsg_on_linear_field():
    from pino.vdic.engine import vsg_strain
    y, x = torch.meshgrid(torch.arange(20.0), torch.arange(25.0), indexing="ij")
    U, V = 1e-3 * x * 1.2 + 2e-4 * y * 1.2, -3e-4 * y * 1.2
    e = vsg_strain(U.double(), V.double(), 1.2, 5)
    assert torch.allclose(e[0], torch.full_like(e[0], 1e-3))
    assert torch.allclose(e[1], torch.full_like(e[1], -3e-4))
    assert torch.allclose(e[2], torch.full_like(e[2], 1e-4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU DIC test")
def test_dic_recovers_affine_shift():
    """Translate + stretch a rendered speckle image and check the DIC displacement."""
    from pino.vdic.engine import SubsetDIC, render
    torch.manual_seed(0)
    h = w = 400
    n = int(0.048 * h * w)
    c = torch.rand(n, 2, device="cuda", dtype=torch.float64) * torch.tensor([h, w], device="cuda")
    ref = render(c, h, w)
    exx = 1e-3
    moved = c.clone()
    moved[:, 1] = c[:, 1] + 1.3 + exx * (c[:, 1] - 200)
    moved[:, 0] = c[:, 0] - 0.7
    dfm = render(moved, h, w)
    dic = SubsetDIC(30)
    rows = cols = np.arange(100, 301, 20)
    u, v, z = dic.run(ref, dfm, rows, cols)
    u_true = torch.tensor(1.3 + exx * (cols - 200.0), device="cuda").expand_as(u)
    assert (u - u_true).abs().max().item() < 0.01
    assert (v + 0.7).abs().max().item() < 0.01
    assert z.min().item() > 0.99


def test_tmcmc_sharp_posterior():
    """TMCMC must not collapse on a very sharp, correlated posterior (full-field case)."""
    from pino.inference.tmcmc import tmcmc
    mu = np.array([0.3, 0.6, 0.45, 0.8])
    sd = np.array([2e-3, 3e-3, 2e-3, 1e-3])
    R = np.eye(4); R[0, 1] = R[1, 0] = 0.8
    Ci = np.linalg.inv(R * np.outer(sd, sd))
    ll = lambda t: -0.5 * np.einsum("ni,ij,nj->n", t - mu, Ci, t - mu)
    samp = lambda n, rng: rng.random((n, 4))
    lpri = lambda t: np.where(np.all((t >= 0) & (t <= 1), 1), 0.0, -np.inf)
    th, logZ, info = tmcmc(ll, samp, lpri, n=1000, scale=0.2, n_mh=2, seed=0)
    assert info["unique_frac"] > 0.3, info["unique_frac"]
    assert np.all(np.abs(th.mean(0) - mu) < 3 * sd / np.sqrt(50))
    assert np.all(np.abs(th.std(0) / sd - 1) < 0.25), th.std(0) / sd
    logZ_true = 0.5 * np.log(np.linalg.det(2 * np.pi * np.linalg.inv(Ci)))
    assert abs(logZ - logZ_true) < 0.5, (logZ, logZ_true)


# ----------------------------------------------------------------------------- improvement runs (T2)
def test_woodbury_ll_matches_dense_mvn():
    """Batched Woodbury likelihood with a per-row diagonal = dense MVN log-density."""
    from pino.inference.discrepancy import woodbury_ll
    g = torch.Generator().manual_seed(0)
    m, k, b = 60, 5, 4
    U = torch.linalg.qr(torch.randn(m, k, generator=g, dtype=torch.float64))[0]
    lam = torch.rand(k, generator=g, dtype=torch.float64) + 0.5
    Dv = torch.rand(b, m, generator=g, dtype=torch.float64) * 0.2 + 0.05
    r = torch.randn(b, m, generator=g, dtype=torch.float64)
    got = woodbury_ll(r, Dv, U, lam)
    for i in range(b):
        S = torch.diag(Dv[i]) + (U * lam) @ U.T
        ref = torch.distributions.MultivariateNormal(torch.zeros(m, dtype=torch.float64), S).log_prob(r[i])
        assert abs(float(got[i] - (ref + 0.5 * m * math.log(2 * math.pi)))) < 1e-8


def test_gp_type2_ml_recovers_hyperparameters():
    """Type-II ML on one draw of a 2-component SE + white GP recovers (a, ell, s) within 25 %.
    (Short length scales: with ell ~ 10 mm on 80 x 20 mm one draw has ~16 independent patches
    and the amplitude scatters by ~35 % from draw to draw.)"""
    from pino.inference.discrepancy import fit_gp_hyper, gp_cov
    g = torch.Generator().manual_seed(1)
    xs, ys = torch.linspace(0, 80, 65, dtype=torch.float64), torch.linspace(0, 20, 17, dtype=torch.float64)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], 1)
    true = torch.tensor([2.0, 3.0, 0.3, 1.0, 4.0, 0.2], dtype=torch.float64)
    K = gp_cov(xy, torch.log(true), 2)
    r = torch.linalg.cholesky(K) @ torch.randn(len(K), generator=g, dtype=torch.float64)
    hp, _ = fit_gp_hyper(xy, r, 2)
    rel = torch.exp(hp) / true - 1
    assert torch.all(rel.abs() < 0.25), rel


def test_tmcmc_with_gp_discrepancy_matches_analytic_posterior():
    """Linear-Gaussian toy: y = A theta + GP discrepancy + noise, flat prior on a wide box.
    TMCMC with the fixed GP covariance reproduces the analytic posterior mean and std."""
    from pino.inference.discrepancy import gp_cov
    from pino.inference.tmcmc import tmcmc
    g = torch.Generator().manual_seed(2)
    x = torch.linspace(0, 10, 80, dtype=torch.float64)[:, None]
    A = torch.cat([torch.ones_like(x), x / 10], 1)                          # offset + slope
    K = gp_cov(x, torch.log(torch.tensor([0.3, 2.0, 0.05], dtype=torch.float64)), 1)
    th_true = torch.tensor([0.4, 0.6], dtype=torch.float64)
    y = A @ th_true + torch.linalg.cholesky(K) @ torch.randn(80, generator=g, dtype=torch.float64)
    Ki = torch.linalg.inv(K)
    Cp = torch.linalg.inv(A.T @ Ki @ A)
    mp = Cp @ A.T @ Ki @ y
    L = torch.linalg.cholesky(K)
    An, yn = A.numpy(), y.numpy()

    def ll(t):
        r = torch.as_tensor(yn[None] - t @ An.T).T
        z = torch.linalg.solve_triangular(L, r, upper=False)
        return (-0.5 * (z * z).sum(0)).numpy()

    samp = lambda n, rng: rng.uniform(-3, 3, (n, 2))
    lpri = lambda t: np.where(np.all(np.abs(t) <= 3, 1), 0.0, -np.inf)
    th, _, _ = tmcmc(ll, samp, lpri, n=2000, scale=0.2, n_mh=3, seed=0)
    sd = np.sqrt(np.diag(Cp.numpy()))
    assert np.all(np.abs(th.mean(0) - mp.numpy()) < 0.15 * sd), (th.mean(0), mp)
    assert np.all(np.abs(th.std(0) / sd - 1) < 0.15), th.std(0) / sd


def test_plate3d_edge_pit_and_intact():
    """3D solid (T5): a pit reaching the loaded end must solve (mid-thickness BC nodes follow
    the deformed mesh); an intact plate gives the uniaxial in-plane solution on both faces."""
    from pino.fe.plate3d import solve_plate3d
    g = Grid(L=G.L, W=G.W, nx=30, ny=25, t0=G.t0)
    nu = 0.3
    uf, uc, _ = solve_plate3d(g, np.ones(g.shape), nu, nz=2)
    u0 = g.u0_hat(nu)
    for u in (uf, uc):
        assert np.isfinite(u).all()
        assert np.abs(u - u0).max() < 1e-6 * np.abs(u0).max(), np.abs(u - u0).max()
    tau = pit_tau(torch.tensor([[0.0, 125.0, 40.0, 0.15]], dtype=torch.float64), g)[0].numpy()
    uf, uc, _ = solve_plate3d(g, tau, nu, nz=2)
    assert np.isfinite(uf).all() and np.isfinite(uc).all()
