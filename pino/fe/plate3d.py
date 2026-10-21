"""3D solid plate with loss removed from one face only (Step 1b), nondimensional E = 1, P = 1.

Flat face at z = 0 (intact side); the corroded face is z = t0 * tau(x, y).  Hex mesh with
nz layers through the thickness, solved with CG + smoothed-aggregation AMG (GAMG) using
the rigid-body near-nullspace.

BCs: left face u1 = 0 (stops rotation about y there), line (x=0, y=0) u2 = 0, mid-thickness
lines of both short faces u3 = 0.  The right face carries a uniform traction of total
force 1 in x.
"""
from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem, la, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc

from ..grid import Grid


def _rigid_body_modes(V):
    """Six rigid-body modes as a PETSc near-nullspace (after the dolfinx elasticity demo)."""
    bs = V.dofmap.index_map_bs
    n = V.dofmap.index_map.size_local
    basis = [la.vector(V.dofmap.index_map, bs=bs, dtype=PETSc.ScalarType) for _ in range(6)]
    b = [v.array for v in basis]
    dofs = [V.sub(i).dofmap.list.ravel() for i in range(3)]
    for i in range(3):
        b[i][dofs[i]] = 1.0
    x = V.tabulate_dof_coordinates()
    d0, d1, d2 = dofs
    x0, x1, x2 = x[d0 // 3, 0], x[d0 // 3, 1], x[d0 // 3, 2]
    b[3][d0] = -x1; b[3][d1] = x0
    b[4][d0] = x2;  b[4][d2] = -x0
    b[5][d2] = x1;  b[5][d1] = -x2
    la.orthonormalize(basis)
    vecs = [PETSc.Vec().createWithArray(v.array[: bs * n], bsize=bs, comm=V.mesh.comm) for v in basis]
    return PETSc.NullSpace().create(vectors=vecs)


def solve_plate3d(grid: Grid, tau_nodes: np.ndarray, nu: float, nz: int = 4, deg: int = 1,
                  nx: int | None = None, ny: int | None = None, rtol=1e-10):
    """tau_nodes on the (ny+1, nx+1) grid. Returns (u_flat, u_corroded): in-plane nondimensional
    displacement u_hat = E t0 u / P on the z = 0 face and on the corroded face, each (ny+1, nx+1, 2)."""
    nx, ny = nx or grid.nx, ny or grid.ny
    L, W, t0 = grid.L, grid.W, grid.t0
    hx, hy = L / nx, W / ny
    dom = mesh.create_box(MPI.COMM_SELF, [np.array([0.0, 0.0, 0.0]), np.array([L, W, t0])],
                          [nx, ny, nz], cell_type=mesh.CellType.hexahedron)
    X = dom.geometry.x
    i = np.rint(X[:, 0] / hx).astype(int)
    j = np.rint(X[:, 1] / hy).astype(int)
    X[:, 2] *= tau_nodes[j, i]                               # remove loss from the top face

    V = fem.functionspace(dom, ("Lagrange", deg, (3,)))
    nu_c = nu
    lam = nu_c / ((1 + nu_c) * (1 - 2 * nu_c))
    mu = 1 / (2 * (1 + nu_c))

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sig(w):
        return lam * ufl.tr(eps(w)) * ufl.Identity(3) + 2 * mu * eps(w)

    tol = 1e-8
    zmid = 0.5 * t0
    V0, _ = V.sub(0).collapse(); V1, _ = V.sub(1).collapse(); V2, _ = V.sub(2).collapse()
    left_f = mesh.locate_entities_boundary(dom, 2, lambda x: np.isclose(x[0], 0.0))
    right_f = mesh.locate_entities_boundary(dom, 2, lambda x: np.isclose(x[0], L))
    d_x = fem.locate_dofs_topological((V.sub(0), V0), 2, left_f)
    d_y = fem.locate_dofs_geometrical((V.sub(1), V1),
                                      lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    def _mid_short_faces(x):
        # mid-thickness nodes of both short faces. The mesh is deformed by tau, so the middle
        # layer sits at 0.5 t0 tau(x, y) there (= zmid when no loss reaches the ends; pits of the
        # prior can reach x = 0 / L, where a fixed zmid would find no nodes).
        ii_ = np.clip(np.rint(x[0] / hx).astype(int), 0, nx)
        jj_ = np.clip(np.rint(x[1] / hy).astype(int), 0, ny)
        return ((np.isclose(x[0], 0.0) | np.isclose(x[0], L))
                & (np.abs(x[2] - 0.5 * t0 * tau_nodes[jj_, ii_]) < 1e-6))

    d_z = fem.locate_dofs_geometrical((V.sub(2), V2), _mid_short_faces)
    bcs = [fem.dirichletbc(fem.Function(V0), d_x, V.sub(0)),
           fem.dirichletbc(fem.Function(V1), d_y, V.sub(1)),
           fem.dirichletbc(fem.Function(V2), d_z, V.sub(2))]
    if len(d_z[0]) == 0:
        raise RuntimeError("no mid-plane nodes on the short faces; use an even nz")

    tags = mesh.meshtags(dom, 2, right_f, np.ones(len(right_f), dtype=np.int32))
    ds = ufl.Measure("ds", domain=dom, subdomain_data=tags)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    q = fem.Constant(dom, np.array([1.0 / (W * t0), 0.0, 0.0]))
    a = ufl.inner(sig(u), eps(v)) * ufl.dx
    rhs = ufl.dot(q, v) * ds(1)

    problem = LinearProblem(a, rhs, bcs=bcs, petsc_options={
        "ksp_type": "cg", "ksp_rtol": rtol, "pc_type": "gamg",
        "pc_gamg_type": "agg", "mg_levels_ksp_type": "chebyshev",
        "mg_levels_pc_type": "jacobi", "ksp_max_it": 5000})
    problem.A.setNearNullSpace(_rigid_body_modes(V))
    uh = problem.solve()
    its = problem.solver.getIterationNumber()
    reason = problem.solver.getConvergedReason()
    if reason < 0:
        raise RuntimeError(f"3D solve diverged (reason {reason}, {its} its)")

    # nodal values on the two faces (Q1 nodes coincide with grid nodes)
    Q1 = fem.functionspace(dom, ("Lagrange", 1, (3,)))
    if deg == 1:
        arr = uh.x.array.reshape(-1, 3)
    else:
        u1 = fem.Function(Q1); u1.interpolate(uh); arr = u1.x.array.reshape(-1, 3)
    xyz = Q1.tabulate_dof_coordinates()
    ii = np.rint(xyz[:, 0] / hx).astype(int)
    jj = np.rint(xyz[:, 1] / hy).astype(int)
    top = tau_nodes[jj, ii] * t0
    out = []
    for face in (np.abs(xyz[:, 2]) < tol, np.abs(xyz[:, 2] - top) < 1e-6):
        g = np.full((ny + 1, nx + 1, 2), np.nan)
        g[jj[face], ii[face]] = t0 * arr[face, :2]          # E = P = 1 -> u_hat = t0 u
        out.append(g)
    return out[0], out[1], its
