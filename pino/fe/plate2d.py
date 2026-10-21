"""2D plane-stress membrane plate solver (FEniCSx 0.9), nondimensional (E = 1, P = 1).

Thickness enters as a DG0 cell value equal to the mean of the four nodal values, the
same rule the PINO residual uses, so FE data and physics loss share one discrete system.

Boundary conditions: left edge u1 = 0 (roller), corner (0, 0) u2 = 0, right edge a
uniform line load of total force P (independent of the local thickness, so f_hat in
the residual is the same vector).  Long edges are traction free.
"""
from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, set_bc)
from mpi4py import MPI
from petsc4py import PETSc

from ..grid import Grid


def _lu_solver_type():
    for name in ("mumps", "superlu_dist", "petsc"):
        try:
            if name == "petsc" or PETSc.Sys.hasExternalPackage(name):
                return name
        except Exception:  # pragma: no cover
            pass
    return "petsc"


class PlateSolver2D:
    def __init__(self, grid: Grid, nx: int | None = None, ny: int | None = None, deg: int = 1,
                 E: float = 1.0, P: float = 1.0, bc: str = "roller"):
        """E, P default to 1 (nondimensional). With other values the output is the
        displacement of a plate of thickness tau (not t0*tau): u = P/E * u_hat.
        bc: "roller" (u1 = 0 on x = 0, u2 = 0 at the corner; the project default) or
        "clamped" (u1 = u2 = 0 on x = 0, as in a gripped specimen)."""
        self.grid = grid
        self.nx, self.ny = nx or grid.nx, ny or grid.ny
        L, W = grid.L, grid.W
        self.hx, self.hy = L / self.nx, W / self.ny
        self.domain = mesh.create_rectangle(
            MPI.COMM_SELF, [np.array([0.0, 0.0]), np.array([L, W])], [self.nx, self.ny],
            cell_type=mesh.CellType.quadrilateral)
        d = self.domain
        self.V = fem.functionspace(d, ("Lagrange", deg, (2,)))
        P1 = fem.functionspace(d, ("Lagrange", 1))
        self.tau_n = fem.Function(P1)
        self.tau = fem.Function(fem.functionspace(d, ("DG", 0)))
        self.nu = fem.Constant(d, PETSc.ScalarType(0.3))

        xy = P1.tabulate_dof_coordinates()[:, :2]
        self._p1_ij = (np.rint(xy[:, 1] / self.hy).astype(int), np.rint(xy[:, 0] / self.hx).astype(int))

        nu = self.nu
        lam = E * nu / (1 - nu ** 2)
        mu = E / (2 * (1 + nu))

        def eps(w):
            return ufl.sym(ufl.grad(w))

        def sig(w):
            return lam * ufl.tr(eps(w)) * ufl.Identity(2) + 2 * mu * eps(w)

        fdim = 1
        left = mesh.locate_entities_boundary(d, fdim, lambda x: np.isclose(x[0], 0.0))
        right = mesh.locate_entities_boundary(d, fdim, lambda x: np.isclose(x[0], L))
        tags = mesh.meshtags(d, fdim, right, np.ones(len(right), dtype=np.int32))
        ds = ufl.Measure("ds", domain=d, subdomain_data=tags)

        V0, _ = self.V.sub(0).collapse()
        V1, _ = self.V.sub(1).collapse()
        dofs_x = fem.locate_dofs_topological((self.V.sub(0), V0), fdim, left)
        if bc == "clamped":
            dofs_y = fem.locate_dofs_topological((self.V.sub(1), V1), fdim, left)
        else:
            dofs_y = fem.locate_dofs_geometrical(
                (self.V.sub(1), V1), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
        self.bcs = [fem.dirichletbc(fem.Function(V0), dofs_x, self.V.sub(0)),
                    fem.dirichletbc(fem.Function(V1), dofs_y, self.V.sub(1))]

        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        q = fem.Constant(d, np.array([P / W, 0.0]))         # total force P on the right edge
        self.a = fem.form(ufl.inner(self.tau * sig(u), eps(v)) * ufl.dx)
        self.L = fem.form(ufl.dot(q, v) * ds(1))

        self.A = fem.petsc.create_matrix(self.a)
        self.b = create_vector(self.L)
        self.b.zeroEntries()
        assemble_vector(self.b, self.L)
        apply_lifting(self.b, [self.a], bcs=[self.bcs])
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b, self.bcs)

        self.ksp = PETSc.KSP().create(MPI.COMM_SELF)
        self.ksp.setType("preonly")
        pc = self.ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType(_lu_solver_type())
        self.uh = fem.Function(self.V)

        Q1 = fem.functionspace(d, ("Lagrange", 1, (2,)))
        self.u1 = fem.Function(Q1)
        q1xy = Q1.tabulate_dof_coordinates()[:, :2]
        qi = np.rint(q1xy[:, 0] / self.hx).astype(int)
        qj = np.rint(q1xy[:, 1] / self.hy).astype(int)
        self._order = np.lexsort((qi, qj))                   # rows = y, columns = x
        self.deg = deg

    def node_grid(self):
        return (np.linspace(0, self.grid.W, self.ny + 1), np.linspace(0, self.grid.L, self.nx + 1))

    def solve(self, tau_nodes: np.ndarray, nu: float) -> np.ndarray:
        """tau_nodes: (ny+1, nx+1) on THIS solver's grid. Returns u_hat (ny+1, nx+1, 2)."""
        self.tau_n.x.array[:] = tau_nodes[self._p1_ij]
        self.tau.interpolate(self.tau_n)                     # cell value = mean of 4 corners
        self.nu.value = nu
        self.A.zeroEntries()
        assemble_matrix(self.A, self.a, bcs=self.bcs)
        self.A.assemble()
        self.ksp.setOperators(self.A)
        self.ksp.solve(self.b, self.uh.x.petsc_vec)
        self.uh.x.scatter_forward()
        if self.deg == 1:
            arr = self.uh.x.array
        else:
            self.u1.interpolate(self.uh)
            arr = self.u1.x.array
        return arr.reshape(-1, 2)[self._order].reshape(self.ny + 1, self.nx + 1, 2).copy()


# ----------------------------------------------------------------- multiprocessing helpers
_SOLVER = None


def _init_worker(grid, bc="roller"):
    global _SOLVER
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _SOLVER = PlateSolver2D(grid, bc=bc)


def _solve_one(args):
    tau, nu = args
    return _SOLVER.solve(tau, nu).astype(np.float64)


def solve_many(grid: Grid, taus: np.ndarray, nus: np.ndarray, workers: int = 8, chunksize=4,
               bc: str = "roller"):
    """Solve a batch in parallel; each worker owns its mesh on MPI.COMM_SELF."""
    import multiprocessing as mp
    PlateSolver2D(grid, bc=bc).solve(np.ones(grid.shape), 0.3)   # warm the JIT cache once
    if workers <= 1:
        _init_worker(grid, bc)
        return np.stack([_solve_one(a) for a in zip(taus, nus)])
    ctx = mp.get_context("fork")
    pool = ctx.Pool(workers, initializer=_init_worker, initargs=(grid, bc))
    try:
        out = pool.map(_solve_one, list(zip(taus, nus)), chunksize=chunksize)
    finally:
        pool.close()          # not terminate(): SIGTERM makes PETSc abort noisily
        pool.join()
    return np.stack(out)
