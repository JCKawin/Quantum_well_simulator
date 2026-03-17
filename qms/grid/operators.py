import numpy as np

from ..utils.backend import cp, get_array_module


def laplacian_matrix(N: int, dx: float, use_gpu: bool = False):
    """Construct finite-difference Laplacian matrix (second derivative operator)."""
    if N < 3:
        raise ValueError("`N` must be at least 3")
    if dx <= 0:
        raise ValueError("`dx` must be positive")

    xp = get_array_module(use_gpu)
    main = -2.0 * xp.ones(N, dtype=xp.float64)
    off = xp.ones(N - 1, dtype=xp.float64)
    L = xp.diag(main) + xp.diag(off, 1) + xp.diag(off, -1)
    return L / (dx**2)


def second_derivative_matrix(N: int, dx: float, use_gpu: bool = False):
    return laplacian_matrix(N, dx, use_gpu=use_gpu)


def second_derivative(psi, dx: float, use_gpu: bool = False):
    """Second derivative using central finite differences (Dirichlet boundaries)."""
    xp = get_array_module(use_gpu)
    psi = xp.asarray(psi, dtype=xp.complex128)

    d2 = xp.zeros_like(psi)
    d2[1:-1] = (psi[2:] - 2.0 * psi[1:-1] + psi[:-2]) / (dx**2)
    d2[0] = 0.0
    d2[-1] = 0.0
    return d2


def apply_boundary_conditions(matrix, type: str):
    """Apply boundary conditions to a finite-difference operator matrix."""
    if cp is not None and isinstance(matrix, cp.ndarray):
        xp = cp
    else:
        xp = np

    A = xp.array(matrix, copy=True)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("`matrix` must be square")

    n = A.shape[0]
    if n < 2:
        raise ValueError("`matrix` size must be at least 2")

    bc = str(type).strip().lower()
    if bc == "dirichlet":
        A[0, :] = 0
        A[-1, :] = 0
        A[:, 0] = 0
        A[:, -1] = 0
        A[0, 0] = 1
        A[-1, -1] = 1
        return A

    if bc == "neumann":
        A[0, :] = 0
        A[-1, :] = 0
        A[0, 0] = -1
        A[0, 1] = 1
        A[-1, -2] = -1
        A[-1, -1] = 1
        return A

    if bc == "periodic":
        A[0, -1] = A[0, 1]
        A[-1, 0] = A[-1, -2]
        return A

    raise ValueError("Unsupported boundary condition type. Use 'dirichlet', 'neumann', or 'periodic'.")

__all__ = [
    "laplacian_matrix",
    "second_derivative_matrix",
    "second_derivative",
    "apply_boundary_conditions",
]
