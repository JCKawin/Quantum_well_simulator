from ..grid import dx_from_grid, laplacian_matrix
from ..utils.backend import get_array_module


def potential_energy_operator(V, use_gpu: bool = False):
    xp = get_array_module(use_gpu)
    v = xp.asarray(V, dtype=xp.float64)
    return xp.diag(v)


def kinetic_energy_operator(x, mass: float = 1.0, hbar: float = 1.0, use_gpu: bool = False):
    if mass <= 0:
        raise ValueError("`mass` must be positive")
    if hbar <= 0:
        raise ValueError("`hbar` must be positive")

    xp = get_array_module(use_gpu)
    x = xp.asarray(x, dtype=xp.float64)
    dx = dx_from_grid(x, use_gpu=use_gpu)
    N = int(x.size)
    lap = laplacian_matrix(N, dx, use_gpu=use_gpu)
    return -(hbar**2) / (2.0 * mass) * lap


def build_hamiltonian(x, V, mass: float = 1.0, hbar: float = 1.0, use_gpu: bool = False):
    xp = get_array_module(use_gpu)
    x = xp.asarray(x, dtype=xp.float64)
    T = kinetic_energy_operator(x, mass=mass, hbar=hbar, use_gpu=use_gpu)
    Vx = V(x) if callable(V) else xp.asarray(V, dtype=xp.float64)
    if Vx.shape != x.shape:
        raise ValueError("Potential values must have the same shape as `x`")
    U = potential_energy_operator(Vx, use_gpu=use_gpu)
    return T + U


def hamiltonian_operator(x, V, mass: float = 1.0, hbar: float = 1.0, use_gpu: bool = False):
    return build_hamiltonian(x=x, V=V, mass=mass, hbar=hbar, use_gpu=use_gpu)

__all__ = [
    "build_hamiltonian",
    "hamiltonian_operator",
    "kinetic_energy_operator",
    "potential_energy_operator",
]
