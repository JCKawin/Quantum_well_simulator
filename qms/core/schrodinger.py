import numpy as np
from scipy.linalg import eigh
from scipy.sparse import diags
from scipy.sparse.linalg import expm

from ..solvers.eigen import finite_difference_solver, schrodinger_time_independent
from ..solvers.time_evolution import schrodinger_time_dependent


class QuantumSimulator1D:
    """Core 1D quantum simulator."""

    def __init__(self, x_min, x_max, N, mass=1.0, hbar=1.0):
        self.x_min = x_min
        self.x_max = x_max
        self.N = N
        self.mass = mass
        self.hbar = hbar

        self.x = np.linspace(x_min, x_max, N)
        self.dx = self.x[1] - self.x[0]
        self.V = np.zeros(N)
        self.H = None

    def set_potential(self, V_func):
        self.V = V_func(self.x)

    def build_hamiltonian(self):
        coeff = -(self.hbar**2) / (2 * self.mass * self.dx**2)
        diagonal = np.full(self.N, -2.0)
        off_diagonal = np.ones(self.N - 1)
        kinetic = diags([off_diagonal, diagonal, off_diagonal], offsets=[-1, 0, 1]).toarray()
        kinetic *= coeff
        self.H = kinetic + np.diag(self.V)

    def solve_eigen(self, num_states=5):
        energies, states = eigh(self.H)
        return energies[:num_states], states[:, :num_states]

    def normalize(self, psi):
        norm = np.sqrt(np.sum(np.abs(psi) ** 2) * self.dx)
        return psi / norm

    def time_evolve(self, psi0, t):
        U = expm(-1j * self.H * t / self.hbar)
        return U @ psi0

    def expectation_x(self, psi):
        return np.sum(np.conj(psi) * self.x * psi) * self.dx

    def expectation_energy(self, psi):
        return np.real(np.conj(psi) @ (self.H @ psi))

    def set_infinite_well(self):
        self.V = np.zeros(self.N)
        self.V[0] = self.V[-1] = 1e10


class InfiniteWell(QuantumSimulator1D):
    def __init__(self, L, N, mass=1.0, hbar=1.0, barrier_height=1e10):
        super().__init__(0.0, float(L), int(N), mass=mass, hbar=hbar)
        self.L = float(L)
        self.barrier_height = float(barrier_height)
        self.set_potential_profile()
        self.build_hamiltonian()

    def set_potential_profile(self):
        self.V = np.zeros(self.N)
        self.V[0] = self.barrier_height
        self.V[-1] = self.barrier_height


class FiniteWell(QuantumSimulator1D):
    def __init__(self, x_min, x_max, N, V0=50.0, well_width=1.0, center=0.0, mass=1.0, hbar=1.0):
        super().__init__(x_min, x_max, N, mass=mass, hbar=hbar)
        self.V0 = float(V0)
        self.well_width = float(well_width)
        self.center = float(center)
        self.set_potential_profile()
        self.build_hamiltonian()

    def set_potential_profile(self):
        left = self.center - self.well_width / 2.0
        right = self.center + self.well_width / 2.0
        self.V = np.where((self.x >= left) & (self.x <= right), 0.0, self.V0)


class HarmonicOscillator(QuantumSimulator1D):
    def __init__(self, x_min, x_max, N, omega=1.0, x0=0.0, mass=1.0, hbar=1.0):
        super().__init__(x_min, x_max, N, mass=mass, hbar=hbar)
        self.omega = float(omega)
        self.x0 = float(x0)
        self.set_potential_profile()
        self.build_hamiltonian()

    def set_potential_profile(self):
        self.V = 0.5 * self.mass * (self.omega**2) * (self.x - self.x0) ** 2


class Barrier(QuantumSimulator1D):
    def __init__(
        self,
        x_min,
        x_max,
        N,
        barrier_height=100.0,
        barrier_width=0.2,
        center=0.0,
        mass=1.0,
        hbar=1.0,
    ):
        super().__init__(x_min, x_max, N, mass=mass, hbar=hbar)
        self.barrier_height = float(barrier_height)
        self.barrier_width = float(barrier_width)
        self.center = float(center)
        self.set_potential_profile()
        self.build_hamiltonian()

    def set_potential_profile(self):
        left = self.center - self.barrier_width / 2.0
        right = self.center + self.barrier_width / 2.0
        self.V = np.zeros(self.N)
        self.V[(self.x >= left) & (self.x <= right)] = self.barrier_height


class DoubleWell(QuantumSimulator1D):
    def __init__(self, x_min, x_max, N, a=1.0, b=5.0, x0=0.0, mass=1.0, hbar=1.0):
        super().__init__(x_min, x_max, N, mass=mass, hbar=hbar)
        self.a = float(a)
        self.b = float(b)
        self.x0 = float(x0)
        self.set_potential_profile()
        self.build_hamiltonian()

    def set_potential_profile(self):
        y = self.x - self.x0
        self.V = self.a * y**4 - self.b * y**2


class CustomPotential(QuantumSimulator1D):
    def __init__(self, x_min, x_max, N, V_func, mass=1.0, hbar=1.0):
        super().__init__(x_min, x_max, N, mass=mass, hbar=hbar)
        self.V_func = V_func
        self.set_potential_profile()
        self.build_hamiltonian()

    def set_potential_profile(self):
        self.V = np.asarray(self.V_func(self.x), dtype=float)
        if self.V.shape != self.x.shape:
            raise ValueError("Custom potential function must return array with same shape as x")

__all__ = [
    "QuantumSimulator1D",
    "InfiniteWell",
    "FiniteWell",
    "HarmonicOscillator",
    "Barrier",
    "DoubleWell",
    "CustomPotential",
    "schrodinger_time_independent",
    "schrodinger_time_dependent",
    "finite_difference_solver",
]
