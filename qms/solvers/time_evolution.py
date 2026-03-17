import scipy.linalg as la

from ..solvers.eigen import solve_eigenvalue_problem
from ..utils.backend import cp, get_array_module


def time_evolve_wavefunction(
	psi0,
	eigenvalues,
	eigenvectors,
	t,
	hbar: float = 1.0,
	use_gpu: bool = False,
):
	if hbar <= 0:
		raise ValueError("`hbar` must be positive")

	xp = get_array_module(use_gpu)
	psi0 = xp.asarray(psi0, dtype=xp.complex128)
	evals = xp.asarray(eigenvalues, dtype=xp.float64)
	evecs = xp.asarray(eigenvectors, dtype=xp.complex128)

	if evecs.ndim != 2:
		raise ValueError("`eigenvectors` must be a 2D matrix (columns are eigenstates)")
	if psi0.ndim != 1 or psi0.shape[0] != evecs.shape[0]:
		raise ValueError("`psi0` must be a vector compatible with `eigenvectors`")
	if evals.ndim != 1 or evals.shape[0] != evecs.shape[1]:
		raise ValueError("`eigenvalues` size must match number of eigenvector columns")

	coeffs = evecs.conj().T @ psi0
	t_arr = xp.asarray(t, dtype=xp.float64)

	if t_arr.ndim == 0:
		phases = xp.exp((-1j * evals * t_arr) / hbar)
		return evecs @ (coeffs * phases)

	phases = xp.exp((-1j * evals[:, None] * t_arr[None, :]) / hbar)
	weighted = coeffs[:, None] * phases
	return evecs @ weighted


def apply_operator(operator, psi, use_gpu: bool = False):
	xp = get_array_module(use_gpu)
	op = xp.asarray(operator, dtype=xp.complex128)
	wf = xp.asarray(psi, dtype=xp.complex128)

	if op.ndim != 2 or op.shape[0] != op.shape[1]:
		raise ValueError("`operator` must be a square matrix")
	if wf.ndim != 1 or wf.shape[0] != op.shape[1]:
		raise ValueError("`psi` must be a vector compatible with `operator`")
	return op @ wf


def schrodinger_time_dependent(psi, H, dt: float, hbar: float = 1.0, use_gpu: bool = False):
	if hbar <= 0:
		raise ValueError("`hbar` must be positive")

	xp = get_array_module(use_gpu)
	psi = xp.asarray(psi, dtype=xp.complex128)
	H = xp.asarray(H, dtype=xp.float64)

	if H.ndim != 2 or H.shape[0] != H.shape[1]:
		raise ValueError("`H` must be a square matrix")
	if psi.ndim != 1 or psi.shape[0] != H.shape[0]:
		raise ValueError("`psi` must be a vector compatible with `H`")

	if xp.__name__ == "numpy":
		U = la.expm((-1j * dt / hbar) * H)
		return U @ psi

	evals, evecs = cp.linalg.eigh(H)
	phases = cp.exp((-1j * dt / hbar) * evals)
	coeffs = evecs.conj().T @ psi
	return evecs @ (phases * coeffs)


def run_time_simulation(
	psi0,
	H,
	t_array,
	hbar: float = 1.0,
	use_gpu: bool = False,
):
	if hbar <= 0:
		raise ValueError("`hbar` must be positive")

	xp = get_array_module(use_gpu)
	psi0 = xp.asarray(psi0, dtype=xp.complex128)
	t_arr = xp.asarray(t_array, dtype=xp.float64)
	if t_arr.ndim != 1:
		raise ValueError("`t_array` must be 1D")

	energies, states = solve_eigenvalue_problem(H, use_gpu=use_gpu)
	psi_t = time_evolve_wavefunction(
		psi0=psi0,
		eigenvalues=energies,
		eigenvectors=states,
		t=t_arr,
		hbar=hbar,
		use_gpu=use_gpu,
	)

	return {
		"t": t_arr,
		"psi_t": psi_t,
		"probability_densities": xp.abs(psi_t) ** 2,
		"energies": energies,
	}

__all__ = [
	"time_evolve_wavefunction",
	"run_time_simulation",
	"apply_operator",
	"schrodinger_time_dependent",
]
