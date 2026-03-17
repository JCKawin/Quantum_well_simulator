import scipy.linalg as la

from ..core.hamiltonian import build_hamiltonian
from ..core.potentials import potential
from ..utils.backend import cp, get_array_module, to_numpy
from ..wavefunction.normalization import normalize_wavefunction


def sort_eigenpairs(eigenvalues, eigenvectors, use_gpu: bool = False):
	xp = get_array_module(use_gpu)
	vals = xp.asarray(eigenvalues)
	vecs = xp.asarray(eigenvectors)
	if vecs.ndim != 2:
		raise ValueError("`eigenvectors` must be a 2D array with eigenvectors in columns")
	idx = xp.argsort(vals)
	return vals[idx], vecs[:, idx]


def solve_eigenvalue_problem(H, num_states: int | None = None, use_gpu: bool = False):
	xp = get_array_module(use_gpu)
	h = xp.asarray(H, dtype=xp.float64)
	if h.ndim != 2 or h.shape[0] != h.shape[1]:
		raise ValueError("`H` must be a square matrix")

	if xp.__name__ == "numpy":
		eigenvalues, eigenvectors = la.eigh(h)
	else:
		eigenvalues, eigenvectors = cp.linalg.eigh(h)

	eigenvalues, eigenvectors = sort_eigenpairs(eigenvalues, eigenvectors, use_gpu=use_gpu)
	if num_states is not None:
		k = int(max(1, min(num_states, h.shape[0])))
		eigenvalues = eigenvalues[:k]
		eigenvectors = eigenvectors[:, :k]
	return eigenvalues, eigenvectors


def compute_energy_levels(eigenvalues, use_gpu: bool = False):
	xp = get_array_module(use_gpu)
	e = xp.asarray(eigenvalues, dtype=xp.float64)
	return xp.sort(e)


def schrodinger_time_independent(H, use_gpu: bool = False):
	return solve_eigenvalue_problem(H, num_states=None, use_gpu=use_gpu)


def finite_difference_solver(
	x,
	num_states: int = 5,
	mass: float = 1.0,
	hbar: float = 1.0,
	potential_values=None,
	potential_kind: str = "finite_square",
	well_width: float = 1.0,
	barrier_height: float = 1.0e6,
	use_gpu: bool = False,
	return_qutip: bool = False,
):
	xp = get_array_module(use_gpu)
	x = xp.asarray(x, dtype=xp.float64)
	n = x.size
	if n < 3:
		raise ValueError("Grid must contain at least 3 points")

	if potential_values is None:
		V = potential(
			x,
			well_width=well_width,
			barrier_height=barrier_height,
			kind=potential_kind,
			use_gpu=use_gpu,
		)
	else:
		V = potential_values

	h = build_hamiltonian(x=x, V=V, mass=mass, hbar=hbar, use_gpu=use_gpu)
	eigvals, eigvecs = solve_eigenvalue_problem(h, num_states=num_states, use_gpu=use_gpu)

	dx = float(to_numpy(x[1] - x[0]))
	for i in range(eigvecs.shape[1]):
		eigvecs[:, i] = normalize_wavefunction(eigvecs[:, i], dx=dx, use_gpu=use_gpu)

	result = {
		"x": x,
		"hamiltonian": h,
		"energies": eigvals,
		"wavefunctions": eigvecs,
		"probability_densities": xp.abs(eigvecs) ** 2,
	}

	if return_qutip:
		import qutip

		result["qutip_hamiltonian"] = qutip.Qobj(to_numpy(h))
		result["qutip_states"] = [
			qutip.Qobj(to_numpy(eigvecs[:, i]).reshape((-1, 1))) for i in range(eigvecs.shape[1])
		]
	return result

__all__ = [
	"solve_eigenvalue_problem",
	"sort_eigenpairs",
	"compute_energy_levels",
	"schrodinger_time_independent",
	"finite_difference_solver",
]
