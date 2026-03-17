from ..utils.backend import get_array_module, to_numpy
from .normalization import normalize_wavefunction


def initialize_gaussian_wavepacket(
	x,
	x0: float,
	k0: float,
	sigma: float,
	use_gpu: bool = False,
):
	if sigma <= 0:
		raise ValueError("`sigma` must be positive")

	xp = get_array_module(use_gpu)
	x = xp.asarray(x, dtype=xp.float64)
	if x.size < 2:
		raise ValueError("`x` must contain at least 2 grid points")

	envelope = xp.exp(-((x - x0) ** 2) / (4.0 * sigma**2))
	phase = xp.exp(1j * k0 * x)
	psi = envelope * phase

	dx = float(to_numpy(x[1] - x[0]))
	return normalize_wavefunction(psi, dx=dx, use_gpu=use_gpu)


def superpose_states(states, coefficients, use_gpu: bool = False):
	xp = get_array_module(use_gpu)

	coeffs = xp.asarray(coefficients, dtype=xp.complex128)
	if coeffs.ndim != 1:
		raise ValueError("`coefficients` must be a 1D array")

	if isinstance(states, (list, tuple)):
		if len(states) == 0:
			raise ValueError("`states` cannot be empty")
		mat = xp.column_stack([xp.asarray(s, dtype=xp.complex128) for s in states])
	else:
		mat = xp.asarray(states, dtype=xp.complex128)
		if mat.ndim != 2:
			raise ValueError("`states` must be 2D (columns are states) or a list of states")

	if mat.shape[1] != coeffs.shape[0]:
		raise ValueError("Number of states must match number of coefficients")
	return mat @ coeffs

__all__ = ["initialize_gaussian_wavepacket", "superpose_states"]
