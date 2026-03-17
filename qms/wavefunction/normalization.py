import mpmath

from ..utils.backend import get_array_module, to_numpy


def normalize_wavefunction(
	psi,
	x=None,
	dx: float | None = None,
	use_gpu: bool = False,
	high_precision: bool = False,
	mp_dps: int = 50,
):
	xp = get_array_module(use_gpu)
	psi = xp.asarray(psi, dtype=xp.complex128)

	if dx is None:
		if x is None:
			raise ValueError("Provide either `x` or `dx`")
		x = xp.asarray(x, dtype=xp.float64)
		if x.size < 2:
			raise ValueError("`x` must contain at least 2 points")
		dx = float(to_numpy(x[1] - x[0]))

	prob = xp.abs(psi) ** 2
	if high_precision:
		mpmath.mp.dps = mp_dps
		prob_np = to_numpy(prob)
		norm2 = mpmath.fsum(mpmath.mpf(str(v)) for v in prob_np) * mpmath.mpf(str(dx))
		if norm2 <= 0:
			raise ValueError("Wavefunction norm is zero or negative")
		norm = float(mpmath.sqrt(norm2))
	else:
		norm = float(xp.sqrt(xp.sum(prob) * dx))
		if norm <= 0.0:
			raise ValueError("Wavefunction norm is zero or negative")
	return psi / norm


def probability_density(psi, use_gpu: bool = False):
	xp = get_array_module(use_gpu)
	psi = xp.asarray(psi, dtype=xp.complex128)
	return xp.abs(psi) ** 2


def inner_product(psi1, psi2, dx: float, use_gpu: bool = False):
	if dx <= 0:
		raise ValueError("`dx` must be positive")
	xp = get_array_module(use_gpu)
	a = xp.asarray(psi1, dtype=xp.complex128)
	b = xp.asarray(psi2, dtype=xp.complex128)
	if a.shape != b.shape:
		raise ValueError("`psi1` and `psi2` must have the same shape")
	return xp.vdot(a, b) * dx


def orthonormalize_wavefunctions(psi_set, dx: float, use_gpu: bool = False):
	if dx <= 0:
		raise ValueError("`dx` must be positive")
	xp = get_array_module(use_gpu)
	states = xp.asarray(psi_set, dtype=xp.complex128)
	if states.ndim != 2:
		raise ValueError("`psi_set` must be a 2D array with states as columns")

	n_points, n_states = states.shape
	ortho = xp.zeros((n_points, n_states), dtype=xp.complex128)

	for i in range(n_states):
		v = states[:, i].copy()
		for j in range(i):
			proj = inner_product(ortho[:, j], v, dx=dx, use_gpu=use_gpu)
			v = v - proj * ortho[:, j]
		norm_v = xp.sqrt(xp.real(inner_product(v, v, dx=dx, use_gpu=use_gpu)))
		if float(to_numpy(norm_v)) <= 1e-14:
			raise ValueError("Linearly dependent wavefunctions found during orthonormalization")
		ortho[:, i] = v / norm_v
	return ortho

__all__ = ["normalize_wavefunction", "probability_density", "inner_product", "orthonormalize_wavefunctions"]
