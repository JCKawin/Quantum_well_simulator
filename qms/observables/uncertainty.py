from ..grid.operators import second_derivative
from ..utils.backend import get_array_module, to_numpy
from ..wavefunction.normalization import inner_product, normalize_wavefunction


def uncertainty_position(psi, x, dx: float, use_gpu: bool = False):
	if dx <= 0:
		raise ValueError("`dx` must be positive")
	xp = get_array_module(use_gpu)
	x_arr = xp.asarray(x, dtype=xp.float64)
	wf = xp.asarray(psi, dtype=xp.complex128)
	if wf.shape != x_arr.shape:
		raise ValueError("`psi` and `x` must have the same shape")

	wf_n = normalize_wavefunction(wf, dx=dx, use_gpu=use_gpu)
	x_mean = inner_product(wf_n, x_arr * wf_n, dx=dx, use_gpu=use_gpu)
	x2_mean = inner_product(wf_n, (x_arr**2) * wf_n, dx=dx, use_gpu=use_gpu)
	var_x = xp.real(x2_mean - x_mean * x_mean)
	var_x = xp.maximum(var_x, 0.0)
	return float(to_numpy(xp.sqrt(var_x)))


def uncertainty_momentum(psi, dx: float, hbar: float = 1.0, use_gpu: bool = False):
	if dx <= 0:
		raise ValueError("`dx` must be positive")
	if hbar <= 0:
		raise ValueError("`hbar` must be positive")

	xp = get_array_module(use_gpu)
	wf = xp.asarray(psi, dtype=xp.complex128)
	wf_n = normalize_wavefunction(wf, dx=dx, use_gpu=use_gpu)

	dpsi = xp.zeros_like(wf_n)
	dpsi[1:-1] = (wf_n[2:] - wf_n[:-2]) / (2.0 * dx)
	dpsi[0] = (wf_n[1] - wf_n[0]) / dx
	dpsi[-1] = (wf_n[-1] - wf_n[-2]) / dx
	p_psi = -1j * hbar * dpsi

	d2psi = second_derivative(wf_n, dx=dx, use_gpu=use_gpu)
	p2_psi = -(hbar**2) * d2psi

	p_mean = inner_product(wf_n, p_psi, dx=dx, use_gpu=use_gpu)
	p2_mean = inner_product(wf_n, p2_psi, dx=dx, use_gpu=use_gpu)

	var_p = xp.real(p2_mean - p_mean * p_mean)
	var_p = xp.maximum(var_p, 0.0)
	return float(to_numpy(xp.sqrt(var_p)))

__all__ = ["uncertainty_position", "uncertainty_momentum"]
