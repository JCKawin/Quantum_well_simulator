import mpmath
import numpy as np
import qutip

from ..utils.backend import get_array_module, to_numpy
from ..wavefunction.normalization import inner_product, normalize_wavefunction, probability_density


def expectation_value(operator, psi, dx: float, use_gpu: bool = False):
    if dx <= 0:
        raise ValueError("`dx` must be positive")
    xp = get_array_module(use_gpu)
    op = xp.asarray(operator, dtype=xp.complex128)
    wf = xp.asarray(psi, dtype=xp.complex128)
    if op.ndim != 2 or op.shape[0] != op.shape[1]:
        raise ValueError("`operator` must be a square matrix")
    if wf.ndim != 1 or wf.shape[0] != op.shape[0]:
        raise ValueError("`psi` must be a vector compatible with `operator`")

    wf_n = normalize_wavefunction(wf, dx=dx, use_gpu=use_gpu)
    exp_val = inner_product(wf_n, op @ wf_n, dx=dx, use_gpu=use_gpu)
    return float(to_numpy(xp.real(exp_val)))


def expectation_position(psi, x, dx: float, use_gpu: bool = False):
    if dx <= 0:
        raise ValueError("`dx` must be positive")
    xp = get_array_module(use_gpu)
    x_arr = xp.asarray(x, dtype=xp.float64)
    wf = xp.asarray(psi, dtype=xp.complex128)
    if wf.shape != x_arr.shape:
        raise ValueError("`psi` and `x` must have the same shape")

    wf_n = normalize_wavefunction(wf, dx=dx, use_gpu=use_gpu)
    return float(to_numpy(xp.real(inner_product(wf_n, x_arr * wf_n, dx=dx, use_gpu=use_gpu))))


def expectation_momentum(psi, dx: float, hbar: float = 1.0, use_gpu: bool = False):
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
    exp_p = inner_product(wf_n, p_psi, dx=dx, use_gpu=use_gpu)
    return float(to_numpy(xp.real(exp_p)))


def expectation_x(
    psi,
    x,
    dx: float | None = None,
    use_gpu: bool = False,
    high_precision: bool = False,
    mp_dps: int = 50,
):
    xp = get_array_module(use_gpu)
    psi = xp.asarray(psi, dtype=xp.complex128)
    x = xp.asarray(x, dtype=xp.float64)

    if dx is None:
        dx = float(to_numpy(x[1] - x[0]))

    if high_precision:
        mpmath.mp.dps = mp_dps
        psi_n = normalize_wavefunction(psi, dx=dx, use_gpu=use_gpu, high_precision=True, mp_dps=mp_dps)
        p = to_numpy(probability_density(psi_n, use_gpu=use_gpu))
        x_np = to_numpy(x)
        ex = mpmath.fsum(mpmath.mpf(str(pi)) * mpmath.mpf(str(xi)) for pi, xi in zip(p, x_np, strict=False))
        return float(ex * mpmath.mpf(str(dx)))

    psi_n = normalize_wavefunction(psi, dx=dx, use_gpu=use_gpu)
    return float(xp.real(xp.sum(xp.conj(psi_n) * x * psi_n) * dx))


def expectation_energy(
    psi,
    hamiltonian,
    x=None,
    dx: float | None = None,
    use_gpu: bool = False,
    use_qutip: bool = False,
):
    xp = get_array_module(use_gpu)
    psi = xp.asarray(psi, dtype=xp.complex128)
    h = xp.asarray(hamiltonian, dtype=xp.complex128)

    if dx is None:
        if x is None:
            raise ValueError("Provide either `x` or `dx`")
        x = xp.asarray(x, dtype=xp.float64)
        dx = float(to_numpy(x[1] - x[0]))

    psi_n = normalize_wavefunction(psi, dx=dx, use_gpu=use_gpu)

    if use_qutip:
        psi_q = qutip.Qobj(to_numpy(psi_n).reshape((-1, 1)))
        h_q = qutip.Qobj(to_numpy(h))
        return float(np.real(qutip.expect(h_q, psi_q)))

    e = xp.vdot(psi_n, h @ psi_n)
    return float(xp.real(e))

__all__ = [
    "expectation_value",
    "expectation_position",
    "expectation_momentum",
    "expectation_x",
    "expectation_energy",
]
