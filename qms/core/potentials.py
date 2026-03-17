from ..utils.backend import get_array_module


def potential(
    x,
    well_width: float = 1.0,
    barrier_height: float = 1.0e6,
    kind: str = "finite_square",
    electric_field: float = 0.0,
    anharmonic: float = 0.0,
    use_gpu: bool = False,
):
    """Build 1D potential profile for a quantum well."""
    xp = get_array_module(use_gpu)
    x = xp.asarray(x, dtype=xp.float64)

    if kind == "finite_square":
        v = xp.where(xp.abs(x) <= (well_width / 2.0), 0.0, barrier_height)
    elif kind == "infinite_square":
        huge_barrier = xp.float64(1.0e12)
        v = xp.where(xp.abs(x) <= (well_width / 2.0), 0.0, huge_barrier)
    elif kind == "harmonic":
        v = 0.5 * (x / max(well_width, 1.0e-15)) ** 2 + anharmonic * x**4
    else:
        raise ValueError(f"Unsupported potential type: {kind}")

    if electric_field != 0.0:
        v = v + electric_field * x

    return v


def potential_infinite_well(
    x,
    L: float,
    barrier_height: float = 1e12,
    use_gpu: bool = False,
):
    if L <= 0:
        raise ValueError("`L` must be positive")
    xp = get_array_module(use_gpu)
    x = xp.asarray(x, dtype=xp.float64)
    return xp.where((x >= 0.0) & (x <= L), 0.0, barrier_height)


def potential_finite_well(x, V0: float, L: float, use_gpu: bool = False):
    if L <= 0:
        raise ValueError("`L` must be positive")
    xp = get_array_module(use_gpu)
    x = xp.asarray(x, dtype=xp.float64)
    return xp.where((x >= 0.0) & (x <= L), 0.0, float(V0))


def potential_custom(x, params, use_gpu: bool = False):
    xp = get_array_module(use_gpu)
    x = xp.asarray(x, dtype=xp.float64)

    if callable(params):
        v = params(x)
        return xp.asarray(v, dtype=xp.float64)

    if not isinstance(params, dict):
        raise ValueError("`params` must be a callable or dict")

    offset = float(params.get("offset", 0.0))
    linear = float(params.get("linear", 0.0))
    quadratic = float(params.get("quadratic", 0.0))
    quartic = float(params.get("quartic", 0.0))
    v = offset + linear * x + quadratic * x**2 + quartic * x**4
    return xp.asarray(v, dtype=xp.float64)

__all__ = [
    "potential",
    "potential_infinite_well",
    "potential_finite_well",
    "potential_custom",
]
