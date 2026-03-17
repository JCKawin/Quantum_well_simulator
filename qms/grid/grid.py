from ..utils.backend import get_array_module, to_numpy


def create_spatial_grid(
	x_min: float,
	x_max: float,
	N: int,
	use_gpu: bool = False,
):
	"""Create a 1D uniform spatial grid."""
	if N < 3:
		raise ValueError("`N` must be at least 3")
	if x_max <= x_min:
		raise ValueError("`x_max` must be greater than `x_min`")

	xp = get_array_module(use_gpu)
	return xp.linspace(x_min, x_max, int(N), dtype=xp.float64)


def dx_from_grid(x, use_gpu: bool = False) -> float:
	"""Compute grid spacing for a uniform grid."""
	xp = get_array_module(use_gpu)
	x = xp.asarray(x, dtype=xp.float64)
	if x.size < 2:
		raise ValueError("Grid must have at least 2 points")
	return float(to_numpy(x[1] - x[0]))


__all__ = ["create_spatial_grid", "dx_from_grid"]
