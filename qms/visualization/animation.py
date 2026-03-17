import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from ..utils.backend import to_numpy


def animate_time_evolution(x, psi_t_series):
	"""Animate probability density over time.

	`psi_t_series` can be shape (N, T) or (T, N).
	"""
	x_np = to_numpy(x).astype(float)
	psi_np = to_numpy(psi_t_series)
	if psi_np.ndim != 2:
		raise ValueError("`psi_t_series` must be 2D")

	if psi_np.shape[0] == x_np.size:
		psi_nt = psi_np
	elif psi_np.shape[1] == x_np.size:
		psi_nt = psi_np.T
	else:
		raise ValueError("`psi_t_series` must have one axis equal to len(x)")

	density = np.abs(psi_nt) ** 2
	ymin = float(np.min(density))
	ymax = float(np.max(density)) * 1.05 if density.size else 1.0
	if ymax <= ymin:
		ymax = ymin + 1.0

	fig, ax = plt.subplots(figsize=(9, 5))
	(line,) = ax.plot(x_np, density[:, 0], color="tab:red")
	ax.set_xlim(float(x_np.min()), float(x_np.max()))
	ax.set_ylim(ymin, ymax)
	ax.set_xlabel("x")
	ax.set_ylabel("|psi(x,t)|^2")
	ax.set_title("Time Evolution")
	ax.grid(alpha=0.3)

	def _update(frame_idx):
		line.set_ydata(density[:, frame_idx])
		ax.set_title(f"Time Evolution (frame {frame_idx})")
		return (line,)

	anim = FuncAnimation(fig, _update, frames=density.shape[1], interval=50, blit=True)
	fig.tight_layout()
	return fig, ax, anim

__all__ = ["animate_time_evolution"]
