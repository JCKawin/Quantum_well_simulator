import matplotlib.pyplot as plt
import numpy as np

from ..utils.backend import to_numpy


def plot_wavefunction(x, psi):
    x_np = to_numpy(x)
    psi_np = to_numpy(psi)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_np, np.real(psi_np), label="Re(psi)")
    ax.plot(x_np, np.imag(psi_np), label="Im(psi)", linestyle="--")
    ax.set_xlabel("x")
    ax.set_ylabel("psi(x)")
    ax.set_title("Wavefunction")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_probability_density(x, psi):
    x_np = to_numpy(x)
    dens = np.abs(to_numpy(psi)) ** 2

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_np, dens, color="tab:purple", label="|psi|^2")
    ax.set_xlabel("x")
    ax.set_ylabel("Probability density")
    ax.set_title("Probability Density")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_potential(x, V):
    x_np = to_numpy(x)
    if callable(V):
        v_np = to_numpy(V(x_np))
    else:
        v_np = to_numpy(V)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_np, v_np, color="black", label="V(x)")
    ax.set_xlabel("x")
    ax.set_ylabel("V(x)")
    ax.set_title("Potential")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_energy_levels(energies):
    e = np.sort(to_numpy(energies).astype(float))

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, en in enumerate(e):
        ax.hlines(en, 0.0, 1.0, colors="tab:blue")
        ax.text(1.02, en, f"E{i}={en:.4g}", va="center", fontsize=9)
    ax.set_xlim(0.0, 1.2)
    ax.set_xticks([])
    ax.set_ylabel("Energy")
    ax.set_title("Energy Levels")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_eigenstates(x, eigenvectors, n_states: int = 3):
    x_np = to_numpy(x)
    evecs = to_numpy(eigenvectors)
    if evecs.ndim != 2:
        raise ValueError("`eigenvectors` must be 2D with states in columns")

    n_plot = max(1, min(int(n_states), evecs.shape[1]))
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(n_plot):
        ax.plot(x_np, evecs[:, i], label=f"n={i}")

    ax.set_xlabel("x")
    ax.set_ylabel("psi_n(x)")
    ax.set_title("Eigenstates")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax

__all__ = [
    "plot_wavefunction",
    "plot_probability_density",
    "plot_potential",
    "plot_energy_levels",
    "plot_eigenstates",
]
