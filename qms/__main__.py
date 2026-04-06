"""
qms.__main__ - Command-line entry point for the Quantum Well Simulator.

Usage:
    python -m qms [options]
    qms-simulate [options]
"""

from __future__ import annotations

import argparse
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend so it works without a display

import matplotlib.pyplot as plt
import numpy as np

from qms.pipeline.simulation import run_full_pipeline, simulate_quantum_well
from qms.utils.io import save_npz_simulation
from qms.visualization.plots import plot_eigenstates, plot_energy_levels, plot_potential


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qms-simulate",
        description="Quantum Well Simulator - solve the 1D Schrödinger equation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Well type
    parser.add_argument(
        "--well",
        choices=["infinite", "finite", "harmonic", "double_well", "barrier"],
        default="finite",
        help="Type of quantum well / potential",
    )

    # Grid parameters
    parser.add_argument("--x-min", type=float, default=-5.0, help="Left boundary of spatial grid")
    parser.add_argument("--x-max", type=float, default=5.0, help="Right boundary of spatial grid")
    parser.add_argument("--n-points", type=int, default=500, help="Number of grid points")

    # Well parameters
    parser.add_argument("--well-width", type=float, default=2.0, help="Width of the well (L)")
    parser.add_argument("--well-depth", type=float, default=50.0, help="Potential barrier height V0 (finite well)")
    parser.add_argument("--omega", type=float, default=1.0, help="Angular frequency (harmonic oscillator)")

    # Physics
    parser.add_argument("--mass", type=float, default=1.0, help="Particle mass (atomic units)")
    parser.add_argument("--hbar", type=float, default=1.0, help="Reduced Planck constant (atomic units)")
    parser.add_argument("--num-states", type=int, default=5, help="Number of eigenstates to compute")

    # Time evolution
    parser.add_argument("--time-evolve", action="store_true", help="Run time evolution of the ground state")
    parser.add_argument("--t-max", type=float, default=10.0, help="Maximum simulation time")
    parser.add_argument("--n-time-steps", type=int, default=100, help="Number of time steps")

    # Output
    parser.add_argument("--save-npz", type=str, default=None, metavar="FILE",
                        help="Save simulation data to a .npz file (e.g. results.npz)")
    parser.add_argument("--save-plot", type=str, default=None, metavar="FILE",
                        help="Save eigenstates plot to an image file (e.g. eigenstates.png)")
    parser.add_argument("--no-summary", action="store_true", help="Suppress printed summary")

    return parser


def run(args: argparse.Namespace) -> dict:
    """Execute the simulation and return the results dict."""
    params: dict = {
        "x_min": args.x_min,
        "x_max": args.x_max,
        "N": args.n_points,
        "well_width": args.well_width,
        "V0": args.well_depth,
        "barrier_height": args.well_depth,
        "mass": args.mass,
        "hbar": args.hbar,
        "num_states": args.num_states,
    }

    well_type = args.well

    if well_type == "harmonic":
        params["omega"] = args.omega

    if well_type == "infinite":
        # Infinite well uses [0, L]; override grid to match
        params["x_min"] = 0.0
        params["x_max"] = args.well_width
        params["L"] = args.well_width

    if well_type == "double_well":
        well_type = "custom"
        a, b = 1.0, 5.0
        params["V"] = lambda x: a * x**4 - b * x**2

    if well_type == "barrier":
        well_type = "custom"
        bw = args.well_width / 4.0
        bh = args.well_depth
        params["V"] = lambda x, _bw=bw, _bh=bh: np.where(np.abs(x) <= _bw, _bh, 0.0)

    config: dict = {
        "well_type": well_type,
        "params": params,
        "num_states": args.num_states,
        "hbar": args.hbar,
        "mass": args.mass,
    }

    if args.time_evolve:
        t_array = np.linspace(0.0, args.t_max, args.n_time_steps)
        config["t_array"] = t_array

    results = run_full_pipeline(config)
    return results


def print_summary(results: dict, args: argparse.Namespace) -> None:
    """Print a human-readable summary of simulation results."""
    static = results["static"]
    energies = np.real(np.asarray(static["energies"]))

    print()
    print("=" * 60)
    print("  Quantum Well Simulator - Results Summary")
    print("=" * 60)
    print(f"  Well type     : {args.well}")
    print(f"  Grid          : [{args.x_min}, {args.x_max}], N={args.n_points}")
    print(f"  Mass / hbar   : {args.mass} / {args.hbar}")
    print(f"  States solved : {len(energies)}")
    print()
    print("  Energy levels (lowest first):")
    for i, e in enumerate(energies):
        print(f"    E_{i} = {e:.6f}")
    print("=" * 60)
    if "time" in results:
        print(f"  Time evolution: t in [0, {args.t_max}], {args.n_time_steps} steps")
    print()


def save_outputs(results: dict, args: argparse.Namespace) -> None:
    """Optionally save NPZ data and/or a plot."""
    static = results["static"]

    if args.save_npz:
        path = args.save_npz
        if not path.endswith(".npz"):
            path += ".npz"
        x_np = np.asarray(static["x"])
        energies_np = np.real(np.asarray(static["energies"]))
        wavefunctions_np = np.asarray(static["wavefunctions"])
        potential_np = np.asarray(static["potential"]) if "potential" in static else np.zeros_like(x_np)
        save_npz_simulation(
            path,
            x=x_np,
            energies=energies_np,
            wavefunctions=np.real(wavefunctions_np),
            potential=potential_np,
        )
        print(f"  Data saved to: {path}")

    if args.save_plot:
        x_np = np.asarray(static["x"])
        energies_np = np.real(np.asarray(static["energies"]))
        wavefunctions_np = np.real(np.asarray(static["wavefunctions"]))
        potential_np = np.asarray(static.get("potential", np.zeros_like(x_np)))

        n_states = wavefunctions_np.shape[1] if wavefunctions_np.ndim == 2 else 1
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: potential + eigenstates
        ax = axes[0]
        v_plot = np.clip(potential_np, None, energies_np.max() * 2 + 1.0)
        ax.plot(x_np, v_plot, "k-", linewidth=2, label="V(x)")
        for i in range(n_states):
            psi = wavefunctions_np[:, i] if wavefunctions_np.ndim == 2 else wavefunctions_np
            scale = (energies_np[i] if i < len(energies_np) else 1.0)
            offset = energies_np[i] if i < len(energies_np) else 0.0
            amplitude = max(abs(psi).max(), 1e-12)
            energy_spacing = abs(energies_np[1] - energies_np[0]) if len(energies_np) > 1 else 1.0
            ax.plot(x_np, psi / amplitude * energy_spacing * 0.4 + offset,
                    label=f"ψ_{i}")
            ax.axhline(offset, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("Energy / ψ(x)")
        ax.set_title("Eigenstates")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Right: energy levels
        ax2 = axes[1]
        for i, e in enumerate(energies_np):
            ax2.hlines(e, 0, 1, colors=f"C{i}", linewidth=2)
            ax2.text(1.05, e, f"E_{i}={e:.4g}", va="center", fontsize=9)
        ax2.set_xlim(0, 1.3)
        ax2.set_xticks([])
        ax2.set_ylabel("Energy")
        ax2.set_title("Energy Levels")
        ax2.grid(axis="y", alpha=0.3)

        fig.suptitle(f"Quantum Well Simulation ({args.well})", fontsize=13)
        fig.tight_layout()
        fig.savefig(args.save_plot, dpi=150)
        plt.close(fig)
        print(f"  Plot saved to: {args.save_plot}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        results = run(args)
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not args.no_summary:
        print_summary(results, args)

    save_outputs(results, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
