"""Smoke tests for the Quantum Well Simulator."""

import os

import numpy as np
import pytest

import qms
from qms.__main__ import build_parser, main
from qms.pipeline.simulation import run_full_pipeline, simulate_quantum_well, run_static_simulation


class TestPackageImports:
    def test_top_level_import(self):
        assert hasattr(qms, "QuantumSimulator1D")
        assert hasattr(qms, "finite_difference_solver")
        assert hasattr(qms, "run_full_pipeline")


class TestPipeline:
    def test_finite_well(self):
        result = simulate_quantum_well("finite", {
            "x_min": -5.0, "x_max": 5.0, "N": 200, "V0": 50.0, "well_width": 2.0,
            "num_states": 3,
        })
        assert "energies" in result
        energies = np.real(np.asarray(result["energies"]))
        assert len(energies) == 3
        # Energy levels should be positive and increasing
        assert energies[0] < energies[1] < energies[2]

    def test_harmonic_oscillator(self):
        result = simulate_quantum_well("harmonic", {
            "x_min": -5.0, "x_max": 5.0, "N": 200, "well_width": 1.0, "num_states": 4,
        })
        energies = np.real(np.asarray(result["energies"]))
        assert len(energies) == 4
        # Harmonic oscillator energy levels are approximately (n+0.5)*hbar*omega
        # Check they're approximately equally spaced
        diffs = np.diff(energies)
        assert np.allclose(diffs, diffs[0], rtol=0.1)

    def test_infinite_well(self):
        result = simulate_quantum_well("infinite", {
            "x_min": 0.0, "x_max": 1.0, "N": 200, "L": 1.0, "num_states": 3,
        })
        energies = np.real(np.asarray(result["energies"]))
        assert len(energies) == 3
        assert energies[0] > 0

    def test_run_full_pipeline_static_only(self):
        config = {
            "well_type": "finite",
            "params": {"x_min": -3.0, "x_max": 3.0, "N": 100, "V0": 20.0, "well_width": 1.0, "num_states": 2},
        }
        results = run_full_pipeline(config)
        assert "static" in results
        assert "time" not in results

    def test_run_full_pipeline_with_time_evolution(self):
        config = {
            "well_type": "finite",
            "params": {"x_min": -3.0, "x_max": 3.0, "N": 100, "V0": 20.0, "well_width": 1.0, "num_states": 2},
            "t_array": np.linspace(0, 1.0, 10),
        }
        results = run_full_pipeline(config)
        assert "static" in results
        assert "time" in results
        psi_t = results["time"]["psi_t"]
        assert psi_t.shape[1] == 10


class TestCLI:
    def test_cli_help(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_cli_finite_well(self, tmp_path):
        npz_file = str(tmp_path / "results")
        ret = main([
            "--well", "finite",
            "--n-points", "100",
            "--num-states", "3",
            "--save-npz", npz_file,
        ])
        assert ret == 0
        data = np.load(npz_file + ".npz")
        assert "energies" in data
        assert len(data["energies"]) == 3

    def test_cli_harmonic(self):
        ret = main([
            "--well", "harmonic",
            "--n-points", "100",
            "--num-states", "4",
            "--no-summary",
        ])
        assert ret == 0

    def test_cli_time_evolve(self, tmp_path):
        npz_file = str(tmp_path / "time_results")
        ret = main([
            "--well", "finite",
            "--n-points", "100",
            "--num-states", "2",
            "--time-evolve",
            "--t-max", "1.0",
            "--n-time-steps", "5",
            "--no-summary",
            "--save-npz", npz_file,
        ])
        assert ret == 0

    def test_cli_save_plot(self, tmp_path):
        plot_file = str(tmp_path / "eigenstates.png")
        ret = main([
            "--well", "finite",
            "--n-points", "100",
            "--num-states", "3",
            "--no-summary",
            "--save-plot", plot_file,
        ])
        assert ret == 0
        assert os.path.exists(plot_file)


class TestPhysics:
    """Verify physically meaningful results."""

    def test_infinite_well_energy_ratios(self):
        """For an infinite square well, E_n ~ n^2 * E_1."""
        result = simulate_quantum_well("infinite", {
            "x_min": 0.0, "x_max": 1.0, "N": 500, "L": 1.0, "num_states": 3,
        })
        energies = np.real(np.asarray(result["energies"]))
        # E_n = n^2 * E_1 for infinite square well
        # E_2 / E_1 ~ 4, E_3 / E_1 ~ 9
        assert abs(energies[1] / energies[0] - 4.0) < 0.1
        assert abs(energies[2] / energies[0] - 9.0) < 0.1

    def test_wavefunctions_normalized(self):
        """Wavefunctions must be normalized (integral of |psi|^2 dx = 1)."""
        x = np.linspace(-5, 5, 300)
        dx = x[1] - x[0]
        V = np.zeros_like(x)
        result = run_static_simulation(x=x, V=V, num_states=3)
        wavefunctions = np.asarray(result["wavefunctions"])
        for i in range(wavefunctions.shape[1]):
            norm = np.sum(np.abs(wavefunctions[:, i]) ** 2) * dx
            assert abs(norm - 1.0) < 1e-6, f"State {i} not normalized: norm={norm}"
