# Quantum Well Simulator

A Python package for solving the 1D time-independent and time-dependent Schrödinger equation for a variety of quantum well potentials.

## Features

- **Multiple potential types**: infinite square well, finite square well, harmonic oscillator, double well, tunnelling barrier, and custom potentials
- **Eigenstate solver**: finite-difference discretisation via `scipy.linalg.eigh`
- **Time evolution**: spectral decomposition method for evolving arbitrary wavefunctions
- **Observables**: expectation values of position, momentum, energy; Heisenberg uncertainty
- **Visualisation**: matplotlib plots of eigenstates, probability densities, energy levels, and animated time evolution
- **GPU support** (optional): CuPy backend for large grids
- **CLI**: `python -m qms` / `qms-simulate` console script

---

## Installation

```bash
# Clone the repository
git clone https://github.com/JCKawin/Quantum_well_simulator.git
cd Quantum_well_simulator

# Install the core package (Python >= 3.10 required)
pip install -e .

# Optional: install extended dependencies (QuTiP, sympy, physipy)
pip install -e ".[full]"

# Optional: GPU support (CUDA 12)
pip install -e ".[gpu]"
```

---

## Running the simulation

### Command-line interface

```bash
# Finite square well (default)
python -m qms

# Harmonic oscillator – 6 energy levels, save plot and data
python -m qms --well harmonic --num-states 6 --save-plot harmonic.png --save-npz harmonic.npz

# Infinite square well
python -m qms --well infinite --well-width 1.0 --num-states 5

# Double-well potential
python -m qms --well double_well --x-min -4 --x-max 4 --num-states 4

# Tunnelling barrier
python -m qms --well barrier --well-depth 100 --num-states 3

# Include time evolution of the ground state
python -m qms --well finite --time-evolve --t-max 10 --n-time-steps 200 --save-npz results.npz
```

### Full option reference

```
usage: qms-simulate [-h]
       [--well {infinite,finite,harmonic,double_well,barrier}]
       [--x-min X_MIN] [--x-max X_MAX] [--n-points N_POINTS]
       [--well-width WELL_WIDTH] [--well-depth WELL_DEPTH] [--omega OMEGA]
       [--mass MASS] [--hbar HBAR] [--num-states NUM_STATES]
       [--time-evolve] [--t-max T_MAX] [--n-time-steps N_TIME_STEPS]
       [--save-npz FILE] [--save-plot FILE] [--no-summary]
```

Run `python -m qms --help` for the full description of every option.

### Python API

```python
import numpy as np
from qms.pipeline.simulation import run_full_pipeline

config = {
    "well_type": "finite",
    "params": {
        "x_min": -5.0, "x_max": 5.0, "N": 500,
        "V0": 50.0, "well_width": 2.0, "num_states": 5,
    },
    "t_array": np.linspace(0, 10, 200),   # optional time evolution
}

results = run_full_pipeline(config)

energies = results["static"]["energies"]
print("Energy levels:", energies)

psi_t = results["time"]["psi_t"]   # shape (N, T)
```

---

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Project structure

```
qms/
├── core/          – potentials, Hamiltonian, Schrödinger classes
├── grid/          – spatial grid & finite-difference operators
├── solvers/       – eigenvalue solver & time evolution
├── wavefunction/  – normalisation, Gaussian wavepackets, superposition
├── observables/   – expectation values & Heisenberg uncertainty
├── visualization/ – matplotlib plots & animations
├── pipeline/      – high-level run_full_pipeline / simulate_quantum_well
├── utils/         – NumPy/CuPy backend, I/O, validation
└── __main__.py    – CLI entry point (python -m qms)
```

---

## License

MIT – see [LICENSE](LICENSE).

