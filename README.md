# Quantum_well_simulator
Quantum_well_simulator

## Real-Time Terminal UI (TUI)

This project now includes a real-time terminal dashboard for exploring quantum-well dynamics.

### Run the TUI

From the project root:

```bash
uv sync
uv run main.py
```

### Start with custom parameters

```bash
uv run main.py --well-type harmonic --well-width 2.2 --points 600 --num-states 8 --state 1 --t-max 24 --fps 30
```

### What the TUI shows

- Potential profile $V(x)$ as an ASCII real-time chart
- Time-dependent probability density $|\psi(x,t)|^2$
- Current state index and energy level
- Live simulation status and update rate

### Keyboard controls

- `space`: pause/resume
- `q`: quit
- `r`: recompute simulation with current parameters
- `n` / `p`: next or previous eigenstate
- `1` / `2` / `3`: switch well type (infinite, finite, harmonic)
- `w` / `s`: increase/decrease well width
- `b` / `v`: increase/decrease barrier height
- `t` / `g`: increase/decrease simulation time window
- `+` / `-`: increase/decrease animation FPS

### CLI arguments

- `--well-type {infinite,finite_well,harmonic}`
- `--x-min`, `--x-max`, `--points`
- `--well-width`, `--barrier-height`, `--anharmonic`
- `--mass`, `--hbar`
- `--num-states`, `--state`
- `--t-max`, `--time-steps`, `--fps`
- `--use-gpu`
