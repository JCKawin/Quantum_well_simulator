from __future__ import annotations

import asyncio
import argparse
import logging
import os
import time
import tracemalloc
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from rich import box
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..pipeline.simulation import simulate_quantum_well
from ..solvers.time_evolution import run_time_simulation
from ..utils.backend import cp, to_numpy
from ..utils.logging import get_logger

try:
    import msvcrt
except Exception:  # pragma: no cover
    msvcrt = None

logger = get_logger(__name__)


def _to_float(v: float, minimum: float = 0.0) -> float:
    return max(minimum, float(v))


class TUILogHandler(logging.Handler):
    def __init__(self, sink: deque[str], max_lines: int = 300):
        super().__init__()
        self.sink = sink
        self.max_lines = max_lines

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.sink.append(msg)
            while len(self.sink) > self.max_lines:
                self.sink.popleft()
        except Exception:
            return


@dataclass
class RuntimeStats:
    started_wall: float = field(default_factory=time.perf_counter)
    started_cpu: float = field(default_factory=time.process_time)
    last_wall: float = field(default_factory=time.perf_counter)
    last_cpu: float = field(default_factory=time.process_time)
    cpu_percent: float = 0.0
    last_recompute_s: float = 0.0
    recompute_count: int = 0
    recompute_total_s: float = 0.0
    last_frame_t: float = 0.0
    cpu_cores: int = field(default_factory=lambda: max(1, int(os.cpu_count() or 1)))
    _last_cpu_sample_wall: float = field(default_factory=time.perf_counter)
    _last_cpu_sample_cpu: float = field(default_factory=time.process_time)

    def update_cpu(self) -> None:
        now_wall = time.perf_counter()
        now_cpu = time.process_time()

        # Sample less frequently and normalize by logical CPU count to avoid
        # misleading "always 100%" readings for a single busy core.
        if (now_wall - self._last_cpu_sample_wall) < 0.75:
            self.last_wall = now_wall
            self.last_cpu = now_cpu
            return

        dw = max(1.0e-9, now_wall - self._last_cpu_sample_wall)
        dc = max(0.0, now_cpu - self._last_cpu_sample_cpu)
        pct = ((dc / dw) * 100.0) / float(self.cpu_cores)
        self.cpu_percent = max(0.0, min(100.0, pct))

        self._last_cpu_sample_wall = now_wall
        self._last_cpu_sample_cpu = now_cpu
        self.last_wall = now_wall
        self.last_cpu = now_cpu

    @property
    def uptime_s(self) -> float:
        return max(0.0, time.perf_counter() - self.started_wall)

    @property
    def avg_recompute_s(self) -> float:
        if self.recompute_count <= 0:
            return 0.0
        return self.recompute_total_s / self.recompute_count


@dataclass
class SimulationGUI:
    fig: plt.Figure | None = None
    _last_t: float = 0.0

    def _is_open(self) -> bool:
        return self.fig is not None and plt.fignum_exists(self.fig.number)

    def close_window(self) -> None:
        if self._is_open():
            plt.close(self.fig)
        self.fig = None

    def render_snapshot(
        self,
        x: np.ndarray,
        potential: np.ndarray,
        state: np.ndarray,
        energies: np.ndarray,
        state_index: int,
        well_type: str,
        t: np.ndarray,
        x_expectation: np.ndarray,
        norms: np.ndarray,
        densities: np.ndarray,
    ) -> None:
        self.close_window()

        self.fig, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
        ax_potential = axes[0, 0]
        ax_density = axes[0, 1]
        ax_state = axes[1, 0]
        ax_energy = axes[1, 1]
        ax_xexp = axes[2, 0]
        ax_norm = axes[2, 1]

        ax_potential.plot(x, potential, color="black", linewidth=2.0)
        ax_potential.set_title(f"Potential ({well_type})")
        ax_potential.set_xlabel("x")
        ax_potential.set_ylabel("V(x)")
        ax_potential.grid(alpha=0.25)

        dimg = ax_density.imshow(
            densities.T,
            aspect="auto",
            origin="lower",
            extent=(float(np.min(x)), float(np.max(x)), float(np.min(t)), float(np.max(t))),
            cmap="viridis",
        )
        ax_density.set_title("Probability Density Map |psi(x,t)|^2")
        ax_density.set_xlabel("x")
        ax_density.set_ylabel("t")
        self.fig.colorbar(dimg, ax=ax_density, shrink=0.9)

        ax_state.plot(x, np.real(state), label="Re(psi_n)", linewidth=1.8)
        ax_state.plot(x, np.imag(state), "--", label="Im(psi_n)", linewidth=1.4)
        ax_state.set_title(f"Selected Eigenstate n={state_index}")
        ax_state.set_xlabel("x")
        ax_state.set_ylabel("psi_n(x)")
        ax_state.grid(alpha=0.25)
        ax_state.legend(loc="upper right")

        e_sorted = np.sort(energies.astype(float))
        for i, e in enumerate(e_sorted):
            ax_energy.hlines(e, 0.0, 1.0, colors="tab:orange", linewidth=1.8)
            ax_energy.text(1.02, e, f"E{i}={e:.5g}", va="center", fontsize=8)
        ax_energy.set_xlim(0.0, 1.3)
        ax_energy.set_xticks([])
        ax_energy.set_ylabel("Energy")
        ax_energy.set_title("Energy Levels")
        ax_energy.grid(axis="y", alpha=0.25)

        ax_xexp.plot(t, x_expectation, color="tab:green", linewidth=1.8)
        ax_xexp.set_title("Expectation Value <x>(t)")
        ax_xexp.set_xlabel("t")
        ax_xexp.set_ylabel("<x>")
        ax_xexp.grid(alpha=0.25)

        ax_norm.plot(t, norms, color="tab:red", linewidth=1.8)
        ax_norm.set_title("Norm Over Time")
        ax_norm.set_xlabel("t")
        ax_norm.set_ylabel("||psi||^2")
        ax_norm.grid(alpha=0.25)

        # Refresh-only behavior: one new window per recompute; no live redrawing.
        plt.show(block=False)


@dataclass
class TUIConfig:
    well_type: str = "finite_well"
    x_min: float = -8.0
    x_max: float = 8.0
    points: int = 450
    well_width: float = 3.0
    barrier_height: float = 80.0
    anharmonic: float = 0.0
    mass: float = 1.0
    hbar: float = 1.0
    num_states: int = 6
    state_index: int = 0
    t_max: float = 18.0
    time_steps: int = 260
    fps: float = 24.0
    use_gpu: bool = False


@dataclass
class QuantumWellTUI:
    config: TUIConfig = field(default_factory=TUIConfig)
    gui: SimulationGUI = field(default_factory=SimulationGUI)
    stats: RuntimeStats = field(default_factory=RuntimeStats)
    _frame_idx: int = 0
    _paused: bool = False
    _quit: bool = False
    _last_error: str = ""
    _dirty: bool = True
    _plot_refresh_requested: bool = True
    _is_recomputing: bool = False
    _progress_total: int = 1
    _progress_done: int = 0
    _progress_label: str = "idle"

    _x: np.ndarray | None = None
    _v: np.ndarray | None = None
    _energies: np.ndarray | None = None
    _states: np.ndarray | None = None
    _densities: np.ndarray | None = None
    _t: np.ndarray | None = None
    _x_expectation: np.ndarray | None = None
    _norms: np.ndarray | None = None

    _input_mode: bool = False
    _input_buffer: str = ""
    _input_status: str = ""
    _show_precise_help: bool = False
    _show_logs: bool = True
    _logs_expanded: bool = False
    _gpu_status: str = "GPU: disabled"
    _last_gpu_probe_s: float = 0.0
    _logs: deque[str] = field(default_factory=lambda: deque(maxlen=300))
    _log_handler: TUILogHandler | None = None

    def _install_tui_logging(self) -> None:
        if self._log_handler is not None:
            return
        self._log_handler = TUILogHandler(self._logs, max_lines=300)
        self._log_handler.setLevel(logging.INFO)
        self._log_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S"))
        logging.getLogger("qms").addHandler(self._log_handler)

    def _uninstall_tui_logging(self) -> None:
        if self._log_handler is None:
            return
        logging.getLogger("qms").removeHandler(self._log_handler)
        self._log_handler = None

    def _update_gpu_status(self) -> None:
        cfg = self.config
        if not cfg.use_gpu:
            self._gpu_status = "GPU: disabled"
            return

        if cp is None:
            self._gpu_status = "GPU: requested but CuPy is unavailable"
            return

        now = time.perf_counter()
        if (now - self._last_gpu_probe_s) < 0.8:
            return
        self._last_gpu_probe_s = now

        try:
            device = cp.cuda.Device()
            free_b, total_b = cp.cuda.runtime.memGetInfo()
            used_b = max(0, int(total_b - free_b))
            pct = (100.0 * used_b / float(total_b)) if total_b > 0 else 0.0
            self._gpu_status = f"GPU {int(device.id)} mem={used_b / 1e9:.2f}/{total_b / 1e9:.2f}GB ({pct:.1f}%)"
        except Exception as exc:
            self._gpu_status = f"GPU status error: {exc}"

    def _begin_progress(self, label: str, total: int) -> None:
        self._is_recomputing = True
        self._progress_label = label
        self._progress_total = max(1, int(total))
        self._progress_done = 0

    def _set_progress(self, done: int, label: str | None = None) -> None:
        if label is not None:
            self._progress_label = label
        total = max(1, int(self._progress_total))
        self._progress_done = max(0, min(total, int(done)))

    def _finish_progress(self, label: str = "simulation ready") -> None:
        self._progress_label = label
        self._progress_done = max(1, int(self._progress_total))
        self._is_recomputing = False

    def _set_param(self, key: str, value: str) -> None:
        cfg = self.config
        k = key.strip().lower()
        v = value.strip()

        if k in {"well", "well_type"}:
            val = v.lower()
            aliases = {"finite": "finite_well", "infinite_well": "infinite", "finite_well": "finite_well", "harmonic": "harmonic", "infinite": "infinite"}
            if val not in aliases:
                raise ValueError("well_type must be one of: infinite, finite_well, harmonic")
            cfg.well_type = aliases[val]
        elif k in {"x_min", "xmin"}:
            cfg.x_min = float(v)
        elif k in {"x_max", "xmax"}:
            cfg.x_max = float(v)
        elif k in {"points", "n", "grid"}:
            cfg.points = max(80, int(v))
        elif k in {"well_width", "width", "l"}:
            cfg.well_width = max(1.0e-6, float(v))
        elif k in {"barrier", "barrier_height", "v0"}:
            cfg.barrier_height = max(0.0, float(v))
        elif k in {"anharmonic", "a4"}:
            cfg.anharmonic = float(v)
        elif k in {"mass", "m"}:
            cfg.mass = max(1.0e-12, float(v))
        elif k in {"hbar"}:
            cfg.hbar = max(1.0e-12, float(v))
        elif k in {"num_states", "states"}:
            cfg.num_states = max(2, int(v))
        elif k in {"state", "state_index", "n_state"}:
            cfg.state_index = max(0, int(v))
        elif k in {"t_max", "tmax"}:
            cfg.t_max = max(1.0e-6, float(v))
        elif k in {"time_steps", "steps"}:
            cfg.time_steps = max(20, int(v))
        elif k in {"fps"}:
            cfg.fps = max(1.0, float(v))
        elif k in {"gpu", "use_gpu"}:
            cfg.use_gpu = v.lower() in {"1", "true", "yes", "on"}
        else:
            raise ValueError(f"unknown parameter: {k}")

    def _apply_precise_input(self, raw: str) -> None:
        if not raw.strip():
            self._input_status = "input ignored (empty)"
            return

        chunks = [c.strip() for c in raw.split(",") if c.strip()]
        if not chunks:
            self._input_status = "input ignored (empty)"
            return

        for chunk in chunks:
            if "=" not in chunk:
                raise ValueError("use key=value format, optionally comma-separated")
            key, value = chunk.split("=", 1)
            self._set_param(key, value)

        self._dirty = True
        self._input_status = "input applied"
        logger.info("precise input applied: %s", raw)

    def recompute(self) -> None:
        t0 = time.perf_counter()
        cfg = self.config
        self._begin_progress("starting", max(1, int(cfg.points)))

        try:
            params = {
                "x_min": cfg.x_min,
                "x_max": cfg.x_max,
                "N": int(max(80, cfg.points)),
                "well_width": _to_float(cfg.well_width, 1.0e-6),
                "barrier_height": _to_float(cfg.barrier_height, 0.0),
                "anharmonic": cfg.anharmonic,
                "mass": _to_float(cfg.mass, 1.0e-12),
                "hbar": _to_float(cfg.hbar, 1.0e-12),
                "num_states": int(max(2, cfg.num_states)),
                "use_gpu": cfg.use_gpu,
            }

            self._set_progress(0, "static solve")
            static = simulate_quantum_well(cfg.well_type, params)
            states = static["wavefunctions"]
            ncols = states.shape[1]
            cfg.state_index = max(0, min(int(cfg.state_index), ncols - 1))
            psi0 = states[:, cfg.state_index]

            self._set_progress(0, "time evolution")
            t_array = np.linspace(0.0, _to_float(cfg.t_max, 1.0e-6), int(max(20, cfg.time_steps)))
            time_result = run_time_simulation(
                psi0=psi0,
                H=static["hamiltonian"],
                t_array=t_array,
                hbar=params["hbar"],
                use_gpu=cfg.use_gpu,
            )

            self._x = to_numpy(static["x"]).astype(float)
            self._v = to_numpy(static["potential"]).astype(float)
            self._energies = to_numpy(static["energies"]).astype(float)
            self._states = to_numpy(states)
            self._densities = to_numpy(time_result["probability_densities"]).astype(float)
            self._t = to_numpy(time_result["t"]).astype(float)

            n_points = int(self._x.size)
            n_frames = int(self._densities.shape[1])
            dx = float(self._x[1] - self._x[0]) if self._x.size > 1 else 1.0
            x_expect = np.zeros(n_frames, dtype=float)
            norms = np.zeros(n_frames, dtype=float)

            self._begin_progress("processing all points", max(1, n_points))
            chunk = max(1, n_points // 80)
            for start in range(0, n_points, chunk):
                stop = min(n_points, start + chunk)
                dens_chunk = self._densities[start:stop, :]
                x_chunk = self._x[start:stop, None]
                x_expect += np.sum(dens_chunk * x_chunk, axis=0)
                norms += np.sum(dens_chunk, axis=0)
                self._set_progress(stop)

            self._x_expectation = x_expect * dx
            self._norms = norms * dx
            self.stats.last_frame_t = float(self._t[0])

            self._frame_idx = 0
            self._last_error = ""
            self._dirty = False

            elapsed = time.perf_counter() - t0
            self.stats.last_recompute_s = elapsed
            self.stats.recompute_total_s += elapsed
            self.stats.recompute_count += 1
            if not self.gui._is_open():
                self._plot_refresh_requested = True
            self._finish_progress("simulation ready")
            logger.info("recompute done in %.4fs", elapsed)
        except Exception:
            self._is_recomputing = False
            self._progress_label = "simulation failed"
            raise

    def _refresh_plot(self) -> None:
        if self._x is None or self._v is None or self._states is None or self._energies is None:
            return
        if self._densities is None or self._t is None or self._x_expectation is None or self._norms is None:
            return

        cfg = self.config
        ncols = self._states.shape[1]
        if ncols <= 0:
            return

        idx = max(0, min(int(cfg.state_index), ncols - 1))
        cfg.state_index = idx
        sel_state = self._states[:, idx]

        self.gui.render_snapshot(
            x=self._x,
            potential=self._v,
            state=sel_state,
            energies=self._energies,
            state_index=idx,
            well_type=cfg.well_type,
            t=self._t,
            x_expectation=self._x_expectation,
            norms=self._norms,
            densities=self._densities,
        )

    def _read_key(self) -> str | None:
        if msvcrt is None:
            return None
        if not msvcrt.kbhit():
            return None
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            if msvcrt.kbhit():
                _ = msvcrt.getwch()
            return None
        if ch in {"\r", "\n", "\b", "\x08", "\x1b"}:
            return ch
        return ch.lower()

    def _handle_key(self, key: str) -> None:
        cfg = self.config

        if self._input_mode:
            if key in {"\r", "\n"}:
                try:
                    self._apply_precise_input(self._input_buffer)
                except Exception as exc:
                    self._input_status = str(exc)
                    logger.error("input parse error: %s", exc)
                self._input_buffer = ""
                self._input_mode = False
            elif key in {"\b", "\x08"}:
                self._input_buffer = self._input_buffer[:-1]
            elif key == "\x1b":
                self._input_mode = False
                self._input_buffer = ""
                self._input_status = "input cancelled"
            elif key.isprintable():
                self._input_buffer += key
            return

        if key == "q":
            self._quit = True
        elif key == " ":
            self._paused = not self._paused
        elif key == "r":
            self._plot_refresh_requested = True
            logger.info("manual matplotlib refresh requested")
        elif key == "h":
            self._show_precise_help = not self._show_precise_help
        elif key == "i":
            self._input_mode = True
            self._input_buffer = ""
            self._input_status = "input mode: key=value,comma-separated"
        elif key == "l":
            self._show_logs = not self._show_logs
        elif key == "e":
            self._logs_expanded = not self._logs_expanded
        elif key == "n":
            cfg.state_index += 1
            self._dirty = True
        elif key == "p":
            cfg.state_index -= 1
            self._dirty = True
        elif key == "w":
            cfg.well_width *= 1.08
            self._dirty = True
        elif key == "s":
            cfg.well_width *= 0.92
            self._dirty = True
        elif key == "b":
            cfg.barrier_height *= 1.15
            self._dirty = True
        elif key == "v":
            cfg.barrier_height *= 0.87
            self._dirty = True
        elif key == "t":
            cfg.t_max *= 1.2
            self._dirty = True
        elif key == "g":
            cfg.t_max *= 0.84
            self._dirty = True
        elif key == "+":
            cfg.fps = min(60.0, cfg.fps + 2.0)
        elif key == "-":
            cfg.fps = max(4.0, cfg.fps - 2.0)
        elif key == "1":
            cfg.well_type = "infinite"
            self._dirty = True
        elif key == "2":
            cfg.well_type = "finite_well"
            self._dirty = True
        elif key == "3":
            cfg.well_type = "harmonic"
            self._dirty = True

    def _render_layout(self) -> Group:
        cfg = self.config

        stats = Table.grid(expand=True)
        stats.add_column(ratio=1)
        stats.add_column(ratio=1)
        stats.add_column(ratio=1)

        current_energy = "n/a"
        if self._energies is not None and self._energies.size:
            idx = max(0, min(cfg.state_index, self._energies.size - 1))
            current_energy = f"{self._energies[idx]:.6g}"

        sim_info = (
            f"Well: {cfg.well_type} | state n={cfg.state_index} | E_n={current_energy}\n"
            f"width={cfg.well_width:.4g}  barrier={cfg.barrier_height:.4g}  t_max={cfg.t_max:.4g}  fps={cfg.fps:.1f}"
        )
        mode = "PAUSED" if self._paused else "RUNNING"
        if self._dirty:
            mode += " (recomputing required)"
        if self._plot_refresh_requested:
            mode += " (plot refresh pending)"

        status_text = Text(f"{mode}", style="bold yellow" if self._paused else "bold green")
        if self._last_error:
            status_text.append(f" | ERROR: {self._last_error}", style="bold red")

        stats.add_row(sim_info, status_text, f"frame={self._frame_idx}")
        top_panel = Panel(stats, title="Quantum Well Real-Time TUI", box=box.ROUNDED)

        if self._x is None or self._v is None or self._densities is None or self._t is None:
            body_panel = Panel("No simulation data yet", title="Status", box=box.ROUNDED)
            controls = self._controls_panel()
            extras = self._render_extra_panels()
            return Group(top_panel, body_panel, controls, extras)

        nframes = int(self._densities.shape[1])
        t_now = float(self._t[0]) if self._t.size else 0.0

        info = Table.grid(expand=True)
        info.add_column(ratio=1)
        info.add_column(ratio=1)
        info.add_row(
            "GUI plots are rendered in a separate matplotlib window.",
            f"t = {t_now:.6g} | frames = {nframes}",
        )
        body = Panel(info, title="GUI Output", box=box.ROUNDED)

        controls = self._controls_panel()
        extras = self._render_extra_panels()
        return Group(top_panel, body, controls, extras)

    def _render_extra_panels(self) -> Group:
        self._update_gpu_status()
        self.stats.update_cpu()
        mem_cur = mem_peak = 0
        if tracemalloc.is_tracing():
            mem_cur, mem_peak = tracemalloc.get_traced_memory()

        perf = Table.grid(expand=True)
        perf.add_column(ratio=1)
        perf.add_column(ratio=1)
        perf.add_column(ratio=1)
        perf.add_row(
            f"uptime={self.stats.uptime_s:.2f}s | cpu~={self.stats.cpu_percent:.1f}% (normalized)",
            f"recompute={self.stats.last_recompute_s:.4f}s avg={self.stats.avg_recompute_s:.4f}s count={self.stats.recompute_count}",
            f"mem(current/peak)={mem_cur / 1e6:.2f}MB/{mem_peak / 1e6:.2f}MB | {self._gpu_status}",
        )
        perf_panel = Panel(perf, title="Resources & Timing", box=box.ROUNDED)

        monitor = Table.grid(expand=True)
        monitor.add_column(ratio=1)
        monitor.add_column(ratio=1)
        cfg = self.config
        monitor.add_row("well_type", str(cfg.well_type))
        monitor.add_row("x_min", f"{cfg.x_min:.6g}")
        monitor.add_row("x_max", f"{cfg.x_max:.6g}")
        monitor.add_row("points", str(int(cfg.points)))
        monitor.add_row("well_width", f"{cfg.well_width:.6g}")
        monitor.add_row("barrier_height", f"{cfg.barrier_height:.6g}")
        monitor.add_row("anharmonic", f"{cfg.anharmonic:.6g}")
        monitor.add_row("mass", f"{cfg.mass:.6g}")
        monitor.add_row("hbar", f"{cfg.hbar:.6g}")
        monitor.add_row("num_states", str(int(cfg.num_states)))
        monitor.add_row("state_index", str(int(cfg.state_index)))
        monitor.add_row("t_max", f"{cfg.t_max:.6g}")
        monitor.add_row("time_steps", str(int(cfg.time_steps)))
        monitor.add_row("fps", f"{cfg.fps:.6g}")
        monitor.add_row("use_gpu", str(bool(cfg.use_gpu)))
        monitor_panel = Panel(monitor, title="Input Monitor", box=box.ROUNDED)

        total = max(1, int(self._progress_total))
        done = max(0, min(total, int(self._progress_done)))
        frac = float(done) / float(total)
        width = 44
        filled = int(frac * width)
        bar = "[" + ("#" * filled) + ("-" * (width - filled)) + "]"
        state = "processing" if self._is_recomputing else ("queued" if self._dirty else "idle")
        progress_text = f"{self._progress_label} | {state}\n{bar} {done}/{total} ({100.0 * frac:.1f}%)"
        progress_panel = Panel(progress_text, title="Simulation Progress", box=box.ROUNDED)

        if self._input_mode:
            input_text = f"input> {self._input_buffer}_"
        else:
            input_text = f"status: {self._input_status or 'ready'}"
        input_panel = Panel(input_text, title="Precise Input", box=box.ROUNDED)

        if not self._show_logs:
            logs_panel = Panel("logs hidden (press l)", title="Logs", box=box.ROUNDED)
        else:
            if self._logs_expanded:
                lines = list(self._logs)[-14:]
            else:
                lines = list(self._logs)[-2:]
            logs_text = "\n".join(lines) if lines else "no logs yet"
            logs_panel = Panel(logs_text, title=f"Logs ({'expanded' if self._logs_expanded else 'compact'})", box=box.ROUNDED)

        precise_help = (
            "Precise controls (use i, then key=value):\n"
            "well_type|well = infinite | finite_well | harmonic\n"
            "x_min|xmin, x_max|xmax, points|n|grid\n"
            "well_width|width|l, barrier_height|barrier|v0, anharmonic|a4\n"
            "mass|m, hbar, num_states|states, state|state_index|n_state\n"
            "t_max|tmax, time_steps|steps, fps, use_gpu|gpu\n"
            "Example: well=harmonic,points=900,state=2,t_max=24,use_gpu=true"
        )
        precise_help_panel = Panel(precise_help, title="Precise Control Options (toggle h)", box=box.ROUNDED)

        if self._show_precise_help:
            return Group(perf_panel, monitor_panel, input_panel, logs_panel, precise_help_panel, progress_panel)
        return Group(perf_panel, monitor_panel, input_panel, logs_panel, progress_panel)

    def _controls_panel(self) -> Panel:
        controls = (
            "Controls: [[space]]=pause/resume  q=quit  r=refresh matplotlib window  i=precise input mode\n"
            "1=infinite  2=finite  3=harmonic  n/p=state  w/s=width  b/v=barrier  t/g=time-window\n"
            "+/-=fps  l=show/hide logs  e=expand logs  h=toggle precise-control popup\n"
            "Precise input format: key=value,key=value (example: well_width=2.35,points=800,state=2)"
        )
        return Panel(controls, title="Keyboard", box=box.ROUNDED)

    async def _input_loop(self) -> None:
        while not self._quit:
            key = self._read_key()
            if key is not None:
                self._handle_key(key)
            await asyncio.sleep(0.02)

    async def _compute_loop(self) -> None:
        while not self._quit:
            if self._paused or not self._dirty:
                await asyncio.sleep(0.05)
                continue

            try:
                await asyncio.to_thread(self.recompute)
            except Exception as exc:  # pragma: no cover
                self._last_error = str(exc)
                self._paused = True
                self._dirty = False
                logger.exception("recompute loop error")
            await asyncio.sleep(0)

    async def _plot_loop(self) -> None:
        while not self._quit:
            try:
                if self._plot_refresh_requested and not self._is_recomputing:
                    self._refresh_plot()
                    self._plot_refresh_requested = False

                # Pump matplotlib's event queue so the window remains responsive.
                if self.gui._is_open():
                    plt.pause(0.001)
            except Exception as exc:  # pragma: no cover
                self._last_error = str(exc)
                self._plot_refresh_requested = False
                logger.exception("plot loop error")

            await asyncio.sleep(0.03)

    async def _tui_loop(self) -> None:
        with Live(self._render_layout(), refresh_per_second=8, screen=True) as live:
            while not self._quit:
                live.update(self._render_layout())
                await asyncio.sleep(0.05)

    async def _run_async(self) -> None:
        tasks = [
            asyncio.create_task(self._input_loop()),
            asyncio.create_task(self._compute_loop()),
            asyncio.create_task(self._plot_loop()),
            asyncio.create_task(self._tui_loop()),
        ]

        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with suppress(asyncio.CancelledError):
                    await task

    def run(self) -> None:
        self._install_tui_logging()
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        try:
            asyncio.run(self._run_async())
        except KeyboardInterrupt:
            self._quit = True
        finally:
            self._uninstall_tui_logging()


def launch_tui_from_cli(args: argparse.Namespace) -> None:
    config = TUIConfig(
        well_type=args.well_type,
        x_min=args.x_min,
        x_max=args.x_max,
        points=args.points,
        well_width=args.well_width,
        barrier_height=args.barrier_height,
        anharmonic=args.anharmonic,
        mass=args.mass,
        hbar=args.hbar,
        num_states=args.num_states,
        state_index=args.state,
        t_max=args.t_max,
        time_steps=args.time_steps,
        fps=args.fps,
        use_gpu=args.use_gpu,
    )
    QuantumWellTUI(config=config).run()


__all__ = ["QuantumWellTUI", "TUIConfig", "launch_tui_from_cli"]