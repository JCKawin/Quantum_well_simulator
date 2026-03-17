from ..core.hamiltonian import build_hamiltonian
from ..core.potentials import potential
from ..solvers.eigen import solve_eigenvalue_problem
from ..solvers.time_evolution import run_time_simulation
from ..utils.backend import get_array_module, to_numpy
from ..wavefunction.normalization import normalize_wavefunction


def run_static_simulation(
	x,
	V,
	mass: float = 1.0,
	hbar: float = 1.0,
	num_states: int = 5,
	use_gpu: bool = False,
):
	xp = get_array_module(use_gpu)
	x = xp.asarray(x, dtype=xp.float64)
	if x.size < 3:
		raise ValueError("`x` must contain at least 3 points")

	h = build_hamiltonian(x=x, V=V, mass=mass, hbar=hbar, use_gpu=use_gpu)
	energies, states = solve_eigenvalue_problem(h, num_states=num_states, use_gpu=use_gpu)

	dx = float(to_numpy(x[1] - x[0]))
	for i in range(states.shape[1]):
		states[:, i] = normalize_wavefunction(states[:, i], dx=dx, use_gpu=use_gpu)

	return {
		"x": x,
		"hamiltonian": h,
		"energies": energies,
		"wavefunctions": states,
		"probability_densities": xp.abs(states) ** 2,
	}


def simulate_quantum_well(well_type: str, params: dict):
	if not isinstance(params, dict):
		raise ValueError("`params` must be a dictionary")

	use_gpu = bool(params.get("use_gpu", False))
	xp = get_array_module(use_gpu)

	if "x" in params:
		x = xp.asarray(params["x"], dtype=xp.float64)
	else:
		x_min = float(params.get("x_min", -1.0))
		x_max = float(params.get("x_max", 1.0))
		n_points = int(params.get("N", 400))
		x = xp.linspace(x_min, x_max, n_points, dtype=xp.float64)

	L = float(params.get("L", params.get("well_width", 1.0)))
	V0 = float(params.get("V0", params.get("barrier_height", 1.0e6)))

	wt = str(well_type).strip().lower()
	if wt in {"infinite", "infinite_well", "infinite_square"}:
		Vx = potential(x, well_width=L, barrier_height=max(V0, 1.0e12), kind="infinite_square", use_gpu=use_gpu)
	elif wt in {"finite", "finite_well", "finite_square"}:
		Vx = potential(x, well_width=L, barrier_height=V0, kind="finite_square", use_gpu=use_gpu)
	elif wt == "harmonic":
		Vx = potential(
			x,
			well_width=L,
			kind="harmonic",
			anharmonic=float(params.get("anharmonic", 0.0)),
			use_gpu=use_gpu,
		)
	elif wt == "custom":
		custom_v = params.get("V", params.get("potential_values"))
		if custom_v is None:
			raise ValueError("For `custom` well_type, provide `params['V']` or `params['potential_values']`")
		if callable(custom_v):
			Vx = xp.asarray(custom_v(x), dtype=xp.float64)
		else:
			Vx = xp.asarray(custom_v, dtype=xp.float64)
	else:
		raise ValueError(f"Unsupported well_type: {well_type}")

	result = run_static_simulation(
		x=x,
		V=Vx,
		mass=float(params.get("mass", 1.0)),
		hbar=float(params.get("hbar", 1.0)),
		num_states=int(params.get("num_states", 5)),
		use_gpu=use_gpu,
	)
	result["well_type"] = wt
	result["potential"] = Vx
	return result


def run_full_pipeline(config: dict):
	if not isinstance(config, dict):
		raise ValueError("`config` must be a dictionary")

	use_gpu = bool(config.get("use_gpu", False))
	hbar = float(config.get("hbar", 1.0))

	if "x" in config and ("V" in config or "potential_values" in config):
		x = config["x"]
		V = config.get("V", config.get("potential_values"))
		static_result = run_static_simulation(
			x=x,
			V=V,
			mass=float(config.get("mass", 1.0)),
			hbar=hbar,
			num_states=int(config.get("num_states", 5)),
			use_gpu=use_gpu,
		)
	else:
		well_type = str(config.get("well_type", "finite_well"))
		params = dict(config.get("params", {}))
		params.setdefault("use_gpu", use_gpu)
		params.setdefault("hbar", hbar)
		params.setdefault("mass", float(config.get("mass", 1.0)))
		params.setdefault("num_states", int(config.get("num_states", 5)))
		static_result = simulate_quantum_well(well_type, params)

	output = {"static": static_result}

	t_array = config.get("t_array")
	if t_array is not None:
		xp = get_array_module(use_gpu)
		psi0 = config.get("psi0")
		if psi0 is None:
			psi0 = static_result["wavefunctions"][:, 0]
		psi0 = xp.asarray(psi0, dtype=xp.complex128)

		time_result = run_time_simulation(
			psi0=psi0,
			H=static_result["hamiltonian"],
			t_array=t_array,
			hbar=hbar,
			use_gpu=use_gpu,
		)
		output["time"] = time_result

	return output

__all__ = ["run_static_simulation", "simulate_quantum_well", "run_full_pipeline"]
