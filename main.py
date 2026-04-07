from __future__ import annotations

import argparse

from qms.visualization.tui import launch_tui_from_cli


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="quantum-well-simulator",
		description="Quantum Well Simulator terminal interface",
	)
	parser.add_argument(
		"--mode",
		choices=["tui"],
		default="tui",
		help="Run mode (default: tui)",
	)
	parser.add_argument("--well-type", default="finite_well", choices=["infinite", "finite_well", "harmonic"])
	parser.add_argument("--x-min", type=float, default=-8.0)
	parser.add_argument("--x-max", type=float, default=8.0)
	parser.add_argument("--points", type=int, default=450)
	parser.add_argument("--well-width", type=float, default=3.0)
	parser.add_argument("--barrier-height", type=float, default=80.0)
	parser.add_argument("--anharmonic", type=float, default=0.0)
	parser.add_argument("--mass", type=float, default=1.0)
	parser.add_argument("--hbar", type=float, default=1.0)
	parser.add_argument("--num-states", type=int, default=6)
	parser.add_argument("--state", type=int, default=0, help="Initial eigenstate index")
	parser.add_argument("--t-max", type=float, default=18.0)
	parser.add_argument("--time-steps", type=int, default=260)
	parser.add_argument("--fps", type=float, default=24.0)
	parser.add_argument("--use-gpu", action="store_true")
	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()

	if args.mode == "tui":
		launch_tui_from_cli(args)


if __name__ == "__main__":
	main()
