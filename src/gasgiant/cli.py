"""Headless CLI: `gasgiant export` and `gasgiant validate`."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from gasgiant.diagnostics import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gasgiant", description="Gas giant map generator")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    exp = sub.add_parser("export", help="render a map set headlessly")
    exp.add_argument("--preset", default="jupiter_like", help="factory preset name or .json path")
    exp.add_argument("--res", type=int, default=None, help="equirect width in pixels (2:1 maps)")
    exp.add_argument("--seed", type=int, default=None, help="override the preset seed")
    exp.add_argument("--name", default=None, help="override the planet name")
    exp.add_argument("--out", type=Path, required=True, help="output map-set directory")

    val = sub.add_parser("validate", help="run seam/pole checks on an exported map set")
    val.add_argument("mapset", type=Path)

    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    if args.command == "export":
        return _export(args)
    return _validate(args)


def _export(args: argparse.Namespace) -> int:
    from gasgiant.engine import Simulation
    from gasgiant.export.manifest import export_mapset
    from gasgiant.params.presets import PresetError, resolve_preset

    try:
        params = resolve_preset(args.preset)
    except PresetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    updates: dict = {}
    if args.seed is not None:
        updates["seed"] = args.seed
    if args.name is not None:
        updates["name"] = args.name
    if updates:
        params = params.model_copy(update=updates)
    if args.res is not None:
        params.export.width = args.res

    started = time.perf_counter()
    sim = Simulation(params)
    manifest_path = export_mapset(sim, args.out)
    elapsed = time.perf_counter() - started
    print(f"exported {params.export.width}x{params.export.width // 2} map set "
          f"to {manifest_path.parent} in {elapsed:.1f}s")
    return 0


def _validate(args: argparse.Namespace) -> int:
    from gasgiant.validate import validate_mapset

    report = validate_mapset(args.mapset)
    print(report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
