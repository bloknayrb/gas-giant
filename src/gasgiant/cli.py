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
    exp.add_argument("--dev-steps", type=int, default=None,
                     help="override the preset's development step count")
    exp.add_argument("--frames", type=int, default=None,
                     help="export an animation sequence of N color frames "
                          "(frame 0 = the map set's color map)")
    exp.add_argument("--steps-per-frame", type=int, default=None,
                     help="sim steps advanced between sequence frames "
                          "(required with --frames)")
    exp.add_argument("--seed", type=int, default=None, help="override the preset seed")
    exp.add_argument("--name", default=None, help="override the planet name")
    exp.add_argument("--out", type=Path, required=True, help="output map-set directory")

    val = sub.add_parser("validate", help="run seam/pole checks on an exported map set")
    val.add_argument("mapset", type=Path)

    pf = sub.add_parser(
        "palette-fit",
        help="fit palette rows from a reference image and bake them into a preset",
    )
    pf.add_argument("--image", type=Path, required=True,
                    help="reference photo (cylindrical/equirect true-color image)")
    pf.add_argument("--preset", default="jupiter_like",
                    help="factory preset name or .json path to start from")
    pf.add_argument("--out", type=Path, required=True, help="output preset .json path")
    pf.add_argument("--anchors", nargs="+", type=float, default=None,
                    help="anchor latitudes in signed degrees (space-separated)")
    pf.add_argument("--bins", type=int, default=90)
    pf.add_argument("--stops", type=int, choices=(3, 5), default=3,
                    help="stops per fitted row")
    pf.add_argument("--fit-mode", choices=("median", "chroma-restore"), default="median")

    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    if args.command == "export":
        return _export(args)
    if args.command == "palette-fit":
        return _palette_fit(args)
    return _validate(args)


def _export(args: argparse.Namespace) -> int:
    from gasgiant.engine import Simulation
    from gasgiant.export.exporter import run_export, run_export_sequence
    from gasgiant.params.presets import PresetError, resolve_preset

    if (args.frames is None) != (args.steps_per_frame is None):
        print("error: --frames and --steps-per-frame must be given together",
              file=sys.stderr)
        return 2
    if args.frames is not None and (args.frames < 1 or args.steps_per_frame < 1):
        print("error: --frames and --steps-per-frame must be >= 1", file=sys.stderr)
        return 2

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
    if args.dev_steps is not None:
        params.sim.dev_steps = args.dev_steps

    # Mask sidecar: load_preset already resolved a relative path against the
    # preset's folder (absolute in memory). A path that doesn't exist is a clear
    # CLI error rather than a silently-disabled mask (the engine's warn+disable
    # is for the checkpoint/GUI case, where a portable preset may outlive its
    # sidecar).
    if params.mask.file is not None and not Path(params.mask.file).is_file():
        print(f"error: mask file not found: {params.mask.file}", file=sys.stderr)
        return 2

    started = time.perf_counter()
    sim = Simulation(params)
    if args.frames is not None:
        run_export_sequence(
            sim, args.out, frames=args.frames, steps_per_frame=args.steps_per_frame
        )
    else:
        run_export(sim, args.out)
    elapsed = time.perf_counter() - started
    seq = f" + {args.frames}-frame sequence" if args.frames is not None else ""
    print(f"exported {params.export.width}x{params.export.width // 2} map set{seq} "
          f"to {args.out} in {elapsed:.1f}s")
    return 0


def _palette_fit(args: argparse.Namespace) -> int:
    """Fit palette rows from a reference image and BAKE them into a preset.

    The palette values are baked into ``appearance.palette_rows`` (a POST-tier
    field); no image path is stored. Decode happens here (the top layer) via
    ``writers.decode_image`` and the model conversion via
    ``params.model.palette_rows_from_fit`` -- keeping the ``palette`` layer clean.
    """
    from gasgiant.export.writers import decode_image
    from gasgiant.palette.fit import DEFAULT_ANCHORS, calibrate
    from gasgiant.params.model import palette_rows_from_fit
    from gasgiant.params.presets import PresetError, resolve_preset, save_preset

    if not args.image.is_file():
        print(f"error: reference image not found: {args.image}", file=sys.stderr)
        return 2
    try:
        params = resolve_preset(args.preset)
    except PresetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not hasattr(params.appearance, "palette_rows"):
        print("error: preset lacks appearance.palette_rows (needs preset format 2+)",
              file=sys.stderr)
        return 2

    try:
        img = decode_image(args.image, color=True)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    anchors = tuple(args.anchors) if args.anchors else DEFAULT_ANCHORS
    doc = calibrate(img, anchors, args.bins, fit_mode=args.fit_mode, stops=args.stops)
    params.appearance.palette_rows = palette_rows_from_fit(doc["palette_rows"])
    save_preset(params, args.out)
    print(f"fit {len(doc['palette_rows'])} palette rows from {args.image} -> {args.out}")
    return 0


def _validate(args: argparse.Namespace) -> int:
    from gasgiant.validate import validate_mapset

    report = validate_mapset(args.mapset)
    print(report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
