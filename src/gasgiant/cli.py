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
    exp.add_argument("--preset", default=None, help="factory preset name or .json path")
    exp.add_argument(
        "--resume", type=Path, default=None,
        help="resume from a saved checkpoint .npz (from `gasgiant checkpoint`) "
             "instead of building from a preset. Mutually exclusive with "
             "--preset/--recipe; combine with --frames to export a sequence from "
             "the resumed state.",
    )
    exp.add_argument(
        "--recipe", default=None,
        help="epoch recipe to overlay (e.g. faded_seb, ochre_ez): a small "
             "parameter overlay reproducing a documented historical atmosphere "
             "state. Precedence: with --preset, the overlay is applied on top of "
             "that preset; without --preset, the recipe's own base preset is used. "
             "Default base (neither given) is jupiter_like",
    )
    exp.add_argument("--res", type=int, default=None, help="equirect width in pixels (2:1 maps)")
    exp.add_argument("--dev-steps", type=int, default=None,
                     help="override the preset's development step count")
    exp.add_argument("--frames", type=int, default=None,
                     help="export an animation sequence of N color frames "
                          "(frame 0 = the map set's color map)")
    exp.add_argument("--steps-per-frame", type=int, default=None,
                     help="sim steps advanced between sequence frames "
                          "(required with --frames)")
    exp.add_argument("--ramp-to", default=None,
                     help="param RAMP target: a factory preset name or .json "
                          "path whose look the sequence interpolates TO over the "
                          "frames (frame 0 = the base look, last frame = this "
                          "target). Requires --frames. The target's seed is "
                          "forced to the base seed (a ramp keeps the same "
                          "developed world); a RESTART-tier diff is rejected")
    exp.add_argument("--all-maps", action="store_true",
                     help="with --frames: also write height (and emission, when "
                          "enabled) per frame into frames/, not just color")
    exp.add_argument("--video", action="store_true",
                     help="with --frames: encode the color frames into an mp4 "
                          "via ffmpeg (must be on PATH)")
    exp.add_argument("--fps", type=int, default=24,
                     help="frames per second for --video (default 24)")
    exp.add_argument("--seed", type=int, default=None, help="override the preset seed")
    exp.add_argument("--name", default=None, help="override the planet name")
    exp.add_argument("--out", type=Path, required=True, help="output map-set directory")

    ckpt = sub.add_parser(
        "checkpoint",
        help="develop a run and save it as a resumable checkpoint .npz",
    )
    ckpt.add_argument("--preset", default=None, help="factory preset name or .json path")
    ckpt.add_argument(
        "--recipe", default=None,
        help="epoch recipe to overlay (same precedence as export's --recipe)",
    )
    ckpt.add_argument("--res", type=int, default=None,
                      help="equirect width in pixels (2:1 maps)")
    ckpt.add_argument("--dev-steps", type=int, default=None,
                      help="override the preset's development step count")
    ckpt.add_argument("--seed", type=int, default=None, help="override the preset seed")
    ckpt.add_argument("--out", type=Path, required=True, help="output checkpoint .npz path")

    sht = sub.add_parser(
        "sheet",
        help="render a seed contact sheet: one small color map per seed in a grid",
    )
    sht.add_argument("--preset", default=None, help="factory preset name or .json path")
    sht.add_argument(
        "--recipe", default=None,
        help="epoch recipe to overlay (same precedence as export's --recipe)",
    )
    sht.add_argument("--seeds", default=None,
                     help="explicit seed list, comma- or space-separated (e.g. \"1,2,3\")")
    sht.add_argument("--count", type=int, default=None,
                     help="number of consecutive seeds to render (with --seed0)")
    sht.add_argument("--seed0", type=int, default=None,
                     help="first seed for --count (default 0)")
    sht.add_argument("--res", type=int, default=256,
                     help="per-cell equirect width in pixels (2:1 cells; default 256)")
    sht.add_argument("--dev-steps", type=int, default=None,
                     help="override the preset's development step count (faster previews)")
    sht.add_argument("--cols", type=int, default=None,
                     help="grid columns (default: ceil(sqrt(N)) for a square-ish sheet)")
    sht.add_argument("--out", type=Path, required=True, help="output contact-sheet .png path")

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
    if args.command == "checkpoint":
        return _checkpoint(args)
    if args.command == "palette-fit":
        return _palette_fit(args)
    if args.command == "sheet":
        return _sheet(args)
    return _validate(args)


def _resolve_params_from_args(args: argparse.Namespace, *, default_preset: str):
    """Resolve the params for a preset-building subcommand (export/checkpoint):
    apply any --recipe overlay onto the base preset, then the CLI overrides
    (--seed/--name/--res/--dev-steps). Returns ``(params, None)`` on success or
    ``(None, exit_code)`` on a user error (message already printed to stderr).

    --preset wins the base choice; without it the recipe's own base is used;
    with neither, ``default_preset`` holds. `--name`/`--dev-steps` are ignored
    when the corresponding attribute is absent from ``args``."""
    from gasgiant.params.presets import (
        PresetError,
        apply_overlay,
        load_recipe,
        resolve_preset,
    )

    overlay: dict | None = None
    base_name = args.preset
    if getattr(args, "recipe", None) is not None:
        try:
            recipe_base, overlay, _meta = load_recipe(args.recipe)
        except PresetError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return None, 2
        if base_name is None:
            base_name = recipe_base
    if base_name is None:
        base_name = default_preset

    try:
        params = resolve_preset(base_name)
        if overlay is not None:
            params = apply_overlay(params, overlay)
    except (PresetError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return None, 2

    updates: dict = {}
    if args.seed is not None:
        updates["seed"] = args.seed
    if getattr(args, "name", None) is not None:
        updates["name"] = args.name
    if updates:
        params = params.model_copy(update=updates)
    if args.res is not None:
        params.export.width = args.res
    if getattr(args, "dev_steps", None) is not None:
        params.sim.dev_steps = args.dev_steps

    # Mask sidecar: load_preset already resolved a relative path against the
    # preset's folder (absolute in memory). A path that doesn't exist is a clear
    # CLI error rather than a silently-disabled mask (the engine's warn+disable
    # is for the checkpoint/GUI case, where a portable preset may outlive its
    # sidecar).
    if params.mask.file is not None and not Path(params.mask.file).is_file():
        print(f"error: mask file not found: {params.mask.file}", file=sys.stderr)
        return None, 2
    return params, None


def _export(args: argparse.Namespace) -> int:
    from gasgiant.engine import Simulation
    from gasgiant.engine.checkpoint import load_checkpoint
    from gasgiant.export.exporter import run_export, run_export_sequence
    from gasgiant.export.video import ffmpeg_available as _ffmpeg_available

    if (args.frames is None) != (args.steps_per_frame is None):
        print("error: --frames and --steps-per-frame must be given together",
              file=sys.stderr)
        return 2
    if args.frames is not None and (args.frames < 1 or args.steps_per_frame < 1):
        print("error: --frames and --steps-per-frame must be >= 1", file=sys.stderr)
        return 2
    if args.ramp_to is not None and args.frames is None:
        print("error: --ramp-to requires --frames/--steps-per-frame", file=sys.stderr)
        return 2

    # --resume and --preset/--recipe are mutually exclusive: one builds a fresh
    # run from a preset, the other reloads a saved run. --preset defaults None,
    # so their simultaneous presence is detectable rather than ambiguous.
    if args.resume is not None and (args.preset is not None or args.recipe is not None):
        print("error: --resume is mutually exclusive with --preset/--recipe",
              file=sys.stderr)
        return 2

    started = time.perf_counter()
    if args.resume is not None:
        if not args.resume.is_file():
            print(f"error: checkpoint not found: {args.resume}", file=sys.stderr)
            return 2
        try:
            # GENERATION_VERSION mismatch / corrupt-file errors surface as clear
            # messages rather than a raw traceback.
            sim = load_checkpoint(args.resume)
        except (ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        params = sim.params
    else:
        params, err = _resolve_params_from_args(args, default_preset="jupiter_like")
        if params is None:
            return err
        sim = Simulation(params)

    if args.frames is not None:
        if args.video and not _ffmpeg_available():
            print("error: --video needs ffmpeg on PATH (not found)", file=sys.stderr)
            return 2
        ramp_to = None
        if args.ramp_to is not None:
            from gasgiant.params.interp import RampError, validate_ramp
            from gasgiant.params.presets import PresetError, resolve_preset

            try:
                ramp_to = resolve_preset(args.ramp_to)
            except (PresetError, ValueError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            # A ramp keeps the SAME developed world: align the target with the
            # base's world-defining knobs (seed + the CLI-overridable RESTART/
            # structural fields dev_steps and width) so a ramp between variants
            # of one preset "just works" and is about LOOK, not world. Any
            # remaining RESTART diff (e.g. a different band layout) is still
            # rejected by validate_ramp below. (validate_ramp also guards the
            # job API for programmatic callers.)
            ramp_to = ramp_to.model_copy(update={"seed": params.seed})
            ramp_to.sim.dev_steps = params.sim.dev_steps
            ramp_to.export.width = params.export.width
            try:
                validate_ramp(params, ramp_to)
            except RampError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
        run_export_sequence(
            sim, args.out, frames=args.frames, steps_per_frame=args.steps_per_frame,
            all_maps=args.all_maps, video=args.video, fps=args.fps, ramp_to=ramp_to,
        )
    else:
        run_export(sim, args.out)
    elapsed = time.perf_counter() - started
    seq = f" + {args.frames}-frame sequence" if args.frames is not None else ""
    src = f" (resumed from {args.resume})" if args.resume is not None else ""
    print(f"exported {params.export.width}x{params.export.width // 2} map set{seq}{src} "
          f"to {args.out} in {elapsed:.1f}s")
    return 0


def _checkpoint(args: argparse.Namespace) -> int:
    """Develop a run to completion and save it as a resumable checkpoint .npz
    (the input to `gasgiant export --resume`). Kept deliberately simple: it
    develops to the preset's ``sim.dev_steps`` and saves there."""
    from gasgiant.engine import Simulation
    from gasgiant.engine.checkpoint import save_checkpoint

    params, err = _resolve_params_from_args(args, default_preset="jupiter_like")
    if params is None:
        return err

    started = time.perf_counter()
    sim = Simulation(params)
    sim.run_to_completion()
    out = args.out if args.out.suffix else args.out.with_suffix(".npz")
    save_checkpoint(sim, out)
    elapsed = time.perf_counter() - started
    print(f"saved checkpoint at step {sim.steps_done} to {out} in {elapsed:.1f}s")
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


def _resolve_seeds(args: argparse.Namespace) -> tuple[list[int] | None, int]:
    """Resolve the seed list for `gasgiant sheet` from either an explicit
    --seeds list or --count/--seed0. Returns ``(seeds, 0)`` on success or
    ``(None, 2)`` on a user error (message printed). --seeds wins when both
    are given."""
    if args.seeds is not None:
        try:
            seeds = [int(tok) for tok in args.seeds.replace(",", " ").split()]
        except ValueError:
            print("error: --seeds must be a comma/space-separated list of integers",
                  file=sys.stderr)
            return None, 2
        if not seeds:
            print("error: --seeds is empty", file=sys.stderr)
            return None, 2
        return seeds, 0
    if args.count is not None:
        if args.count < 1:
            print("error: --count must be >= 1", file=sys.stderr)
            return None, 2
        s0 = args.seed0 if args.seed0 is not None else 0
        return list(range(s0, s0 + args.count)), 0
    print("error: provide --seeds \"1,2,3\" or --count N [--seed0 K]", file=sys.stderr)
    return None, 2


def _sheet(args: argparse.Namespace) -> int:
    """Render a seed contact sheet: reuse ONE Simulation across all seeds
    (no per-seed GPU leak), re-seeding it per iteration, and tile the color
    maps into a grid PNG."""
    from gasgiant.engine import Simulation
    from gasgiant.export.sheet import run_sheet
    from gasgiant.params.presets import (
        PresetError,
        apply_overlay,
        load_recipe,
        resolve_preset,
    )

    if args.res < 2 or args.res % 2 != 0:
        print("error: --res must be a positive even number (2:1 maps)", file=sys.stderr)
        return 2
    if args.cols is not None and args.cols < 1:
        print("error: --cols must be >= 1", file=sys.stderr)
        return 2

    overlay: dict | None = None
    base_name = args.preset
    if args.recipe is not None:
        try:
            recipe_base, overlay, _meta = load_recipe(args.recipe)
        except PresetError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if base_name is None:
            base_name = recipe_base
    if base_name is None:
        base_name = "gas_giant_warm"
    try:
        params = resolve_preset(base_name)
        if overlay is not None:
            params = apply_overlay(params, overlay)
    except (PresetError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if params.mask.file is not None and not Path(params.mask.file).is_file():
        print(f"error: mask file not found: {params.mask.file}", file=sys.stderr)
        return 2

    seeds, err = _resolve_seeds(args)
    if seeds is None:
        return err

    out = args.out if args.out.suffix else args.out.with_suffix(".png")
    started = time.perf_counter()
    run_sheet(
        Simulation, params, seeds, out,
        width=args.res, dev_steps=args.dev_steps, cols=args.cols,
    )
    elapsed = time.perf_counter() - started
    print(f"wrote {len(seeds)}-seed contact sheet ({args.res}px cells) to {out} "
          f"in {elapsed:.1f}s")
    return 0


def _validate(args: argparse.Namespace) -> int:
    from gasgiant.validate import validate_mapset

    report = validate_mapset(args.mapset)
    print(report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
