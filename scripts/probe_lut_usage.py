"""Which part of the palette LUT does a preset actually USE?

A carefully authored ramp is wasted if the sim's tracer only ever lands in one
slice of it -- the dark end is then dead code and the planet reads as
the bright end alone. (Exactly that happened to ember_dwarf v1: a dark plum low
end that never appeared, so a "dim magenta" object rendered as uniform lava red.)

Trick: replace the palette with a LINEAR GRAYSCALE ramp, so the rendered
luminance IS the palette index. The histogram then reads directly as "how much
of the disc sits at each point along the LUT".

    uv run python scripts/probe_lut_usage.py --preset ember_dwarf
"""
from __future__ import annotations

import argparse

import numpy as np

# Reuse the sibling preview tool's override parser rather than keeping a second
# copy: it walks any nesting depth (``poles.north.strength``), which a local
# rpartition-based version did NOT -- it built the attribute name "poles.north"
# and raised AttributeError, exactly on the polar levers worth A/B-ing against a
# LUT histogram. Sibling-script imports are house precedent (see
# scripts/build_legacy_presets.py).
from preview_globe import apply_overrides  # noqa: E402

from gasgiant.engine import Simulation
from gasgiant.gl import GpuContext
from gasgiant.params.model import GradientStop, PaletteRow
from gasgiant.params.presets import resolve_preset

GRAY = [GradientStop(pos=p, color=[p, p, p]) for p in (0.0, 0.25, 0.5, 0.75, 1.0)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", required=True)
    ap.add_argument("--sim-res", type=int, default=1024)
    ap.add_argument("--dev-steps", type=int, default=None)
    ap.add_argument("--set", dest="sets", action="append", default=[],
                    metavar="section.field=value", help="param override")
    ap.add_argument("--keep-detail", action="store_true",
                    help="leave the detail pass on (default: off, so the "
                         "histogram reflects the SIM/band structure alone)")
    args = ap.parse_args()

    p = resolve_preset(args.preset)
    apply_overrides(p, args.sets)
    p.sim.resolution = args.sim_res
    if args.dev_steps is not None:
        p.sim.dev_steps = args.dev_steps
    # Neutralize everything that would color or reshape the output, so the only
    # thing left driving luminance is the palette INDEX.
    p.appearance = p.appearance.model_copy(update={
        "palette_rows": [PaletteRow(latitude=0.0, stops=GRAY)],
        "storm_tints": [GradientStop(pos=x, color=[x, x, x]) for x in (0.0, 0.5, 1.0)],
        "haze_amount": 0.0, "contrast": 1.0, "saturation": 1.0, "gamma": 1.0,
        "chroma_variance": 0.0, "hue_variance": 0.0, "chroma_aging": 0.0,
        "detail_chroma": 0.0, "polar_tint_strength": 0.0, "polar_canvas_value": 0.0,
    })
    if not args.keep_detail:
        p.detail = p.detail.model_copy(update={"intensity": 0.0})

    gpu = GpuContext.headless()
    sim = Simulation(p, gpu)
    sim.run_to_completion(chunk=64)
    tex, _ = sim.ensure_preview(1024)
    lum = np.asarray(gpu.read_texture(tex), dtype=np.float32)[..., 0].ravel()

    print(f"{args.preset}: LUT index distribution "
          f"(detail {'on' if args.keep_detail else 'off'})")
    print(f"  min {lum.min():.3f}  p01 {np.percentile(lum, 1):.3f}  "
          f"p50 {np.percentile(lum, 50):.3f}  "
          f"p99 {np.percentile(lum, 99):.3f}  max {lum.max():.3f}")
    print(f"  mean {lum.mean():.3f}")
    edges = np.linspace(0.0, 1.0, 11)
    hist, _ = np.histogram(np.clip(lum, 0, 1), bins=edges)
    frac = hist / hist.sum()
    for i in range(len(hist)):
        bar = "#" * int(round(frac[i] * 60))
        print(f"  [{edges[i]:.1f},{edges[i+1]:.1f})  {frac[i]*100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
