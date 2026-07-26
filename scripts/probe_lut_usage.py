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

# The fields the neutralization block below overwrites. An `--set` on any of them
# is REFUSED rather than applied, because overrides run LAST (see the note in
# main()) and would therefore win -- silently breaking the one invariant that
# makes this tool mean anything, that rendered luminance IS the palette index.
#
# This is the mirror image of the bug the ordering fixed, and the more dangerous
# half. Applying overrides first made `--set sim.resolution=4096` silently
# IGNORED: wrong, but the printed histogram was still a real histogram. Applying
# them last makes `--set appearance.chroma_aging=0.35` silently EFFECTIVE: a
# nonzero chroma_aging modulates the very red channel this tool reads back, so the
# output stops being a LUT-index distribution while the header still labels it
# one -- and preset knee positions are authored from exactly these numbers.
#
# For a chroma A/B, use the REAL palette and compare mean on-disc RGB (how
# chroma_aging was actually measured on ember_dwarf), or scripts/preview_globe.py
# --set. Not this tool: it has no palette left to tint.
_NEUTRALIZED = frozenset(
    f"appearance.{f}" for f in (
        "palette_rows", "storm_tints", "haze_amount", "contrast", "saturation",
        "gamma", "chroma_variance", "hue_variance", "chroma_aging",
        "detail_chroma", "polar_tint_strength", "polar_canvas_value",
    )
)


def _reject_neutralized(sets: list[str], keep_detail: bool) -> None:
    blocked = dict.fromkeys(_NEUTRALIZED)
    if not keep_detail:
        blocked["detail.intensity"] = None    # zeroed unless --keep-detail
    for spec in sets:
        path = spec.partition("=")[0].strip()
        if path in blocked:
            raise SystemExit(
                f"refusing --set {path}: this tool overwrites that field to make "
                f"rendered luminance equal the palette INDEX, and overrides are "
                f"applied last, so the override would win and the printed "
                f"histogram would no longer be a LUT-index distribution.\n"
                f"For a chroma/appearance A/B use scripts/preview_globe.py --set "
                f"(real palette) and compare mean on-disc RGB."
                + ("" if keep_detail or not path.startswith("detail.")
                   else "\nFor detail.intensity specifically: pass --keep-detail.")
            )


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

    # Before any GL work: an override that would invalidate the measurement is a
    # usage error, and failing in under a second beats failing after a 4096 develop.
    _reject_neutralized(args.sets, args.keep_detail)

    p = resolve_preset(args.preset)
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
    # Overrides go LAST, matching preview_globe's precedence. Applied first they
    # would be printed as accepted and then silently clobbered by the lines above:
    # `--set sim.resolution=4096` would report 4096 and measure at 1024, i.e.
    # hand back the 13.0% histogram while the operator reads it as the 20.2% one.
    #
    # Ordering last is only SAFE because _reject_neutralized() already refused any
    # override that targets a field the block above overwrites -- otherwise this
    # line would hand the operator a silently invalid histogram, which is worse
    # than the ignored-override bug it fixes. The two must stay together.
    apply_overrides(p, args.sets)

    gpu = GpuContext.headless()
    sim = Simulation(p, gpu)
    sim.run_to_completion(chunk=64)
    tex, _ = sim.ensure_preview(1024)
    lum = np.asarray(gpu.read_texture(tex), dtype=np.float32)[..., 0].ravel()

    # Resolution and dev_steps belong in the header, not just in the invocation:
    # the numbers this tool prints move materially with sim-res (ember_dwarf's
    # ember fraction is 13.0% at 1024 vs 20.2% at 4096), so a pasted histogram
    # without them is unattributable.
    print(f"{args.preset}: LUT index distribution "
          f"(sim-res {p.sim.resolution}, dev_steps {p.sim.dev_steps}, "
          f"detail {'on' if args.keep_detail else 'off'})")
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
