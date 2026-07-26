"""Render one showcase image per factory preset for the README.

Each preset is developed for a fixed number of steps (default 1000) on a
reduced sim grid so the whole set renders in a couple of hours under software
GL (llvmpipe); the exported color map is downsampled to an 8-bit sRGB PNG under
``docs/img/presets/``.

Usage (needs a GL 4.3 context; llvmpipe works):

    LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a \
        uv run python scripts/render_readme_examples.py

This is a one-off doc-asset generator, not part of the test suite.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from gasgiant.engine import Simulation
from gasgiant.export.exporter import run_export
from gasgiant.params.presets import resolve_preset

PRESETS = [
    "gas_giant_warm",
    "jupiter_like",
    "jupiter_vorticity",
    "saturn_pale",
    "ice_giant",
    "neptune",
    "ember_dwarf",
]

# Presets whose committed image must NOT use the reduced default sim grid.
# ember_dwarf is unusually resolution-sensitive: its tears are made by convective
# excursions, and the fraction of the disc clearing the ember knee grows by about
# half again between sim-res 1024 and its shipped 4096 (13.0% -> 20.2% of the disc
# above LUT index 0.80, measured with scripts/probe_lut_usage.py --keep-detail).
# The reduced grid understates it badly, so it is always rendered at its shipped
# resolution regardless of --sim-res.
SIM_RES_OVERRIDES = {"ember_dwarf": 4096}


def _to_srgb8_png(src_png16: Path, dst_png8: Path, width: int) -> None:
    """Downsample a 16-bit RGB color map to an 8-bit sRGB PNG of ``width`` px."""
    img = cv2.imread(str(src_png16), cv2.IMREAD_UNCHANGED)  # BGR, uint16
    if img is None:
        raise RuntimeError(f"could not read {src_png16}")
    h = width // 2
    resized = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
    out8 = (resized.astype(np.float64) / 65535.0 * 255.0 + 0.5).astype(np.uint8)
    dst_png8.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_png8), out8, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev-steps", type=int, default=1000)
    ap.add_argument("--sim-res", type=int, default=1024,
                    help="sim grid width (reduced for tractable software-GL renders)")
    ap.add_argument("--export-res", type=int, default=2048,
                    help="equirect map width fed to the exporter")
    ap.add_argument("--out-width", type=int, default=2048,
                    help="width of the final 8-bit README PNG")
    ap.add_argument("--presets", nargs="+", default=PRESETS,
                    metavar="NAME", help="subset of presets to render")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("docs/img/presets"))
    ap.add_argument("--work-dir", type=Path,
                    default=Path("out/readme_examples"))
    args = ap.parse_args()

    for name in args.presets:
        t0 = time.perf_counter()
        params = resolve_preset(name)
        # STRUCTURAL, not advisory: a plain `uv run python scripts/render_readme_examples.py`
        # would otherwise re-render ember_dwarf at the reduced default and overwrite
        # its committed image with the very render this file calls misrepresentative.
        # A comment cannot enforce that, so the override lives in code.
        sim_res = SIM_RES_OVERRIDES.get(name, args.sim_res)
        params.sim.resolution = sim_res
        params.sim.dev_steps = args.dev_steps
        params.export.width = args.export_res
        sim = Simulation(params)
        work = args.work_dir / name
        run_export(sim, work)
        dst = args.out_dir / f"{name}.png"
        _to_srgb8_png(work / "color.png", dst, args.out_width)
        sim.gpu.release()
        print(f"[{name}] {args.dev_steps} steps @ {sim_res} -> {dst} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
