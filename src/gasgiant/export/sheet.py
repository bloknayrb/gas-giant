"""Seed contact sheet: render one small image per seed into a grid.

An iteration aid -- develop the SAME preset under a range of seeds and tile the
color maps so you can eyeball which worlds are worth a full export. Opt-in
tooling: no default/shader/params changes, so it carries no byte-identity risk.

Two correctness points:

- ONE ``Simulation`` is reused across every seed. Building N throwaway sims
  would leak GPU memory (each owns solver + preview textures until released);
  instead a single sim is re-seeded per iteration via ``update_params`` (a
  RESTART-tier diff that re-inits in place) and released exactly ONCE at the end.
- The engine object arrives through a ``build_sim`` factory, never an import:
  ``export`` sits BELOW ``engine`` in the layer order (see exporter.py), so the
  CLI passes ``Simulation`` in and tests pass a fake. ``compose_grid`` and the
  writer are pure/GL-free and unit-tested directly.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from gasgiant.export.writers import write_png8_rgb
from gasgiant.jobs import Progress

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from gasgiant.params.model import PlanetParams


def compose_grid(
    images: list[np.ndarray],
    cols: int,
    pad: int = 8,
    bg: tuple[float, float, float] = (0.05, 0.05, 0.05),
) -> np.ndarray:
    """Tile equal-sized (H, W, 3) float images into a ``cols``-wide grid.

    Pure numpy (no GL): unit-testable without a context. Every image must share
    the first image's (H, W); a mismatch is a clear error rather than a silent
    crop. The last row is padded out with the ``bg`` color when the image count
    isn't a multiple of ``cols``. ``pad`` pixels of ``bg`` surround every cell
    (including the outer border). Returns an (Hgrid, Wgrid, 3) float32 array."""
    if not images:
        raise ValueError("compose_grid needs at least one image")
    if cols < 1:
        raise ValueError(f"cols must be >= 1, got {cols}")
    h, w = images[0].shape[:2]
    for i, im in enumerate(images):
        if im.ndim != 3 or im.shape[2] < 3:
            raise ValueError(f"image {i}: expected (H, W, 3+), got {im.shape}")
        if im.shape[:2] != (h, w):
            raise ValueError(
                f"image {i}: size {im.shape[:2]} != first image {(h, w)} "
                "(contact-sheet cells must be uniform)"
            )
    pad = max(0, int(pad))
    n = len(images)
    rows = math.ceil(n / cols)
    grid_h = rows * h + (rows + 1) * pad
    grid_w = cols * w + (cols + 1) * pad
    canvas = np.empty((grid_h, grid_w, 3), dtype=np.float32)
    canvas[:] = np.asarray(bg, dtype=np.float32)
    for idx, im in enumerate(images):
        r, c = divmod(idx, cols)
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        canvas[y : y + h, x : x + w] = im[..., :3]
    return canvas


def sheet_job(
    build_sim: Callable[..., Any],
    base_params: PlanetParams,
    seeds: list[int],
    out_path: Path,
    *,
    width: int = 256,
    dev_steps: int | None = None,
    cols: int | None = None,
    pad: int = 8,
    bg: tuple[float, float, float] = (0.05, 0.05, 0.05),
    gpu: Any | None = None,
) -> Iterator[Progress]:
    """Render one color map per seed and compose them into a contact sheet PNG.

    ``build_sim(params, gpu)`` constructs the engine facade (the CLI passes
    ``engine.Simulation``; a test passes a fake). Exactly ONE sim is built --
    from ``base_params`` -- and then RE-SEEDED per iteration with
    ``update_params(base.model_copy(update={"seed": s}))`` so no per-seed GPU
    memory leaks; it is ``release()``d once in the ``finally``.

    ``dev_steps`` (if given) overrides ``base_params.sim.dev_steps`` for every
    seed (faster, coarser previews). ``cols`` defaults to ceil(sqrt(N)) for a
    roughly square sheet. Yields ``Progress`` after each seed and once for the
    final compose+write, so the CLI can print a bar and a GUI could drive it.
    """
    seeds = list(seeds)
    if not seeds:
        raise ValueError("sheet_job needs at least one seed")
    if width < 2 or width % 2 != 0:
        raise ValueError(f"width must be a positive even number (2:1 maps), got {width}")
    if cols is None:
        cols = math.ceil(math.sqrt(len(seeds)))
    if cols < 1:
        raise ValueError(f"cols must be >= 1, got {cols}")

    base = base_params
    if dev_steps is not None:
        if dev_steps < 0:
            raise ValueError(f"dev_steps must be >= 0, got {dev_steps}")
        base = base.model_copy(deep=True)
        base.sim.dev_steps = dev_steps

    total = len(seeds) + 1  # + the compose/write slice
    images: list[np.ndarray] = []
    sim = build_sim(base, gpu)
    try:
        for i, s in enumerate(seeds):
            # ONE sim, re-seeded in place (RESTART-tier): update_params releases
            # and rebuilds the solver on the SAME sim -- never a new Simulation.
            sim.update_params(base.model_copy(update={"seed": s}))
            sim.run_to_completion()
            color = sim.render_maps(width)["color"]
            images.append(np.ascontiguousarray(color[..., :3], dtype=np.float32))
            yield Progress(i + 1, total, f"seed {s} ({i + 1}/{len(seeds)})")
    finally:
        # The no-leak property: the single sim is torn down exactly once, whether
        # the loop finished or raised.
        sim.release()

    grid = compose_grid(images, cols, pad=pad, bg=bg)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_png8_rgb(out_path, grid)
    yield Progress(total, total, f"wrote {out_path}")


def run_sheet(
    build_sim: Callable[..., Any],
    base_params: PlanetParams,
    seeds: list[int],
    out_path: Path,
    *,
    width: int = 256,
    dev_steps: int | None = None,
    cols: int | None = None,
    gpu: Any | None = None,
) -> None:
    """Drain ``sheet_job`` synchronously (CLI / tests)."""
    for _ in sheet_job(
        build_sim, base_params, seeds, out_path,
        width=width, dev_steps=dev_steps, cols=cols, gpu=gpu,
    ):
        pass
