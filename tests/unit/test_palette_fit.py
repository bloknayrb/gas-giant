"""T13: palette-from-image promotion.

Covers the library fit function (``gasgiant.palette.fit.calibrate``) and the
CLI ``palette-fit`` handler. The real layering gate is ``uv run lint-imports``
(``palette.fit`` may import only ``palette.*`` + numpy/scipy); this file adds a
lightweight import-surface note plus behavioral coverage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from gasgiant.cli import _palette_fit
from gasgiant.palette.fit import calibrate
from gasgiant.params.model import PaletteRow
from gasgiant.params.presets import load_preset, resolve_preset


def _make_image(warm: bool, h: int = 200, w: int = 400) -> np.ndarray:
    """A synthetic equirect gradient (bright top -> dark bottom) with per-pixel
    texture so the per-latitude luminance quartiles are non-degenerate. ``warm``
    biases red up / blue down; the opposite for a cool image."""
    rng = np.random.default_rng(0)
    lat = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None]
    v = np.clip(np.broadcast_to(lat, (h, w)) * 0.8 + rng.random((h, w), np.float32) * 0.2, 0, 1)
    img = np.stack([v, v, v], axis=2).astype(np.float32)
    if warm:
        img[..., 0] = np.clip(img[..., 0] * 1.3, 0, 1)
        img[..., 2] = img[..., 2] * 0.6
    else:
        img[..., 2] = np.clip(img[..., 2] * 1.3, 0, 1)
        img[..., 0] = img[..., 0] * 0.6
    return img


def _mean_channel(doc: dict, ch: int) -> float:
    return float(
        np.mean([s["color"][ch] for row in doc["palette_rows"] for s in row["stops"]])
    )


def test_calibrate_deterministic():
    img = _make_image(warm=True)
    assert calibrate(img) == calibrate(img)


def test_calibrate_warm_vs_cool():
    warm = calibrate(_make_image(warm=True))
    cool = calibrate(_make_image(warm=False))
    # Robust, non-brittle property: the warm reference yields redder / less-blue
    # rows than the cool one (compared as population means over every stop).
    assert _mean_channel(warm, 0) > _mean_channel(cool, 0)
    assert _mean_channel(warm, 2) < _mean_channel(cool, 2)


def test_palette_import_surface():
    """``palette.fit`` must not reach up into higher layers (the enforcing gate
    is lint-imports; this AST scan of the actual import statements catches the
    obvious regressions in-process)."""
    import ast

    import gasgiant.palette.fit as fit

    tree = ast.parse(Path(fit.__file__).read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    for mod in modules:
        top = mod.split(".")[0]
        assert top in ("gasgiant", "numpy", "scipy", "__future__"), f"unexpected import {mod}"
        if top == "gasgiant":
            assert mod.startswith("gasgiant.palette"), f"palette.fit imports up-layer {mod}"


def test_palette_fit_cli_roundtrip(tmp_path):
    img_path = tmp_path / "ref.png"
    bgr = np.clip(_make_image(warm=True)[..., ::-1] * 255.0, 0, 255).astype(np.uint8)
    assert cv2.imwrite(str(img_path), bgr)

    out = tmp_path / "fitted.json"
    args = argparse.Namespace(
        image=img_path, preset="jupiter_like", out=out,
        anchors=None, bins=90, stops=3, fit_mode="median",
    )
    assert _palette_fit(args) == 0
    assert out.is_file()

    before = resolve_preset("jupiter_like").appearance.palette_rows
    after = load_preset(out).appearance.palette_rows
    assert after != before  # the fit baked new rows
    assert all(isinstance(r, PaletteRow) and r.stops for r in after)
    for r in after:
        for s in r.stops:
            assert all(0.0 <= c <= 1.0 for c in s.color)


def test_palette_fit_cli_missing_image(tmp_path):
    args = argparse.Namespace(
        image=tmp_path / "nope.png", preset="jupiter_like", out=tmp_path / "o.json",
        anchors=None, bins=90, stops=3, fit_mode="median",
    )
    assert _palette_fit(args) == 2
    assert not (tmp_path / "o.json").exists()
