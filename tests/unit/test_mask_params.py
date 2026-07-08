"""T11 imported paint mask: params, decode, and path-resolution policy.

All GPU-free. The forced-MASK no-op byte-identity and per-target behavior live
in tests/gpu/test_mask.py; here we cover the model, the ``decode_image`` sidecar
reader, the preset/session/CLI path-resolution sites, and the widget mapping.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from gasgiant import cli
from gasgiant.export.writers import decode_image
from gasgiant.params.model import MaskParams, PlanetParams
from gasgiant.params.presets import (
    load_preset,
    resolve_mask_path,
    save_preset,
)


def _write_mask(path: Path, w: int = 32, h: int = 16) -> Path:
    """Write a deterministic 2:1 grayscale PNG at ``path``."""
    grad = np.linspace(0, 255, w * h, dtype=np.float32).reshape(h, w).astype(np.uint8)
    assert cv2.imwrite(str(path), grad)
    return path


# -- MaskParams model -----------------------------------------------------------


def test_mask_defaults_are_all_no_op():
    m = MaskParams()
    assert m.file is None
    assert m.band_fade == 0.0
    assert m.emission_gain == 0.0
    assert m.detail_gain == 0.0


def test_planet_params_composes_mask():
    assert isinstance(PlanetParams().mask, MaskParams)


@pytest.mark.parametrize("field", ["band_fade", "emission_gain", "detail_gain"])
def test_mask_gain_bounds(field):
    with pytest.raises(ValidationError):
        MaskParams(**{field: 1.5})
    with pytest.raises(ValidationError):
        MaskParams(**{field: -0.1})


def test_mask_fields_carry_no_rand_metadata():
    # A rand draw on a POST art-direction field would reshuffle every later
    # randomize draw AND could silently enable a mask target.
    for name in ("file", "band_fade", "emission_gain", "detail_gain"):
        extra = MaskParams.model_fields[name].json_schema_extra
        assert isinstance(extra, dict)
        assert "rand" not in extra, name


def test_mask_fields_have_literate_help():
    for name in ("file", "band_fade", "emission_gain", "detail_gain"):
        assert MaskParams.model_fields[name].description


# -- decode_image ---------------------------------------------------------------


def test_decode_image_valid_2to1(tmp_path):
    arr = decode_image(_write_mask(tmp_path / "m.png"))
    assert arr.shape == (16, 32)
    assert arr.dtype == np.float32
    assert float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0


def test_decode_image_rejects_non_2to1(tmp_path):
    square = tmp_path / "square.png"
    assert cv2.imwrite(str(square), np.zeros((32, 32), np.uint8))
    with pytest.raises(ValueError, match="2:1 equirect"):
        decode_image(square)


def test_decode_image_missing_file_raises(tmp_path):
    with pytest.raises(OSError):
        decode_image(tmp_path / "does_not_exist.png")


def test_decode_image_converts_color_to_gray(tmp_path):
    color = tmp_path / "color.png"
    assert cv2.imwrite(str(color), np.zeros((16, 32, 3), np.uint8))
    arr = decode_image(color)
    assert arr.ndim == 2 and arr.shape == (16, 32)


# -- path resolution: load / save / relativize ----------------------------------


def test_resolve_mask_path_relative_to_base(tmp_path):
    p = PlanetParams()
    p.mask.file = "sub/paint.png"
    resolve_mask_path(p, tmp_path)
    assert p.mask.file == str((tmp_path / "sub" / "paint.png").resolve())


def test_resolve_mask_path_none_stays_none(tmp_path):
    p = PlanetParams()
    resolve_mask_path(p, tmp_path)
    assert p.mask.file is None


def test_load_preset_resolves_relative_mask_to_absolute(tmp_path):
    preset_dir = tmp_path / "presets"
    preset_dir.mkdir()
    _write_mask(preset_dir / "paint.png")
    # A hand-written preset carrying a RELATIVE sidecar path.
    doc = {
        "preset_format": 1,
        "params": {**PlanetParams(name="masked").model_dump(), "mask": {
            "file": "paint.png", "band_fade": 0.5,
            "emission_gain": 0.0, "detail_gain": 0.0,
        }},
    }
    path = preset_dir / "look.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    loaded = load_preset(path)
    assert loaded.mask.file == str((preset_dir / "paint.png").resolve())
    assert loaded.mask.band_fade == 0.5


def test_save_preset_relativizes_and_copies_sidecar(tmp_path):
    # Mask lives in a SEPARATE folder from where the preset is saved.
    assets = tmp_path / "assets"
    assets.mkdir()
    mask = _write_mask(assets / "brush.png")
    p = PlanetParams()
    p.mask.file = str(mask.resolve())
    p.mask.detail_gain = 0.4

    preset_dir = tmp_path / "out"
    path = preset_dir / "portable.json"
    save_preset(p, path)

    # The JSON stores just the sidecar filename (portable) ...
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["params"]["mask"]["file"] == "brush.png"
    # ... the PNG was copied next to it ...
    assert (preset_dir / "brush.png").is_file()
    # ... the caller's in-memory params still hold the ABSOLUTE path ...
    assert p.mask.file == str(mask.resolve())
    # ... and a reload resolves back to an absolute path next to the preset.
    reloaded = load_preset(path)
    assert reloaded.mask.file == str((preset_dir / "brush.png").resolve())
    assert reloaded.mask.detail_gain == 0.4


def test_session_save_keeps_absolute_path(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    mask = _write_mask(assets / "s.png")
    p = PlanetParams()
    p.mask.file = str(mask.resolve())
    session = tmp_path / "session.json"
    save_preset(p, session, name="session", relativize_mask=False)
    doc = json.loads(session.read_text(encoding="utf-8"))
    assert doc["params"]["mask"]["file"] == str(mask.resolve())
    # No sidecar copied next to the session doc.
    assert not (tmp_path / "s.png").exists()


# -- CLI resolve + missing-file error -------------------------------------------


def test_cli_export_errors_on_missing_mask(tmp_path, capsys):
    preset_dir = tmp_path / "p"
    preset_dir.mkdir()
    p = PlanetParams()
    p.mask.file = str((preset_dir / "gone.png").resolve())  # never created
    save_preset(p, preset_dir / "look.json")  # stores "gone.png", no sidecar to copy

    rc = cli.main([
        "export", "--preset", str(preset_dir / "look.json"),
        "--out", str(tmp_path / "out"),
    ])
    assert rc == 2  # errored BEFORE any GL/Simulation work
    assert "mask file not found" in capsys.readouterr().err


# -- checkpoint / engine: missing file -> disabled, never crash -----------------


class _FakeSim:
    """The subset of Simulation state _sync_mask/set_mask touch, without GL.
    set_mask(None) never allocates a texture, so the missing-file disable path
    is exercisable GPU-free (the checkpoint-restore contract)."""

    def __init__(self, params: PlanetParams) -> None:
        self.params = params
        self._mask_tex = None
        self._post_dirty = False
        self._emission_preview_dirty = False

    from gasgiant.engine.facade import Simulation as _S
    set_mask = _S.set_mask
    _sync_mask = _S._sync_mask
    del _S


def test_sync_mask_missing_file_disables_without_crash(caplog):
    p = PlanetParams()
    p.mask.file = "/definitely/not/here/mask.png"
    sim = _FakeSim(p)
    sim._sync_mask()  # must not raise
    assert sim._mask_tex is None
    assert sim._post_dirty is True


def test_sync_mask_none_file_is_disabled(tmp_path):
    sim = _FakeSim(PlanetParams())
    sim._sync_mask()
    assert sim._mask_tex is None


# -- GUI widget mapping ---------------------------------------------------------


def test_leaf_kind_maps_optional_str():
    panels = pytest.importorskip("gasgiant.app.panels")
    info = MaskParams.model_fields["file"]
    assert panels.leaf_kind("file", info, None) == "optional_str"
