"""The 32K export cap and the guards that keep the paths which CANNOT serve it
from being reached by accident.

All non-GPU: every guard here is specified to fire before any GL or dev work,
so exercising them must not need a context.
"""

from __future__ import annotations

import pytest

from gasgiant.app import main as app_main
from gasgiant.cli import _range_error, main
from gasgiant.engine.facade import RENDER_MAPS_MAX_WIDTH, ResolutionUnsupportedError
from gasgiant.export.exporter import H264_MAX_DIM, TILE, enumerate_tiles, export_sequence_job
from gasgiant.params.model import PlanetParams


def _width_bounds() -> tuple[int, int]:
    info = PlanetParams.model_fields["export"].annotation.model_fields["width"]
    lo = next(m.ge for m in info.metadata if hasattr(m, "ge"))
    hi = next(m.le for m in info.metadata if hasattr(m, "le"))
    return int(lo), int(hi)


# -- the cap itself ------------------------------------------------------------

def test_export_width_accepts_32768() -> None:
    assert _width_bounds()[1] == 32768
    p = PlanetParams()
    p.export.width = 32768  # validate_assignment=True, so this is the real check
    assert p.export.width == 32768


def test_export_width_still_rejects_above_the_cap() -> None:
    with pytest.raises(ValueError):
        PlanetParams().export.width = 32769


def test_export_resolutions_are_ascending_and_unique() -> None:
    widths = [w for w, _ in app_main.EXPORT_RESOLUTIONS]
    assert widths == sorted(widths), "combo entries must ascend"
    assert len(widths) == len(set(widths)), "duplicate width in the combo"
    assert 32768 in widths, "the raised cap should be reachable from the GUI"


# -- CLI range guards: a clean error, never a pydantic traceback ----------------

@pytest.mark.parametrize("flag,value", [
    ("--res", "99999"),      # above export.width's cap
    ("--res", "1"),          # below its floor
    ("--dev-steps", "99999"),  # the same defect on the adjacent assignment
])
def test_out_of_range_override_exits_2(tmp_path, capsys, flag, value) -> None:
    rc = main(["export", "--preset", "jupiter_like", flag, value,
               "--out", str(tmp_path / "x")])
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("error:"), f"expected a CLI error, got: {err!r}"
    assert "Traceback" not in err
    assert flag in err


@pytest.mark.parametrize("value", [512, 2048, 16384, 32768])
def test_in_range_widths_pass_the_guard(value) -> None:
    """The guard is checked directly rather than through a full `main()` run:
    driving the CLI with an in-range width would actually render, and at 32768
    that is minutes of GPU work inside the non-GPU tier."""
    assert _range_error("export", "width", "--res", value) is None


def test_guard_reports_the_schema_bounds_not_a_literal() -> None:
    lo, hi = _width_bounds()
    msg = _range_error("export", "width", "--res", hi + 1)
    assert msg is not None and f"({lo}..{hi})" in msg


def test_checkpoint_subcommand_shares_the_guard(tmp_path, capsys) -> None:
    rc = main(["checkpoint", "--preset", "jupiter_like", "--res", "99999",
               "--out", str(tmp_path / "c.npz")])
    assert rc == 2
    assert "out of range" in capsys.readouterr().err


# -- video guard ---------------------------------------------------------------

class _StubSim:
    """Enough of a Simulation for the guards that run before any GL work."""

    def __init__(self, width: int) -> None:
        self.params = PlanetParams()
        self.params.export.width = width


def test_sequence_video_refused_above_h264_max() -> None:
    job = export_sequence_job(_StubSim(32768), object(), 4, 8, video=True)
    with pytest.raises(ValueError, match="H.264"):
        next(job)


def test_sequence_video_allowed_at_h264_max() -> None:
    """The guard must not fire AT the limit -- 16384 is a legal H.264 dimension."""
    job = export_sequence_job(_StubSim(H264_MAX_DIM), object(), 4, 8, video=True)
    with pytest.raises(Exception) as exc:
        next(job)
    assert "H.264" not in str(exc.value)


def test_sequence_without_video_is_not_width_capped() -> None:
    """A 32K still-frame sequence is fine; only the mp4 encode has the limit."""
    job = export_sequence_job(_StubSim(32768), object(), 4, 8, video=False)
    with pytest.raises(Exception) as exc:
        next(job)
    assert "H.264" not in str(exc.value)


# -- render_maps cap -----------------------------------------------------------

def test_render_maps_refuses_above_its_cap() -> None:
    """Fires before run_to_completion, so no GL context is needed to see it."""
    sim = _StubSim(2048)
    with pytest.raises(ResolutionUnsupportedError, match="tiled"):
        type(sim).render_maps = None  # guard against accidental real call
        from gasgiant.engine.facade import Simulation
        Simulation.render_maps(sim, RENDER_MAPS_MAX_WIDTH + 1)


def test_render_maps_cap_is_at_or_below_the_export_cap() -> None:
    """The whole point of the cap is that export.width may legally exceed it."""
    assert _width_bounds()[1] > RENDER_MAPS_MAX_WIDTH


# -- tile enumeration at the new cap -------------------------------------------

@pytest.mark.parametrize("w,h", [
    (32768, 16384),  # the new cap
    (16384, 8192),   # the old one
    (24576, 12288),  # V7's reachable proxy
    (1000, 500),     # not a multiple of TILE
    (512, 256),      # smaller than one tile
])
def test_tiles_cover_the_plane_exactly(w, h) -> None:
    tiles = enumerate_tiles(w, h)
    assert len(tiles) == ((w + TILE - 1) // TILE) * ((h + TILE - 1) // TILE)
    assert len(set(tiles)) == len(tiles), "duplicate tile corner"
    covered = 0
    for x, y in tiles:
        tw, th = min(TILE, w - x), min(TILE, h - y)
        assert tw > 0 and th > 0, "tile starts outside the plane"
        covered += tw * th
    assert covered == w * h, "tiles must tile the plane with no gap or overlap"


def test_tiles_are_row_major_so_bands_complete() -> None:
    """Band-completeness is the property a streaming/banded consumer relies on:
    every tile of band k is emitted before any tile of band k+1."""
    ys = [y for _x, y in enumerate_tiles(4096, 2048)]
    assert ys == sorted(ys)
