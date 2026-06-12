from __future__ import annotations

import pytest

from gasgiant.engine import Simulation
from gasgiant.export.exporter import export_job, run_export
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu


def _params() -> PlanetParams:
    p = PlanetParams(seed=77)
    p.sim.resolution = 512
    p.sim.dev_steps = 20
    p.export.width = 2048  # multiple tiles (2x1 at TILE=1024)
    return p


def test_tiled_export_produces_valid_mapset(gpu, tmp_path):
    from gasgiant.validate import validate_mapset

    sim = Simulation(_params(), gpu)
    out = tmp_path / "mapset"
    run_export(sim, out)
    assert (out / "color.png").is_file()
    assert (out / "height.exr").is_file()
    assert (out / "mapset.json").is_file()
    report = validate_mapset(out)
    assert report.ok, report.summary()


def test_tiles_are_seamless(gpu, tmp_path):
    """No discontinuity at the tile boundary (x=1024)."""
    import numpy as np

    from gasgiant.export.writers import read_png16

    sim = Simulation(_params(), gpu)
    out = tmp_path / "mapset"
    run_export(sim, out)
    img = read_png16(out / "color.png")
    boundary = np.abs(img[:, 1024].astype(np.float32) - img[:, 1023].astype(np.float32)).mean()
    interior = np.abs(img[:, 512].astype(np.float32) - img[:, 511].astype(np.float32)).mean()
    assert boundary < max(3.0 * interior, 1e-3), (
        f"tile seam: boundary diff {boundary:.5f} vs interior {interior:.5f}"
    )


def test_cancellation_removes_partial_output(gpu, tmp_path):
    sim = Simulation(_params(), gpu)
    out = tmp_path / "mapset"
    keep = out / "users_own_file.txt"
    out.mkdir(parents=True)
    keep.write_text("precious")

    job = export_job(sim, out)
    next(job)  # start it
    next(job)
    job.close()  # cancel

    assert not (out / "color.png").exists()
    assert not (out / "mapset.json").exists()
    assert keep.read_text() == "precious"  # never touches the user's files


def _emission_params() -> PlanetParams:
    p = _params()
    p.emission.thermal_strength = 0.6
    p.emission.lightning_strength = 0.4
    p.emission.aurora_strength = 0.5
    return p


def test_emission_export_round_trip(gpu, tmp_path):
    import numpy as np

    from gasgiant.export.manifest import read_manifest
    from gasgiant.export.writers import read_exr_rgba
    from gasgiant.validate import validate_mapset

    sim = Simulation(_emission_params(), gpu)
    out = tmp_path / "mapset"
    run_export(sim, out)
    assert (out / "emission.exr").is_file()
    manifest = read_manifest(out)
    entry = manifest["maps"]["emission"]
    assert entry["channels"] == 4
    assert entry["colorspace"] == "non-color"
    assert len(entry["aurora_color"]) == 3
    em = read_exr_rgba(out / "emission.exr")
    assert em.shape == (1024, 2048, 4)
    assert np.isfinite(em).all() and em.min() >= 0.0
    assert em[..., :3].max() > 0.0 and em[..., 3].max() > 0.0
    report = validate_mapset(out)
    assert report.ok, report.summary()


def test_emission_tiles_match_render_maps(gpu, tmp_path):
    """The tiled emission must equal the one-shot render_maps emission —
    the tile-apron contract extended to the new map."""
    import numpy as np

    from gasgiant.export.writers import read_exr_rgba

    sim = Simulation(_emission_params(), gpu)
    out = tmp_path / "mapset"
    run_export(sim, out)
    tiled = read_exr_rgba(out / "emission.exr")
    whole = sim.render_maps(2048)["emission"]
    np.testing.assert_array_equal(tiled, whole)


def test_default_export_writes_no_emission(gpu, tmp_path):
    from gasgiant.export.manifest import read_manifest

    sim = Simulation(_params(), gpu)
    out = tmp_path / "mapset"
    run_export(sim, out)
    assert not (out / "emission.exr").exists()
    assert "emission" not in read_manifest(out)["maps"]


def test_cancellation_removes_emission(gpu, tmp_path):
    sim = Simulation(_emission_params(), gpu)
    out = tmp_path / "mapset"
    job = export_job(sim, out)
    next(job)
    next(job)
    job.close()
    assert not (out / "emission.exr").exists()


def test_snapshot_isolates_export_from_live_edits(gpu, tmp_path):
    """Mutating live params mid-export must not affect the output."""
    import numpy as np

    from gasgiant.export.writers import read_png16

    sim = Simulation(_params(), gpu)
    sim.run_to_completion()

    out_a = tmp_path / "a"
    run_export(sim, out_a)

    out_b = tmp_path / "b"
    job = export_job(sim, out_b)
    for i, _ in enumerate(job):
        if i == 1:
            # Mid-export POST-tier edit (would shift colors without a snapshot).
            mutated = sim.params.model_copy(deep=True)
            mutated.appearance.haze_amount = 0.9
            sim.update_params(mutated)
    a = read_png16(out_a / "color.png")
    b = read_png16(out_b / "color.png")
    np.testing.assert_array_equal(a, b)
