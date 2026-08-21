"""export.emission_half: a per-export format choice for emission.exr.

Default off, so a default export is unchanged. On, emission.exr is written as
HALF and the manifest says so -- which the validator and the Blender importer
both have to cope with.
"""

from __future__ import annotations

import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams, ProjectionKind, SolverType

pytestmark = pytest.mark.gpu


def _params(half: bool, width: int = 512) -> PlanetParams:
    p = PlanetParams(seed=5)
    p.solver.type = SolverType.KINEMATIC
    p.sim.resolution = 512
    p.sim.dev_steps = 12
    p.export.width = width
    # emission.enabled is derived from the strengths; aurora is also the channel
    # whose RAW (un-log-compressed) alpha makes half quantization interesting.
    p.emission.aurora_strength = 2.0
    p.export.emission_half = half
    return p


def test_emission_half_off_is_byte_identical(gpu, tmp_path):
    """With the lever off, two independent exports agree byte for byte.

    Read this for exactly what it is: no in-test comparison CAN reach the
    pre-lever code, so this pins DETERMINISM of the default path, not
    off-vs-before identity. The evidence for the latter is the 4096
    jupiter_like SHA-256 pair and p05 --check in the PR body, neither of which
    CI re-runs.

    Keep "identical" in the name regardless -- gpu-smoke selects on
    ``-k "identical or noop or no_op"``, so renaming drops this out of the
    PR-blocking job.
    """
    from gasgiant.export.exporter import run_export

    a, b = tmp_path / "a", tmp_path / "b"
    run_export(Simulation(_params(False), gpu), a)
    run_export(Simulation(_params(False), gpu), b)
    assert (a / "emission.exr").read_bytes() == (b / "emission.exr").read_bytes()
    assert (a / "color.png").read_bytes() == (b / "color.png").read_bytes()


@pytest.mark.parametrize("projection", [ProjectionKind.EQUIRECT, ProjectionKind.CUBE])
def test_emission_half_writes_half_and_declares_it(gpu, tmp_path, projection):
    """Buffer dtype, on-disk pixel type and manifest token must all agree -- the
    manifest is the only thing a downstream reader can consult."""
    import OpenEXR

    from gasgiant.export.exporter import run_export
    from gasgiant.export.manifest import read_manifest

    p = _params(True)
    p.export.projection = projection
    out = tmp_path / "half"
    run_export(Simulation(p, gpu), out)

    entry = read_manifest(out)["maps"]["emission"]
    assert entry["format"] == "exr16f"

    files = (
        [out / rel for rel in entry["faces"].values()]
        if projection is ProjectionKind.CUBE
        else [out / entry["file"]]
    )
    for f in files:
        with OpenEXR.File(str(f)) as fh:
            assert fh.channels()["RGBA"].pixels.dtype.name == "float16", f.name


def test_half_emission_is_smaller_than_full(gpu, tmp_path):
    from gasgiant.export.exporter import run_export

    sizes = {}
    for half in (False, True):
        out = tmp_path / f"h{int(half)}"
        run_export(Simulation(_params(half), gpu), out)
        sizes[half] = (out / "emission.exr").stat().st_size
    assert sizes[True] < sizes[False], sizes


def test_half_emission_still_validates(gpu, tmp_path):
    """Including the raw aurora ALPHA channel, whose float16 quantization step
    is what makes a naive absolute seam floor fail on rounding alone."""
    from gasgiant.export.exporter import run_export
    from gasgiant.validate import validate_mapset

    out = tmp_path / "half"
    run_export(Simulation(_params(True), gpu), out)
    report = validate_mapset(out)
    assert report.ok, report.summary()
    assert any("emission_alpha" in c.name for c in report.checks), (
        "the alpha channel must actually be under test"
    )


def test_unknown_map_format_raises_instead_of_silently_passing(tmp_path, monkeypatch):
    """Pre-existing hazard: an unrecognized format fell through BOTH read
    branches, so the map was never added to `maps` and Report.ok -- an all()
    over checks it never contributed -- stayed True.

    The manifest schema's format enum stops a truly bogus token at read time, so
    the reachable case is a token that IS in the enum but has no read branch --
    exactly what adding "exr16f" to the enum and forgetting seams.py would have
    produced. Simulated by patching the manifest the validator reads.
    """
    from gasgiant.export import manifest as manifest_mod
    from gasgiant.validate import validate_mapset

    monkeypatch.setattr(manifest_mod, "read_manifest", lambda _d: {
        "schema_version": 1,
        "projection": "equirectangular",
        "maps": {"color": {"file": "color.png", "format": "exr8f", "channels": 3}},
    })
    with pytest.raises(ValueError, match="unknown format"):
        validate_mapset(tmp_path)


def test_sequence_frames_match_the_base_map_precision(gpu, tmp_path):
    """Frame emission EXRs must be written at the SAME precision as the base map.

    The sequence allocates its own frame buffers, separately from export_job's,
    so the dtype has to be threaded to both. It was not: with emission_half on,
    the base emission.exr came out half while every frame stayed float32 -- no
    error, no failing test, and no way for a reader to notice, because the
    manifest's frames block carries no format token at all.
    """
    import OpenEXR

    from gasgiant.export.exporter import run_export_sequence

    out = tmp_path / "seq"
    run_export_sequence(
        Simulation(_params(True), gpu), out, frames=3, steps_per_frame=4, all_maps=True,
    )

    def _dtype(path):
        with OpenEXR.File(str(path)) as fh:
            return fh.channels()["RGBA"].pixels.dtype.name

    base = _dtype(out / "emission.exr")
    assert base == "float16"
    for i in range(1, 3):
        frame = out / "frames" / f"emission_{i:04d}.exr"
        assert frame.is_file(), frame
        assert _dtype(frame) == base, f"{frame.name} is {_dtype(frame)}, base is {base}"
