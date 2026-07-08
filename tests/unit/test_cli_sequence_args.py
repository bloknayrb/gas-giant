"""CLI sequence-flag validation: rejected before any GL work, so these run
in the non-GPU tier."""

from __future__ import annotations

from gasgiant.cli import main


def test_frames_requires_steps_per_frame(tmp_path):
    assert main(["export", "--frames", "4", "--out", str(tmp_path / "x")]) == 2


def test_steps_per_frame_requires_frames(tmp_path):
    assert main(["export", "--steps-per-frame", "8", "--out", str(tmp_path / "x")]) == 2


def test_frames_must_be_positive(tmp_path):
    rc = main([
        "export", "--frames", "0", "--steps-per-frame", "8",
        "--out", str(tmp_path / "x"),
    ])
    assert rc == 2


def test_steps_per_frame_must_be_positive(tmp_path):
    rc = main([
        "export", "--frames", "4", "--steps-per-frame", "0",
        "--out", str(tmp_path / "x"),
    ])
    assert rc == 2


def test_ramp_to_requires_frames(tmp_path):
    rc = main([
        "export", "--ramp-to", "saturn_pale", "--out", str(tmp_path / "x"),
    ])
    assert rc == 2


def test_sequence_export_rejects_cube_projection():
    """A cube-projection sequence has nothing to sequence (frame 0 writes six
    face files, no color.png): export_sequence_job fails fast, before any
    dev/GL work, instead of crashing at the frame-0 copy and deleting the
    completed cube export's manifest."""
    from pathlib import Path

    import pytest

    from gasgiant.export.exporter import export_sequence_job
    from gasgiant.params.model import PlanetParams, ProjectionKind

    p = PlanetParams(seed=0)
    p.export.projection = ProjectionKind.CUBE

    class _Sim:  # only .params is touched before the guard raises
        params = p

    job = export_sequence_job(_Sim(), Path("unused"), frames=2, steps_per_frame=1)
    with pytest.raises(ValueError, match="equirect"):
        next(job)
