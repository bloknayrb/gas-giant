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
