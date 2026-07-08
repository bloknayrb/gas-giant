"""T7 sequence-export upgrades: all-maps per-frame output + exact cancellation
cleanup of the new per-map files.

Structural assertions only (file set / manifest shape / cleanup) — no
hash comparisons: emission rides the vorticity-adjacent derive and per-map
frames are as noise-bound as color. Determinism is covered for the color path
in test_export_sequence.py.
"""

from __future__ import annotations

import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu


def _params(width: int = 512, dev_steps: int = 12, emission: bool = True) -> PlanetParams:
    p = PlanetParams(seed=7)
    p.sim.resolution = 512  # the model's lower bound (256 fails validation)
    p.sim.dev_steps = dev_steps
    p.export.width = width
    # emission.enabled is a derived property (any strength > 0)
    p.emission.thermal_strength = 0.6 if emission else 0.0
    p.emission.lightning_strength = 0.0
    p.emission.aurora_strength = 0.0
    return p


def test_all_maps_sequence_writes_full_file_set(gpu, tmp_path):
    from gasgiant.export.exporter import run_export_sequence
    from gasgiant.export.manifest import read_manifest

    sim = Simulation(_params(emission=True), gpu)
    out = tmp_path / "seq"
    run_export_sequence(sim, out, frames=3, steps_per_frame=4, all_maps=True)

    frames = out / "frames"
    for i in range(3):
        assert (frames / f"frame_{i:04d}.png").is_file(), f"missing color frame {i}"
        assert (frames / f"height_{i:04d}.png").is_file(), f"missing height frame {i}"
        assert (frames / f"emission_{i:04d}.exr").is_file(), f"missing emission frame {i}"
    # no stray frame index past the requested count
    assert not (frames / "frame_0003.png").exists()

    m = read_manifest(out)  # schema-validated on read
    assert m["frames"]["count"] == 3
    assert m["frames"]["files"] == [f"frames/frame_{i:04d}.png" for i in range(3)]
    assert m["frames"]["maps"]["height"] == [f"frames/height_{i:04d}.png" for i in range(3)]
    assert m["frames"]["maps"]["emission"] == [
        f"frames/emission_{i:04d}.exr" for i in range(3)
    ]
    assert "video" not in m["frames"]
    # frame-0 height in frames/ is a 16-bit gray PNG (not the base float EXR)
    from gasgiant.export.writers import read_png16

    h0 = read_png16(frames / "height_0000.png")
    assert h0.ndim == 2 and h0.shape == (256, 512)


def test_all_maps_without_emission_writes_no_emission_frames(gpu, tmp_path):
    from gasgiant.export.exporter import run_export_sequence
    from gasgiant.export.manifest import read_manifest

    sim = Simulation(_params(emission=False), gpu)
    out = tmp_path / "seq"
    run_export_sequence(sim, out, frames=3, steps_per_frame=4, all_maps=True)

    frames = out / "frames"
    assert not list(frames.glob("emission_*.exr"))
    assert all((frames / f"height_{i:04d}.png").is_file() for i in range(3))
    m = read_manifest(out)
    assert "emission" not in m["frames"]["maps"]
    assert m["frames"]["maps"]["height"] == [f"frames/height_{i:04d}.png" for i in range(3)]


def test_all_maps_cancellation_removes_exactly_new_files(gpu, tmp_path):
    from gasgiant.export.exporter import export_sequence_job

    sim = Simulation(_params(emission=True), gpu)
    out = tmp_path / "seq"
    out.mkdir(parents=True)
    keep = out / "users_own_file.txt"
    keep.write_text("precious")

    job = export_sequence_job(sim, out, frames=4, steps_per_frame=4, all_maps=True)
    saw_frame_1 = False
    for prog in job:
        # per-tile progress now; frame 0's per-map files are already on disk
        if prog.message.startswith("frame 1"):
            saw_frame_1 = True
            break
    assert saw_frame_1
    job.close()  # finally-block cleanup

    # every file the job created is gone (color + the new per-map frame-0 files)
    frames = out / "frames"
    assert not frames.exists() or not any(frames.iterdir())
    assert not (out / "mapset.json").exists()
    assert not (out / "color.png").exists()
    assert not (out / "height.exr").exists()
    assert not (out / "emission.exr").exists()
    # the user's own file is untouched
    assert keep.read_text() == "precious"
