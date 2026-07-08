"""Sequence export: Simulation.extend_run + export_sequence_job.

Determinism scope (per testing policy): the kinematic path is byte-exact, so
the golden A/B test hash-compares kinematic sequence files. The vorticity
path carries SOR LSB noise that COMPOUNDS across frames, so vorticity gets
STRUCTURAL assertions only (frame count/naming, frames differ pairwise,
manifest schema-valid, cancellation cleanup) — never hash comparisons.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu


def _kin_params(dev_steps: int = 20, width: int = 512) -> PlanetParams:
    p = PlanetParams(seed=42)
    p.sim.resolution = 512
    p.sim.dev_steps = dev_steps
    p.export.width = width
    return p


def _vort_params() -> PlanetParams:
    p = _kin_params()
    p.solver.type = "vorticity"
    return p


# -- Simulation.extend_run ----------------------------------------------------


def test_extend_run_advances_exact_steps(gpu):
    p = _kin_params()
    sim = Simulation(p, gpu)
    sim.run_to_completion()
    assert sim.steps_done == p.sim.dev_steps
    sim.extend_run(7)
    assert sim.steps_done == p.sim.dev_steps + 7
    assert sim.is_developed


def test_extend_run_accumulates(gpu):
    p = _kin_params()
    sim = Simulation(p, gpu)
    sim.run_to_completion()
    sim.extend_run(3)
    sim.extend_run(4)
    assert sim.steps_done == p.sim.dev_steps + 7
    assert sim.steps_target == p.sim.dev_steps + 7


def test_extend_run_rejects_negative(gpu):
    sim = Simulation(_kin_params(), gpu)
    with pytest.raises(ValueError):
        sim.extend_run(-1)


def test_extend_run_zero_is_noop(gpu):
    p = _kin_params()
    sim = Simulation(p, gpu)
    sim.run_to_completion()
    sim.extend_run(0)
    assert sim.steps_done == p.sim.dev_steps


# -- sequence export ----------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sequence_structure_vorticity(gpu, tmp_path):
    """STRUCTURAL only for vorticity (SOR LSB noise compounds across frames)."""
    import itertools

    from gasgiant.export.exporter import run_export_sequence
    from gasgiant.export.manifest import read_manifest
    from gasgiant.export.writers import read_png16

    sim = Simulation(_vort_params(), gpu)
    out = tmp_path / "seq"
    run_export_sequence(sim, out, frames=4, steps_per_frame=6)

    files = [out / "frames" / f"frame_{i:04d}.png" for i in range(4)]
    assert all(f.is_file() for f in files)
    assert not (out / "frames" / "frame_0004.png").exists()
    # frame 0 is a byte duplicate of the mapset color map
    assert files[0].read_bytes() == (out / "color.png").read_bytes()
    # the base mapset is intact and the sim advanced 3 * 6 steps past dev
    assert (out / "height.exr").is_file()
    assert sim.steps_done == sim.params.sim.dev_steps + 18
    # frames differ pairwise (the sim actually advanced between renders)
    imgs = [read_png16(f) for f in files]
    for a, b in itertools.combinations(range(4), 2):
        assert (imgs[a] != imgs[b]).any(), f"frames {a} and {b} are identical"
    # manifest frames block, schema-validated by read_manifest
    m = read_manifest(out)
    assert m["frames"]["count"] == 4
    assert m["frames"]["steps_per_frame"] == 6
    assert m["frames"]["files"] == [f"frames/frame_{i:04d}.png" for i in range(4)]


def test_sequence_kinematic_golden_determinism(gpu, tmp_path):
    """Two fresh runs of the same 8-frame kinematic sequence are hash-identical
    (the kinematic path is byte-exact; never do this for vorticity)."""
    from gasgiant.export.exporter import run_export_sequence

    hashes = []
    for run in ("a", "b"):
        sim = Simulation(_kin_params(), gpu)
        out = tmp_path / run
        run_export_sequence(sim, out, frames=8, steps_per_frame=5)
        hashes.append(
            [_sha256(out / "color.png")]
            + [_sha256(out / "frames" / f"frame_{i:04d}.png") for i in range(8)]
        )
    assert hashes[0] == hashes[1]


def test_default_export_writes_no_frames(gpu, tmp_path):
    from gasgiant.export.exporter import run_export
    from gasgiant.export.manifest import read_manifest

    sim = Simulation(_kin_params(), gpu)
    out = tmp_path / "plain"
    run_export(sim, out)
    assert "frames" not in read_manifest(out)
    assert not (out / "frames").exists()


def test_sequence_cancellation_cleans_up(gpu, tmp_path):
    from gasgiant.export.exporter import export_sequence_job

    sim = Simulation(_vort_params(), gpu)
    out = tmp_path / "seq"
    keep = out / "users_own_file.txt"
    out.mkdir(parents=True)
    keep.write_text("precious")
    # Files the job does NOT write on this config (rings/flow are off) but
    # that a PREVIOUS export into the same folder could have left: cleanup
    # must not delete them (it removes only the files THIS job writes).
    foreign_rings = out / "rings.exr"
    foreign_rings.write_bytes(b"not-ours")
    foreign_flow = out / "flow.exr"
    foreign_flow.write_bytes(b"also-not-ours")

    job = export_sequence_job(sim, out, frames=4, steps_per_frame=6)
    saw_frame_1 = False
    for prog in job:
        if prog.message.startswith("frame 1"):
            saw_frame_1 = True  # rendering frame 1; cancel mid-sequence
            break
    assert saw_frame_1
    job.close()

    assert not (out / "mapset.json").exists()
    assert not (out / "color.png").exists()
    frames_dir = out / "frames"
    assert not frames_dir.exists() or not any(frames_dir.iterdir())
    assert keep.read_text() == "precious"  # never touches the user's files
    assert foreign_rings.read_bytes() == b"not-ours"
    assert foreign_flow.read_bytes() == b"also-not-ours"


def test_sequence_cancellation_with_pending_encodes_cleans_up(gpu, tmp_path):
    """Cancel AFTER frame 1's encodes were submitted (the first frame-2 tile
    message): the finally block must drain the pool, then remove the frame
    files that were written/in flight -- the concurrent-cancel path the
    off-thread encode pool introduced."""
    from gasgiant.export.exporter import export_sequence_job

    sim = Simulation(_vort_params(), gpu)
    out = tmp_path / "seq_pending"
    job = export_sequence_job(sim, out, frames=4, steps_per_frame=6)
    saw = False
    for prog in job:
        if prog.message.startswith("frame 2 tile"):
            saw = True  # frame 1's encode futures are submitted (maybe pending)
            break
    assert saw
    job.close()

    assert not (out / "mapset.json").exists()
    assert not (out / "color.png").exists()
    frames_dir = out / "frames"
    assert not frames_dir.exists() or not any(frames_dir.iterdir())


def test_sequence_rejects_bad_args(gpu, tmp_path):
    from gasgiant.export.exporter import export_sequence_job

    sim = Simulation(_kin_params(), gpu)
    with pytest.raises(ValueError):
        next(export_sequence_job(sim, tmp_path / "x", frames=0, steps_per_frame=6))
    with pytest.raises(ValueError):
        next(export_sequence_job(sim, tmp_path / "x", frames=4, steps_per_frame=0))
