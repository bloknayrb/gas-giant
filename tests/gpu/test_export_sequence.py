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


def test_sequence_frame_channels_identical(gpu, tmp_path):
    """Frames 1..N carry the same channel order as the derive that produced them.

    Frame 0 is a byte copy of color.png, whose channel order IS pinned (by
    test_cube_export.py::test_default_export_identical) -- but frames 1..N go
    through export_sequence_job's own per-frame tile loop, and every other
    sequence test compares frames only to EACH OTHER, so a red/blue swap there
    is invisible to the whole suite. Kinematic, so this is byte-exact.
    """
    import numpy as np

    from gasgiant.export.exporter import derive_tile, run_export_sequence
    from gasgiant.export.writers import read_png16

    frames, spf = 3, 5
    p = _kin_params()
    w, h = p.export.width, p.export.width // 2
    out = tmp_path / "seq"
    run_export_sequence(Simulation(p, gpu), out, frames=frames, steps_per_frame=spf)

    # Rebuild frame `fi` from a fresh run of the same kinematic sequence.
    ref_sim = Simulation(_kin_params(), gpu)
    ref_sim.run_to_completion()
    tile_color = gpu.texture2d((1024, 1024), 4, "f4")
    tile_height = gpu.texture2d((1024, 1024), 1, "f4")
    tile_detail = gpu.texture2d((1024, 1024), 1, "f4", linear=True)
    try:
        for fi in range(1, frames):
            ref_sim.extend_run(spf)
            snap = ref_sim.create_snapshot()
            try:
                derive_tile(
                    ref_sim, snap, snap.params, 0, 0, w, h,
                    tile_color, tile_height, tile_detail, None,
                )
                ref = gpu.read_texture(tile_color)[:h, :w, :3]
                ref_u16 = (np.clip(ref, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
            finally:
                snap.release()
            ref_norm = ref_u16.astype(np.float32) / np.float32(65535.0)
            np.testing.assert_array_equal(
                read_png16(out / "frames" / f"frame_{fi:04d}.png"), ref_norm,
                err_msg=f"frame {fi}: stored channels differ from the derive",
            )
    finally:
        tile_color.release()
        tile_height.release()
        tile_detail.release()


# -- double-buffered frame assembly -------------------------------------------


def _slow(fn, delay=0.05):
    """Wrap a writer so every encode is genuinely still in flight when the next
    frame renders. At 512 px an encode is sub-millisecond, so without this the
    overlap the double-buffering exists to manage never actually happens and the
    tests below prove nothing."""
    import time as _t

    def _w(*a, **k):
        _t.sleep(delay)
        return fn(*a, **k)

    return _w


def test_slow_encodes_do_not_change_output(gpu, tmp_path, monkeypatch):
    """Delaying every encode must not alter a single output byte.

    Frame buffers are handed to the pool WITHOUT a copy, so a set may only be
    refilled once its own encodes have finished. If that wait were tracked on a
    shared COUNT instead of per set, a slow color write plus fast completions
    elsewhere would drop the count and let the renderer overwrite a buffer
    mid-write -- producing a frame file spliced from two frames, with no
    exception and no other failing test.

    So the delays are ASYMMETRIC by design: the color write is much slower than
    the height and emission writes, which is exactly the shape that makes a
    shared count go slack. A uniform delay would not construct it.
    Kinematic, so byte-exact A/B is legitimate.
    """
    from gasgiant.export import exporter
    from gasgiant.export.exporter import run_export_sequence

    def _run(out, slow):
        if slow:
            for name, delay in (
                ("write_png16_bgr_u16", 0.15),   # the long pole, as in a real export
                ("write_png16_gray_u16", 0.01),
                ("write_exr_rgba", 0.01),
            ):
                monkeypatch.setattr(exporter, name, _slow(getattr(exporter, name), delay))
        run_export_sequence(
            Simulation(_kin_params(), gpu), out, frames=4, steps_per_frame=5,
            all_maps=True,
        )
        return {p.name: _sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}

    fast = _run(tmp_path / "fast", slow=False)
    slow = _run(tmp_path / "slow", slow=True)

    assert set(fast) == set(slow)
    assert len(fast) >= 4 + 4 + 2, f"expected color+height frames and a base set, got {sorted(fast)}"
    differing = [n for n in fast if fast[n] != slow[n]]
    assert not differing, f"slow encodes changed these files: {differing}"


def test_failed_final_frame_encode_fails_the_export(gpu, tmp_path, monkeypatch):
    """A worker exception on the LAST frame must reach the caller.

    The final drain is the only thing that calls .result() on those futures --
    concurrent.futures.wait() and a bare done() loop both return silently on a
    failed future, which would let a broken encode reach a written manifest and
    a reported-success export.
    """
    from gasgiant.export import exporter
    from gasgiant.export.exporter import run_export_sequence

    real = exporter.write_png16_bgr_u16
    calls = {"n": 0}

    def _fail_on_last(*a, **k):
        calls["n"] += 1
        if calls["n"] >= 4:          # frame 0 is a file copy; this is frame 3
            raise OSError("disk full")
        return real(*a, **k)

    monkeypatch.setattr(exporter, "write_png16_bgr_u16", _fail_on_last)

    out = tmp_path / "boom"
    with pytest.raises(OSError, match="disk full"):
        run_export_sequence(
            Simulation(_kin_params(), gpu), out, frames=4, steps_per_frame=5,
        )
    assert not (out / "mapset.json").exists(), "a failed encode must not leave a manifest"


def test_sequence_cancellation_with_slow_encodes_cleans_up(gpu, tmp_path, monkeypatch):
    """The existing cancellation test runs at 512 px where an encode is
    sub-millisecond, so a future is essentially never actually in flight. With
    buffers no longer copied, that machinery is load-bearing -- cancel while an
    encode genuinely holds a live frame buffer."""
    from gasgiant.export import exporter
    from gasgiant.export.exporter import export_sequence_job

    monkeypatch.setattr(
        exporter, "write_png16_bgr_u16", _slow(exporter.write_png16_bgr_u16, 0.2)
    )

    out = tmp_path / "seq_slow"
    job = export_sequence_job(Simulation(_vort_params(), gpu), out, frames=4, steps_per_frame=6)
    for prog in job:
        if prog.message.startswith("frame 2"):
            break
    job.close()

    assert not (out / "mapset.json").exists()
    assert not (out / "color.png").exists()
    frames_dir = out / "frames"
    assert not frames_dir.exists() or not any(frames_dir.iterdir())
