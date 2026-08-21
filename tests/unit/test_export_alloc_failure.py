"""A failed host/GL allocation must not leak the export snapshot.

export_job creates the snapshot (exporter.py) BEFORE the whole-map np.empty
buffers and the tile textures, but releases it only in a `finally` those
allocations sit outside of. A MemoryError there -- the exact 8 GiB emission
allocation the host-memory work exists to shrink -- would leak the snapshot's
cloned GL textures (~400 MB VRAM at sim.resolution 4096). The GUI catches
MemoryError and keeps running, so the leak accumulates across retries.

No GL: the guard is specified to run before any tile is derived, so a stub
gpu whose texture2d raises exercises it exactly.
"""

from __future__ import annotations

import pytest

from gasgiant.export.exporter import export_job
from gasgiant.params.model import PlanetParams, ProjectionKind


class _Snap:
    def __init__(self, params):
        self.params = params
        self.released = 0

    def release(self):
        self.released += 1


class _Tex:
    def __init__(self):
        self.released = 0

    def release(self):
        self.released += 1


class _BoomGpu:
    """Texture allocation succeeds ``ok_count`` times, then fails -- like a
    driver running out of VRAM partway through, or a later host allocation
    throwing after some textures already exist."""

    def __init__(self, exc, ok_count=0):
        self._exc = exc
        self._ok = ok_count
        self.made: list[_Tex] = []

    def texture2d(self, *a, **k):
        if len(self.made) >= self._ok:
            raise self._exc
        self.made.append(_Tex())
        return self.made[-1]


class _Sim:
    def __init__(self, params, gpu):
        self.params = params
        self.gpu = gpu
        self.snap = _Snap(params)
        self.steps_done = 0
        self.steps_target = 0

    def tick(self, _n):
        return False           # development already complete

    def create_snapshot(self):
        return self.snap


@pytest.mark.parametrize("exc", [MemoryError("no room"), RuntimeError("driver lost")])
@pytest.mark.parametrize("projection", [ProjectionKind.EQUIRECT, ProjectionKind.CUBE])
def test_allocation_failure_releases_the_snapshot(tmp_path, exc, projection):
    p = PlanetParams()
    p.export.width = 512
    p.export.projection = projection
    sim = _Sim(p, _BoomGpu(exc))

    with pytest.raises(type(exc)):
        for _ in export_job(sim, tmp_path / "out", 512):
            pass

    assert sim.snap.released == 1, (
        f"{projection}: snapshot leaked when allocation raised {type(exc).__name__}"
    )



@pytest.mark.parametrize("projection", [ProjectionKind.EQUIRECT, ProjectionKind.CUBE])
def test_partially_acquired_textures_are_released_too(tmp_path, projection):
    """A throw PART WAY through the acquisition block must release the textures
    already created, not just the snapshot.

    Releasing only the snapshot leaves ~24 MB of tile textures stranded per
    attempt, and the GUI catches the error and keeps running -- so retrying a
    too-large export accumulates exactly the leak this guard exists to stop.
    The cube job is the likelier path: its multi-GiB face buffers are allocated
    AFTER its textures.
    """
    p = PlanetParams()
    p.export.width = 512
    p.export.projection = projection
    # Emission on so BOTH jobs request a third texture -- otherwise the cube job
    # asks for only two and nothing throws.
    p.emission.aurora_strength = 1.0
    gpu = _BoomGpu(MemoryError("no room"), ok_count=2)
    sim = _Sim(p, gpu)

    with pytest.raises(MemoryError):
        for _ in export_job(sim, tmp_path / "out", 512):
            pass

    assert len(gpu.made) == 2, "test needs a partial acquisition to be meaningful"
    assert all(t.released == 1 for t in gpu.made), (
        f"{projection}: stranded textures "
        f"{[t.released for t in gpu.made]}"
    )
    assert sim.snap.released == 1
