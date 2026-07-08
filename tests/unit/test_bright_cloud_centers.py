"""bright_cloud_centers: the CPU-side elongated-bright-cloud list the
cirrus-fiber synthesis masks on. The predicate mirrors the stamp's soft-streak
branch (asp > 1): non-hero, aspect > 1, positive brightness."""

from __future__ import annotations

import math

from gasgiant.engine.snapshot import MAX_BRIGHT_CLOUDS, bright_cloud_centers
from gasgiant.sim.vortices import (
    KIND_BARGE,
    KIND_DEBRIS,
    KIND_HERO,
    KIND_OVAL,
    KIND_PEARL,
    Vortex,
    VortexRegistry,
)


def _v(kind, aspect=1.0, brightness=0.0, lat=0.3, lon=1.0, r_core=0.05):
    return Vortex(lat=lat, lon=lon, r_core=r_core, strength=1.0, kind=kind,
                  brightness=brightness, aspect=aspect)


def test_predicate_selects_only_elongated_bright_nonheroes():
    reg = VortexRegistry(vortices=[
        _v(KIND_PEARL, aspect=3.5, brightness=0.32),   # companion  -> in
        _v(KIND_OVAL, aspect=4.0, brightness=0.32),    # accent     -> in
        _v(KIND_HERO, aspect=2.2, brightness=0.5),     # hero       -> out (kind)
        _v(KIND_HERO, aspect=2.2, brightness=-0.3),    # dark hero  -> out
        _v(KIND_OVAL, aspect=1.0, brightness=0.6),     # round oval -> out (aspect)
        _v(KIND_BARGE, aspect=2.0, brightness=-0.4),   # dark barge -> out (brightness)
        _v(KIND_DEBRIS, aspect=1.0, brightness=0.9),   # debris     -> out (aspect)
    ])
    out = bright_cloud_centers(reg)
    assert len(out) == 2
    for (_, _, _, rc, asp) in out:
        assert rc == 0.05
        assert asp in (3.5, 4.0)


def test_positions_reflect_drifted_lon():
    lat, lon = 0.2, -2.1
    reg = VortexRegistry(vortices=[
        _v(KIND_PEARL, aspect=3.0, brightness=0.3, lat=lat, lon=lon),
    ])
    ((x, y, z, _, _),) = bright_cloud_centers(reg)
    assert math.isclose(x, math.cos(lat) * math.cos(lon), abs_tol=1e-12)
    assert math.isclose(y, math.sin(lat), abs_tol=1e-12)
    assert math.isclose(z, math.cos(lat) * math.sin(lon), abs_tol=1e-12)


def test_capped_at_max_in_registry_order():
    reg = VortexRegistry(vortices=[
        _v(KIND_PEARL, aspect=2.0 + 0.1 * i, brightness=0.3)
        for i in range(MAX_BRIGHT_CLOUDS + 4)
    ])
    out = bright_cloud_centers(reg)
    assert len(out) == MAX_BRIGHT_CLOUDS
    # Registry order: the first MAX qualifying entries, not the last.
    assert [round(a, 3) for (_, _, _, _, a) in out] == [
        round(2.0 + 0.1 * i, 3) for i in range(MAX_BRIGHT_CLOUDS)
    ]
