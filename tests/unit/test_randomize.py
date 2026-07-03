from __future__ import annotations

import pytest
from pydantic import BaseModel

from gasgiant.params.model import PlanetParams
from gasgiant.params.randomize import randomize


def test_deterministic():
    a = randomize(seed=42)
    b = randomize(seed=42)
    assert a == b


def test_different_seeds_differ():
    assert randomize(seed=1) != randomize(seed=2)


def test_values_in_declared_ranges():
    p = randomize(seed=7)

    def walk(model: type[BaseModel], obj, prefix=""):
        for name, info in model.model_fields.items():
            ann = info.annotation
            value = getattr(obj, name)
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                walk(ann, value, f"{prefix}{name}.")
                continue
            extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
            rand = extra.get("rand")
            if rand is None:
                continue
            lo, hi = rand
            assert lo - 0.5 <= value <= hi + 0.5, f"{prefix}{name}={value} outside [{lo}, {hi}]"

    walk(PlanetParams, p)


def test_locked_fields_keep_base_value_without_shifting_others():
    base = PlanetParams()
    base.bands.count = 33
    locked = randomize(seed=5, base=base, locked={"bands.count"})
    unlocked = randomize(seed=5, base=base)
    assert locked.bands.count == 33
    # The lock consumed its draw, so every other field matches the unlocked roll.
    assert locked.bands.width_jitter == unlocked.bands.width_jitter
    assert locked.appearance.haze_amount == unlocked.appearance.haze_amount


def test_randomize_sets_seed():
    assert randomize(seed=99).seed == 99


def test_randomize_output_is_pinned():
    """Golden guard pinning the exact ``randomize(seed)`` output for a fixed
    seed. This is a BEHAVIORAL PIN of the CURRENT declaration order, NOT a
    correctness check: the randomize walk draws one RNG value per ``rand`` field
    in StormsParams (and every other model) in DECLARATION order, so reordering a
    ``rand``-bearing field -- or inserting/removing one mid-list -- silently
    changes the roll for every field after it. That is exactly what the Phase 4
    Storms sub-grouping did (it moved ``wake_turbulence`` and shifted
    ``oval_density``/``barge_density``/``pearls_count``) with no test to catch it.
    If this test fails after a field reorder, that is the signal: either restore
    the order or re-baseline these constants deliberately (and note the
    randomize-output change in the PR)."""
    p = randomize(seed=12345)
    # A field early in the tree (guards the whole draw sequence upstream)...
    assert p.bands.width_jitter == pytest.approx(0.5965876168020049, abs=1e-9)
    # ...and the Storms fields the Phase 4 reorder disturbed.
    assert p.storms.oval_density == pytest.approx(1.1674481331441742, abs=1e-9)
    assert p.storms.barge_density == pytest.approx(1.1486749689049411, abs=1e-9)
    assert p.storms.pearls_count == 3
    assert p.storms.wake_turbulence == pytest.approx(1.1276795025451216, abs=1e-9)
