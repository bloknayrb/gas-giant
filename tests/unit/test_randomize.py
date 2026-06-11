from __future__ import annotations

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
