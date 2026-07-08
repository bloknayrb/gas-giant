"""Checkpoint registry-field completeness guard.

Every Vortex dataclass field must be either in _REG_FIELDS or in the
explicitly CPU-side set that load_checkpoint restores by name (cooldown,
ttl). Added after the F06 wake_lat_off field silently missed serialization
(restored heroes lost their wake offset and the resumed run diverged) — the
same gap had already hidden the aspect field for a month.
"""

from __future__ import annotations

import dataclasses

from gasgiant.engine.checkpoint import _REG_FIELDS
from gasgiant.sim.vortices import Vortex

_SEPARATELY_SERIALIZED = {"cooldown", "ttl", "origin"}


def test_every_vortex_field_has_a_serialization_home():
    all_fields = {f.name for f in dataclasses.fields(Vortex)}
    covered = set(_REG_FIELDS) | _SEPARATELY_SERIALIZED
    missing = all_fields - covered
    assert not missing, (
        f"Vortex field(s) {sorted(missing)} are not checkpoint-serialized: "
        "add to _REG_FIELDS (and bump GENERATION_VERSION) or to the "
        "explicit CPU-side set in load_checkpoint."
    )
    phantom = set(_REG_FIELDS) - all_fields
    assert not phantom, f"_REG_FIELDS names non-existent field(s): {sorted(phantom)}"
