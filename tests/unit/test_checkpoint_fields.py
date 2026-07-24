"""Checkpoint registry-field completeness guard.

Every Vortex dataclass field must be either in _REG_FIELDS or in the
explicitly CPU-side set that load_checkpoint restores by name (cooldown,
ttl). Added after the F06 wake_lat_off field silently missed serialization
(restored heroes lost their wake offset and the resumed run diverged) — the
same gap had already hidden the aspect field for a month.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from gasgiant.engine.checkpoint import _REG_FIELDS, GENERATION_VERSION, load_checkpoint
from gasgiant.sim.vortices import Vortex

_SEPARATELY_SERIALIZED = {"cooldown", "ttl", "origin", "cast_ref"}


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


def test_stale_generation_version_is_refused_loudly(tmp_path):
    """A checkpoint from an older generation must be REFUSED, not resumed with
    zero-defaulted registry columns: the v9 bump (GRS pass, 2026-07-16) changed
    generation output for every emergence-on preset (jet-derived wake frame,
    bow_gain, hero-relative accents), so a v8 warm checkpoint resuming here
    would silently mix old-generation tracers with new-generation stamps. The
    version gate fires before any GL work, so gpu=None proves the refusal is
    unconditional."""
    stale = tmp_path / "stale.npz"
    np.savez(stale, generation_version=np.int64(GENERATION_VERSION - 1))
    with pytest.raises(ValueError, match="generation_version"):
        load_checkpoint(stale, None)
