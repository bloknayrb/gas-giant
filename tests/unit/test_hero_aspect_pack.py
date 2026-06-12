"""Unit test (no GPU): hero_aspect packs into SSBO slot 9."""

from __future__ import annotations

import numpy as np

from gasgiant.sim.vortices import KIND_HERO, Vortex, VortexRegistry


def test_aspect_packs_into_ssbo_slot9():
    """Hero with aspect=2.0 must appear at out[:,9]; others default to 1.0."""
    # Build a registry with one hero (aspect=2.0) and one oval (aspect=1.0 default).
    reg = VortexRegistry()
    # Hero vortex
    reg.vortices.append(
        Vortex(
            lat=float(np.deg2rad(-22.5)),
            lon=0.0,
            r_core=0.10,
            strength=0.045,
            kind=KIND_HERO,
            tint=0.9,
            brightness=0.05,
            wake_dir=1.0,
            aspect=2.0,
        )
    )
    # Non-hero vortex (white oval) — aspect stays at default 1.0
    reg.vortices.append(
        Vortex(
            lat=float(np.deg2rad(15.0)),
            lon=1.0,
            r_core=0.03,
            strength=0.01,
            kind=2.0,  # VKIND_BARGE-ish; anything != KIND_HERO
        )
    )

    out = reg.pack_ssbo()
    assert out.shape[1] == 12, f"Expected 12 columns, got {out.shape[1]}"

    # Hero is vortex 0
    hero_aspect_packed = float(out[0, 9])
    assert hero_aspect_packed == 2.0, (
        f"Hero aspect packed as {hero_aspect_packed!r}, expected 2.0 (float32 exact)"
    )

    # Non-hero is vortex 1 — must have default aspect 1.0
    oval_aspect_packed = float(out[1, 9])
    assert oval_aspect_packed == 1.0, (
        f"Oval aspect packed as {oval_aspect_packed!r}, expected 1.0 (default)"
    )
