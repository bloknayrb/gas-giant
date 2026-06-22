"""Fast (no-GPU) tripwire pinning WHERE the M3 baroclinic source is injected.

The whole fix for the band-crossing distortion was to move the external-vorticity
injection OUT of the force pass (which writes the persistent advected q state, where
it accumulated unbounded) and INTO the recover pass (the Poisson RHS omega_rel = q-f,
where it is bounded and coherent). The only current guard on that invariant is a
9-minute GPU suite. This static check reads the two kernel sources directly and fails
in milliseconds if a future edit re-introduces injection into the force/q pass.
"""
from __future__ import annotations

from importlib.resources import files

_KERNELS = "gasgiant.sim.kernels"


def _kernel_source(name: str) -> str:
    return (files(_KERNELS) / name).read_text(encoding="utf-8")


def test_injection_lives_in_recover_not_force():
    recover = _kernel_source("omega_recover.comp")
    force = _kernel_source("omega_force.comp")

    # The Poisson-RHS pass owns the coupling uniforms and the overlay add.
    assert "u_external_gain" in recover
    assert "u_external_omega" in recover
    assert "u_external_gain != 0.0" in recover, (
        "the recover pass must gate the overlay on the exact-zero gain compare"
    )

    # The force pass (which writes the persistent q state) must NOT touch the
    # external source -- that was the accumulation bug.
    assert "u_external_gain" not in force, (
        "omega_force.comp re-declares the baroclinic uniform: injection has leaked "
        "back into the q/force pass (the original accumulation bug)"
    )
    assert "u_external_omega" not in force
