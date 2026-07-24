"""Resolution-invariant development scaling.

When ``SimParams.resolution_invariant`` is on, the sim auto-adjusts its time-axis
settings so a run developed at ``resolution`` reproduces (as closely as the physics
allows) what the same settings would produce at ``reference_resolution`` -- letting a
user iterate at a low resolution and render at a high one without the outcome changing.

Why any of this is needed: the timestep already scales as ``dt in proportion to
1/resolution`` (:func:`gasgiant.sim.solver.compute_dt`), but every step COUNT and every
per-step RATE is authored in fixed *steps*. So raising resolution shrinks the developed
physical time (``dev_steps * dt``) and rebalances forcing against advection. This module
derives a single scale factor ``s = resolution / reference_resolution`` and the effective
step counts + per-step coefficients that hold physical time and the physical
forcing/relaxation balance fixed.

Two invariants the rest of the engine relies on:

* ``s == 1`` (or the flag off) is a STRICT no-op: every helper returns its input
  unchanged via a structural short-circuit -- NOT arithmetic identity -- so the
  default-off / authored-at-reference paths stay byte-identical (p05 hash gate, the
  dev-0 omega capture).
* Stored params are never mutated; callers read effective values at the point of use.

Scope (honest limit): fully effective for *nudge-dominated* presets (short
``vort_relax_tau``, small ``vort_inject``). *Turbulence-dominated* presets carry a
grid-Nyquist-locked hyperviscosity and an active 2-D inverse cascade whose resolved
inertial range grows with resolution, so their large scale is intrinsically
resolution-dependent and this scaling only *reduces* -- never eliminates -- the drift.
See ``docs/architecture.md``.
"""

from __future__ import annotations

from gasgiant.params.model import PlanetParams


def scale_factor(params: PlanetParams) -> float:
    """``s = resolution / reference_resolution`` when the feature is on, else ``1.0``.

    ``1.0`` is returned (never computed as ``res/res``) whenever the flag is off or the
    reference is unusable, so downstream ``s == 1.0`` short-circuits are exact."""
    sim = params.sim
    if not sim.resolution_invariant:
        return 1.0
    ref = sim.reference_resolution
    if ref <= 0 or sim.resolution == ref:
        return 1.0
    return sim.resolution / ref


def effective_dev_steps(params: PlanetParams) -> int:
    """The development-run length the solver should actually step: ``dev_steps``
    at the reference resolution, scaled to ``round(dev_steps * s)`` so the same
    physical time (``steps * dt``, and ``dt`` in proportion to ``1/resolution``)
    elapses at any resolution. The single source of truth threaded into the seeded
    timeline (drift compensation, merger + outbreak scheduling) so it never
    desyncs from ``steps_target``. Returns raw ``dev_steps`` when the feature is
    off (``s == 1``).

    Floored to ``>= 1`` for a nonzero request: a strong downscale (``s`` as small
    as ``512/8192 = 0.0625``) can ``round`` a small ``dev_steps`` (1..8) down to
    ``0``, which would silently render the undeveloped step-0 field as if
    developed. One step is the honest minimum for a run the user asked to
    develop; a genuine ``dev_steps == 0`` still returns ``0``."""
    n = params.sim.dev_steps
    eff = scale_duration(n, scale_factor(params))
    return max(1, eff) if n > 0 else eff


def scale_duration(n_steps: int, s: float) -> int:
    """A step count that measures a physical-time DURATION (dev_steps, event
    lifetimes, warmup): ``round(n * s)`` so the same physical time elapses at any
    resolution. Structural no-op at ``s == 1``."""
    if s == 1.0:
        return n_steps
    return round(n_steps * s)


def scale_decay_fraction(f: float, s: float) -> float:
    """A deterministic per-step decay/relaxation fraction ``f`` in ``[0, 1)`` (applied
    as ``x <- x * (1 - f)`` or ``x += (target - x) * f`` once per step).

    Decay-exact remap ``f' = 1 - (1 - f) ** (1 / s)`` so the retained fraction over the
    (``s``-scaled) run length matches the reference. This is REQUIRED over the linear
    ``f / s``: for ``s < 1`` (iterating below the authoring resolution) ``f / s`` can
    exceed 1 and invert/blow up a ``mix()``; the exponential form stays in ``[0, 1)``.
    Structural no-op at ``s == 1``."""
    if s == 1.0:
        return f
    retained = 1.0 - f
    if retained <= 0.0:
        return f
    return 1.0 - retained ** (1.0 / s)


def scale_rate(c: float, s: float) -> float:
    """A per-step ADDITIVE rate coefficient that modifies the PERSISTENT state
    directly (no ``dt`` weighting) -- e.g. ``vort_psi_drag`` (``q += c * psi_eddy``
    in omega_force.comp). Its cumulative effect over the run is ``~ c * steps``, so
    to hold it fixed as the run length scales by ``s`` the coefficient scales
    ``c / s`` (linear).

    Distinct from :func:`scale_decay_fraction`: that is for a MULTIPLICATIVE decay
    written as an explicit ``[0, 1)`` retained fraction (``x <- x * (1 - f)``).
    ``vort_psi_drag`` reads like a fraction but is a rate coefficient with
    ``hi = 20`` -- the per-mode decay it INDUCES is ``c / (k^2 + 1/L_d^2)`` (small
    for realistic ``c``), so linear ``c / s`` is the correct first-order remap and,
    unlike ``scale_decay_fraction``, it is continuous across the whole ``[0, 20]``
    range (the fraction helper silently no-ops at ``c >= 1``). Also distinct from a
    per-step term consumed with ``dt`` weighting (an instantaneous velocity/source
    that advects the state), whose effect is ALREADY invariant (``dt * steps`` is
    fixed) and must NOT be scaled -- see the baroclinic gain in ``engine/facade.py``.
    Structural no-op at ``s == 1``."""
    if s == 1.0:
        return c
    return c / s


def scale_relax_tau(tau: float, s: float) -> float:
    """A relaxation timescale expressed in STEPS (``tau``; per-step fraction
    ``1 / tau``). Apply the decay-exact correction to the implied fraction and invert:
    ``tau' = 1 / (1 - (1 - 1/tau) ** (1/s))``. Equivalent to scaling the fraction, kept
    as its own helper because several uniforms are authored as tau, not as a fraction.
    Structural no-op at ``s == 1``."""
    if s == 1.0 or tau <= 0.0:
        return tau
    f2 = scale_decay_fraction(1.0 / tau, s)
    return 1.0 / f2 if f2 > 0.0 else tau


def scale_stochastic_amp(amp: float, s: float) -> float:
    """A white-in-time stochastic forcing amplitude (``vort_inject``,
    ``hero_wake_turb``): ``amp / sqrt(s)`` so the
    variance-injection RATE per unit physical time is preserved (independent draws add
    in variance). A deterministic ``amp / s`` would over/under-inject. The ``sqrt``
    exponent is the first-principles white-noise value; Phase-0 calibration may refine
    it per term. Structural no-op at ``s == 1``."""
    if s == 1.0:
        return amp
    return amp / (s ** 0.5)
