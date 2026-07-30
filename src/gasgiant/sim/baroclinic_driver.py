"""Evolving baroclinic source driver for M3 coupling.

Owns a validated 2-layer baroclinic CPU solver spun to a finite-amplitude
warm start, then advanced in lockstep with the v1.6 turbulence solver. Each
cadence it re-derives the coherent geostrophic vorticity source (the EVOLVING
imprint, not the spike's static stamp) and resamples it to the equirect grid.
On lower-layer outcrop it holds the last good state and reports it.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path

from gasgiant.params.model import baroclinic_effective_width
from gasgiant.sim import baroclinic_cache as bcache
from gasgiant.sim import baroclinic_source as bsrc
from gasgiant.sim import shallow_water_ref as ref

log = logging.getLogger(__name__)


class BaroclinicWarmupError(RuntimeError):
    """The warmup outcropped before reaching a finite-amplitude state — the
    DOCUMENTED graceful-degrade signal for driver construction. Subclasses
    RuntimeError so any existing ``except RuntimeError`` caller keeps working,
    while letting the facade catch this *expected* degrade distinctly from a
    genuine unexpected RuntimeError (which must propagate loudly). Mirrors the
    IncoherentSourceError/ValueError pattern in baroclinic_source."""


class BaroclinicOutcropError(RuntimeError):
    """The lower layer outcropped DURING the run, after a clean warmup.

    Distinct from BaroclinicWarmupError because it is raised from `advance`
    rather than from construction, and the facade handles the two at different
    points. Before this existed `advance` swallowed the outcrop and only latched
    a flag, so the facade's mid-run handler could never fire: `baroclinic_status`
    kept reporting 'active' while the source was frozen on its last good state --
    silently reinstating the static stamp this driver exists to replace."""


class BaroclinicSourceDriver:
    """Warm baroclinic state plus the derivation that reads a source off it.

    The constructor takes ONLY inputs the warm state depends on. Everything
    consumed when a source is derived -- the output grid and the smoothing
    sigma -- is an argument to `current_source` instead, because the warmup is
    the expensive part (~52 s on a 192x96 grid at the 8000 steps every shipped
    preset uses; this signature's own `warmup_steps` default of 9000 predates
    the pfield and only applies to a caller that passes nothing) and the facade
    caches the driver on exactly those constructor inputs.

    Keeping the split in the signatures is what makes the cache honest. Stored
    as attributes, a derivation-time input has to be BOTH excluded from the
    facade's cache key AND re-pushed onto every reused driver; miss the second
    half and the artist's slider silently does nothing. As arguments the value
    is supplied fresh at each call and the mistake is unrepresentable.

    `cache_dir` opts into the on-disk warm-state cache (`baroclinic_cache`),
    which turns a repeat visit to a configuration into a ~50 ms load instead of
    a ~52 s warmup, across process restarts. Default None = no disk cache, so a
    test that means to exercise the real warmup gets one and nothing writes to
    the user's home unless a caller asks for it. The facade opts in.
    """

    def __init__(self, warmup_steps: int = 9000, seed: int = 0,
                 m_zonal: int | None = None,
                 gp2: float | None = None,
                 latitude: float | None = None,
                 width: float | None = None,
                 phase_jitter: float = 0.0,
                 spectrum_width: int = 0,
                 cache_dir: Path | None = None) -> None:
        # None sentinels rather than `= bsrc.GP2` defaults: a default argument
        # binds ONCE at def time, which would freeze the module constants and
        # silently break the tests that monkeypatch them to force an outcrop.
        m_zonal = bsrc.M_ZONAL if m_zonal is None else m_zonal
        gp2 = bsrc.GP2 if gp2 is None else gp2
        latitude = bsrc.PHI_TEST_DEG if latitude is None else latitude
        width = bsrc.BAND_HALFWIDTH_DEG if width is None else width
        self.outcropped = False
        # ONE effective width feeds both the seeding envelope and the source
        # mask; if they disagreed the mask would clip the storms it exists to
        # pass. Clamped to keep the band off the equator (the rule lives in the
        # params layer so validation_warnings can surface it); inert at the
        # default band.
        self.width = baroclinic_effective_width(latitude, width)
        self.lat_band, self.taper = bsrc.mask_band_for(latitude, self.width)
        self.st = ref.baroclinic_test_state(
            W=bsrc.SRC_W, H=bsrc.SRC_H, unstable=True, seed=seed,
            gp1=bsrc.GP1, gp2=gp2, xi_unstable=bsrc.XI,
            m_zonal=m_zonal,
            phi_test_deg=latitude, band_halfwidth_deg=self.width,
            phase_jitter=phase_jitter, spectrum_width=spectrum_width,
            pert_amp_frac=1e-3, dt_safety=0.30, nu4=0.0,
        )
        # The key carries the RUNTIME module constants, not only their on-disk
        # text: bsrc.GP1/XI/SRC_W/SRC_H are read above and the tests monkeypatch
        # them, which a source fingerprint alone cannot see (see baroclinic_cache).
        key = bcache.warm_cache_key(
            warmup_steps=warmup_steps, seed=seed, m_zonal=m_zonal, gp2=gp2,
            latitude=latitude, width=self.width,
            phase_jitter=phase_jitter, spectrum_width=spectrum_width,
            gp1=bsrc.GP1, xi=bsrc.XI, src_w=bsrc.SRC_W, src_h=bsrc.SRC_H,
        ) if cache_dir is not None else None
        cached = bcache.load_warm_state(key, cache_dir) if key is not None else None
        if cached is not None:
            for field, arr in cached.items():
                setattr(self.st, field, arr)
        else:
            try:
                self.advance(warmup_steps)
            except BaroclinicOutcropError as exc:
                raise BaroclinicWarmupError(
                    f"BaroclinicSourceDriver: warmup outcropped within {warmup_steps} "
                    f"steps -- the source never reached a finite-amplitude state. "
                    f"Reduce warmup_steps or eddy_scale. ({exc})"
                ) from exc
            # Only a SURVIVING warmup is cached. An outcropped state is useless
            # to a later run, and the facade already remembers failing keys.
            if key is not None:
                bcache.save_warm_state(
                    key, {f: getattr(self.st, f) for f in bcache.EVOLVING_FIELDS},
                    cache_dir)
        # Post-warmup snapshot: a reused driver (cache hit on a RESTART rebuild)
        # restores this so every development run starts from the identical
        # baroclinic state -- deterministic regardless of prior preview ticks.
        self._warm_st = copy.deepcopy(self.st)

    def advance(self, n: int) -> None:
        """Advance the baroclinic solver n steps.

        On lower-layer outcrop (PositivityViolation -- the ONLY ValueError
        reachable from step_2layer's explicit call tree) latch `outcropped`, keep
        the last good state, and RAISE BaroclinicOutcropError so the caller can
        degrade visibly. Swallowing it here is what let a dead source masquerade
        as a live one. Any OTHER exception (a genuine bug) propagates untouched.
        """
        if self.outcropped:
            raise BaroclinicOutcropError(
                "baroclinic solver already outcropped; no further advance is "
                "possible from the held state")
        for _ in range(n):
            try:
                ref.step_2layer(self.st)
            except ref.PositivityViolation as exc:
                log.warning("baroclinic lower-layer outcrop; holding last good "
                            "state: %s", exc)
                self.outcropped = True
                raise BaroclinicOutcropError(str(exc)) from exc

    def reset(self) -> None:
        """Restore the post-warmup state. Called when a cached driver is reused
        for a new development run so the result is independent of how far a live
        preview was ticked before a RESTART-tier edit."""
        self.st = copy.deepcopy(self._warm_st)
        self.outcropped = False

    def current_source(self, grid_w: int, grid_h: int, smooth_sigma: float):
        """Coherent unit-std evolving source on the equirect grid (grid_h, grid_w).
        Passes the coherence gate (raises if the source is a checkerboard).

        Both arguments are derivation-only -- neither touches the warm state, so
        changing either must NOT cost a re-warmup. See the class docstring.
        """
        zeta = bsrc.geostrophic_vorticity_source(
            self.st, smooth_sigma=smooth_sigma,
            lat_band=self.lat_band, taper=self.taper)
        # in_band: the gate must follow the band, or a steered `latitude`
        # puts the storms outside the rows it samples and it grades noise.
        bsrc.assert_coherent(zeta, in_band=True)
        return bsrc.resample_to_equirect(zeta, grid_w, grid_h)

    @property
    def eddy_var(self) -> float:
        return ref.eddy_interface_var(self.st)
