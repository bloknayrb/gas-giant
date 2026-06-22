"""Evolving baroclinic source driver for M3 coupling.

Owns a validated 2-layer baroclinic CPU solver spun to a finite-amplitude
warm start, then advanced in lockstep with the v1.6 turbulence solver. Each
cadence it re-derives the coherent geostrophic vorticity source (the EVOLVING
imprint, not the spike's static stamp) and resamples it to the equirect grid.
On lower-layer outcrop it holds the last good state.
"""
from __future__ import annotations

import copy

from gasgiant.sim import baroclinic_source as bsrc
from gasgiant.sim import shallow_water_ref as ref


class BaroclinicSourceDriver:
    def __init__(self, grid_w: int, grid_h: int,
                 warmup_steps: int = 9000, seed: int = 0,
                 m_zonal: int = bsrc.M_ZONAL) -> None:
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.outcropped = False
        self.st = ref.baroclinic_test_state(
            W=bsrc.SRC_W, H=bsrc.SRC_H, unstable=True, seed=seed,
            gp1=bsrc.GP1, gp2=bsrc.GP2, xi_unstable=bsrc.XI,
            m_zonal=m_zonal,
            pert_amp_frac=1e-3, dt_safety=0.30, nu4=0.0,
        )
        self.advance(warmup_steps)
        if self.outcropped:
            raise RuntimeError(
                f"BaroclinicSourceDriver: warmup outcropped within {warmup_steps} "
                f"steps -- the source never reached a finite-amplitude state. "
                f"Reduce warmup_steps or xi_unstable."
            )
        # Post-warmup snapshot: a reused driver (cache hit on a RESTART rebuild)
        # restores this so every development run starts from the identical
        # baroclinic state -- deterministic regardless of prior preview ticks.
        self._warm_st = copy.deepcopy(self.st)

    def advance(self, n: int) -> None:
        """Advance the baroclinic solver n steps; on outcrop (ValueError) latch
        `outcropped` and stop stepping (the last good state is retained)."""
        for _ in range(n):
            if self.outcropped:
                return
            try:
                ref.step_2layer(self.st)
            except ValueError:
                self.outcropped = True
                return

    def reset(self) -> None:
        """Restore the post-warmup state. Called when a cached driver is reused
        for a new development run so the result is independent of how far a live
        preview was ticked before a RESTART-tier edit."""
        self.st = copy.deepcopy(self._warm_st)
        self.outcropped = False

    def current_source(self):
        """Coherent unit-std evolving source on the equirect grid (grid_h, grid_w).
        Passes the coherence gate (raises if the source is a checkerboard)."""
        zeta = bsrc.geostrophic_vorticity_source(self.st, smooth_sigma=bsrc.SMOOTH_SIGMA)
        bsrc.assert_coherent(zeta)
        return bsrc.resample_to_equirect(zeta, self.grid_w, self.grid_h)

    @property
    def eddy_var(self) -> float:
        return ref.eddy_interface_var(self.st)
