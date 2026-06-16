"""Evolving baroclinic source driver for M3 coupling.

Owns a validated 2-layer baroclinic CPU solver spun to a finite-amplitude
warm start, then advanced in lockstep with the v1.6 turbulence solver. Each
cadence it re-derives the coherent geostrophic vorticity source (the EVOLVING
imprint, not the spike's static stamp) and resamples it to the equirect grid.
On lower-layer outcrop it holds the last good state.
"""
from __future__ import annotations

from gasgiant.sim import baroclinic_source as bsrc
from gasgiant.sim import shallow_water_ref as ref


class BaroclinicSourceDriver:
    def __init__(self, grid_w: int, grid_h: int,
                 warmup_steps: int = 9000, seed: int = 0) -> None:
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.outcropped = False
        self.st = ref.baroclinic_test_state(
            W=bsrc.SRC_W, H=bsrc.SRC_H, unstable=True, seed=seed,
            gp1=bsrc.GP1, gp2=bsrc.GP2, xi_unstable=bsrc.XI,
            pert_amp_frac=1e-3, dt_safety=0.30, nu4=0.0,
        )
        self.advance(warmup_steps)

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

    def current_source(self):
        """Coherent unit-std evolving source on the equirect grid (grid_h, grid_w).
        Passes the coherence gate (raises if the source is a checkerboard)."""
        zeta = bsrc.geostrophic_vorticity_source(self.st)
        bsrc.assert_coherent(zeta)
        return bsrc.resample_to_equirect(zeta, self.grid_w, self.grid_h)

    @property
    def eddy_var(self) -> float:
        return ref.eddy_interface_var(self.st)
