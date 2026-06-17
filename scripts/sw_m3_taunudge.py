"""M3 fast-nudge experiment (last cheap lever before closing the coupling).

Hypothesis (the only injection-knob the adversarial review left untested):
a FAST nudge -- small vort_relax_tau -- makes the relax-toward-target outrace
the nonlinear advective fold, so the coherent baroclinic source HOLDS as low-m
structure instead of being shredded to broadband. Cost: a fast nudge also
laminarizes v1.6's own turbulence, so coherence may rise only by erasing texture.

This sweeps vort_relax_tau and reports ALL FOUR gates the review locked, so a
"pass" cannot be a laminar collapse:
  HERO     banded_coherent_fraction(coupled)/base  > 1.05
  CO-1     texture floor: 0.7 <= highfreq(coupled)/highfreq(base) <= 1.4  (bidir)
  CO-2     max single-mode share of coherent band <= 0.60  (>=~3 effective modes)
  SOURCE   source dominant zonal m <= M_GATE_MAX
ALL FOUR must hold at the SAME tau for a window to exist. Exploratory at RES
(default 1024) for speed; confirm any winner at 2048 with the frozen gate params.

Usage: py -3 scripts/sw_m3_taunudge.py [RES] [GAIN] [TAUS_CSV]
"""
from __future__ import annotations

import sys

import numpy as np

from gasgiant.engine.baroclinic_coupling import run_coupled
from gasgiant.engine.facade import Simulation
from gasgiant.gl import GpuContext
from gasgiant.params.model import SolverType
from gasgiant.params.presets import load_factory_preset
from gasgiant.render.m3_metrics import banded_coherent_fraction, highfreq_energy
from gasgiant.sim import baroclinic_source as bsrc
from gasgiant.sim.baroclinic_driver import BaroclinicSourceDriver

RES = int(sys.argv[1]) if len(sys.argv) >= 2 else 1024
GAIN = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.8
TAUS = ([float(x) for x in sys.argv[3].split(",")] if len(sys.argv) >= 4
        else [600.0, 150.0, 40.0, 10.0])
WARMUP, BARO_PER_UPDATE, UPDATE_EVERY = 8000, 150, 32
ACTIVE_DEG, M_HI = (20.0, 55.0), 12


def _params(tau: float):
    p = load_factory_preset("jupiter_vorticity")
    p = p.model_copy(update={
        "sim": p.sim.model_copy(update={"resolution": RES}),
        "solver": p.solver.model_copy(update={"vort_relax_tau": tau}),
    })
    p.solver.type = SolverType.VORTICITY
    return p


def _single_mode_share(img: np.ndarray) -> float:
    """Max share of any single zonal mode m in 1..M_HI of the coherent-band
    power, aggregated over active rows. ~1 => one clean ripple (hollow pass)."""
    lum = img[..., :3].mean(axis=2).astype(np.float64) if img.ndim == 3 else img
    H, _ = lum.shape
    lat = 90.0 - (np.arange(H) + 0.5) / H * 180.0
    active = (np.abs(lat) >= ACTIVE_DEG[0]) & (np.abs(lat) <= ACTIVE_DEG[1])
    rows = lum[active]
    eddy = rows - rows.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(eddy, axis=1)) ** 2
    hi = min(M_HI, power.shape[1] - 1)
    band = power[:, 1:hi + 1].sum(axis=0)  # total per mode across rows
    return float(band.max() / (band.sum() + 1e-12))


def _render(sim: Simulation) -> np.ndarray:
    return np.clip(sim.render_maps(RES)["color"][..., :3], 0, 1).astype(np.float32)


def main() -> None:
    gpu = GpuContext.headless()
    gpu.make_current()
    print(f"=== fast-nudge sweep  RES={RES} GAIN={GAIN}  taus={TAUS} ===\n")
    print(f"{'tau':>6} | {'hero ratio':>10} | {'tex ratio':>9} | "
          f"{'1mode':>6} | {'src m':>5} | verdict")
    print("-" * 60)

    any_pass = False
    for tau in TAUS:
        base = Simulation(_params(tau), gpu)
        base_rgb = _render(base)
        base._release_sim()

        sim = Simulation(_params(tau), gpu)
        w, h = sim.solver.equirect.size
        driver = BaroclinicSourceDriver(grid_w=w, grid_h=h, warmup_steps=WARMUP, seed=0)
        m_src = bsrc.dominant_zonal_m(driver.current_source())[0]
        run_coupled(sim, driver, gain=GAIN, update_every=UPDATE_EVERY,
                    baro_steps_per_update=BARO_PER_UPDATE)
        coupled_rgb = _render(sim)
        sim._release_sim()

        hero = (banded_coherent_fraction(base_rgb, ACTIVE_DEG, M_HI),
                banded_coherent_fraction(coupled_rgb, ACTIVE_DEG, M_HI))
        hero_ratio = hero[1] / (hero[0] + 1e-12)
        tex_ratio = highfreq_energy(coupled_rgb) / (highfreq_energy(base_rgb) + 1e-12)
        one_mode = _single_mode_share(coupled_rgb)

        g_hero = hero_ratio > 1.05
        g_tex = 0.7 <= tex_ratio <= 1.4
        g_mode = one_mode <= 0.60
        g_src = m_src <= bsrc.M_GATE_MAX
        ok = g_hero and g_tex and g_mode and g_src
        any_pass = any_pass or ok
        print(f"{tau:>6.0f} | {hero_ratio:>10.3f} | {tex_ratio:>9.3f} | "
              f"{one_mode:>6.3f} | {m_src:>5} | "
              f"{'PASS' if ok else 'fail'} "
              f"(H{'+' if g_hero else '-'} "
              f"T{'+' if g_tex else '-'} "
              f"M{'+' if g_mode else '-'} "
              f"S{'+' if g_src else '-'})")

    print("\nWINDOW EXISTS" if any_pass else "\nNO WINDOW -> fast-nudge lever FALSIFIED")
    print("DONE_TAUNUDGE")
    sys.exit(0 if any_pass else 1)


if __name__ == "__main__":
    main()
