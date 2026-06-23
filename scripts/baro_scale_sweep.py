"""CPU crux sweep for shrinking the M3 baroclinic eddy scale (THROWAWAY).

Falsification gate before any GPU work: drive the real 2-layer baroclinic solver
across (gp2, H, xi, m_zonal, smooth_sigma) and report, per config:
  outcrop? + step-of-outcrop, emergent dominant zonal m, emergent-vs-seed,
  eddy_var, coherence-gate pass, gp1/gp2.

Goal: a config whose emergent dominant m is ~1.5-2x the baseline (~m7) that stays
coherent and does NOT outcrop before the preset warmup (8000). If none exists ->
falsified, fall back to gain-only.

Run: ./.venv/Scripts/python.exe scripts/baro_scale_sweep.py
"""
from __future__ import annotations

import time

import numpy as np

from gasgiant.sim import baroclinic_source as bsrc
from gasgiant.sim import shallow_water_ref as ref

PHI_TEST = np.radians(45.0)
OMEGA = 7.292e-5
A = 6.4e6


def predicted_m(gp2: float, H_each: float) -> float:
    """Zonal wavenumber at the f-plane Phillips K_max for this config."""
    f0 = 2.0 * OMEGA * np.sin(PHI_TEST)
    Htot = 2.0 * H_each
    kd = np.sqrt(4.0 * f0 * f0 / (gp2 * Htot))
    kmax = kd * np.sqrt(np.sqrt(2.0) - 1.0)
    return float(kmax * A * np.cos(PHI_TEST))


def run_config(gp2, H_each, xi, m_zonal, smooth, warmup, seed=0):
    """Build + advance the baroclinic solver; return a result dict."""
    t0 = time.time()
    st = ref.baroclinic_test_state(
        W=bsrc.SRC_W, H=bsrc.SRC_H, unstable=True, seed=seed,
        gp1=bsrc.GP1, gp2=gp2, H1_mean=H_each, H2_mean=H_each,
        m_zonal=m_zonal, xi_unstable=xi,
        pert_amp_frac=1e-3, dt_safety=0.30, nu4=0.0,
    )
    outcrop_step = None
    for i in range(warmup):
        try:
            ref.step_2layer(st)
        except ref.PositivityViolation:  # real outcrop only; a stray bug stays loud
            outcrop_step = i
            break
    res = {
        "gp2": gp2, "H": H_each, "xi": xi, "m_seed": m_zonal,
        "smooth": smooth, "m_pred": round(predicted_m(gp2, H_each), 1),
        "gp1_gp2": round(bsrc.GP1 / gp2, 2),
        "outcrop_step": outcrop_step, "secs": round(time.time() - t0, 1),
    }
    if outcrop_step is not None:
        res.update(m_emg=None, eddy_var=None, coherent=None)
        return res
    zeta = bsrc.geostrophic_vorticity_source(st, smooth_sigma=smooth)
    m_emg, _ = bsrc.dominant_zonal_m(zeta)
    res.update(
        m_emg=m_emg,
        eddy_var=float(f"{ref.eddy_interface_var(st):.3e}"),
        coherent=bool(m_emg <= bsrc.M_GATE_MAX),
    )
    return res


def fmt(r):
    o = f"out@{r['outcrop_step']}" if r["outcrop_step"] is not None else "OK"
    return (f"gp2={r['gp2']:.3f} H={r['H']:.0f} xi={r['xi']:.1f} "
            f"m_seed={r['m_seed']:2d} smooth={r['smooth']:.2f} | "
            f"m_pred={r['m_pred']:5.1f} m_emg={str(r['m_emg']):>4} "
            f"coh={str(r['coherent']):>5} evar={r['eddy_var']} "
            f"g1/g2={r['gp1_gp2']:.2f} {o} ({r['secs']}s)")


# Stage A: confirm the baseline reproduces ~m5-7, then a focused shrink grid.
WARMUP = 8000
configs = [
    # (gp2,    H_each, xi,  m_zonal(track pred), smooth)
    (0.300,  12500.0, 3.0, 5, 2.50),   # current baseline
]
# gp2-down ladder (track m_zonal to predicted, lower smooth as eddies shrink)
for gp2 in (0.150, 0.075):
    mp = predicted_m(gp2, 12500.0)
    configs.append((gp2, 12500.0, 3.0, max(5, round(mp)), float(np.clip(2.5 * 7.0 / mp, 0.8, 2.5))))
# H-down ladder (cleaner margin per physics review)
for H in (6250.0, 3125.0):
    mp = predicted_m(0.300, H)
    configs.append((0.300, H, 3.0, max(5, round(mp)), float(np.clip(2.5 * 7.0 / mp, 0.8, 2.5))))
# combos + xi taming
for gp2, H, xi in ((0.150, 6250.0, 3.0), (0.150, 6250.0, 2.0), (0.075, 12500.0, 2.0)):
    mp = predicted_m(gp2, H)
    configs.append((gp2, H, xi, max(5, round(mp)), float(np.clip(2.5 * 7.0 / mp, 0.8, 2.5))))

print(f"warmup={WARMUP}  SRC={bsrc.SRC_W}x{bsrc.SRC_H}  gp1={bsrc.GP1}  gate={bsrc.M_GATE_MAX}")
print("-" * 110)
for c in configs:
    print(fmt(run_config(*c, warmup=WARMUP)), flush=True)
