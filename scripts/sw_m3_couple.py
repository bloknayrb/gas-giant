"""M3 coupled render gate (replaces the mis-targeted T9 direct-render gate).

Builds a v1.6 baseline and an EVOLVING-baroclinic-coupled render, writes a blind
panel (baseline | source | coupled) + key, and prints PASS/FAIL on:
  (1) source coherence (dominant zonal m <= M_GATE_MAX),
  (2) latitude concentration: coupled / baseline > 1.05,
  (3) texture preservation: 0.5 <= highfreq(coupled)/highfreq(baseline) <= 2.0.
Also prints the cadence benchmark + residency recommendation.

Usage: py -3 scripts/sw_m3_couple.py [RENDER_RES] [GAIN]
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from gasgiant.engine.baroclinic_coupling import residency_recommendation, run_coupled
from gasgiant.engine.facade import Simulation
from gasgiant.gl import GpuContext
from gasgiant.params.model import SolverType
from gasgiant.params.presets import load_factory_preset
from gasgiant.render.m3_metrics import highfreq_energy, latitude_concentration
from gasgiant.sim import baroclinic_source as bsrc
from gasgiant.sim.baroclinic_driver import BaroclinicSourceDriver

OUT = Path("out/audit/m3/gate")
RES = int(sys.argv[1]) if len(sys.argv) >= 2 else 2048
GAIN = float(sys.argv[2]) if len(sys.argv) >= 3 else 1.2


def _u8(rgb01: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((np.clip(rgb01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _params():
    p = load_factory_preset("jupiter_vorticity")
    p = p.model_copy(update={"sim": p.sim.model_copy(update={"resolution": RES})})
    p.solver.type = SolverType.VORTICITY
    return p


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gpu = GpuContext.headless()
    gpu.make_current()

    print("=== baseline (v1.6, no source) ===")
    base = Simulation(_params(), gpu)
    base_rgb = np.clip(base.render_maps(RES)["color"][..., :3], 0, 1).astype(np.float32)
    cv2.imwrite(str((OUT / "gate_baseline.png").resolve()), _u8(base_rgb))
    base._release_sim()

    print("=== coupled (evolving baroclinic source) ===")
    sim = Simulation(_params(), gpu)
    w, h = sim.solver.equirect.size
    driver = BaroclinicSourceDriver(grid_w=w, grid_h=h, warmup_steps=9000, seed=0)
    src = driver.current_source()
    m_src = bsrc.dominant_zonal_m(src)[0]
    stats = run_coupled(sim, driver, gain=GAIN, update_every=32,
                        baro_steps_per_update=400)
    coupled_rgb = np.clip(sim.render_maps(RES)["color"][..., :3], 0, 1).astype(np.float32)
    cv2.imwrite(str((OUT / "gate_coupled.png").resolve()), _u8(coupled_rgb))
    sim._release_sim()

    sgray = (src / (np.abs(src).max() or 1.0)) * 0.5 + 0.5
    cv2.imwrite(str((OUT / "gate_source.png").resolve()),
                (np.clip(sgray, 0, 1) * 255).astype(np.uint8))

    base_conc = latitude_concentration(base_rgb)
    coupled_conc = latitude_concentration(coupled_rgb)
    tex_ratio = highfreq_energy(coupled_rgb) / (highfreq_energy(base_rgb) + 1e-12)

    coherent = m_src <= bsrc.M_GATE_MAX
    concentrates = coupled_conc / (base_conc + 1e-12) > 1.05
    texture_ok = 0.5 <= tex_ratio <= 2.0
    residency = residency_recommendation(stats)

    print(f"\nsource dominant m         = {m_src}  (gate <= {bsrc.M_GATE_MAX})  "
          f"{'OK' if coherent else 'FAIL'}")
    print(f"latitude concentration    = base {base_conc:.3f} -> coupled "
          f"{coupled_conc:.3f}  {'OK' if concentrates else 'FAIL'}")
    print(f"texture ratio (coupled/base) = {tex_ratio:.3f}  "
          f"{'OK' if texture_ok else 'FAIL'}")
    print(f"cadence: baro {stats.baro_seconds:.1f}s + upload {stats.upload_seconds:.2f}s "
          f"vs v1.6 {stats.v16_seconds:.1f}s ({stats.source_updates} updates"
          f"{', OUTCROPPED' if stats.baro_outcropped else ''}) -> {residency}")

    verdict = "PASS" if (coherent and concentrates and texture_ok) else "FAIL"
    print(f"\nM3-COUPLING GATE: {verdict}")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
