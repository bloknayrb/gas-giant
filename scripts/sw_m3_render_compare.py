"""M3 RENDER COMPARISON — the validated instability, rendered from each layer.

Finding (sw_m3_saturate): the validated baroclinic config (gp1=0.05, xi=3, no
forcing, 192x96) develops vigorous LOWER-layer eddies (Ro2~0.55) while the TOP
layer stays an order of magnitude quieter (Ro1~0.013). The production encoder
(sw_encode) renders the TOP layer -> laminar. The eddies live in the layer we
are not drawing.

This script snapshots the saturated-but-pre-outcrop state and renders it three
ways so we can SEE which is the fidelity winner:
  1. top   = current encoder (top layer)                       -> laminar
  2. lower = same encoder on the LOWER layer (h2,u2,v2)        -> the eddies
  3. blend = top-layer thickness (bands/altitude) + LOWER-layer
             vorticity as the hero detail channel              -> deck + storms

Writes out/audit/m3/cmp_{top,lower,blend}.png. No verdict logic; this is a
visual A/B/C for the fidelity decision.

Usage: py -3 scripts/sw_m3_render_compare.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from gasgiant.gl import GpuContext  # noqa: E402
from gasgiant.params.presets import load_factory_preset  # noqa: E402
from gasgiant.render.maps import MapDeriver  # noqa: E402
from gasgiant.sim import shallow_water_ref as ref  # noqa: E402
from gasgiant.sim import sw_encode  # noqa: E402

OUT = Path("out/audit/m3")
RES = 4096
W_GRID, H_GRID = 192, 96
TARGET_STEP = 10500          # saturated (Ro2~0.49), safely before outcrop (~12500)


def _u8(rgb01: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((np.clip(rgb01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _blend_tracer(h1, u1, v1, h2, u2, v2, g, h_eq1):
    """Top-layer thickness for bands/altitude; LOWER-layer vorticity for the hero
    detail/polarity channels (where the eddies are)."""
    H, W = h1.shape
    h_anom = h1 - np.asarray(h_eq1, dtype=h1.dtype)
    z2 = ref.vorticity(u2, v2, g)
    z2c = 0.5 * (z2[0:H] + z2[1:H + 1])
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    rgba[..., 0] = sw_encode._norm(h_anom)          # banded color (top)
    rgba[..., 1] = sw_encode._norm(h1)              # altitude (top)
    rgba[..., 2] = sw_encode._norm(np.abs(z2c))     # hero detail = LOWER eddies
    rgba[..., 3] = sw_encode._norm(z2c)             # storm polarity = LOWER
    return rgba


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f0 = 2.0 * 7.292e-5 * np.sin(np.radians(45.0))

    print(f"Building + stepping validated config to step {TARGET_STEP} (CPU)...")
    st = ref.baroclinic_test_state(
        W=W_GRID, H=H_GRID, unstable=True, seed=0,
        gp1=0.05, gp2=0.3, xi_unstable=3.0, pert_amp_frac=1e-3,
        dt_safety=0.30, nu4=0.0,
    )
    for n in range(TARGET_STEP):
        ref.step_2layer(st)
    z1 = ref.vorticity(st.u1, st.v1, st.g); z1 = z1 - z1.mean(axis=1, keepdims=True)
    z2 = ref.vorticity(st.u2, st.v2, st.g); z2 = z2 - z2.mean(axis=1, keepdims=True)
    ro1 = float(np.std(z1)) / abs(f0)
    ro2 = float(np.std(z2)) / abs(f0)
    print(f"  snapshot: Ro1(top)={ro1:.4f}  Ro2(lower)={ro2:.4f}  min_h2={st.h2.min():.1f}")

    gpu = GpuContext.headless()
    gpu.make_current()
    p = load_factory_preset("jupiter_vorticity")
    deriver = MapDeriver(gpu)

    encodings = {
        "top":   sw_encode.to_tracer_fields(st.h1, st.u1, st.v1, st.g, st.h_eq1),
        "lower": sw_encode.to_tracer_fields(st.h2, st.u2, st.v2, st.g, st.h_eq2),
        "blend": _blend_tracer(st.h1, st.u1, st.v1, st.h2, st.u2, st.v2, st.g, st.h_eq1),
    }
    for name, tracer in encodings.items():
        rgb = deriver.derive_from_tracer(tracer, RES, p.appearance, seed=p.seed)
        path = (OUT / f"cmp_{name}.png").resolve()
        cv2.imwrite(str(path), _u8(np.clip(rgb[..., :3], 0, 1)))
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
