"""Dump RAW M3 fields (no derive.comp, no procedural detail) to see the true
eddy structure: grid-scale noise vs coherent vortices.

Re-runs the validated config to the saturated snapshot and writes normalized
grayscale PNGs of the bare fields, upscaled NEAREST (honest pixels):
  raw_zeta1.png  top-layer eddy vorticity
  raw_zeta2.png  lower-layer eddy vorticity (Ro~0.5; is it stripes or vortices?)
  raw_hanom2.png lower interface anomaly h2-h_eq2 (the large-scale mode)
  raw_zeta2_nu.png lower-layer eddy vorticity WITH light nu4 (enstrophy sink test)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from gasgiant.sim import shallow_water_ref as ref  # noqa: E402

OUT = Path("out/audit/m3")
W, H = 192, 96
TARGET = 10500


def gray(field2d, name, up=8):
    a = field2d - field2d.mean(axis=1, keepdims=True)   # eddy (non-zonal)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    n = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
    img = (n * 255).astype(np.uint8)
    img = cv2.resize(img, (a.shape[1] * up, a.shape[0] * up), interpolation=cv2.INTER_NEAREST)
    p = (OUT / name).resolve()
    cv2.imwrite(str(p), img)
    print(f"  wrote {p}  (dominant zonal m: {_dom_m(a)})")


def _dom_m(eddy2d):
    """Dominant zonal wavenumber of the band-row eddy field (energy-weighted)."""
    row = eddy2d[eddy2d.shape[0] // 2]            # mid-latitude band row
    sp = np.abs(np.fft.rfft(row)) ** 2
    sp[0] = 0
    return int(np.argmax(sp))


def cz(u, v, g):
    z = ref.vorticity(u, v, g)
    return 0.5 * (z[0:H] + z[1:H + 1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"stepping validated config to {TARGET} (nu4=0)...")
    st = ref.baroclinic_test_state(W=W, H=H, unstable=True, seed=0, gp1=0.05,
                                   gp2=0.3, xi_unstable=3.0, pert_amp_frac=1e-3,
                                   dt_safety=0.30, nu4=0.0)
    for _ in range(TARGET):
        ref.step_2layer(st)
    gray(cz(st.u1, st.v1, st.g), "raw_zeta1.png")
    gray(cz(st.u2, st.v2, st.g), "raw_zeta2.png")
    gray(st.h2, "raw_hanom2.png")   # gray() removes the zonal mean -> interface eddy

    print(f"stepping with nu4=0.05 (enstrophy sink) to {TARGET}...")
    st2 = ref.baroclinic_test_state(W=W, H=H, unstable=True, seed=0, gp1=0.05,
                                    gp2=0.3, xi_unstable=3.0, pert_amp_frac=1e-3,
                                    dt_safety=0.30, nu4=0.05)
    ok = TARGET
    for n in range(TARGET):
        try:
            ref.step_2layer(st2)
        except (ValueError, AssertionError):
            ok = n
            print(f"  nu4=0.05 trapped at step {n}")
            break
    z2 = cz(st2.u2, st2.v2, st2.g); z2e = z2 - z2.mean(axis=1, keepdims=True)
    f0 = 2.0 * 7.292e-5 * np.sin(np.radians(45.0))
    print(f"  nu4=0.05 reached step {ok}, lower Ro2={np.std(z2e)/abs(f0):.4f}")
    gray(cz(st2.u2, st2.v2, st2.g), "raw_zeta2_nu.png")


if __name__ == "__main__":
    main()
