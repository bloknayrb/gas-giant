"""Geometry probe for the Task-0 band-fill STOP: where should Y_SHEET sit?

The frozen Y_SHEET = LY/2 maps to the hero latitude (-24 deg) — the bracket's
SEAT, a u~0 crossing by design. The billow chain physically rides the belt/wake
SHEAR LINE (max |du/dy| flank), which also carries real zonal advection.
This probe prints the profile around the box and the transit clocks for
candidate Y_SHEET placements so the re-derivation decision is informed.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import config as C

y, u, om = C.warm_profile_window()
lat_deg = np.degrees(C.LAT0 + (y - C.LY / 2.0))

print(f"box lat range: {lat_deg[0]:.1f} .. {lat_deg[-1]:.1f} deg   "
      f"LY={C.LY:.4f} rad = {C.LY/C.RC:.1f} rc")
print(f"{'y/LY':>6} {'lat':>7} {'u':>8} {'du/dy':>8}")
for frac in np.arange(0.20, 0.81, 0.05):
    i = int(frac * C.NY)
    print(f"{frac:6.2f} {lat_deg[i]:7.2f} {u[i]:8.3f} {om[i]:8.2f}")

# interior window away from the Tukey-blended edges
lo, hi = C.EDGE_BLEND * 2, C.NY - C.EDGE_BLEND * 2
ii = np.arange(lo, hi)
i_shear = ii[np.argmax(np.abs(om[ii]))]
i_umax = ii[np.argmax(np.abs(u[ii]))]
print(f"\nmax |du/dy| interior: {np.abs(om[i_shear]):.2f} at lat {lat_deg[i_shear]:.2f} "
      f"(y/LY {y[i_shear]/C.LY:.3f}), u there = {u[i_shear]:.3f}")
print(f"max |u| interior:     {np.abs(u[i_umax]):.3f} at lat {lat_deg[i_umax]:.2f} "
      f"(y/LY {y[i_umax]/C.LY:.3f}), du/dy there = {om[i_umax]:.2f}")

print("\ncandidate Y_SHEET placements (need |u| for advection AND |du/dy| context):")
print(f"{'y/LY':>6} {'lat':>7} {'u':>8} {'du/dy':>8} {'strip->band':>12} {'band-fill':>10}")
for i in [i_shear, i_umax] + [int(f * C.NY) for f in (0.35, 0.40, 0.45, 0.55, 0.60, 0.65)]:
    us = abs(u[i])
    s2b = (C.BAND_X[0] - C.STRIP_X[1]) / max(us, 1e-6) / C.DT
    bf = (C.BAND_X[1] - C.BAND_X[0]) / max(us, 1e-6) / C.DT
    print(f"{y[i]/C.LY:6.3f} {lat_deg[i]:7.2f} {u[i]:8.3f} {om[i]:8.2f} {s2b:12.0f} {bf:10.0f}")
