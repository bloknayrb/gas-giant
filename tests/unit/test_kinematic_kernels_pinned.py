"""Source-hash guard for the kinematic solver's GLSL kernels.

Pins SHA1 hashes of psi.comp, velocity.comp, advect.comp and every .glsl file
they #include (one level of transitivity). An accidental edit to any of these
files will make this test fail, forcing the author to consciously update the
hashes and re-run scripts/p05_baseline_hash.py to advance the GPU baseline.

Pinned to guard the kinematic path's byte-identity (v1.6 P0.5).
If you INTENTIONALLY change a kinematic kernel, update these hashes AND
re-run scripts/p05_baseline_hash.py to advance the baseline.
"""

from __future__ import annotations

import hashlib
import importlib.resources as ir

_PKG = "gasgiant.sim.kernels"

# SHA1 hashes pinned at v1.6 P1 implementation (2026-06-14).
# To regenerate: uv run python -c "
#   import hashlib, importlib.resources as ir
#   pkg = 'gasgiant.sim.kernels'
#   for f in ['psi.comp','velocity.comp','advect.comp',
#             'noise3d.glsl','common.glsl','vortex_stamp.glsl',
#             'band_mod.glsl','wave_stamp.glsl']:
#       t = ir.files(pkg).joinpath(f).read_text(encoding='utf-8')
#       print(f, hashlib.sha1(t.encode()).hexdigest())
# "
_PINNED: dict[str, str] = {
    # Updated 2026-07-03 for the placement-chirality fixes (review F12/F06,
    # montage user-approved 2026-07-03): hero wake wedge reads the new
    # wake_lat_off lane, defaults westward, and is windowed to |across| 2.5
    # so it can no longer leak into the psi_feather polar band.
    "psi.comp":          "4e905a61164cea74991442622d38742482f0edeb",
    "velocity.comp":     "a5edeb117303788431b9d1ab686f0dddae402fd6",
    "advect.comp":       "a44a0061d19ac2769c45b308500e6405f8663fd1",
    "noise3d.glsl":      "971a4a110900ff63237eb7ae030edc18ea23bc1a",
    "common.glsl":       "48c13b438e4e893b32b594234ef965bdfeac1cad",
    # Updated 2026-06-29 for the convective white-plume outbreak stamp branch
    # (KIND_OUTBREAK ring, default-off). Re-updated 2026-06-29 cutting the
    # KIND_OUTBREAK cool push 0.15->0.07 (lead-knot visibility pass). Both edits
    # touch ONLY the KIND_OUTBREAK branch, which never fires without outbreak
    # vortices -> byte-identical kinematic GPU output for the no-outbreak case.
    # Updated 2026-07-03 (review F06, approved with the chirality montage):
    # tracer-side hero wake mirrors psi.comp — wake_lat_off lane read,
    # |across| 2.5 locality window. INTENTIONAL pixel change on presets with
    # heroes; P0.5 baseline advanced the same day (scripts/p05_baseline_hash.py).
    "vortex_stamp.glsl": "57524532a8c40952941a750b0b01921869e12a3c",
    "band_mod.glsl":     "278a7379ae63c7cc59e4ab8b61c7dc783c099fd6",
    "wave_stamp.glsl":   "11094b91e32fd4f59cd5db8bc26b630d05306e47",
}


def _sha1(filename: str) -> str:
    text = ir.files(_PKG).joinpath(filename).read_text(encoding="utf-8")
    return hashlib.sha1(text.encode()).hexdigest()


def test_kinematic_kernel_sources_unchanged():
    """All kinematic GLSL sources must match their pinned SHA1s.

    If this test fails after an intentional kernel edit:
    1. Re-run the snippet in the module docstring to get new hashes.
    2. Update _PINNED above.
    3. Re-run scripts/p05_baseline_hash.py --check (or capture a new baseline)
       to confirm byte-identical GPU output or document the intentional change.
    """
    current = {fname: _sha1(fname) for fname in _PINNED}
    mismatches = {
        fname: (expected, current[fname])
        for fname, expected in _PINNED.items()
        if current[fname] != expected
    }
    assert not mismatches, (
        "Kinematic kernel source(s) changed — update _PINNED and re-run "
        "scripts/p05_baseline_hash.py to advance the GPU baseline.\n"
        + "\n".join(
            f"  {f}: expected {exp}\n          got     {got}"
            for f, (exp, got) in mismatches.items()
        )
    )
