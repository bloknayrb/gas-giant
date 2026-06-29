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
    "psi.comp":          "c2e7dcf5422aad158d47189c4e95baca48c0450e",
    "velocity.comp":     "a5edeb117303788431b9d1ab686f0dddae402fd6",
    "advect.comp":       "a44a0061d19ac2769c45b308500e6405f8663fd1",
    "noise3d.glsl":      "971a4a110900ff63237eb7ae030edc18ea23bc1a",
    "common.glsl":       "48c13b438e4e893b32b594234ef965bdfeac1cad",
    "vortex_stamp.glsl": "3aa81ad720796a07e04d726a0c9578d725545c00",
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
