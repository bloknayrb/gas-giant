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
#             'band_mod.glsl','wave_stamp.glsl','hero_q.glsl']:
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
    # Updated 2026-07-10 (hero_emergence, GRS-realism pack): pass 2's relaxation
    # lines compile as a HERO_EMERGENCE preprocessor VARIANT (define selected when
    # storms.hero_emergence > 0); the #else branches carry the pre-feature lines
    # verbatim, so the default program text is unchanged after preprocessing —
    # byte-identical by construction. (An earlier runtime-guarded cut moved the
    # jupiter@1024 p05 hash via FMA-contraction changes on shared expressions;
    # the variant conversion is the fix, per the CLAUDE.md gated-out rule.)
    "advect.comp":       "239d5022eeab06c8ea747a1614e9f00c55d04040",
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
    # Updated 2026-07-08 (neptune cirrus-streak lever): the OVAL/PEARL stamp
    # else-branch gained an `asp > 1.0` path (soft collar-free feathered glow +
    # flow-frame noise modulation) for elongated bright accent/companion clouds.
    # asp==1.0 (every existing preset's accents/companions) short-circuits it =>
    # byte-identical kinematic output; p05 baseline unchanged (9/9 match).
    # Updated 2026-07-10 (storms.hero_emergence, the GRS-realism pack — see
    # docs/superpowers/specs/2026-07-09-hero-emergence-design.md): the whole
    # feature (heroRelaxWeight rim fade + band flush, plateau fill + radial
    # identity, ring/collar/moat remap, quiet-storm fades, hero_q.glsl include)
    # compiles under `#ifdef HERO_EMERGENCE` with the pre-feature lines
    # verbatim in the #else branches, so the default program text is unchanged
    # after preprocessing => byte-identical by construction; p05 9/9. (An
    # earlier runtime-guarded cut moved the jupiter@1024 hash via
    # FMA-contraction changes on shared expressions — hence the variant rule.)
    # Re-pinned same day: COMMENT-ONLY fix above heroRelaxWeight (the old text
    # claimed advect.comp runtime-guards the call; the guard is variant
    # compilation). Zero code change — the compiled default program and the
    # p05 hashes are untouched.
    "vortex_stamp.glsl": "0ab171fa175e7243e6382ea81f884f9ced50072b",
    # New 2026-07-10 with hero_emergence: heroEllipQ, the shared elliptical-q
    # helper for the variant-only heroRelaxWeight/heroAnchorWindow. Entirely
    # #ifdef HERO_EMERGENCE => contributes nothing to the default program.
    "hero_q.glsl": "0d116e76423ac56301e74907bf2b2a81aaa659fa",
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
