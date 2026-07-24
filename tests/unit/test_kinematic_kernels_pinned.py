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
#   for f in ['psi.comp','velocity.comp','advect.comp','init.comp',
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
    # Updated 2026-07-15 (GRS hero-interaction pass, Phase 1): the hero wake
    # wedge gains a HERO_EMERGENCE variant arm extending its length (rc*7 ->
    # mix to rc*10) with the pre-feature text verbatim in #else — default
    # program text unchanged after preprocessing; p05 9/9 verified same day.
    # Updated 2026-07-24 (M2-B per-storm emergence family): the HERO_EMERGENCE
    # wake-length arm reads a per-hero local `emergence_v` (default = the global
    # uniform), overridden from THIS hero's binding-5 CastLevers row under
    # `#ifdef CAST_LEVERS` — the VELOCITY wedge must track the per-storm TRACER
    # wedge or a placed hero folds a mismatched wake. Default program: the local
    # equals the uniform, so numerically identical (p05 9/9 same day).
    "psi.comp":          "b03d6a49c3bd14f40716253ce2927dd412c43eff",
    "velocity.comp":     "a5edeb117303788431b9d1ab686f0dddae402fd6",
    # Updated 2026-07-10 (hero_emergence, GRS-realism pack): pass 2's relaxation
    # lines compile as a HERO_EMERGENCE preprocessor VARIANT (define selected when
    # storms.hero_emergence > 0); the #else branches carry the pre-feature lines
    # verbatim, so the default program text is unchanged after preprocessing —
    # byte-identical by construction. (An earlier runtime-guarded cut moved the
    # jupiter@1024 p05 hash via FMA-contraction changes on shared expressions;
    # the variant conversion is the fix, per the CLAUDE.md gated-out rule.)
    # Updated 2026-07-15 (GRS hero-interaction pass, Phase 1): the band-target
    # lookup gains a HERO_EMERGENCE variant arm sampling the hero-deflected
    # latitude (heroBandDeflect — belt bowing around the oval), verbatim
    # pre-feature #else => default program text unchanged after preprocessing.
    "advect.comp":       "c76ee3ba979c656e5751bd4a3890bdef04708b5b",
    # New pin 2026-07-15: init.comp gained the SAME heroBandDeflect variant
    # arm as advect.comp (both must shape the SAME relaxation target). It was
    # never pinned before this pass; it is a kinematic kernel and #includes
    # vortex_stamp.glsl, so it belongs here.
    "init.comp":         "ea86b0344a599329f458096adddbe6ff7608bc0c",
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
    # Updated 2026-07-15 (GRS hero-interaction pass, Phase 1 — plan
    # ancient-snuggling-meadow): all inside HERO_EMERGENCE arms (open-spiral
    # interior lane + off-center knot, deterministic moat shear-asymmetry with
    # the west-arc carve, wake-sector relaxation release in heroRelaxWeight,
    # the new heroBandDeflect helper) or a new variant arm around the wake
    # wedge (extended + dimmed; verbatim pre-feature #else). Default program
    # text unchanged after preprocessing; p05 9/9 verified same day.
    # Re-pinned same day: the moat-asymmetry east/west weights were built on a
    # wrong frame reading (h1 = cross(j, c) points ANTI-east, so hth=0 is
    # local west) — the carve landed on the upstream arc; caught by the new
    # collar-arc asymmetry GPU test. Variant-arm-only fix.
    # Re-pinned 2026-07-15 (Phase 2 retune, same pass): anatomy inversion
    # (bright annulus 1.12/k34 hugging the plateau, diffuse dark collar
    # 1.30/k12 outside), rim relax fade narrowed off the annulus (0.95/k10),
    # flush rise steepened (1.55,1.9) at x8 with the outer fade kept WIDE at
    # (2.7,3.4) — a pulled-in outer let wound arcs survive in the 2.8-3.4
    # shell (measured as upstream fold variance at parity with the wake) —
    # mottle mute 0.35/fscale 0.9. ALL inside HERO_EMERGENCE
    # mix-endpoints/arms; e=0 legacy endpoints and the default program text
    # unchanged; p05 9/9 verified same day.
    # Re-pinned same day (Checkpoint-1 feedback): bright-collar base raised
    # under emergence (0.22 -> mix to 0.31; m5 hero-contrast tripwire), and
    # leading-side smoothing in heroRelaxWeight (upstream weight suppresses
    # the rim-fade erosion x(1-0.65 upw) and boosts the flush x(1+0.6 upw) so
    # the belt approaches the storm laminar and band-parallel, deflecting
    # cleanly — "tighten up the leading side"). Variant-arm-only.
    # Re-pinned 2026-07-15 (Round A, per-latitude adversarial reviews): belt
    # bow gains a CPU boundary gate (bow_gain, SSBO slot [3i+2].w — no phantom
    # wrap where no boundary exists) plus FLANK-only shed/raggedness (|cos az|
    # weighting: the E/W painted-ride arcs open and vary, the load-bearing
    # N/S apex bow is untouched — an un-weighted first cut zeroed the apex
    # and the bow test caught it); collar closure-breaking (downstream carve
    # 0.8 over a wider arc + seeded amplitude/width lobes on both rings);
    # core polarity (radial deep-darkening REMOVED, uniform plateau lift
    # +0.10e, hot off-center knot 0.14/0.10, T3 rim fade 0.60->0.30).
    # Variant-arm-only; default text unchanged; p05 9/9 same day.
    # Re-pinned same day (A2): dark collar gains its own amplitude lobes +
    # downstream tear (ringmod) and eases -0.16 -> mix to -0.125; rim-tint
    # ring DE-DOUBLED (co-located with the dark collar at 1.30/k12 — the old
    # 1.09 inner line + 1.27 collar pair read as two drawn ellipses).
    # NOTE the residual visible boundary ring was subsequently ROOT-CAUSED as
    # EMERGENT wound tracer (controlled renders with rim_contrast=0 AND
    # rim_tint=0 keep it; palette-notch warming does not remove it) — the
    # stamps are exonerated; regularity of the wound boundary is a
    # texture/fray question, not a stamp one. Variant-arm-only.
    # Re-pinned 2026-07-15 (Round B de-bullseye + interior legibility, plan
    # ancient-snuggling-meadow): heroRelaxWeight gains a hero-local
    # meridional frame — belt-side flush pinch (inner rise to ~1.19,
    # protected by a uniform full-strength floor q 2.05-2.35 + the wide
    # outer fade), low-order wound-boundary raggedness (width lobes +
    # one-sided inward radius wobble + one seeded ring-break arc + per-arc
    # erosion depth); the dark collar's lobes deepen (floor 0.10) and gain
    # an equatorward cut paired with the pinch; the interior gains T3-space
    # spiral banding (lane3, pitch 13), a hotter knot (T3 0.24), and a
    # storm-within-a-storm dark nucleus. ALL inside HERO_EMERGENCE arms;
    # default program text unchanged after preprocessing; p05 9/9 verified
    # same day.
    # Re-pinned same day (round-B calibration fix pass, reference-anchored
    # review): heroBandDeflect outer fade azimuth-BLENDED — equatorward arc
    # recovers by q~1.6 (the flush relaxes toward the DEFLECTED target, so
    # the bow's reach WAS the pale moat's width and the belt-side pinch
    # measured as a no-op), flanks keep (1.45,2.0) (an all-azimuth tighten
    # broke the bow/flush co-design there — the wake-fold test's upstream
    # window read the target-vs-flow disagreement annulus as folds); interior
    # amplitudes to their calibration bounds (lane3 0.30, knot T3 0.32,
    # nucleus 0.45 — measured interior luminance std 4.5 vs reference 18.9).
    # Variant-arm-only; default text unchanged; p05 9/9.
    # Re-pinned same day (user: "is it too perfect of an oval?" — yes): a
    # low-order SHAPE deformation of the hero outline itself. R(theta)
    # divides q/qrim/qcol in the stamp anatomy AND q in heroRelaxWeight
    # (matched seeds): equatorward flattening 0.11 e (belt presses the north
    # rim flat) + seeded m=2/3 breathing 0.05/0.04 e, so plateau, rings,
    # collar and the release/flush windows share ONE imperfect envelope.
    # heroBandDeflect and the vorticity-side windows deliberately stay
    # elliptical (bow calibration + anchor basin). Variant-arm-only.
    # Re-pinned same day: the deformation became a user lever —
    # storms.hero_shape (intensity, default 1.0 = the calibrated egg) +
    # hero_shape_seed (dedicated "hero-shape:<seed>" substream), via new
    # variant-declared uniforms u_hero_shape / u_hero_shape_phase replacing
    # the hard-coded amplitudes' shared noise-offset phases. Variant-arm-only.
    # Re-pinned same day: seeded lobes 0.05/0.04 -> 0.075/0.055 (at ~2 px the
    # seed dial was sub-perceptual — a 3-seed strip rendered near-identical;
    # the flatten is deterministic, so the lobes ARE the re-roll), and the
    # heroRelaxWeight cull 3.6 -> 4.2 so the deformed flush fade COMPLETES on
    # max-bulge azimuths instead of truncating with a relax-rate step arc.
    # Far field beyond raw ~4.0 still returns exactly 1.0. Variant-arm-only.
    # Updated 2026-07-16 (storms.hero_taper, plan ancient-snuggling-meadow):
    # both shape blocks restructured from a single-expression Rr to guarded
    # accumulate (`Rr = 1.0; if shape>0 ...; if taper>0 ...`) and gained the
    # deterministic upstream WEDGE term (0.25 amp, 6.75*c^4*(1-c^2) window on
    # the squashed upstream cosine, Rr floor 0.4 inside the taper guard).
    # Variant-arm-only; taper=0 output verified BYTE-identical across the
    # restructure by the cross-commit capture (4 kinematic emergence scenes,
    # dev 0+60, shape 1.0 and 0.0 — p05 cannot see HERO_EMERGENCE, so the
    # capture is the binding gate; p05 9/9 unchanged same day). Constants
    # cross-pinned by tests/unit/test_hero_shape_constants.py. Re-pinned same
    # day: uct sign — cross(j, c) points ANTI-east (the F06 chirality trap),
    # the first cut put the wedge downstream; caught by the new
    # test_hero_taper_is_upstream_wedge before commit. Taper-guard-only.
    # Re-pinned 2026-07-16 (taper equilibrium mechanisms, S2 calibration):
    # geometric deformation alone measured ~0 at dev 700 (psi low-passes the
    # wedge harmonics; wound material re-parks on smooth streamlines; the
    # dev-60 field shows the wedge plainly). Three taper-guarded holds so
    # the equilibrium keeps it: erosion hold on the wedge arc (release
    # suppressed x(1-0.7*e*twr)), wedge flush (x12 fast-relax from just
    # outside the DEFORMED annulus — only the flush rate outruns advective
    # re-supply), and the heroBandDeflect CONVERGENCE (the bow's outer
    # recovery pulls in up to 35% on the wedge arc — the hollow closes at
    # the stagnation point; the percept lives on THIS contour, measured).
    # All under u_hero_taper > 0 guards / twr = 0 off; cross-commit capture
    # 4/4 byte-identical, p05 9/9 unchanged.
    # Re-pinned 2026-07-16 (PR-43 review fixes): heroBandDeflect gains the
    # heroRelaxWeight-style q > 0.05 atan(0,0) guard (GLSL-undefined at the
    # exact hero-center texel; bw masks the center so output is unchanged on
    # atan(0,0)==0 GPUs); stale `westw` renamed `wakew` (it keys off wdir —
    # the downstream arc, east on warm — not compass west); one flush-window
    # comment qualified. Comment/rename + emergence-arm-only guard; default
    # program text unchanged, p05 9/9 same day.
    # Updated 2026-07-23 (M2 per-storm CastLevers, part 2 — GPU wiring): the six
    # hero stamp levers (rim_contrast/rim_tint/rim_warp/mottle/tint_var + the
    # wake-block wake_detail) are hoisted into per-hero LOCALS that default to the
    # global uniforms; a `#ifdef CAST_LEVERS` block (new binding-5 CastLevers SSBO)
    # overrides them from THIS hero's own row. Default program: locals == the
    # uniforms, so numerically identical — p05 9/9 verified same day (byte-identical
    # by construction; the CAST_LEVERS variant compiles only when a cast hero
    # overrides a lever).
    # Updated 2026-07-24 (M2-B per-storm emergence family): emergence/shape/taper
    # join the hoisted per-hero locals in all FOUR hero scopes (stamp anatomy,
    # wake block, heroRelaxWeight, heroBandDeflect), each `#ifdef CAST_LEVERS`-
    # overridden from vec4_2 of this hero's row (the row grew 2 -> 3 vec4, so every
    # index moved 2*i -> 3*i). heroRelaxWeight's cross-hero max() combine could no
    # longer scale by one global after the loop, so each accumulator carries the
    # emergence of the hero that owns it and compares on the SCALED magnitude with a
    # raw tie-break — arithmetically identical whenever every hero shares one
    # emergence. Verified by `uv run python scripts/m2b_emergence_hash.py --check`
    # (bare / shape+taper / rim+wake / TWO CONTENDING heroes / aspect+interior,
    # all 5 byte-identical across the change) — that is the gate for THIS file's
    # variant arms; p05 9/9 cannot see them at all (every p05 config has
    # hero_emergence 0, so HERO_EMERGENCE never compiles).
    # Re-pinned 2026-07-24 (M2-B review): heroRelaxWeight's hero loop now skips
    # heroes at emergence 0 outright. Without that skip their scaled candidate
    # (0) TIED the initial accumulator and the raw tie-break handed them the
    # slot, so an opted-out storm's raw wake window rode `flush *= 1.0 - wrel`
    # into a DIFFERENT, emergent hero's flush. Byte-identical for a shared
    # emergence > 0 (the guard is then always true) — same 5 hashes, p05 9/9.
    "vortex_stamp.glsl": "e1f250b233d880d191de2d8c0da50500513f6f65",
    # NOT kinematic — pinned for the opposite reason (PR-43 test review,
    # 2026-07-16): vortex_omega.glsl hosts the vorticity-only hero levers
    # (solid core, emergence ring/skirt, shape/taper/flow-aspect arms), whose
    # levers-OFF output NO byte gate can observe (p05 renders kinematic-only
    # configs; SOR noise forbids byte asserts on developed vorticity output;
    # the dev-0 omega byte-captures live in session scratchpads and die with
    # the session). The source pin is the standing off-state gate: any edit
    # here is a CONSCIOUS re-pin whose author re-runs the dev-0 omega capture
    # discipline (see tests/gpu/test_hero_emergence.py::_dev0_omega).
    # Pinned 2026-07-16 (PR-43 review fixes commit): comment-only corrections
    # same day (wake window frame note — wake_dir is flow-derived under
    # emergence, east on warm; K>1 GRS-recipe claim replaced with the S2
    # falsified verdict; skirt peak 0.9 -> 1.0 and ~70% -> ~76% cancellation
    # prose; renorm figures re-qualified to the S1 calibration scene).
    # Updated 2026-07-23 (M2 per-storm CastLevers, part 2 — GPU wiring): the hero
    # solid-core lever is hoisted into a per-hero local `solidcore_v` defaulting to
    # u_hero_solid_core; a `#ifdef CAST_LEVERS` line overrides it from this hero's
    # binding-5 CastLevers row (the callers omega_init.comp/omega_force.comp declare
    # the buffer). Default path == u_hero_solid_core (byte-identical); the levers-off
    # gate is the dev-0 omega byte-capture (test_hero_emergence.py::_dev0_omega),
    # re-run this pass. Variant-arm-only otherwise.
    # Updated 2026-07-24 (M2-B per-storm emergence family): emergence/shape/taper
    # hoisted into per-hero locals inside the HERO_EMERGENCE arm of the solid-core
    # branch, `#ifdef CAST_LEVERS`-overridden from vec4_2 (index 2*i -> 3*i). Per
    # storm the omega side therefore acts only for a hero whose own solid_core > 0,
    # mirroring the existing global coupling. Default path == the uniforms; the
    # off-state gate is the dev-0 omega byte-capture (test_cast_levers.py /
    # test_hero_emergence.py::_dev0_omega), re-run this pass.
    # 2026-07-24 (M2-C): heroAnchorWindow/heroWakeWindow are UNCHANGED; two
    # CAST_LEVERS-only companions (heroAnchorBoost/heroWakeInject) were added
    # beside them, folding each hero's own emergence into its candidate before
    # the cross-hero max so a placed hero anchors and injects at ITS value
    # rather than the scene max. omega_force keeps the legacy lines verbatim in
    # its #else arms.
    #
    # The companions are a separate variant arm rather than a rewrite because
    # bit-identity for a uniform scene was MEASURED AND FAILED: scaling each
    # candidate before the max is exact in isolation (rounding is monotone), but
    # the legacy site `1.0 + 60.0*E*wa` can be FMA-contracted and interposing
    # max() forces the product to round first. 9.5e-07 on 0.007% of pixels after
    # ONE step, amplified by the chaotic field to O(OMEGA_CEILING) in q by step
    # 40. Preserving expression SHAPE was not enough — see
    # scripts/m2c_omega_equiv.py, which is the gate for this path (p05 is
    # kinematic and cannot reach omega_force; the dev-0 omega carve-out cannot
    # see a step>0 pass). That script now measures maxdiff 0 across 40 steps.
    "vortex_omega.glsl": "37e4bd9de5a74ed94f7cdd14f5b2805b8ba3eb0e",
    # New 2026-07-10 with hero_emergence: heroEllipQ, the shared elliptical-q
    # helper for the variant-only heroRelaxWeight/heroAnchorWindow. Entirely
    # #ifdef HERO_EMERGENCE => contributes nothing to the default program.
    "hero_q.glsl": "0d116e76423ac56301e74907bf2b2a81aaa659fa",
    "band_mod.glsl":     "278a7379ae63c7cc59e4ab8b61c7dc783c099fd6",
    # Updated 2026-07-15 (waves.festoon_hero_strength, Round B of the GRS
    # hero-interaction pass): a second festoon train rooted on the band edge
    # nearest the hero — plumes only, T3 only, per-plume amplitude jitter.
    # Entirely `#ifdef FESTOON2` (predicate in solver._domain_defines:
    # strength > 0 AND a facade-selected root edge exists), so the default
    # program text is unchanged after preprocessing — byte-identical by
    # construction; p05 9/9 verified same day.
    "wave_stamp.glsl":   "97e01d66d370e640867619e083f69610505cfd7e",
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
