// vortex_omega.glsl — Gaussian-vortex vorticity accumulator (v1.6 P2).
//
// Mirrors vorticity_ref.py:vortex_omega_ref() exactly.
// Included by P3's omega compute kernel; this file is a pure helper —
// it declares no layout/binding itself and needs no standalone dispatch.
//
// Sign convention: ω = +∇²ψ  (same as vorticity_ref.py; see P2 docs).
//
// Analytic formula (∇²f(d) = f''(d) + cot(d)·f'(d) on the 2-sphere):
//   ψ = S·exp(−q²), q = d/r_core
//   f'  = −(2Sq/r)·exp(−q²)
//   f'' = (S/r²)·(4q²−2)·exp(−q²)
//   ω   = (S/r²)·[(4q²−2) − 2·(d/tan d)]·exp(−q²)
//
// The d/tan(d) form avoids the cot(d) singularity at d=0 and is
// numerically stable.  A small-d series guard (1 − d²/3) is kept for
// exactness near the vortex centre.
//
// SSBO layout is identical to vortex_stamp.glsl (shared binding):
//   [3*i+0]  vec4(x, y, z, r_core)
//   [3*i+1]  vec4(strength, kind, tint, brightness)
//   [3*i+2]  vec4(wake_dir, aspect, 0, 0)
//
// The q computation (great-circle distance + hero-aspect branch) is
// copied verbatim from vortex_stamp.glsl lines 28-43 to stay in sync.

// ---------------------------------------------------------------------------
// Coriolis parameter:  f(φ) = f₀·sin φ
// ---------------------------------------------------------------------------

float coriolis(float lat, float f0) {
    return f0 * sin(lat);
}

// ---------------------------------------------------------------------------
// d/tan(d) with small-d series fallback
//
// Mirrors vorticity_ref.py:  lim_{d→0} d/tan(d) = 1 − d²/3 + …
// ---------------------------------------------------------------------------

float d_over_tand(float d) {
    if (d < 1e-4) {
        return 1.0 - d * d / 3.0;
    }
    return d / tan(d);
}

// Hero kind sentinel (matches vortex_stamp.glsl, which shares this SSBO) and the
// solid-body blend amount. Declared here so both includers (omega_init.comp,
// omega_force.comp) see them; default 0.0 keeps the Gaussian path byte-identical.
const float VKIND_HERO = 1.0;
const float VKIND_OVAL = 0.0;
// KIND_OVAL entries below this core radius (radians) don't visibly bullseye and
// stay Gaussian regardless of u_oval_solid_core. This excludes the small-storm-
// scale KIND_OVAL spots; large white ovals plus any merge-product / KH-debris
// ovals at/above the threshold get the patch. (Pearls are KIND_PEARL, not OVAL,
// so they never reach this gate.) Matches test_oval_solid_core.py.
const float OVAL_SOLID_MIN_R = 0.035;
uniform float u_hero_solid_core;
uniform float u_oval_solid_core;
#ifdef HERO_EMERGENCE
// GRS-realism pack strength (annular-ring morph + core anchor). Compiled as a
// preprocessor variant: the default program text is the pre-feature kernel.
uniform float u_hero_emergence;
// storms.hero_shape (+ its dedicated seed substream): the hero ring/skirt
// carry the SAME low-order outline deformation as the tracer anatomy — the
// visible boundary is wound material riding THESE streamlines, so a tracer-
// only deformation is invisible at develop time (measured: a 3-seed strip
// rendered near-identical until the flow deformed too).
uniform float u_hero_shape;
uniform vec3 u_hero_shape_phase;
// storms.hero_taper: the upstream wedge deforms the ring/skirt too, for the
// same reason as the lobes above (the flow owns the outline).
uniform float u_hero_taper;
// storms.hero_flow_aspect (K): author the FLOW wider than the tracer anatomy.
// psi = inv-Laplacian(omega) is intrinsically rounder than the ring (Poisson
// low-pass: boundary streamlines measure 1.24-1.40 from a ring held at 2.2),
// the dye rides psi, and ambient shear is ~6x below the Kida requirement —
// so the developed dye reads ~1.5 no matter what the stamps say. Widening
// only the ring/skirt EW metric by K pre-compensates the FLOW — but the S2
// calibration (2026-07-16) falsified K>1 as a GRS recipe: every erasure
// window stays anatomy-metric, so the widened eddy dilutes its own dye and
// the ENVELOPE stretches while the reference wants the CORE more elongated
// (2.9) than the envelope (1.8). The shipped recipe raises authored
// hero_aspect instead (build_warm_preset.py); K stays a default-1.0
// envelope-trim lever with the dilution documented in
// test_hero_flow_aspect_flow_stays_anchored_at_hi.
uniform float u_hero_flow_aspect;
// Net-circulation renorm for the widened ring+skirt, CPU-computed by
// spherical quadrature (sim/flow_renorm.py) from the ACTUAL hero r_core.
// The tangent-plane 1/K under-corrects: the curvature weight inflates the
// wide skirt more than the ring, and the net is the ~24% residual of a ~76%
// cancellation — plain 1/K leaves a 16% net deficit at K=2 on the S1
// calibration scene (r_core 0.062 / aspect 2.2; larger at warm's baked 2.9)
// — the taper's visible band-shift class. Exactly 1.0 whenever the lever is
// off.
uniform float u_hero_flow_renorm;
#endif
// heroEllipQ for the variant-only heroAnchorWindow below (include-once, and
// the include's whole body is #ifdef HERO_EMERGENCE). Needs vortex_data,
// which the includer declares before this file.
#include "hero_q.glsl"

// ---------------------------------------------------------------------------
// Accumulated ω for all vortices at sphere point p.
//
// p             — unit sphere position (normalised xyz)
// vortex_data   — same SSBO as vortex_stamp.glsl (must be declared by caller)
// u_vortex_count — uniform declared by caller
//
// Magnitude-based cull: skip a vortex contribution only when |contrib| is
// below EPS * |reference_scale| (= |S/r_core²|).  A fixed q-cutoff would
// truncate contributions from large-core vortices (GRS r_core ~ 0.15 rad).
// ---------------------------------------------------------------------------

#ifdef HERO_EMERGENCE
// Hero anchor window in [0,1]: 1 inside any hero's core+ring, fading to 0 past
// the collar. Used by omega_force SUBPASS 0 to locally BOOST the q-nudge so the
// prognostic hero core stays glued to the (drifting) registry position — without
// it the core wanders ~0.2 rad from where the tracer stamp paints the red fill,
// smearing the spot into a muddy arc.
float heroAnchorWindow(vec3 p) {
    float w = 0.0;
    for (int i = 0; i < u_vortex_count; ++i) {
        vec4 b = vortex_data[3 * i + 1];
        if (b.y != VKIND_HERO) continue;
        // Window deliberately WIDER than the visible anatomy (the compacted
        // collar ends ~1.5): this is a capture basin, not a visual feature —
        // it must exceed the core's free-drift excursion (~0.2 rad at 512
        // res) or the wandering core escapes the boosted nudge and never gets
        // pulled back (the anchor test's 0.04-T3 failure mode), and it must
        // cover the whole shield skirt (ends 2.6) so the skirt is held in
        // place too. Costs no footprint: the boost only speeds relaxation
        // toward the band target out there.
        w = max(w, 1.0 - smoothstep(1.6, 2.8, heroEllipQ(p, i, 2.8)));
    }
    return w;
}

// Hero wake-wedge window in [0,1]: the turbulent-wake sector DOWNSTREAM of
// each hero (same frame the tracer wedge in vortex_stamp.glsl uses). This
// function compiles only under HERO_EMERGENCE, where wake_dir is flow-
// derived by vortices.py::_hero_wake_frame — EAST on gas_giant_warm — or
// forced by storms.hero_wake_dir; "hardwired westward" is only the
// emergence-off legacy fallback, a configuration this code never runs in.
// wake_lat_off likewise tracks the jet under emergence. (A west-assuming
// read of this window already mis-measured one taper review — judge in
// wake_dir terms, never compass terms.)
// Used by omega_force SUBPASS 0 to localize wake eddy-vorticity injection:
// psi.comp's velocity wake is FEATHER-ONLY in vorticity mode (the definitive
// psi at storm latitudes is SOR-solved from prognostic q), so without a
// q-side source the wake sector has no folding velocity at all and the
// tracer-side relaxation release preserves nothing but laminar shear. Hard
// windows on both axes => exactly 0 in the far field.
float heroWakeWindow(vec3 p) {
    float w = 0.0;
    for (int i = 0; i < u_vortex_count; ++i) {
        vec4 b = vortex_data[3 * i + 1];
        if (b.y != VKIND_HERO) continue;
        vec4 a = vortex_data[3 * i];
        float rc = a.w;
        float down = vortex_data[3 * i + 2].x;
        float woff = vortex_data[3 * i + 2].z;
        float vlat = asin(clamp(a.y, -1.0, 1.0));
        float vlon = atan(a.z, a.x);
        float plat = asin(clamp(p.y, -1.0, 1.0));
        float plon = atan(p.z, p.x);
        float dlon = mod(plon - vlon + 3.0 * PI, 2.0 * PI) - PI;
        float an = dlon * down / max(rc, 1e-4);
        float across = (plat - (vlat + woff)) / max(rc * 1.8, 1e-4);
        // Rises just off the collar (the ring's own shear folds the boundary;
        // the injection owns the wedge beyond), decays to exactly 0 by 9 rc.
        if (an > 0.8 && an < 9.0 && abs(across) < 2.0) {
            float rise = smoothstep(0.8, 1.8, an);
            float fall = 1.0 - smoothstep(6.0, 9.0, an);
            float aw = (1.0 - smoothstep(1.4, 2.0, abs(across)))
                     * exp(-across * across);
            w = max(w, rise * fall * aw);
        }
    }
    return w;
}
#endif  // HERO_EMERGENCE

float vortexOmegaAccum(vec3 p) {
    float omega = 0.0;
    const float EPS = 1e-6;

    for (int i = 0; i < u_vortex_count; ++i) {
        vec4 a = vortex_data[3 * i];
        vec4 b = vortex_data[3 * i + 1];

        float S      = b.x;          // streamfunction strength
        float r_core = a.w;          // core radius (radians)

        // --- great-circle distance + q (verbatim from vortex_stamp.glsl) ---
        float d   = acos(clamp(dot(p, a.xyz), -1.0, 1.0));
        float asp = vortex_data[3 * i + 2].y;
        float q;
        if (asp == 1.0) {
            q = d / r_core;
        } else {
            vec3 c  = a.xyz;
            vec3 ew = cross(vec3(0.0, 1.0, 0.0), c);
            float ewl = length(ew);
            if (ewl < 1e-4) {
                q = d / r_core;
            } else {
                vec3 e1 = ew / ewl;
                vec3 e2 = cross(c, e1);
                // Gate the tangent-plane metric to the near hemisphere: at the
                // antipode both components vanish -> q~0 would stamp a phantom
                // vorticity patch on the far side. Far points get a large q.
                q = (dot(p, c) > 0.0)
                  ? length(vec2(dot(p, e1) / asp, dot(p, e2))) / r_core
                  : 1e3;
            }
        }
        // For elliptical vortices d is the round great-circle distance while
        // q uses the aspect-stretched metric.  The d/tan(d) term uses the
        // true great-circle d (metric curvature, independent of ellipse shape).

        float exp_q2 = exp(-q * q);
        float scale  = S / (r_core * r_core);   // reference scale

        // term1: radial curvature  f''(d)
        float term1 = scale * (4.0 * q * q - 2.0) * exp_q2;
        // term2: geodesic-metric correction  cot(d)·f'(d) = −(2S/r²)·(d/tan d)·exp(−q²)
        float term2 = -2.0 * scale * d_over_tand(d) * exp_q2;

        float contrib = term1 + term2;

        // Solid-body hero core (vorticity mode): the Gaussian-Laplacian vorticity
        // is center-peaked, giving DIFFERENTIAL rotation that winds the passive
        // tracer into a center-draining whirlpool. A near-uniform vorticity patch
        // (flat disk, smooth edge) instead gives RIGID solid-body interior
        // rotation -> the hero reads as a coherent GRS-like oval (spiral arms
        // only OUTSIDE it). Blended so u_hero_solid_core=0 is byte-identical to
        // the Gaussian; =1 is the full patch.
        // Per-storm solid-core lever: the global uniform by default; the
        // CAST_LEVERS variant reads THIS hero's own row (the caller -- omega_init
        // / omega_force -- declares the CastLevers buffer at binding 5, exactly as
        // it declares Vortices). Default path == u_hero_solid_core (byte-identical).
        float solidcore_v = u_hero_solid_core;
#ifdef CAST_LEVERS
        solidcore_v = cast_lever_data[2 * i + 1].z;
#endif
        if (b.y == VKIND_HERO && solidcore_v > 0.0) {
            // ('patch' is a reserved GLSL keyword — use 'disk'.)
            float disk = -2.5 * scale * (1.0 - smoothstep(0.80, 1.15, q));
            // Hero emergence: morph the solid disk toward an ANNULAR RING of
            // vorticity — the real GRS profile (Voyager/Juno): the ~430 km/h
            // winds live in a peripheral ring while the interior is STAGNANT
            // ("currents inside it seem stagnant, with little inflow or
            // outflow"). By Stokes' theorem a ring with zero enclosed vorticity
            // induces v~0 inside -> the quiescent core HOLDS its stamped red
            // fill (physics and relaxation agree), while the ring's intense
            // shear folds the tracer at the BOUNDARY — genuine emergent
            // raggedness exactly where the real storm has it, no injected
            // noise. (Amplitude/placement rationale lives with the code below.)
#ifdef HERO_EMERGENCE
            // Ring placed so its shear peaks just inside the visible oval edge
            // (the plateau fill's edge sits at q~1.0, the authored hero_radius):
            // folding happens at the color boundary, matching the real GRS.
            // Amplitude -6.0: the compacted ring has ~0.42x the old annulus
            // area, so circulation is scaled to keep v_peak just outside the
            // rim in the same league as the calibrated disk (CFL-safe — the
            // advection CFL is velocity-based, and v_peak is unchanged).
            {
                // Outline shape deformation on the FLOW: same R(theta), same
                // seeds as vortex_stamp.glsl, applied to the ring/skirt q so
                // the streamlines themselves are egg-shaped and the emergent
                // wound boundary inherits the deformed outline. The anchor
                // window (heroAnchorWindow) deliberately stays elliptical —
                // capture-basin stability. R modulates radius by <=~13% at
                // shape 1: net circulation moves a few percent around the
                // calibrated ~76% skirt cancellation, far from the falsified
                // full-shield regime; watch for roll-up per the note below.
                float qh = q;
                float tcomp = 1.0;
                if (u_hero_shape > 0.0 || u_hero_taper > 0.0
                        || u_hero_flow_aspect != 1.0) {
                    vec3 hcs = a.xyz;
                    vec3 hews = cross(vec3(0.0, 1.0, 0.0), hcs);
                    float hewls = length(hews);
                    // Pole fallback: flow_aspect silently no-ops here too —
                    // fine, the 60-deg jet confine leaves no hero near a pole.
                    if (hewls > 1e-4) {
                        vec3 hs1 = hews / hewls;
                        // Flow-metric base: the shape/taper deforms below ride
                        // whatever metric the ring is authored on.
                        float qb = q;
                        float aspf = asp;
                        if (u_hero_flow_aspect != 1.0) {
                            // Rebuild the hero q on the K-widened ellipse (EW
                            // component only; same tangent-plane construction
                            // and hemisphere gate as the loop head). NOTE at
                            // hero_aspect 1.0 the loop-head q is great-circle
                            // while qb is tangent-plane: the ~10%-weight
                            // solid-disk residual and the ring then differ by
                            // O(d^2/6) ~ 0.4% at the skirt edge — accepted.
                            aspf = asp * u_hero_flow_aspect;
                            vec3 hs2b = cross(hcs, hs1);
                            float xq = dot(p, hs1) / aspf;
                            float yq = dot(p, hs2b);
                            qb = (dot(p, hcs) > 0.0)
                               ? length(vec2(xq, yq)) / r_core : 1e3;
                            tcomp = u_hero_flow_renorm;
                        }
                        float Rrs = 1.0;
                        if (u_hero_shape > 0.0) {
                            vec3 hs2 = cross(hcs, hs1);
                            float hths = atan(dot(p, hs2), dot(p, hs1));
                            float thps = 3.14159265 - hths;   // 0 = local EAST
                            float neqs = (a.y < 0.0) ? max(sin(hths), 0.0)
                                                     : max(-sin(hths), 0.0);
                            vec3 sphs = u_hero_shape_phase;
                            Rrs -= u_hero_shape * u_hero_emergence
                                   * (0.11 * neqs * neqs
                                      - 0.075 * sin(2.0 * thps + sphs.x)
                                      - 0.055 * sin(3.0 * thps + sphs.y));
                        }
                        if (u_hero_taper > 0.0) {
                            // Upstream-signed SQUASHED cosine over r_core*qb
                            // = |squashed offset| in the FLOW metric (aspf,
                            // qb): the wedge deforms the ring it rides, so it
                            // lives in the ring's metric; at K=1 these are
                            // the loop-head asp/q exactly. hs1 = cross(j, c)
                            // points ANTI-east (the F06 chirality trap), so
                            // +wdir * dot = upstream. Matches the stamp and
                            // relax wedges exactly (constants pinned by the
                            // blocks-agree unit test).
                            float wdir_h = vortex_data[3 * i + 2].x;
                            float uct = clamp(wdir_h * dot(p, hs1)
                                              / (aspf * max(r_core * qb, 1e-5)),
                                              -1.0, 1.0);
                            float tc = max(uct, 0.0);
                            float tc2 = tc * tc;
                            float tw = 6.75 * tc2 * tc2 * (1.0 - tc2);
                            Rrs -= 0.25 * u_hero_taper * u_hero_emergence * tw;
                            // Same slider-space clamp, same guard placement
                            // as the tracer sites.
                            Rrs = max(Rrs, 0.4);
                            // CIRCULATION CONSERVATION: the wedge shrinks
                            // the ring/skirt area by mean(Rrs^2) ~ 1 -
                            // 2*0.25*t*e*mean(w) (mean(w) = 0.21 over the
                            // circle) -> net circulation would drop ~7-10%
                            // and the FAR FIELD with it — measured as a
                            // coherent band shift 25+ deg away at all
                            // longitudes (corr 0.988 across doses) plus a
                            // chaotic north-jet re-roll, both through the
                            // global Poisson solve. Renormalize ring+skirt
                            // UNIFORMLY (cancellation fraction untouched);
                            // the local deficit stays where the wedge is,
                            // the planet-scale moment does not move.
                            tcomp *= 1.0 / (1.0 - 0.105 * u_hero_taper
                                                  * u_hero_emergence);
                        }
                        qh = qb / Rrs;
                    }
                }
                float ring = -6.0 * scale
                           * (smoothstep(0.29, 0.55, qh) - smoothstep(0.78, 1.04, qh));
                // PARTIAL SHIELD skirt: the single-signed ring (like the solid
                // disk it morphs from) carries NET circulation, so its velocity
                // decays like 1/r and winds the whole neighborhood into a
                // pinwheel many spot-radii wide — the "effect much bigger than
                // the storm" tell. (The kinematic Gaussian never had this: a
                // Laplacian-of-Gaussian is self-shielded.) A gentle opposite-
                // signed annulus cancels ~76% of it, taming the far-field
                // winding; the tracer-side band-flush (heroRelaxWeight) erases
                // the slower residual. Deliberately PARTIAL and WIDE/WEAK
                // (peak 1.0 vs the ring's 6.0): a full-strength concentrated
                // shield rolls up into its own companion cyclone and the
                // resulting near-dipole self-propels off the anchor (observed:
                // a second creamy swirl displacing the core). Enclosed
                // circulation at the rim — the peripheral wind speed — is
                // untouched. Physically: the counter-flowing jets deflected
                // around the Hollow; Cassini shows bands running parallel
                // again by ~1.5-2 spot radii — hence this PULL-IN (was
                // 0.7 x [1.4..2.0,2.6]): the skirt now ends by 2.4 with the
                // amplitude raised so net cancellation stays ~68% by
                // int(w*q dq) — plan review caught that the naive pull-in
                // dropped it to 54%, STRENGTHENING the pinwheel 42%. If the
                // concentrated skirt shows any roll-up, revert the OUTER
                // window toward (2.0,2.6); NEVER lower the amplitude (that
                // reduces cancellation further).
                // 0.9 -> 1.0 (cancellation 68% -> 76%): reference-anchored
                // review found band edges still curving concentrically 1.5
                // oval-widths out (the whirlpool signature) — the residual
                // circulation's far field. Still PARTIAL (peak 17% of the
                // ring's 6.0); watch for roll-up per the fallback note above.
                ring += 1.0 * scale
                      * (smoothstep(1.05, 1.35, qh) - smoothstep(1.8, 2.4, qh));
                // Apply the circulation renorm to ring+skirt TOGETHER
                // (cancellation fraction untouched). tcomp = 1.0 exactly
                // whenever taper is off -> byte-identical.
                ring *= tcomp;
                disk = mix(disk, ring, u_hero_emergence);
            }
#endif
            contrib = mix(contrib, disk, solidcore_v);
        } else if (b.y == VKIND_OVAL && u_oval_solid_core > 0.0
                   && r_core >= OVAL_SOLID_MIN_R) {
            // Same patch for large ovals: kills the per-oval mini-bullseye that
            // otherwise accumulates over a long dev run. Small ovals (below the
            // threshold) and every other kind keep the byte-identical Gaussian.
            float disk = -2.5 * scale * (1.0 - smoothstep(0.80, 1.15, q));
            contrib = mix(contrib, disk, u_oval_solid_core);
        }

        // Magnitude-based cull (safe for large r_core / GRS)
        if (abs(contrib) < EPS * abs(scale)) {
            continue;
        }

        omega += contrib;
    }

    return omega;
}
