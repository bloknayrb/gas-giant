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
#endif

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
        vec4 a = vortex_data[3 * i];
        float d = acos(clamp(dot(p, a.xyz), -1.0, 1.0));
        float asp = vortex_data[3 * i + 2].y;
        float q;
        if (asp == 1.0) {
            q = d / a.w;
        } else {
            vec3 c = a.xyz;
            vec3 ew = cross(vec3(0.0, 1.0, 0.0), c);
            float ewl = length(ew);
            if (ewl < 1e-4) {
                q = d / a.w;
            } else {
                vec3 e1 = ew / ewl;
                vec3 e2 = cross(c, e1);
                q = (dot(p, c) > 0.0)
                  ? length(vec2(dot(p, e1) / asp, dot(p, e2))) / a.w
                  : 1e3;
            }
        }
        // Deliberately WIDER than the visible anatomy (the compacted collar
        // ends ~1.5): this is a capture basin, not a visual feature — it must
        // exceed the core's free-drift excursion (~0.2 rad at 512 res) or the
        // wandering core escapes the boosted nudge and never gets pulled back
        // (the anchor test's 0.04-T3 failure mode), and it must cover the
        // whole shield skirt (ends 2.6) so the skirt is held in place too.
        // Costs no footprint: the boost only speeds relaxation toward the
        // band target out there.
        w = max(w, 1.0 - smoothstep(1.6, 2.8, q));
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
        if (b.y == VKIND_HERO && u_hero_solid_core > 0.0) {
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
            // noise. Amplitude 1.8x the disk's roughly conserves circulation
            // (ring occupies ~55% of the disk area), keeping v_peak at the rim
            // in the same league as the calibrated disk (CFL-safe).
#ifdef HERO_EMERGENCE
            // Ring placed so its shear peaks just inside the visible oval edge
            // (the plateau fill's edge sits at q~1.0, the authored hero_radius):
            // folding happens at the color boundary, matching the real GRS.
            // Amplitude -6.0: the compacted ring has ~0.42x the old annulus
            // area, so circulation is scaled to keep v_peak just outside the
            // rim in the same league as the calibrated disk (CFL-safe — the
            // advection CFL is velocity-based, and v_peak is unchanged).
            {
                float ring = -6.0 * scale
                           * (smoothstep(0.29, 0.55, q) - smoothstep(0.78, 1.04, q));
                // PARTIAL SHIELD skirt: the single-signed ring (like the solid
                // disk it morphs from) carries NET circulation, so its velocity
                // decays like 1/r and winds the whole neighborhood into a
                // pinwheel many spot-radii wide — the "effect much bigger than
                // the storm" tell. (The kinematic Gaussian never had this: a
                // Laplacian-of-Gaussian is self-shielded.) A gentle opposite-
                // signed annulus cancels ~70% of it, taming the far-field
                // winding; the tracer-side band-flush (heroRelaxWeight) erases
                // the slower residual. Deliberately PARTIAL and WIDE/WEAK
                // (peak 0.7 vs the ring's 6.0): a full-strength concentrated
                // shield rolls up into its own companion cyclone and the
                // resulting near-dipole self-propels off the anchor (observed:
                // a second creamy swirl displacing the core). Enclosed
                // circulation at the rim — the peripheral wind speed — is
                // untouched. Physically: the counter-flowing jets deflected
                // around the Hollow; Cassini shows bands running parallel
                // again by ~1.5-2 spot radii.
                ring += 0.7 * scale
                      * (smoothstep(1.05, 1.4, q) - smoothstep(2.0, 2.6, q));
                disk = mix(disk, ring, u_hero_emergence);
            }
#endif
            contrib = mix(contrib, disk, u_hero_solid_core);
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
