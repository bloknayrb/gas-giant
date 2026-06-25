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
uniform float u_hero_solid_core;

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
                q = length(vec2(dot(p, e1) / asp, dot(p, e2))) / r_core;
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
            contrib = mix(contrib, disk, u_hero_solid_core);
        }

        // Magnitude-based cull (safe for large r_core / GRS)
        if (abs(contrib) < EPS * abs(scale)) {
            continue;
        }

        omega += contrib;
    }

    return omega;
}
