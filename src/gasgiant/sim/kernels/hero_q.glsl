// Shared hero-emergence helper: aspect-corrected elliptical q for vortex i at
// sphere point p — the same metric vortexStamp/vortexOmegaAccum compute inline
// (their copies live in DEFAULT program text and must stay verbatim; this
// helper serves the variant-only heroRelaxWeight/heroAnchorWindow so the two
// don't carry drifting clones of the ~20-line block).
//
// Entirely #ifdef HERO_EMERGENCE: the GLSL preprocessor strips it from the
// default variant, so including this file changes no compiled default program.
// Requires the caller to have the Vortices SSBO (vortex_data) in scope.
#ifdef HERO_EMERGENCE
// Returns q, or 1e3 when q provably exceeds qmax. Far-field cull: with
// {e1,e2,c} orthonormal, dot(p,e1)^2 + dot(p,e2)^2 = 1 - cd^2, so
// q >= sqrt(1-cd^2) / (max(asp,1) * r) — and on the asp==1 great-circle path
// d >= sin d gives the same bound. Two multiplies reject the vast majority of
// pixels before the frame math and acos (both callers run per pixel per step
// in hot kernels, and the window they apply is zero past their qmax anyway).
float heroEllipQ(vec3 p, int i, float qmax) {
    vec4 a = vortex_data[3 * i];
    float asp = vortex_data[3 * i + 2].y;
    float cd = dot(p, a.xyz);
    float s2 = 1.0 - cd * cd;
    float lim = qmax * max(asp, 1.0) * a.w;
    if (s2 > lim * lim) return 1e3;
    if (asp == 1.0) {
        return acos(clamp(cd, -1.0, 1.0)) / a.w;
    }
    vec3 c = a.xyz;
    vec3 ew = cross(vec3(0.0, 1.0, 0.0), c);
    float ewl = length(ew);
    if (ewl < 1e-4) {
        return acos(clamp(cd, -1.0, 1.0)) / a.w;
    }
    vec3 e1 = ew / ewl;
    vec3 e2 = cross(c, e1);
    // Near-hemisphere gate: the tangent-plane metric collapses at the antipode.
    return (cd > 0.0)
        ? length(vec2(dot(p, e1) / asp, dot(p, e2))) / a.w
        : 1e3;
}
#endif  // HERO_EMERGENCE
