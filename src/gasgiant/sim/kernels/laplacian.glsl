// laplacian.glsl — Discrete spherical Laplacian (equirect + AE polar patch).
//
// Provides:
//   float laplacianPsi(sampler2D field, ivec2 px, ivec2 size)
//
// Branches on DOMAIN:
//   DOMAIN == 0  equirect (longitude-periodic, latitude-clamped)
//   DOMAIN != 0  azimuthal-equidistant polar patch (both axes clamped)
//
// Sign convention (CRITICAL, locked in P2 vorticity_ref.py):
//   ω = +∇²ψ    (Poisson equation: ∇²ψ = +ω)
//
// ── DOMAIN == 0 (equirect) ────────────────────────────────────────────────────
//
// Grid convention:
//   Row y=0 is the NORTH pole; y increases southward.
//   lat(px) = +π/2 − (px.y + 0.5)/H · π        (descending latitude)
//
// Discrete spherical Laplacian (naive central-difference, matches velocity.comp):
//
//   ∇²ψ = (1/cos²φ) · (ψ[x+1,y] − 2ψ[x,y] + ψ[x−1,y]) / Δλ²
//         + (ψ[x,y+1] − 2ψ[x,y] + ψ[x,y−1]) / Δφ²
//         + tanφ · (ψ[x,y+1] − ψ[x,y−1]) / (2Δφ)
//
// SIGN of the tanφ term (+, not −):
//   In the descending-row convention, ∂ψ/∂φ = (ψ[y−1] − ψ[y+1]) / (2Δφ)
//   (row y−1 is at HIGHER latitude).  velocity.comp line 43 confirms:
//       dpsi_dphi = (psiAt(x, px.y−1) − psiAt(x, px.y+1)) / (2·dlat)
//   The full Laplacian includes −tanφ · ∂ψ/∂φ (continuous).  Substituting:
//       −tanφ · (ψ[y−1] − ψ[y+1]) / (2Δφ)
//     = +tanφ · (ψ[y+1] − ψ[y−1]) / (2Δφ)          ← y+1 is southward (descending)
//   Written in the same order as the numerics:
//       = +tanφ · (ψ[x,y+1] − ψ[x,y−1]) / (2Δφ)
//   This is the + sign above.  Verified by the P3a GPU sign-lock test against
//   the Y_2^0 spherical harmonic (eigenvalue −6).
//
// Δλ = 2π/W,  Δφ = π/H.
// wrapX for longitude (periodic), clampY for latitude (boundary rows clamped).
// cosφ floored at 0.017 (mirrors velocity.comp / vorticity_ref.py).
//
// Reused by omega_force.comp (∇⁴ = laplacianPsi(laplacianPsi(·)) hyperviscosity)
// and by the P3b SOR Poisson solve.
//
// ── DOMAIN != 0 (AE polar patch) ─────────────────────────────────────────────
//
// Patch grid: Cartesian (s,t) in radians, s,t ∈ [−rho_max, rho_max].
//   rho   = length(st)   — colatitude (geodesic distance from pole)
//   theta = atan(t, s)   — longitude
//
// Laplace-Beltrami in AE (s,t) coordinates (mirrors vorticity_ref.laplacian_patch):
//
//   ∇²ψ = c_ss·ψ_ss + c_tt·ψ_tt + c_st·ψ_st + c_g·(s·ψ_s + t·ψ_t)/rho
//
//   c_ss = s²/rho² + t²/sin²rho
//   c_tt = t²/rho² + s²/sin²rho
//   c_st = 2·s·t·(1/rho² − 1/sin²rho)
//   c_g  = cos(rho)/sin(rho) − rho/sin²rho
//
// rho floored at 1e-6, sin(rho) floored at 1e-6.
// dstep = 2·u_rho_max / size.x  (st-radians per pixel).
// BOTH axes clamped (patch is NOT periodic) — wrapX is already a clamp for
// DOMAIN != 0 (see common.glsl).  clampY also clamps.
// POLE_SIGN does not enter the Laplacian.
// Verified by P6b GPU sign-lock test against Y_2^0 (eigenvalue −6).

float laplacianPsi(sampler2D field, ivec2 px, ivec2 size) {
#if DOMAIN == 0
    int W = size.x;
    int H = size.y;

    float dlam = 2.0 * PI / float(W);   // longitude step
    float dphi = PI / float(H);          // latitude step (|Δφ| between rows)

    // Pixel-center latitude: descending, row 0 = north pole.
    float lat = 0.5 * PI - (float(px.y) + 0.5) / float(H) * PI;

    float cosl = max(cos(lat), 0.017);
    float tanl = tan(lat);

    // Stencil neighbours.
    float c   = texelFetch(field, ivec2(wrapX(px.x,     W), clampY(px.y,     H)), 0).r;
    float xp1 = texelFetch(field, ivec2(wrapX(px.x + 1, W), clampY(px.y,     H)), 0).r;
    float xm1 = texelFetch(field, ivec2(wrapX(px.x - 1, W), clampY(px.y,     H)), 0).r;
    float yp1 = texelFetch(field, ivec2(wrapX(px.x,     W), clampY(px.y + 1, H)), 0).r;
    float ym1 = texelFetch(field, ivec2(wrapX(px.x,     W), clampY(px.y - 1, H)), 0).r;

    // Longitude part: (1/cos²φ) · d²ψ/dλ²
    float d2_lam = (xp1 - 2.0 * c + xm1) / (dlam * dlam);
    float lon_term = d2_lam / (cosl * cosl);

    // Latitude part: d²ψ/dφ²
    float d2_phi = (yp1 - 2.0 * c + ym1) / (dphi * dphi);

    // tanφ correction: +tanφ · (ψ[y+1] − ψ[y−1]) / (2Δφ)
    // (+ sign under descending-row convention — see header comment)
    float d1_phi_desc = (yp1 - ym1) / (2.0 * dphi);
    float tan_term = tanl * d1_phi_desc;

    return lon_term + d2_phi + tan_term;

#else
    // ── AE polar patch Laplacian (mirrors vorticity_ref.laplacian_patch) ──────
    int W = size.x;
    int H = size.y;

    // st-radians per pixel (patch is square: size.x == size.y assumed)
    float dstep = 2.0 * u_rho_max / float(W);

    // (s, t) at pixel centre (radians of colatitude on each axis)
    vec2 st = patchStFromPix(vec2(px) + 0.5, size);
    float s = st.x;
    float t = st.y;

    float rho = max(length(st), 1e-6);
    float sinr = max(sin(rho), 1e-6);

    float rho2   = rho * rho;
    float sin2   = sinr * sinr;
    float inv_r2 = 1.0 / rho2;
    float inv_s2 = 1.0 / sin2;

    // AE Laplacian coefficients (finite at rho→0 → flat Laplacian)
    float c_ss = s * s * inv_r2 + t * t * inv_s2;
    float c_tt = t * t * inv_r2 + s * s * inv_s2;
    float c_st = 2.0 * s * t * (inv_r2 - inv_s2);
    float c_g  = cos(rho) / sinr - rho * inv_s2;

    // Stencil neighbours — BOTH axes clamped (patch is not periodic).
    // wrapX is already clamp for DOMAIN != 0 (see common.glsl); clampY clamps.
    float c    = texelFetch(field, ivec2(wrapX(px.x,     W), clampY(px.y,     H)), 0).r;
    float xp1  = texelFetch(field, ivec2(wrapX(px.x + 1, W), clampY(px.y,     H)), 0).r;
    float xm1  = texelFetch(field, ivec2(wrapX(px.x - 1, W), clampY(px.y,     H)), 0).r;
    float yp1  = texelFetch(field, ivec2(wrapX(px.x,     W), clampY(px.y + 1, H)), 0).r;
    float ym1  = texelFetch(field, ivec2(wrapX(px.x,     W), clampY(px.y - 1, H)), 0).r;

    // Cross-stencil for psi_st (4 corners, both axes clamped)
    float xp1yp1 = texelFetch(field, ivec2(wrapX(px.x + 1, W), clampY(px.y + 1, H)), 0).r;
    float xp1ym1 = texelFetch(field, ivec2(wrapX(px.x + 1, W), clampY(px.y - 1, H)), 0).r;
    float xm1yp1 = texelFetch(field, ivec2(wrapX(px.x - 1, W), clampY(px.y + 1, H)), 0).r;
    float xm1ym1 = texelFetch(field, ivec2(wrapX(px.x - 1, W), clampY(px.y - 1, H)), 0).r;

    // Central differences in st (dstep = st-radians per pixel).
    // x = column/s axis (px.x ± 1), y = row/t axis (px.y ± 1).
    float psi_s  = (xp1 - xm1) / (2.0 * dstep);
    float psi_t  = (yp1 - ym1) / (2.0 * dstep);
    float psi_ss = (xp1 - 2.0 * c + xm1) / (dstep * dstep);
    float psi_tt = (yp1 - 2.0 * c + ym1) / (dstep * dstep);
    float psi_st = (xp1yp1 - xp1ym1 - xm1yp1 + xm1ym1) / (4.0 * dstep * dstep);

    // Radial gradient term: (s·ψ_s + t·ψ_t) / rho
    float psi_rho = (s * psi_s + t * psi_t) / rho;

    return c_ss * psi_ss + c_tt * psi_tt + c_st * psi_st + c_g * psi_rho;
#endif
}
