// laplacian.glsl — Discrete spherical Laplacian for the equirect domain.
//
// Provides:
//   float laplacianPsi(sampler2D field, ivec2 px, ivec2 size)
//
// DOMAIN == 0 ONLY. Applying to patch domains is unsupported (never included
// in a DOMAIN != 0 shader).
//
// Sign convention (CRITICAL, locked in P2 vorticity_ref.py):
//   ω = +∇²ψ    (Poisson equation: ∇²ψ = +ω)
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

float laplacianPsi(sampler2D field, ivec2 px, ivec2 size) {
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
}
