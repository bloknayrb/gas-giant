// sw_cont_cons.glsl — shared primitives for the M2 conservative FCT continuity
// kernel (continuity_step_conservative).  Include AFTER sw_common.glsl.
// Requires the includer to declare:
//   uniform sampler2D u_h;   // (W,H)
//   uniform sampler2D u_u;   // (W,H)
//   uniform sampler2D u_v;   // (W,H+1)
//   uniform float u_dt, u_h_floor, u_a;
//   uniform ivec2 u_size;
//
// Ports _mass_fluxes / _apply_fluxes / _positive_lowflux_scales from
// shallow_water_ref.py EXACTLY.

float cc_h(int i, int j, int W) { return texelFetch(u_h, ivec2(wrapX(i, W), j), 0).r; }
float cc_u(int i, int j, int W) { return texelFetch(u_u, ivec2(wrapX(i, W), j), 0).r; }
float cc_v(int i, int k, int W) { return texelFetch(u_v, ivec2(wrapX(i, W), k), 0).r; }

// --- low-order (donor) and high-order (centred) face fluxes -----------------
// Zonal east face of cell (i,j): donor = h[i] if u>=0 else h[i+1].
float fx_low_e(int i, int j, int W) {
    float uc = cc_u(i, j, W);
    float hd = (uc >= 0.0) ? cc_h(i, j, W) : cc_h(i + 1, j, W);
    return hd * uc;
}
float fx_high_e(int i, int j, int W) {
    float uc = cc_u(i, j, W);
    return 0.5 * (cc_h(i, j, W) + cc_h(i + 1, j, W)) * uc;
}
// Meridional v-face row k (1..H-1): donor = h[k-1] if v>=0 else h[k].
float fy_low_k(int i, int k, int W, int H) {
    if (k <= 0 || k >= H) return 0.0;
    float vc = cc_v(i, k, W);
    float hd = (vc >= 0.0) ? cc_h(i, k - 1, W) : cc_h(i, k, W);
    return hd * vc;
}
float fy_high_k(int i, int k, int W, int H) {
    if (k <= 0 || k >= H) return 0.0;
    float vc = cc_v(i, k, W);
    return 0.5 * (cc_h(i, k - 1, W) + cc_h(i, k, W)) * vc;
}

// --- _positive_lowflux_scales: per-cell s_cell given the cell's 4 LOW faces ---
// Returns s_cell in [0,1] for cell (i,j).  Uses the donor-low fluxes.
float s_cell_low(int i, int j, int W, int H) {
    float pref = u_dt / (u_a * cosCenter(j, H));
    float avail = max(cc_h(i, j, W) - u_h_floor, 0.0);
    float dlam = u_size.x > 0 ? (2.0 * PI / float(W)) : 0.0;
    float dphi = PI / float(H);

    float fxe = fx_low_e(i,     j, W);       // east face of cell i
    float fxw = fx_low_e(i - 1, j, W);       // west face = east face of i-1
    float out_xe = max(fxe, 0.0) / dlam;
    float out_xw = max(-fxw, 0.0) / dlam;

    float fyn = fy_low_k(i, j,     W, H);    // north face (row j)
    float fys = fy_low_k(i, j + 1, W, H);    // south face (row j+1)
    float fyn_c = fyn * cosVface(j,     H);
    float fys_c = fys * cosVface(j + 1, H);
    float out_yn = max(fyn_c, 0.0) / dphi;
    float out_ys = max(-fys_c, 0.0) / dphi;

    float out_total = pref * (out_xe + out_xw + out_yn + out_ys);
    return min(1.0, avail / (out_total + 1e-30));
}
