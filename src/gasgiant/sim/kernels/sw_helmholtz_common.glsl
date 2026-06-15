// sw_helmholtz_common.glsl — shared L_sym action + analytic diagonal for the
// M2 GPU Helmholtz kernels.  Include AFTER sw_common.glsl.  Requires the
// includer to declare:
//   uniform sampler2D u_dh;     // current dh, center (W,H)
//   uniform sampler2D u_Href;   // reference depth, (1,H)
//   uniform float u_alpha, u_a, u_dlam, u_dphi;
//
// Ports helmholtz_apply / grad_faces / divergence_helmholtz / _helmholtz_diagonal
// from shallow_water_ref.py EXACTLY.

// H_ref_lat[row], clamped at the latitude boundary (used only via cos_v=0 poles).
float hh_href(int row, int H) {
    int r = (row < 0) ? 0 : ((row >= H) ? H - 1 : row);
    return texelFetch(u_Href, ivec2(0, r), 0).r;
}

// gx at east u-face of cell (row,col): (dh[row,col+1]-dh[row,col])/(a cos_c dlam).
float hh_gx_face(int col, int row, int W, int H) {
    int ie = wrapX(col + 1, W);   // branch-form wrapX -- NEVER ((x%w)+w)%w
    float dh_c = texelFetch(u_dh, ivec2(col, row), 0).r;
    float dh_e = texelFetch(u_dh, ivec2(ie, row), 0).r;
    float cc = cosCenter(row, H);
    return (dh_e - dh_c) / (u_a * cc * u_dlam);
}

// gy at v-face row k (1..H-1): (dh[k-1,col]-dh[k,col])/(a dphi).  Poles -> 0.
float hh_gy_face(int col, int k, int W, int H) {
    if (k <= 0 || k >= H) return 0.0;
    float dh_n = texelFetch(u_dh, ivec2(col, k - 1), 0).r;
    float dh_s = texelFetch(u_dh, ivec2(col, k),     0).r;
    return (dh_n - dh_s) / (u_a * u_dphi);
}

// L_sym(dh) at cell (i,j): dh - alpha * divergence_helmholtz(grad_faces(dh)).
float hh_apply(int i, int j, int W, int H) {
    float dh_c = texelFetch(u_dh, ivec2(i, j), 0).r;

    // Zonal divergence term.
    float Hx = hh_href(j, H);
    float gxE = hh_gx_face(i, j, W, H);
    int iw = wrapX(i - 1, W);
    float gxW = hh_gx_face(iw, j, W, H);
    float dFx = (Hx * gxE - Hx * gxW) / (u_a * u_dlam);

    // Meridional divergence term.
    float HvN = 0.5 * (hh_href(j - 1, H) + hh_href(j, H));
    float HvS = 0.5 * (hh_href(j, H)     + hh_href(j + 1, H));
    float cvN = cosVface(j, H);
    float cvS = cosVface(j + 1, H);
    float gyN = hh_gy_face(i, j,     W, H);
    float gyS = hh_gy_face(i, j + 1, W, H);
    float dFy = (HvS * cvS * gyS - HvN * cvN * gyN) / (u_a * u_dphi);

    float cc = cosCenter(j, H);
    float div_H = (dFx - dFy) / cc;
    return dh_c - u_alpha * div_H;
}

// Analytic diagonal D[j] = coefficient of dh[j,i] in hh_apply (ports _helmholtz_diagonal).
//   D = 1 - alpha*(cz + cm)
//   cz = Hx / (a dlam cos_c) * (-2 / (a cos_c dlam))
//   cm = -[ Hv[j+1]*cos_v[j+1]*(1/(a dphi)) - Hv[j]*cos_v[j]*(-1/(a dphi)) ] / (a dphi) / cos_c
float hh_diagonal(int j, int H) {
    float cc = cosCenter(j, H);
    float Hx = hh_href(j, H);
    float cz = Hx / (u_a * u_dlam * cc) * (-2.0 / (u_a * cc * u_dlam));

    float HvJ  = 0.5 * (hh_href(j - 1, H) + hh_href(j, H));     // Hv[j]
    float HvJ1 = 0.5 * (hh_href(j, H)     + hh_href(j + 1, H)); // Hv[j+1]
    float cvJ  = cosVface(j, H);
    float cvJ1 = cosVface(j + 1, H);
    float coeff_gyjp1 = HvJ1 * cvJ1 * (1.0 / (u_a * u_dphi));
    float coeff_gyj   = HvJ  * cvJ  * (-1.0 / (u_a * u_dphi));
    float dFy_coeff = (coeff_gyjp1 - coeff_gyj) / (u_a * u_dphi);
    float cm = -dFy_coeff / cc;

    return 1.0 - u_alpha * (cz + cm);
}
