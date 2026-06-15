// sw_common.glsl — Metric helpers for the M1 single-layer GPU shallow-water solver.
// Include after #version in each sw_*.comp kernel.
// Graduated from swp_common.glsl (M0.5).  DO NOT change wrapX to the modulo
// form: ((x%w)+w)%w is incorrect on some drivers for non-power-of-2 w.

#define PI 3.14159265358979

// cos(phi) at cell-centre row `row` (0 = northernmost).
// phi_c[row] = 0.5*PI - (row + 0.5)*PI/H  (descending latitude).
float cosCenter(int row, int H) {
    return cos(0.5 * PI - (float(row) + 0.5) * PI / float(H));
}

// cos(phi) at v-face row `row` (0 = north pole, H = south pole).
// phi_v[row] = 0.5*PI - row*PI/H; zeroed at both poles.
float cosVface(int row, int H) {
    if (row <= 0 || row >= H) return 0.0;
    return cos(0.5 * PI - float(row) * PI / float(H));
}

// Periodic wrap in X (longitude).
// NOTE: The two-step ((x%w)+w)%w form is avoided because some drivers treat
// % as unsigned for signed int arguments on non-power-of-2 w, producing wrong
// results (e.g. wrapX(-1,96)→63 instead of 95).  Branch-based form is safe on
// all conformant implementations and the branch is trivially branch-predicted.
int wrapX(int x, int w) {
    if (x < 0)  return x + w;
    if (x >= w) return x - w;
    return x;
}

// Clamp wrap in Y (latitude) — no meridional periodicity in single-layer SW.
// Returns the clamped row index (boundary rows repeat at poles).
int wrapY(int y, int h) {
    if (y < 0)  return 0;
    if (y >= h) return h - 1;
    return y;
}
