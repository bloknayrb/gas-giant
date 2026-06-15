// sw_common.glsl — Metric helpers for the M1 single-layer GPU shallow-water solver.
// Include after #version in each sw_*.comp kernel.
// Graduated from swp_common.glsl (M0.5).  DO NOT change wrapX to the modulo
// form: ((x%w)+w)%w is incorrect on some drivers for non-power-of-2 w.

#define PI 3.14159265358979

// cos(phi) at cell-centre row `row` (0 = northernmost).
// Uses sin((row+0.5)*PI/H) == cos(PI/2 - ...) because GPU cos near PI/2 loses
// ~74 ULPs near the poles; sin near 0 is accurate (M3-carry lesson, 2026-06).
float cosCenter(int row, int H) {
    return sin((float(row) + 0.5) * PI / float(H));
}

// cos(phi) at v-face row `row` (0 = north pole, H = south pole).
// Same sin-based form for pole accuracy; zeroed at both poles.
float cosVface(int row, int H) {
    if (row <= 0 || row >= H) return 0.0;
    return sin(float(row) * PI / float(H));
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
