// swp_common.glsl — Metric helpers for the M0.5 GPU shallow-water probe.
// Include after #version in each swp_*.comp kernel.

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
int wrapX(int x, int w) {
    return ((x % w) + w) % w;
}
