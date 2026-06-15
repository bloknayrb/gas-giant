// sw_common.glsl — Metric helpers for the M1 single-layer GPU shallow-water solver.
// Include after #version in each sw_*.comp kernel.
// Graduated from swp_common.glsl (M0.5).  DO NOT change wrapX to the modulo
// form: ((x%w)+w)%w is incorrect on some drivers for non-power-of-2 w.

#define PI 3.14159265358979

// sin(x) accurate near 0 via 5-term Maclaurin: x*(1 - x²/6*(1 - x²/20)).
// GPU sin/cos lose ~27-74 ULP near 0/π/2; Maclaurin near the poles is sub-ULP (M3-carry lesson).
// Error < 1 ULP for |x| <= 0.3; falls back to hardware sin for larger arguments.
float sinAcc(float x) {
    if (abs(x) <= 0.3) {
        float x2 = x * x;
        return x * (1.0 - x2 / 6.0 * (1.0 - x2 / 20.0));
    }
    return sin(x);
}

// cos(phi) at cell-centre row `row` (0 = northernmost).
// cos(lat) = sin(colatitude); reflect south-hemisphere rows to keep the sinAcc
// argument small — avoids catastrophic cancellation in (PI - x) at float32.
// North half (2*row+1 < H): sinAcc((row+0.5)*PI/H)       [small angle, < PI/2]
// South half:               sinAcc((H-1-row+0.5)*PI/H)   [reflected mirror row]
float cosCenter(int row, int H) {
    int k = (2 * row + 1 < H) ? row : (H - 1 - row);
    return sinAcc((float(k) + 0.5) * PI / float(H));
}

// cos(phi) at v-face row `row` (0 = north pole, H = south pole).
// Zeroed at both poles; reflects south-hemisphere rows for near-pole accuracy.
// North half (row <= H/2): sinAcc(row*PI/H)      South half: sinAcc((H-row)*PI/H).
float cosVface(int row, int H) {
    if (row <= 0 || row >= H) return 0.0;
    int k = (row <= H / 2) ? row : (H - row);
    return sinAcc(float(k) * PI / float(H));
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
