// Shared helpers for the sim kernels, parameterized by the DOMAIN define:
//   DOMAIN 0  equirect main grid (texel centers, lat descending with +y,
//             lon periodic in x) — matches core/domain.py
//   DOMAIN 1  north polar patch, azimuthal-equidistant, square
//   DOMAIN 2  south polar patch
// Patch coords: st = (px-center) scaled to [-rho_max, rho_max] radians of
// colatitude; lon = atan(st.y, st.x); lat = POLE_SIGN * (pi/2 - |st|).
// AE patches have uniform radial scale and azimuthal scale rho/sin(rho).

#ifndef DOMAIN
#define DOMAIN 0
#endif

const float PI = 3.14159265358979;

#if DOMAIN != 0
uniform float u_rho_max;  // patch angular radius, radians of colatitude
#if DOMAIN == 1
const float POLE_SIGN = 1.0;
#else
const float POLE_SIGN = -1.0;
#endif
#endif

vec2 lonlatAtPos(vec2 pixPos, ivec2 size) {
#if DOMAIN == 0
    float lon = (pixPos.x / float(size.x)) * 2.0 * PI - PI;
    float lat = 0.5 * PI - (pixPos.y / float(size.y)) * PI;
    return vec2(lon, lat);
#else
    vec2 st = (pixPos / vec2(size) * 2.0 - 1.0) * u_rho_max;
    float rho = length(st);
    float lon = (rho < 1e-6) ? 0.0 : atan(st.y, st.x);
    float lat = POLE_SIGN * (0.5 * PI - rho);
    return vec2(lon, lat);
#endif
}

vec2 lonlatAt(ivec2 px, ivec2 size) {
    return lonlatAtPos(vec2(px) + 0.5, size);
}

vec3 spherePoint(vec2 ll) {
    float cl = cos(ll.y);
    return vec3(cl * cos(ll.x), sin(ll.y), cl * sin(ll.x));
}

// Profile LUT lookup coordinate for a latitude (profiles sampled from +pi/2
// at v=0 down to -pi/2 at v=1).
vec2 latProfileUV(float lat) {
    return vec2(clamp((0.5 * PI - lat) / PI, 0.0, 1.0), 0.5);
}

int wrapX(int x, int w) {
#if DOMAIN == 0
    return ((x % w) + w) % w;
#else
    return clamp(x, 0, w - 1);
#endif
}

int clampY(int y, int h) {
    return clamp(y, 0, h - 1);
}

#if DOMAIN != 0
// Patch-space velocity (d(st)/dt, radians) from physical (u east, v north):
//   east:  dtheta/dt = u / sin(rho); d(st) = rho * dtheta * e_theta
//   north: drho/dt = -POLE_SIGN * v;  d(st) = drho * e_rho
vec2 patchVelocity(vec2 st, vec2 vel_en) {
    float rho = length(st);
    if (rho < 1e-5) return vec2(0.0);
    vec2 er = st / rho;
    vec2 et = vec2(-er.y, er.x);
    float metric = rho / max(sin(rho), 1e-5);
    return vel_en.x * metric * et + (-POLE_SIGN * vel_en.y) * er;
}

vec2 patchPixFromSt(vec2 st, ivec2 size) {
    return (st / u_rho_max * 0.5 + 0.5) * vec2(size);
}

vec2 patchStFromPix(vec2 pixPos, ivec2 size) {
    return (pixPos / vec2(size) * 2.0 - 1.0) * u_rho_max;
}
#endif
