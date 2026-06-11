// Shared helpers for the sim kernels. Conventions match core/domain.py:
// texel centers, lat descending with +y, lon periodic in x.

const float PI = 3.14159265358979;

vec2 lonlatAt(ivec2 px, ivec2 size) {
    float lon = ((float(px.x) + 0.5) / float(size.x)) * 2.0 * PI - PI;
    float lat = 0.5 * PI - ((float(px.y) + 0.5) / float(size.y)) * PI;
    return vec2(lon, lat);
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
    return ((x % w) + w) % w;
}

int clampY(int y, int h) {
    return clamp(y, 0, h - 1);
}
