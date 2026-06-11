// 3D Worley (cellular) noise, F1 distance — convective closed-cell texture.
// Hash from a common integer-mix construction; 3x3x3 neighborhood search.

vec3 worleyHash3(vec3 p) {
    p = vec3(
        dot(p, vec3(127.1, 311.7, 74.7)),
        dot(p, vec3(269.5, 183.3, 246.1)),
        dot(p, vec3(113.5, 271.9, 124.6))
    );
    return fract(sin(p) * 43758.5453123);
}

// Returns F1 in ~[0, 1] (distance to nearest feature point).
float worleyF1(vec3 p) {
    vec3 ip = floor(p);
    vec3 fp = fract(p);
    float f1 = 1e9;
    for (int k = -1; k <= 1; ++k)
    for (int j = -1; j <= 1; ++j)
    for (int i = -1; i <= 1; ++i) {
        vec3 g = vec3(float(i), float(j), float(k));
        vec3 o = worleyHash3(ip + g);
        vec3 r = g + o - fp;
        f1 = min(f1, dot(r, r));
    }
    return sqrt(f1);
}
