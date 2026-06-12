// Oklab (Ottosson) <-> sRGB for the CHROMA_FX block. No #version line:
// this file is inlined into derive.comp (inside #ifdef CHROMA_FX, so the
// default program variant strips it at the GLSL-preprocessor level).
//
// Constants must match gasgiant/palette/gradient.py — the GPU<->Python
// parity test pins them (the Python side inverts numerically; these are
// Ottosson's published inverse matrices, ~1e-7 apart). mat3 constructors
// are COLUMN-major: each triplet below is one column.
//
// Numerics, both NaN classes closed deliberately:
// - srgb_to_oklab clamps its input (upstream contrast/saturation can push
//   channels outside [0,1]; pow(negative) is undefined in GLSL).
// - oklab_to_srgb cubes LMS via x*x*x — NEVER pow — because scaled chroma
//   drives LMS negative in the normal operating regime, and takes
//   max(linear, 0) BEFORE the 1/2.4 transfer pow (mirrors
//   gradient._linear_to_srgb), then clamps the output (gamut clip).

vec3 oklab_srgb_to_lin(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}

vec3 oklab_lin_to_srgb(vec3 c) {
    c = max(c, 0.0);
    return clamp(
        mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c)),
        0.0, 1.0);
}

vec3 srgb_to_oklab(vec3 c) {
    const mat3 M1 = mat3(
        0.4122214708, 0.2119034982, 0.0883024619,
        0.5363325363, 0.6806995451, 0.2817188376,
        0.0514459929, 0.1073969566, 0.6299787005);
    const mat3 M2 = mat3(
        0.2104542553, 1.9779984951, 0.0259040371,
        0.7936177850, -2.4285922050, 0.7827717662,
        -0.0040720468, 0.4505937099, -0.8086757660);
    vec3 lms = M1 * oklab_srgb_to_lin(c);   // >= 0: clamped input, positive M1
    return M2 * pow(lms, vec3(1.0 / 3.0));
}

vec3 oklab_to_srgb(vec3 lab) {
    const mat3 M2i = mat3(
        1.0, 1.0, 1.0,
        0.3963377774, -0.1055613458, -0.0894841775,
        0.2158037573, -0.0638541728, -1.2914855480);
    const mat3 M1i = mat3(
        4.0767416621, -1.2684380046, -0.0041960863,
        -3.3077115913, 2.6097574011, -0.7034186147,
        0.2309699292, -0.3413193965, 1.7076147010);
    vec3 lms = M2i * lab;
    lms = lms * lms * lms;
    return oklab_lin_to_srgb(M1i * lms);
}
