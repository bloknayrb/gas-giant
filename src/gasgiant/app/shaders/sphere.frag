#version 430
// Ray-traced sphere preview: equirect UV computed per pixel from the surface
// direction vector (no mesh, no UV seam, no pole pinch). The continuous QA
// view for seam/pole artifacts.

#include "agx.glsl"

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_color;
uniform float u_yaw;        // orbit camera angles, radians
uniform float u_pitch;
uniform float u_zoom;       // 1 = default framing
uniform vec3 u_light_dir;   // normalized, world space
uniform int u_mode;         // view transform mode

const float PI = 3.14159265358979;

void main() {
    // Orbit camera at distance d looking at the origin.
    float d = 3.0 / max(u_zoom, 0.05);
    float cp = cos(u_pitch), sp = sin(u_pitch);
    float cy = cos(u_yaw), sy = sin(u_yaw);
    vec3 eye = d * vec3(cp * cy, sp, cp * sy);
    vec3 fwd = normalize(-eye);
    vec3 right = normalize(cross(fwd, vec3(0.0, 1.0, 0.0)));
    vec3 up = cross(right, fwd);

    vec2 ndc = v_uv * 2.0 - 1.0;
    float focal = 2.4;
    vec3 ray = normalize(fwd * focal + right * ndc.x + up * ndc.y);

    // Ray-sphere intersection, unit sphere at origin.
    float b = dot(eye, ray);
    float c = dot(eye, eye) - 1.0;
    float disc = b * b - c;
    if (disc < 0.0) {
        frag_color = vec4(0.05, 0.05, 0.07, 1.0);
        return;
    }
    float t = -b - sqrt(disc);
    vec3 p = eye + t * ray;      // = surface normal on a unit sphere
    vec3 n = p;

    float lon = atan(p.z, p.x);
    float lat = asin(clamp(p.y, -1.0, 1.0));
    vec2 uv = vec2(lon / (2.0 * PI) + 0.5, 0.5 - lat / PI);
    // textureLod: derivative-based mip selection would show a seam line at
    // the atan2 wrap; the preview texture has no mips anyway.
    vec3 albedo = textureLod(u_color, uv, 0.0).rgb;

    float diffuse = max(dot(n, u_light_dir), 0.0);
    float wrap = max((dot(n, u_light_dir) + 0.3) / 1.3, 0.0);  // soft terminator
    vec3 lit = albedo * (0.04 + 0.96 * mix(diffuse, wrap, 0.35));

    // Cheap limb darkening: grazing view angles dim slightly.
    float facing = max(dot(n, -ray), 0.0);
    lit *= mix(0.55, 1.0, pow(facing, 0.35));

    frag_color = vec4(viewTransform(clamp(lit, 0.0, 1.0), u_mode), 1.0);
}
