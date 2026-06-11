// Vortex tracer stamps, shared by init.comp and the MacCormack correct pass:
// storms are not just velocity features — anticyclones continually regenerate
// bright cloud tops (and the GRS its red core), so the same stamp pattern that
// initializes the tracers is also the relaxation target, evaluated at the
// vortices' CURRENT (drifted) positions.
//
// Requires the Vortices SSBO (binding 2) and u_vortex_count to be declared by
// the includer? No — declared here; include once per kernel.

layout(std430, binding = 2) readonly buffer Vortices {
    vec4 vortex_data[];  // pairs: [x,y,z,r_core], [strength, kind, tint, brightness]
};
uniform int u_vortex_count;

const float VKIND_BARGE = 2.0;

// Accumulates the vortex stamp deltas at sphere point p:
//   dT0 brightness, dT1 dome/depression, dT3 tint.
vec3 vortexStamp(vec3 p) {
    float dT0 = 0.0;
    float dT1 = 0.0;
    float dT3 = 0.0;
    for (int i = 0; i < u_vortex_count; ++i) {
        vec4 a = vortex_data[2 * i];
        vec4 b = vortex_data[2 * i + 1];
        float d = acos(clamp(dot(p, a.xyz), -1.0, 1.0));
        float q = d / a.w;
        if (q < 3.0) {
            float core = exp(-q * q);
            float ring = exp(-(q - 1.2) * (q - 1.2) * 4.0);  // collar annulus
            // Anticyclones (everything but barges) are domes; cyclones dips.
            float dome = (b.y == VKIND_BARGE) ? -1.0 : 1.0;
            dT0 += b.w * core - 0.3 * abs(b.w) * ring;
            dT1 += dome * 0.15 * core;
            dT3 += b.z * core;
        }
    }
    return vec3(dT0, dT1, dT3);
}
