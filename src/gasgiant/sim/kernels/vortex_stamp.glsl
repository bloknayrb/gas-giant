// Vortex tracer stamps, shared by init.comp and the MacCormack correct pass:
// storms are not just velocity features — anticyclones continually regenerate
// bright cloud tops (and the GRS its red core), so the same stamp pattern that
// initializes the tracers is also the relaxation target, evaluated at the
// vortices' CURRENT (drifted) positions.

layout(std430, binding = 2) readonly buffer Vortices {
    // Triples: [x,y,z,r_core], [strength,kind,tint,brightness], [wake_dir,-,-,-]
    vec4 vortex_data[];
};
uniform int u_vortex_count;

const float VKIND_HERO = 1.0;
const float VKIND_BARGE = 2.0;

// Accumulates the vortex stamp deltas at sphere point p:
//   dT0 brightness, dT1 dome/depression, dT3 tint.
vec3 vortexStamp(vec3 p) {
    float dT0 = 0.0;
    float dT1 = 0.0;
    float dT3 = 0.0;
    for (int i = 0; i < u_vortex_count; ++i) {
        vec4 a = vortex_data[3 * i];
        vec4 b = vortex_data[3 * i + 1];
        float d = acos(clamp(dot(p, a.xyz), -1.0, 1.0));
        float q = d / a.w;
        if (q < 3.0) {
            float core = exp(-q * q);
            // Anticyclones (everything but barges) are domes; cyclones dips.
            float dome = (b.y == VKIND_BARGE) ? -1.0 : 1.0;
            dT1 += dome * 0.15 * core;
            dT3 += b.z * core;
            if (b.y == VKIND_HERO) {
                // GRS anatomy: dark thin perimeter ring at the spot edge,
                // bright collar (the Red Spot Hollow) outside it.
                dT0 += b.w * core
                     - 0.16 * exp(-(q - 1.0) * (q - 1.0) * 16.0)
                     + 0.22 * exp(-(q - 1.55) * (q - 1.55) * 5.0);
            } else {
                float ring = exp(-(q - 1.2) * (q - 1.2) * 4.0);  // collar annulus
                dT0 += b.w * core - 0.3 * abs(b.w) * ring;
            }
        }
        // Hero wake: the turbulent folded-filament field downstream reads
        // BRIGHT gray-white in true color — a tracer signature, matching the
        // velocity wedge in psi.comp.
        if (b.y == VKIND_HERO) {
            float down = vortex_data[3 * i + 2].x;
            float rc = a.w;
            float vlat = asin(clamp(a.y, -1.0, 1.0));
            float vlon = atan(a.z, a.x);
            float plat = asin(clamp(p.y, -1.0, 1.0));
            float plon = atan(p.z, p.x);
            float dlon = mod(plon - vlon + 3.0 * PI, 2.0 * PI) - PI;
            float along = dlon * down;
            float across = (plat - vlat) / max(rc * 1.6, 1e-4);
            if (along > rc * 0.5 && along < rc * 6.0) {
                float w = exp(-across * across) * (1.0 - along / (rc * 6.0));
                dT0 += 0.16 * w;   // bright churned clouds
                dT3 -= 0.20 * w;   // cool gray-white, not belt-colored
            }
        }
    }
    return vec3(dT0, dT1, dT3);
}
