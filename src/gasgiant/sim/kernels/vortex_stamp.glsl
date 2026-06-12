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
uniform float u_rim_contrast;

const float VKIND_HERO = 1.0;
const float VKIND_BARGE = 2.0;
const float VKIND_POLAR = 5.0;
const float VKIND_DEBRIS = 7.0;

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
            if (b.y == VKIND_DEBRIS) {
                // Merger debris: a bright turbulent collar just outside the
                // product's own collar ring (q 1.2), cool gray-white like the
                // hero wake, slight height churn. Zero psi contribution; the
                // ambient flow folds it into filaments as it decays. NOTE the
                // shared dome/tint lines below are skipped — an ungated
                // debris dome would punch a broad T1 dome (and a thermal-
                // emission hole) over the merge site.
                float ring = exp(-(q - 1.5) * (q - 1.5) * 6.0);
                dT0 += b.w * ring;
                dT3 -= 0.5 * b.w * ring;
                dT1 += 0.04 * b.w * (core - 0.5 * ring);
                continue;
            }
            // Barges and polar cyclones are dips (cyclonic); the rest domes.
            // Polar dips matter doubly: the polar tint is gated by LOW cloud
            // tops, so cyclone interiors go structurally blue (PIA21641).
            // (Outbreaks keep the dome: their bright plume rides on a real
            // convective tower.)
            float dome = (b.y == VKIND_BARGE || b.y == VKIND_POLAR) ? -1.0 : 1.0;
            dT1 += dome * 0.15 * core;
            dT3 += b.z * core;
            if (b.y == VKIND_HERO) {
                // GRS anatomy: dark thin perimeter ring at the spot edge,
                // bright collar (the Red Spot Hollow) outside it.
                dT0 += b.w * core
                     - 0.16 * u_rim_contrast * exp(-(q - 1.0) * (q - 1.0) * 16.0)
                     + 0.22 * u_rim_contrast * exp(-(q - 1.55) * (q - 1.55) * 5.0);
            } else if (b.y == VKIND_POLAR) {
                // Dark eye, dark body, and a bright pearly wisp annulus —
                // neighboring annuli overlap into the chaotic bright
                // filigree between cyclones.
                dT0 += b.w * core
                     - 0.10 * exp(-q * q * 9.0)
                     + 0.14 * exp(-(q - 2.0) * (q - 2.0) * 2.2);
                dT1 -= 0.06 * exp(-q * q * 9.0);
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
