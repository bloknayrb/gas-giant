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
uniform float u_hero_mottle;       // interior brightness churn (T0); 0 disables
uniform float u_hero_tint_var;     // interior tint festoon (T3); 0 disables
// Seeded noise offset for the hero interior fbm. Declared here (not reusing the
// includer's u_detail_offset) because this file is #included BEFORE that uniform
// is declared in init.comp/advect.comp, so it is not yet in scope. Set from the
// same seeded self._detail_offset, so determinism is identical.
uniform vec3 u_hero_noise_offset;

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
        float asp = vortex_data[3 * i + 2].y;
        float q;
        if (asp == 1.0) {
            q = d / a.w;
        } else {
            vec3 c = a.xyz;
            vec3 ew = cross(vec3(0.0, 1.0, 0.0), c);
            float ewl = length(ew);
            if (ewl < 1e-4) {
                q = d / a.w;
            } else {
                vec3 e1 = ew / ewl;
                vec3 e2 = cross(c, e1);
                q = length(vec2(dot(p, e1) / asp, dot(p, e2))) / a.w;
            }
        }
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
                // Interior turbulent churn: a flow-scale fbm breaks up the
                // smooth Gaussian core so the spot reads as churning cloud, not
                // an airbrushed blob. Windowed to the interior (q<1) so it never
                // fights the perimeter ring/collar. Stamped into the relaxation
                // target => advect.comp folds it into filaments (motion-safe).
                // Guarded => byte-identical when off (fbm never evaluated).
                if (u_hero_mottle > 0.0) {
                    float win = core * (1.0 - smoothstep(0.6, 1.0, q));
                    float fscale = a.w > 0.0 ? 9.0 / a.w : 9.0;
                    // Fixed amplitude in the rim/collar league (~0.16/0.22), NOT
                    // scaled by the dim hero core brightness (b.w~0.05) which
                    // would render the churn invisible. The interior window keeps
                    // it off the perimeter ring.
                    dT0 += 0.15 * u_hero_mottle * win
                         * fbm(p * fscale + u_hero_noise_offset.yzx, 4, 2.0, 0.5);
                }
                // Interior color festoon: modulate the warm-red tint with a
                // decorrelated fbm so the spot carries salmon/white mottle
                // instead of flat red. Signed => spot mean tint ~preserved.
                if (u_hero_tint_var > 0.0) {
                    float winT = core * (1.0 - smoothstep(0.55, 1.0, q));
                    float fscaleT = a.w > 0.0 ? 7.0 / a.w : 7.0;
                    dT3 += b.z * u_hero_tint_var * winT
                         * fbm(p * fscaleT + u_hero_noise_offset.zxy + 13.0, 3, 2.0, 0.5);
                }
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
            if (along > 0.0 && along < rc * 6.0) {
                float ramp = smoothstep(rc * 0.5 * asp, rc * asp, along);
                float w = exp(-across * across) * (1.0 - along / (rc * 6.0)) * ramp;
                dT0 += 0.16 * w;   // bright churned clouds
                dT3 -= 0.20 * w;   // cool gray-white, not belt-colored
            }
        }
    }
    return vec3(dT0, dT1, dT3);
}
