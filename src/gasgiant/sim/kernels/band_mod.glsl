// Band-stamp modifiers shared by init.comp and the relaxation target in
// advect.comp (both must shape the SAME target or relaxation slowly erases
// the init-time look):
//   - within-band longitudinal color drift (real belts hold several hues
//     at once, varying slowly with longitude)
//   - the faded sector (SEB fade): a longitude window of one belt relaxes
//     toward a pale tone
//   - the latitude-contrast envelope: banding collapses toward the mid tone
//     poleward of ~45 deg (the real contrast profile; mottle replaces bands)
// Applies to the band stamp ONLY — vortex/wave stamps are added afterwards,
// so storms keep full contrast inside the collapsed-banding region (as on
// the real planets).
//
// Requires noise3d.glsl (fbm) and common.glsl (spherePoint) to be included
// first. All uniforms default to 0 == feature off == v1-identical stamps.

uniform float u_band_variance;   // T0 amplitude of the longitudinal drift
uniform vec3 u_variance_offset;  // seeded noise offset
uniform float u_env_strength;    // 0..1 contrast collapse at the envelope end
uniform float u_fade_amp;        // 0..1 faded-sector strength
uniform vec4 u_fade_sector;      // lat_lo, lat_hi, center lon, halfwidth (rad)

const float ENV_START = 0.7854;  // 45 deg
const float ENV_END = 1.2566;    // 72 deg
const float T0_MID = 0.54;       // 0.5 * (ZONE_VALUE + BELT_VALUE)
const float T1_MID = 0.55;

void bandStampMod(inout float t0, inout float t1, vec3 p, vec2 ll) {
    if (u_band_variance > 0.0) {
        // Anisotropic: slow along longitude, band-confined across latitude
        // (spherePoint puts latitude on y).
        float drift = fbm(p * vec3(0.9, 4.0, 0.9) + u_variance_offset, 3, 2.0, 0.5);
        t0 += u_band_variance * drift;
    }
    if (u_fade_amp > 0.0 && ll.y > u_fade_sector.x && ll.y < u_fade_sector.y) {
        float dlon = abs(atan(sin(ll.x - u_fade_sector.z), cos(ll.x - u_fade_sector.z)));
        float w = 1.0 - smoothstep(0.55 * u_fade_sector.w, u_fade_sector.w, dlon);
        t0 = mix(t0, T0_MID + 0.10, u_fade_amp * w);  // pale gray-tan, belt vanishes
    }
    if (u_env_strength > 0.0) {
        float env = 1.0 - u_env_strength * smoothstep(ENV_START, ENV_END, abs(ll.y));
        t0 = T0_MID + (t0 - T0_MID) * env;
        t1 = T1_MID + (t1 - T1_MID) * env;
    }
}
