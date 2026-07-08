// Vortex tracer stamps, shared by init.comp and the MacCormack correct pass:
// storms are not just velocity features — anticyclones continually regenerate
// bright cloud tops (and the GRS its red core), so the same stamp pattern that
// initializes the tracers is also the relaxation target, evaluated at the
// vortices' CURRENT (drifted) positions.

layout(std430, binding = 2) readonly buffer Vortices {
    // Triples: [x,y,z,r_core], [strength,kind,tint,brightness],
    //          [wake_dir, aspect, wake_lat_off, -]
    vec4 vortex_data[];
};
uniform int u_vortex_count;
uniform float u_rim_contrast;
uniform float u_hero_mottle;       // interior brightness churn (T0); 0 disables
uniform float u_hero_tint_var;     // interior tint festoon (T3); 0 disables
uniform float u_hero_rim_warp;     // lumpy-oval boundary warp; 0 disables (byte-identical)
uniform float u_hero_rim_tint;     // dark reddish collar rim; 0 disables (byte-identical)
uniform float u_hero_wake_detail;  // wake filament structure; 0 disables (byte-identical)
// Seeded noise offset for the hero interior fbm. Declared here (not reusing the
// includer's u_detail_offset) because this file is #included BEFORE that uniform
// is declared in init.comp/advect.comp, so it is not yet in scope. Set from the
// same seeded self._detail_offset, so determinism is identical.
uniform vec3 u_hero_noise_offset;

const float VKIND_HERO = 1.0;
const float VKIND_BARGE = 2.0;
const float VKIND_POLAR = 5.0;
const float VKIND_OUTBREAK = 6.0;
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
                // The tangent-plane projection collapses to q~0 at the antipode
                // (p perpendicular to e1,e2) -> the vortex would re-stamp a
                // phantom on the far side. Gate to the near hemisphere; far
                // points get a large q so nothing stamps. Byte-identical for the
                // near field (the stamp is already ~0 by 90 deg at these radii).
                q = (dot(p, c) > 0.0)
                  ? length(vec2(dot(p, e1) / asp, dot(p, e2))) / a.w
                  : 1e3;
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
            if (b.y == VKIND_OUTBREAK) {
                // Convective white-plume train (Great-White-Spot / SEB-revival):
                // a brilliant ammonia-white turbulent patch, NOT a coherent
                // vortex. Like debris (zero psi -> the belt shear folds the train
                // into a planet-girdling streak) but brighter and a tighter ring.
                // Crucially NO DOME: the original outbreak fell through to the
                // anticyclone dome+collar and read as a 2nd GRS; the plume is a
                // high BRIGHT cloud that reads as churn, not a rotating storm. The
                // shared dome/collar lines below are skipped.
                float ring = exp(-(q - 1.0) * (q - 1.0) * 9.0);
                dT0 += b.w * (1.0 * core + ring);    // bright solid plume + halo (high value)
                dT3 -= 0.07 * b.w * (core + ring);   // barely cool -- ammonia WHITE, not blue
                                                     // (was 0.15; cut so the brighter lead
                                                     // knot reads neutral, not blue, since
                                                     // this push scales with brightness b.w)
                dT1 += 0.05 * b.w * core;            // a little high cloud, not a dome
                continue;
            }
            // Barges and polar cyclones are dips (cyclonic); the rest domes.
            // Polar dips matter doubly: the polar tint is gated by LOW cloud
            // tops, so cyclone interiors go structurally blue (PIA21641).
            float dome = (b.y == VKIND_BARGE || b.y == VKIND_POLAR) ? -1.0 : 1.0;
            dT1 += dome * 0.15 * core;
            dT3 += b.z * core;
            if (b.y == VKIND_HERO) {
                // GRS anatomy: dark thin perimeter ring at the spot edge,
                // bright collar (the Red Spot Hollow) outside it.
                // Lumpy-oval boundary: warp the q feeding the ring/collar with a
                // LOW-AZIMUTHAL-WAVENUMBER (few-lobe) perturbation so the edge is
                // an irregular oval, not a flawless ring. Lobes are functions of
                // the azimuth theta only => scale-invariant (no fixed pixel
                // frequency that over-scallops in close-up). Rim and collar use
                // decorrelated phases so they do not wobble in lockstep. Guarded
                // => the trig is never evaluated when off (byte-identical).
                float qrim = q;
                float qcol = q;
                // Hero-frame azimuth, shared by the rim_warp lobes (lumpy edge) and the
                // rim_tint collar break. Same e1/e2 east-west frame the aspect path uses.
                // Computed ONCE when either lever is on (both feed off the identical
                // frame, so there is no point building it twice); hth_ok guards the
                // degenerate pole-aligned case. Guarded => byte-identical when both off.
                float hth = 0.0;
                bool hth_ok = false;
                if (u_hero_rim_warp > 0.0 || u_hero_rim_tint > 0.0) {
                    vec3 hc = a.xyz;
                    vec3 hew = cross(vec3(0.0, 1.0, 0.0), hc);
                    float hewl = length(hew);
                    if (hewl > 1e-4) {
                        vec3 h1 = hew / hewl;
                        vec3 h2 = cross(hc, h1);
                        hth = atan(dot(p, h2), dot(p, h1));
                        hth_ok = true;
                    }
                }
                if (u_hero_rim_warp > 0.0 && hth_ok) {
                    // Seeded phases from the hero noise offset (deterministic).
                    vec3 ph = u_hero_noise_offset * 6.2831853;
                    // Incommensurate wavenumbers {2,3,5} => quasi-irregular, few lobes
                    // around the oval. Normalised to ~[-1,1]. Decorrelated rim/collar
                    // phases so they do not wobble in lockstep.
                    float wr = ( 0.55 * sin(2.0 * hth + ph.x)
                               + 0.30 * sin(3.0 * hth + ph.y)
                               + 0.20 * sin(5.0 * hth + ph.z));
                    float wc = ( 0.55 * sin(2.0 * hth + ph.y + 1.7)
                               + 0.30 * sin(3.0 * hth + ph.z + 0.6)
                               + 0.20 * sin(5.0 * hth + ph.x + 2.9));
                    qrim += u_hero_rim_warp * 0.20 * wr;
                    qcol += u_hero_rim_warp * 0.20 * wc;
                }
                dT0 += b.w * core
                     - 0.16 * u_rim_contrast * exp(-(qrim - 1.0) * (qrim - 1.0) * 16.0)
                     + 0.22 * u_rim_contrast * exp(-(qcol - 1.55) * (qcol - 1.55) * 5.0);
                // Dark reddish collar (the Red Spot Hollow rim): redden (T3 up,
                // toward the salmon storm-tint LUT) and darken (T0 down) the
                // perimeter annulus so the spot has a discrete dark-red rim.
                // Rides on the warped qrim so the tint follows the lumpy edge.
                // Guarded => byte-identical when off.
                if (u_hero_rim_tint > 0.0) {
                    float rring = exp(-(qrim - 1.08) * (qrim - 1.08) * 11.0);
                    // Azimuthal break-up: the real Red Spot Hollow is a soft, broken,
                    // asymmetric moat -- dark on some arcs, faint on others -- not a
                    // uniform ring. Modulate the DARKENING (not the reddening) with a
                    // few-lobe seeded function of the shared hero-frame azimuth so the
                    // collar reads as a moat, not a drawn-on outline. Rides the warped
                    // qrim, so radius AND darkness are both irregular (no new ring).
                    float azw = 1.0;
                    if (hth_ok) {
                        vec3 tph = u_hero_noise_offset * 6.2831853;
                        float lobe = ( 0.6 * sin(hth + tph.x)
                                     + 0.3 * sin(2.0 * hth + tph.y)
                                     + 0.2 * sin(3.0 * hth + tph.z));
                        azw = clamp(0.35 + 0.65 * (0.5 + 0.5 * lobe), 0.35, 1.0);
                    }
                    dT3 += u_hero_rim_tint * 0.55 * rring;            // redden -- unchanged
                    dT0 -= u_hero_rim_tint * 0.16 * rring * azw;      // deeper (was 0.12) + broken
                }
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
            } else if (asp > 1.0) {
                // Elongated bright stamp = wispy cirrus streak (Neptune bright-cloud
                // class): a SOFT, longer-tailed feathered glow with NO dark collar
                // ring (a collar reads as a hard stamped rim; a pure Gaussian reads
                // as an opaque puff), BROKEN into multiple thin flow-parallel
                // filaments -- real Neptune cirrus is "combed fibers", not one lobe.
                // An anisotropic fbm does the combing: high frequency ACROSS the
                // streak (fine strands), low frequency ALONG it (long tails). The
                // fibers ride the streak's own east-west frame. asp==1.0 never
                // enters here => byte-identical when off.
                // Soft feathered glow, plus a mild flow-frame noise modulation so the
                // streak is not a perfectly uniform lobe. (A crisp multi-strand "combed
                // fiber" texture cannot be stamped here -- the sim advects and diffuses
                // fine tracer detail into a wash over the dev run; true fibrous cirrus
                // needs a render-time synthesis pass on a dedicated cloud mask. See
                // docs/roadmap.md.) asp==1.0 never enters here => byte-identical when off.
                float glow = 0.6 * core + 0.4 * exp(-q * 1.3);
                vec3 cc = a.xyz;
                vec3 few = cross(vec3(0.0, 1.0, 0.0), cc);
                float fewl = length(few);
                if (fewl > 1e-4) {
                    vec3 f1 = few / fewl;            // east-west == along the streak
                    vec3 f2 = cross(cc, f1);         // cross-streak (latitude)
                    float strand = fbm(vec3(dot(p, f1) * 6.0, dot(p, f2) * 44.0, 3.1)
                                       + vec3(11.3, 4.7, 8.1), 3, 2.0, 0.5);
                    glow *= clamp(0.4 + 0.95 * strand, 0.0, 1.35);
                }
                dT0 += b.w * glow;
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
            float woff = vortex_data[3 * i + 2].z;  // equatorward wake bias (F06)
            float rc = a.w;
            float vlat = asin(clamp(a.y, -1.0, 1.0));
            float vlon = atan(a.z, a.x);
            float plat = asin(clamp(p.y, -1.0, 1.0));
            float plon = atan(p.z, p.x);
            float dlon = mod(plon - vlon + 3.0 * PI, 2.0 * PI) - PI;
            float along = dlon * down;
            float across = (plat - (vlat + woff)) / max(rc * 1.6, 1e-4);
            // Wake filaments: fray the smooth wedge into ragged folded streaks so it
            // reads as turbulence, not a blob. The wake is the relaxation TARGET, so a
            // structured target makes a structured wake (advect.comp folds it; the
            // velocity wake in psi.comp supplies the along-flow folding). All
            // rc-normalized => scale-invariant. Guarded => byte-identical when off.
            float wseed = u_hero_noise_offset.x * 6.3;
            float an = along / max(rc, 1e-4);              // 0..6 downstream, rc-invariant
            if (u_hero_wake_detail > 0.0) {
                // (1) Ragged envelope: low-freq wobble of the wedge centreline/width,
                // applied BEFORE w so the silhouette both widens and narrows.
                across += u_hero_wake_detail * 0.30
                        * fbm(vec3(an * 0.5, 0.0, wseed + 11.0), 2, 2.0, 0.5);
            }
            // Same latitude window as psi.comp's wedge: keep the stamp
            // strictly local (the raw Gaussian tail never truly reaches zero).
            if (along > 0.0 && along < rc * 6.0 && abs(across) < 2.5) {
                float ramp = smoothstep(rc * 0.5 * asp, rc * asp, along);
                float win = 1.0 - smoothstep(2.0, 2.5, abs(across));
                float w = exp(-across * across) * win * (1.0 - along / (rc * 6.0)) * ramp;
                if (u_hero_wake_detail > 0.0) {
                    // (2) Intermittent flow-aligned filaments: anisotropic fbm (low
                    // along-freq, higher across-freq => downstream streaks), sheared so
                    // they fan along the curving flow, thresholded so there are clear
                    // lanes between filaments (not uniform mottle). Clamped so the
                    // factor stays in [0,1] (snoise can exceed +/-1 -> would sign-flip w).
                    float sh  = across + 0.25 * an;
                    float fil = fbm(vec3(an * 0.30, sh * 1.7, wseed), 4, 2.0, 0.5);
                    float streak = clamp(smoothstep(-0.2, 0.6, fil), 0.0, 1.0);
                    w *= mix(1.0, streak, u_hero_wake_detail);
                }
                dT0 += 0.16 * w;   // bright churned clouds
                dT3 -= 0.20 * w;   // cool gray-white, not belt-colored
            }
        }
    }
    return vec3(dT0, dT1, dT3);
}
