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
// heroEllipQ for the variant-only heroRelaxWeight below; the include's whole
// body is #ifdef HERO_EMERGENCE so the default program is unchanged.
#include "hero_q.glsl"
uniform float u_rim_contrast;
uniform float u_hero_mottle;       // interior brightness churn (T0); 0 disables
uniform float u_hero_tint_var;     // interior tint festoon (T3); 0 disables
uniform float u_hero_rim_warp;     // lumpy-oval boundary warp; 0 disables (byte-identical)
uniform float u_hero_rim_tint;     // dark reddish collar rim; 0 disables (byte-identical)
uniform float u_hero_wake_detail;  // wake filament structure; 0 disables (byte-identical)
#ifdef HERO_EMERGENCE
// GRS-realism pack strength. The whole feature compiles as a preprocessor
// VARIANT (HERO_EMERGENCE define, selected when storms.hero_emergence > 0), so
// the default program text is the pre-feature kernel — byte-identical by
// construction, per the project rule (gated out, not branch-guarded).
uniform float u_hero_emergence;
#endif
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
#ifdef HERO_EMERGENCE
                if (u_hero_rim_warp > 0.0 || u_hero_rim_tint > 0.0
                    || u_hero_emergence > 0.0) {
#else
                if (u_hero_rim_warp > 0.0 || u_hero_rim_tint > 0.0) {
#endif
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
                // Hero fill profile (emergence): the real GRS is a FILLED oval —
                // its red is a near-flat plateau across the whole spot (PIA21775),
                // not a Gaussian stain fading from the center, and the red reaches
                // nearly to the bright hollow margin (thin pale collar, NOT a small
                // egg in a big bright basin). The plateau EDGE sits at q~1.0 — the
                // authored hero_radius — with the thin ring/collar hugging just
                // outside (an earlier cut inflated the edge to q~1.55 with the
                // collar at 2.0, which made the storm and its whole influence
                // zone read much bigger than its authored size). Under emergence:
                //  - the tint/brightness fill morphs core -> a WIDE plateau riding
                //    the warped qrim (lumpy oval), with an fbm-frayed edge so
                //    filaments peel instead of a sticker cutoff;
                //  - radial identity: inner half slightly darker (EBSCO: "in the
                //    inner half of the Spot, reflectivity was lower"), outer
                //    annulus grades paler (Juno: pale salmon rim);
                //  - faint concentric wrapped lanes (the Voyager low-contrast
                //    internal banding), azimuth-wobbled so they read as wound
                //    cloud lanes, not target rings.
                // The dark perimeter ring and bright collar shift OUT to the new
                // edge with emergence (mix below). Variant-gated => the default
                // program never contains any of this.
#ifdef HERO_EMERGENCE
                float fill = core;
                {
                    float plate = 1.0 - smoothstep(0.62, 1.0, qrim);
                    // fbm edge fray, only where its Gaussian window is
                    // non-negligible (< 2% past qrim 1.7; plate is already 0
                    // there) — this runs per pixel per step as the relaxation
                    // target, so skipping the dead outer annulus matters.
                    if (qrim < 1.7) {
                        // Fray/wisp frequencies ride the compaction (x~1.55) so
                        // the interior character keeps its proportions on the
                        // smaller oval.
                        float ffrq = mix(5.0, 8.0, u_hero_emergence);
                        float efray = fbm(p * (a.w > 0.0 ? ffrq / a.w : ffrq)
                                          + u_hero_noise_offset.yxz, 3, 2.0, 0.5);
                        plate = clamp(plate + 0.6 * u_hero_emergence * efray
                                      * exp(-(qrim - 0.84) * (qrim - 0.84) * 6.0),
                                      0.0, 1.0);
                    }
                    fill = mix(core, plate, u_hero_emergence);
                    // Upgrade the shared Gaussian tint/dome to the plateau fill
                    // (delta form: the shared lines above already added *core).
                    dT3 += b.z * (fill - core);
                    dT1 += 0.15 * (fill - core);
                    // Radial identity: deeper red + slightly darker core, paler
                    // salmon outer annulus. NOTE the storm_tints LUT is
                    // NON-MONOTONIC (tan -> dark brown ~0.72 -> salmon 1.0): the
                    // deep brick red lives at LOWER T3 than the pale salmon, so
                    // the core LOWERS T3 (toward brick) and the annulus RAISES it
                    // (toward pale salmon; the LUT lookup clamps at 1).
                    // The LUT index and the blend WEIGHT are both |T3|-driven, so
                    // "pale edge" is reached by dropping T3 steeply (weight fades,
                    // cream base shows through), NOT by raising it (that saturates
                    // at pure salmon and reads as rust). Center: slight drop into
                    // the brick zone at full weight = deep. Edge: strong drop to
                    // ~0.3 = pale wash.
                    // Deep center via VALUE (T0 down at full salmon weight), not
                    // via T3: the weight coupling makes a lowered T3 fade pale.
                    // CORE POLARITY (per-latitude reviews, 4/6 flagged, 2 as
                    // the single worst deviation): the reference core is the
                    // BRIGHTEST, warmest element in frame — ours read as a
                    // recessed dull hole. The old radial 'deep' darkening is
                    // GONE (it also drew a shared-center contour), replaced
                    // by a uniform plateau lift; the off-center knot supplies
                    // the hot patch, tint_var the mottle.
                    dT0 += 0.10 * u_hero_emergence * plate;
                    dT3 -= 0.30 * u_hero_emergence * b.z
                         * smoothstep(0.45, 0.97, q) * plate;
                    dT0 += 0.06 * u_hero_emergence
                         * smoothstep(0.45, 0.97, q) * plate;
                    // OPEN wrapped lanes, windowed off the very center. The
                    // integer +hth term (m=1, 2pi-periodic => branch-cut safe
                    // at th=+-pi) turns the q-only phase — closed concentric
                    // ellipses by functional form, the "etched onion ring"
                    // tell — into one continuous wound spiral arm, matching
                    // the GRS's open low-contrast internal banding.
                    if (hth_ok) {
                        vec3 lph = u_hero_noise_offset * 9.42;
                        float lane = sin(q * 6.0 + hth
                                         + 1.1 * sin(hth + lph.x) + lph.y);
                        // 0.07 -> 0.09: ONE legible spiral (reference shows
                        // faint but readable internal striation; at 0.07 the
                        // interior read as unstructured noise).
                        dT0 += 0.09 * u_hero_emergence * lane * plate
                             * smoothstep(0.16, 0.32, q);
                        // T3-space internal spiral banding: the T0 lane above
                        // is swamped by derive's |T3|~0.9 tint blend (measured
                        // — interior structure MUST ride T3, which moves the
                        // LUT index AND the blend weight). Rectified DIP —
                        // crests pull T3 from ~0.9 toward the palette's
                        // bright-orange shoulder (raising T3 saturates at the
                        // top stop). Tighter pitch than the T0 lane (~3 wound
                        // arcs across the plateau); fresh inner phase lph.y —
                        // lph.x already drives the T0 lane and would correlate.
                        float lane3 = sin(q * 13.0 + hth
                                          + 1.1 * sin(hth + lph.y) + lph.z);
                        float wq = smoothstep(0.12, 0.28, q)
                                 * (1.0 - smoothstep(0.82, 1.0, q));
                        dT3 -= 0.22 * u_hero_emergence * b.z
                             * (0.5 + 0.5 * lane3) * plate * wq;
                        // Off-center bright knot (HST/Cassini: the interior's
                        // brightest patch sits off-center): seeded azimuth,
                        // 0.3-rc offset, one open Gaussian — no closed
                        // contour. Law of cosines in the (q, hth) polar frame.
                        float q_off2 = q * q + 0.09
                                     - 0.6 * q * cos(hth - lph.z);
                        float knot = exp(-3.0 * q_off2);
                        // T3 carries the knot (0.10 -> 0.24: fully saturated
                        // salmon at raised blend weight); the T0 lift rides
                        // along (0.14 -> 0.18).
                        dT0 += 0.18 * u_hero_emergence * knot * plate;
                        dT3 += 0.24 * u_hero_emergence * knot * plate;
                        // Storm-within-a-storm nucleus: a compact dark inner
                        // eye at 0.25 rc, decorrelated from the bright knot
                        // (phase offset -2.6), dipping T3 toward the brown
                        // notch — the reference core's visible sub-storm.
                        // Never max this AND lane3 together (stacked dips
                        // past the notch read as a pale wash).
                        float q_off2b = q * q + 0.0625
                                      - 0.5 * q * cos(hth - lph.z - 2.6);
                        dT3 -= 0.30 * u_hero_emergence * b.z
                             * exp(-6.0 * q_off2b) * plate;
                    }
                }
                // GRS annulus anatomy (mix so small lever values stay near
                // the legacy anatomy): a THIN bright annulus hugging the
                // plateau edge (the real Hollow margin is ~0.2 R), then ONE
                // diffuse dark collar outside it — the reference order. The
                // pre-pass anatomy inverted this (etched dark ring at 1.05
                // sharpening to k~38 = the "onion ring" tell, bright basin at
                // 1.30): the dark ring is repurposed as the outer diffuse
                // collar (out to 1.30, softened toward k 12 — its tail ends
                // ~1.54, clear of the 1.55 flush rise) and the bright ring
                // tightens onto the plateau (1.12, k toward 34). Endpoints
                // start-values from plan review (fill ~79% of the hollow at
                // e=0.9; 1.42 read as low-60s% perceived); calibration-owned.
                float ring_q = mix(1.0, 1.30, u_hero_emergence);
                float col_q  = mix(1.55, 1.12, u_hero_emergence);
                float ring_k = mix(16.0, 12.0, u_hero_emergence);
                float col_k  = mix(5.0, 34.0, u_hero_emergence);
                // Moat shear-asymmetry (deterministic, reference-keyed — NOT
                // the seeded-random azw lobes below): the GRS moat is wider
                // poleward and upstream, pinched equatorward, and torn open on
                // the downstream (wake) arc. Directions come from the hero
                // frame: h1 = cross(j, c) points ANTI-east, so hth=0 is local
                // WEST and +cos(hth) is the west component; wake_dir supplies
                // the downstream sign (east-positive), a.y the hemisphere.
                // carve hands the downstream collar arc to the wake instead of
                // leaving a uniform drawn ring; it also fades the rim-tint
                // moat below (same tear).
                float carve = 1.0;
                float ringmod = 1.0;
                if (hth_ok) {
                    float wdir  = vortex_data[3 * i + 2].x;
                    float polew = (a.y < 0.0) ? max(-sin(hth), 0.0)
                                              : max(sin(hth), 0.0);
                    float equw  = (a.y < 0.0) ? max(sin(hth), 0.0)
                                              : max(-sin(hth), 0.0);
                    // upstream = -wake_dir in east terms = +wdir on the
                    // anti-east h1 axis; downstream is the negation.
                    float eastw = max(cos(hth) * wdir, 0.0);
                    // Downstream opening WIDENED and DEEPENED (0.3,0.9/0.55
                    // -> 0.1,0.8/0.8): the per-latitude reviews still read
                    // the collar as a closed 360-degree ring — the moat must
                    // visibly open into the wake, not just dim there.
                    float westw = smoothstep(0.1, 0.8, -cos(hth) * wdir);
                    col_q += u_hero_emergence * (0.10 * polew + 0.06 * eastw);
                    col_k *= 1.0 + 0.9 * u_hero_emergence * equw;
                    carve = 1.0 - 0.8 * u_hero_emergence * westw;
                    // Closure-breaking raggedness: seeded few-lobe modulation
                    // of BOTH ring amplitudes (decorrelated from the rim_warp
                    // lobes) so neither annulus holds constant width/value
                    // around its arc — "constant gap = drawn ring".
                    vec3 cph = u_hero_noise_offset * 17.3;
                    carve *= 0.78 + 0.22 * sin(2.0 * hth + cph.x)
                                  * sin(1.0 * hth + cph.y);
                    ring_k *= 1.0 + 0.45 * u_hero_emergence
                                    * sin(3.0 * hth + cph.z);
                    // The DARK collar needs its own break-up: with the detail
                    // overlay off it read as a crisp CLOSED ellipse — width
                    // lobes alone don't open it. Decorrelated amplitude lobes
                    // + the same downstream tear the bright collar gets.
                    // Lobe depth 0.28 -> 0.45 with floor 0.10 (round-B review:
                    // at 0.28 the collar only dimmed to 0.33x — it must
                    // visibly FAIL on some arcs, not thin), plus an
                    // equatorward amplitude cut paired with the belt-side
                    // flush pinch (heroRelaxWeight): the pinched contact must
                    // be belt-against-bright-rim, not belt-against-etched-
                    // dark-ring, so the collar yields on the arc the flush
                    // re-imposes hardest.
                    ringmod = (0.55 + 0.45 * sin(2.0 * hth + cph.y + 2.1)
                                    * sin(3.0 * hth + cph.x + 0.7))
                            * (1.0 - 0.6 * u_hero_emergence * westw)
                            * (1.0 - 0.55 * u_hero_emergence * equw);
                }
                // Quiet hollow: the real Red Spot Hollow is only slightly
                // brighter than the bands (Juno close-ups), not a glowing
                // basin — soften both the dark ring and the bright collar
                // with the lever.
                float quiet = 1.0 - 0.5 * u_hero_emergence;
                // Collar base raised under emergence (0.22 -> 0.31 at e=1):
                // with quiet, rim_contrast 1.3 and the ring/annulus overlap
                // dilution, the flat 0.22 washed the thin annulus below the
                // m5 hero-contrast tripwire (0.20 < 0.22) — user asked for
                // more pop (Checkpoint 1). Raising THIS base, not
                // rim_contrast, keeps the dark collar unamplified.
                dT0 += b.w * fill
                     - mix(0.16, 0.125, u_hero_emergence) * quiet * ringmod * u_rim_contrast
                       * exp(-(qrim - ring_q) * (qrim - ring_q) * ring_k)
                     + mix(0.22, 0.31, u_hero_emergence) * quiet * carve * u_rim_contrast
                       * exp(-(qcol - col_q) * (qcol - col_q) * col_k);
#else
                dT0 += b.w * core
                     - 0.16 * u_rim_contrast * exp(-(qrim - 1.0) * (qrim - 1.0) * 16.0)
                     + 0.22 * u_rim_contrast * exp(-(qcol - 1.55) * (qcol - 1.55) * 5.0);
#endif
                // Dark reddish collar (the Red Spot Hollow rim): redden (T3 up,
                // toward the salmon storm-tint LUT) and darken (T0 down) the
                // perimeter annulus so the spot has a discrete dark-red rim.
                // Rides on the warped qrim so the tint follows the lumpy edge.
                // Guarded => byte-identical when off.
                if (u_hero_rim_tint > 0.0) {
#ifdef HERO_EMERGENCE
                    // DE-DOUBLED under emergence: co-located with the dark
                    // collar (ring_q endpoint 1.30) so the reddening and the
                    // darkening form ONE diffuse boundary — the reference has
                    // a single reddish edge dissolving into shear, and the old
                    // inner line at 1.09 + collar at 1.27 pair read as "two
                    // inky concentric drawn ellipses" (adversarial re-check;
                    // the inner ring sampled continuous, value 127-153, no
                    // gap). Sharpness matches the diffuse collar (k 12), not
                    // the old etched 26.
                    float rt_q = mix(1.08, 1.30, u_hero_emergence);
                    float rt_k = mix(11.0, 12.0, u_hero_emergence);
                    float rring = exp(-(qrim - rt_q) * (qrim - rt_q) * rt_k);
#else
                    float rring = exp(-(qrim - 1.08) * (qrim - 1.08) * 11.0);
#endif
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
#ifdef HERO_EMERGENCE
                    // Under emergence the moat sits ON the enlarged pale margin —
                    // the real hollow boundary is subtle wisps, not a heavy dark
                    // ring — so both moat terms soften with the lever. carve
                    // (declared with the collar above) tears the downstream arc
                    // open the same way it tears the bright collar.
                    float moat = (1.0 - 0.6 * u_hero_emergence) * carve;
                    dT3 += u_hero_rim_tint * 0.55 * rring * moat;     // redden
                    dT0 -= u_hero_rim_tint * 0.16 * rring * azw * moat;  // darken + broken
#else
                    dT3 += u_hero_rim_tint * 0.55 * rring;            // redden -- unchanged
                    dT0 -= u_hero_rim_tint * 0.16 * rring * azw;      // deeper (was 0.12) + broken
#endif
                }
                // Interior turbulent churn: a flow-scale fbm breaks up the
                // smooth Gaussian core so the spot reads as churning cloud, not
                // an airbrushed blob. Windowed to the interior (q<1) so it never
                // fights the perimeter ring/collar. Stamped into the relaxation
                // target => advect.comp folds it into filaments (motion-safe).
                // Guarded => byte-identical when off (fbm never evaluated).
                if (u_hero_mottle > 0.0) {
                    float win = core * (1.0 - smoothstep(0.6, 1.0, q));
#ifdef HERO_EMERGENCE
                    // Under emergence the interior is the whole plateau, not just
                    // the Gaussian core: extend the churn window with it — but
                    // MUTED (the real interior motion is "small and random",
                    // ~3%-contrast wisps, not loud churn). Wisp frequency rides
                    // the compaction so the texture keeps its proportions.
                    win = max(win, fill * (1.0 - smoothstep(0.78, 1.04, qrim)))
                        * (1.0 - 0.35 * u_hero_emergence);
                    // fscale boost 0.9 -> 0.4: at x1.9 the churn was so fine
                    // it read as sensor noise (reference-anchored review);
                    // coarser wisps read as circulation.
                    float fscale = (a.w > 0.0 ? 9.0 / a.w : 9.0)
                                 * (1.0 + 0.4 * u_hero_emergence);
#else
                    float fscale = a.w > 0.0 ? 9.0 / a.w : 9.0;
#endif
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
#ifdef HERO_EMERGENCE
                    winT = max(winT, fill * (1.0 - smoothstep(0.75, 1.0, qrim)))
                         * (1.0 - 0.5 * u_hero_emergence);
                    float fscaleT = (a.w > 0.0 ? 7.0 / a.w : 7.0)
                                  * (1.0 + 0.55 * u_hero_emergence);
#else
                    float fscaleT = a.w > 0.0 ? 7.0 / a.w : 7.0;
#endif
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
#ifdef HERO_EMERGENCE
            // Emergence: the wedge extends toward 9 rc (the reference wake
            // stays identifiable 2-3 oval diameters west) and the painted
            // gray-white DIMS — under the pack the wake's brightness should
            // come from REAL advected belt/zone material folded by the wake
            // forcing (omega_force) inside the relaxation release
            // (heroRelaxWeight), not from this stamp. Variant arm only; the
            // #else arm below is the pre-feature text verbatim.
            float wlen = mix(6.0, 9.0, u_hero_emergence);
            float wdim = 1.0 - 0.6 * u_hero_emergence;
            if (along > 0.0 && along < rc * wlen && abs(across) < 2.5) {
                float ramp = smoothstep(rc * 0.5 * asp, rc * asp, along);
                float win = 1.0 - smoothstep(2.0, 2.5, abs(across));
                float w = exp(-across * across) * win * (1.0 - along / (rc * wlen)) * ramp;
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
                dT0 += 0.16 * w * wdim;   // bright churned clouds (dimmed)
                dT3 -= 0.20 * w * wdim;   // cool gray-white, not belt-colored
            }
#else
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
#endif
        }
    }
    return vec3(dT0, dT1, dT3);
}

#ifdef HERO_EMERGENCE
// Per-pixel relaxation multiplier that makes the hero's edge FLOW-NEGOTIATED
// and its NEIGHBORHOOD band-flushed.
//
// The relaxation forcing (advect.comp pass 2) re-imposes the analytic hero
// stamp every step, so the flow can never own the storm -> it reads as stamped.
// Two hero-local modifications, both gated by u_hero_emergence:
//   RIM BAND (q ~ 1.0, the plateau edge): FADE relaxation so advection (the
//     annular ring's shear) folds the tracer into a ragged, filament-shedding
//     boundary. The interior keeps FULL relaxation — the real GRS interior is
//     stagnant and must hold its plateau fill.
//   FLUSH ANNULUS (q ~ 1.55-3.4): BOOST relaxation toward the band stamp. Over
//     a dev run even a weak orbital flow winds the whole neighborhood into
//     concentric arcs, and with relax_tau ~2000 nothing ever erases them —
//     while on the real planet the jets sweep wound material downstream and
//     the bands re-assert (Cassini PIA07782: band-parallel beyond ~1.5 spot
//     radii, thin collar hugging the oval). The boost is that flushing,
//     hero-local (the hero stamp is ~0 out there, so the target IS the band).
//     With the plateau edge at q~1.0, q IS spot radii — flush starts right
//     where Cassini shows the bands re-asserting, and is done by ~3, so the
//     storm's influence zone stays a small multiple of the storm itself.
// Far-field pixels (no hero within q<3.6) return exactly 1.0.
//
// Compiled and called only in the HERO_EMERGENCE variant (advect.comp's pass-2
// relaxation lines select rk = u_relax_k * heroRelaxWeight(p) under the same
// #ifdef, with no runtime guard) — the default program contains none of this
// by construction, so it stays byte-identical.
float heroRelaxWeight(vec3 p) {
    float infl = 0.0;    // strongest rim-band fade at this pixel, in [0,1.4]
    float flush = 0.0;   // strongest neighborhood-flush boost, in [0,1]
    float wrel = 0.0;    // strongest wake-sector release window, in [0,1]
    for (int i = 0; i < u_vortex_count; ++i) {
        vec4 b = vortex_data[3 * i + 1];
        if (b.y != VKIND_HERO) continue;
        // Wake-sector relaxation RELEASE. Advection must OWN the wake: the
        // smooth stamped wedge is the relaxation TARGET, so full-rate
        // relaxation (and the flush boost below) actively erases every fold
        // the wake forcing creates each step — the old "single laminar hook"
        // was this mechanism's designed steady state, not an undertuned
        // lever. Computed BEFORE the q cull: at aspect 2.2 the 9-rc wake tail
        // sits at q~4.1, past the 3.6 rim/flush cull. Hard windows on both
        // axes => exactly 0 in the far field (the locality byte-identity
        // tests assert_array_equal outside the neighborhood), with smooth
        // ramps strictly inside so no relax-weight arc prints.
        vec4 wa = vortex_data[3 * i];
        float rc_h = wa.w;
        float asp_h = vortex_data[3 * i + 2].y;
        float down = vortex_data[3 * i + 2].x;
        float woff = vortex_data[3 * i + 2].z;
        float vlat = asin(clamp(wa.y, -1.0, 1.0));
        float vlon = atan(wa.z, wa.x);
        float plat = asin(clamp(p.y, -1.0, 1.0));
        float plon = atan(p.z, p.x);
        float dlon = mod(plon - vlon + 3.0 * PI, 2.0 * PI) - PI;
        {
            float an = dlon * down / max(rc_h, 1e-4);
            float across = (plat - (vlat + woff)) / max(rc_h * 1.8, 1e-4);
            // along > 1.5 rc keeps the flush pincer's hollow rim clean (the
            // release starts where the wake lives, not at the collar).
            if (an > 1.5 && an < 9.0 && abs(across) < 2.0) {
                float rise = smoothstep(1.5, 2.5, an);
                float fall = 1.0 - smoothstep(6.0, 9.0, an);  // relaminarize
                float aw = (1.0 - smoothstep(1.4, 2.0, abs(across)))
                         * exp(-across * across);
                wrel = max(wrel, rise * fall * aw);
            }
        }
        float q = heroEllipQ(p, i, 3.6);
        if (q > 3.6) continue;   // strictly local to the storm neighborhood
        // Upstream (leading-side) weight, from the aspect-normalized azimuth
        // in the wake frame: 1 on the arc the flow ARRIVES from, 0 downstream.
        // The reference's leading side is smoothly compressed laminar flow —
        // the belt parts around the oval and rejoins — while all the ragged
        // shedding lives downstream. Used below to (a) suppress the rim-fade
        // erosion (the boundary stays target-held = smooth) and (b) boost the
        // flush (wound arcs on the approach are erased faster, so the belt
        // reads band-parallel right up to the deflection).
        float xe = dlon * down / max(asp_h, 1.0);
        float yn = plat - vlat;
        float upw = smoothstep(0.15, 0.7,
                               -xe / max(length(vec2(xe, yn)), 1e-5));
        // Hero-local meridional/azimuthal frame, shared by the boundary
        // raggedness and the flush shaping below. m ~ sin(hero-frame
        // azimuth) via the squashed elliptical metric (heroEllipQ divides
        // only the east axis by aspect, so yn/(rc*q) is exact); eqs points
        // equatorward — the belt side for a hollow-straddling hero. az uses
        // dlon, NOT the wdir-flipped xe: seeded lobes must not mirror when
        // the wake direction flips. The ternary guards atan(0,0)
        // (GLSL-undefined) at the exact center pixel — a NaN here would
        // advect outward through the relax weight.
        float m     = clamp(yn / max(rc_h * q, 1e-5), -1.0, 1.0);
        float eqs   = (wa.y < 0.0) ? 1.0 : -1.0;
        float beltw = smoothstep(0.15, 0.7,  m * eqs);
        float zonew = smoothstep(0.15, 0.7, -m * eqs);
        float az    = (q > 0.05) ? atan(yn, dlon) : 0.0;
        vec3  fph   = u_hero_noise_offset * 23.1;
        // Narrowed OFF the bright annulus (center 0.95, k 10 — was 1.0/3.8):
        // the fade must release the PLATEAU EDGE to the flow without also
        // releasing the thin annulus at ~1.16, which already fights the
        // plateau fray and the collar carve (three shredders on one
        // 0.15-q-wide feature washed it out).
        // Low-order raggedness of the emergent WOUND boundary (the dark
        // ellipse is root-caused as wound tracer in this released band —
        // stamps exonerated): the fbm fray below is per-pixel with a
        // uniform azimuthal MEAN, so the wound band kept constant
        // width/strength around the ellipse and read as drawn. Three seeded
        // low-order terms break the mean: WIDTH lobes (primary, +-40% on
        // the release sharpness), a one-sided INWARD radius wobble (inward
        // only — pushing the center outward re-admits the bright annulus at
        // 1.12 to the flow, undoing the narrowed-off-the-annulus retune;
        // rim_bump(1.12) <= 0.8 off the break arc holds by construction),
        // and ONE seeded break arc where the boundary is handed to the flow
        // outright — lobed dimming alone never breaks the closed-curve
        // squint read; the reference boundary visibly fails on arcs.
        float rl  = 0.55 * sin(2.0 * az + fph.z)
                  + 0.45 * sin(3.0 * az + fph.x + 1.9);
        float rl2 = 0.6 * sin(2.0 * az + fph.y + 0.8)
                  + 0.4 * sin(4.0 * az + fph.z + 2.4);
        float rq  = 0.95 - 0.10 * u_hero_emergence * max(rl, 0.0);
        float rk  = 10.0 * (1.0 + 0.4 * u_hero_emergence * rl2);
        float rim_bump = exp(-(q - rq) * (q - rq) * rk);
        float brk = smoothstep(0.5, 0.9, sin(az + fph.y * 1.3));
        rim_bump = max(rim_bump,
                       brk * exp(-(q - 1.14) * (q - 1.14) * 14.0));
        if (q < 2.2) {
            // Per-azimuth erosion: some arcs keep the ring, others dissolve ->
            // the boundary is ragged, not a uniformly-softened circle.
            // Flow-scale fbm on the seeded hero offset (deterministic). Only
            // evaluated where rim_bump is non-negligible (< 0.6% past q 2.2) —
            // the flush annulus beyond gets the smoothsteps only.
            float fscale = rc_h > 0.0 ? 9.0 / rc_h : 9.0;
            float ero = clamp(0.15 + 1.4 * fbm(p * fscale + u_hero_noise_offset.zyx + 5.0,
                                               4, 2.0, 0.5),
                              0.0, 1.4);
            // Per-arc erosion DEPTH (low-order, same rl2 lobes as the width
            // mod): some arcs hand the boundary to the flow, others stay
            // target-held — the wound band's mean strength varies around
            // the ellipse instead of only its pixel-scale fray.
            ero = clamp(ero * (0.95 + 0.30 * u_hero_emergence * rl2),
                        0.0, 1.4);
            // Leading side stays SMOOTH: the erosion (which hands rim arcs
            // to the flow) is suppressed on the upstream arc, so the belt's
            // approach meets a target-held, cleanly-deflected boundary
            // instead of a ragged one (Checkpoint-1 feedback: "tighten up
            // the leading side so the belt flows smoothly around the storm").
            // This suppression stays LAST.
            ero *= 1.0 - 0.65 * upw;
            infl = max(infl, rim_bump * ero);
        }
        // Tight-but-strong Hollow (user constraint), now MERIDIONALLY
        // SHAPED (round-B de-bullseye): a radially-uniform flush halo was
        // the bullseye's outermost ring. Belt-side (equatorward) the inner
        // rise pulls IN to ~1.19 with a shorter rise — the band tone
        // asserts right against the bright rim, the reference's hard pinch
        // (paired with the equw dark-collar cut in the stamp, so the
        // re-imposed contact is belt-against-bright-rim). Zone-side widens
        // only MODESTLY (+0.12: a wider clean south moat reads emptier, not
        // more natural — its structure comes from occupants, not less
        // erasure). Seeded 2/3-lobe wobble on the non-belt arcs keeps the
        // inner boundary off an analytic curve.
        // The wound-arc cleanup role is protected two ways: the OUTER fade
        // stays WIDE (2.7,3.4) on every azimuth (pulling it in let wound
        // arcs survive in the 2.8-3.4 shell — measured), and a uniform
        // full-strength FLOOR from q 2.05-2.35 caps how far out any
        // azimuthal reduction reaches — poleward survival is confined to
        // the q<~2.35 moat where it reads as structure, not pinwheel arcs.
        float fl   = 0.6 * sin(2.0 * az + fph.x)
                   + 0.4 * sin(3.0 * az + fph.y);
        float qin  = 1.55 + u_hero_emergence
                          * (-0.40 * beltw + 0.12 * zonew
                             + 0.08 * fl * (1.0 - beltw));
        float rise = mix(0.35, 0.20, beltw);
        float shaped = smoothstep(qin, qin + rise, q)
                     * (1.0 - smoothstep(2.7, 3.4, q));
        float floorf = smoothstep(2.05, 2.35, q)
                     * (1.0 - smoothstep(2.7, 3.4, q));
        flush = max(flush, max(shaped, floorf));
    }
    // Wake release: fade relaxation inside the wedge (CAPPED at 0.75 — the
    // floor prevents long-run homogenization into mud; through-flow refreshes
    // the wedge ~4x per dev run at the measured jet speed) and exempt the
    // wake sector from the flush so the boost cannot stomp the very folds the
    // release preserves.
    infl = max(infl, 0.75 * wrel);
    flush *= 1.0 - wrel;
    // Fade in the rim band (down to 0), boost in the flush annulus (up to
    // x12: tau_eff ~ relax_tau/12 ~ 170 steps -> the wound arcs decay well
    // within the dev run while the bands re-assert; still << 1 per step,
    // stability untouched). The boost is paired with the partial vorticity
    // shield: the residual circulation still winds the annulus, just slowly
    // enough for this flush to win. NON-directional by design (an earlier
    // upstream-only x1.6 was too subtle): the reference-anchored review
    // wants bands re-asserting on EVERY azimuth — leading side laminar AND
    // the belt visibly re-closing downstream — with the wake wedge the only
    // exemption (the wrel release above already carves it out).
    return clamp(1.0 - u_hero_emergence * infl, 0.0, 1.0)
         + 11.0 * u_hero_emergence * flush;
}

// Belt bowing (Red Spot Hollow geometry): pull the SAMPLED latitude of the
// band-target lookup toward the hero inside an annular window, so every band
// boundary within ~2.3 core radii bows tightly around the oval and recovers
// within ~one storm width — instead of being painted STRAIGHT through the
// hollow, which the flush would then re-impose every step. With the
// deflection the flush COOPERATES: it re-imposes the bowed band. Applied
// identically by init.comp and advect.comp (both must shape the SAME target,
// the band_mod.glsl rule); ordering pinned here: deflect the true latitude
// FIRST, the caller adds its fbm band warp after. Amplitude: pull 0.75 at
// e=1 bows a boundary ~1.6 deg from the hero center by ~1 core radius (solve
// lat_s(L)=L_b) — the reference's tight-but-strong bow; displacement capped
// at 1.1 rc. The jet velocity profile is deliberately NOT deflected (the
// vorticity solver's jets already respond dynamically; warping the analytic
// profile would double-count).
float heroBandDeflect(vec3 p, float lat) {
    float lat_s = lat;
    for (int i = 0; i < u_vortex_count; ++i) {
        vec4 b = vortex_data[3 * i + 1];
        if (b.y != VKIND_HERO) continue;
        // Boundary gate (CPU-derived, vortices.py::_hero_bow_gain): 0 when no
        // band boundary sits within the bow's reach. Without it the sampling
        // displacement PAINTS a phantom wrap out of whatever latitude
        // gradient exists — the per-latitude adversarial reviews found it as
        // a sweep-invariant "red hook" at edge placements and a symmetric
        // funnel deep in the zone.
        float gate = vortex_data[3 * i + 2].w;
        if (gate <= 0.0) continue;
        float q = heroEllipQ(p, i, 2.3);
        if (q > 2.3) continue;
        vec4 a = vortex_data[3 * i];
        float vlat = asin(clamp(a.y, -1.0, 1.0));
        // Outer fade tightened (1.6,2.3)->(1.45,2.0): only the collar zone
        // bows; bands beyond stay horizontal (reference: nothing but the
        // collar itself is circular).
        float bw = smoothstep(0.8, 1.2, q) * (1.0 - smoothstep(1.45, 2.0, q));
        // The painted wrap must not read authored: (a) downstream SHED — the
        // bow opens toward the wake so the wrapped strand hands off to
        // advected material instead of riding the collar at constant width
        // through 180+ degrees (radius-locked ride = the stamp fingerprint);
        // (b) seeded few-lobe raggedness so width/reach vary along the arc
        // like torn cloud, not a drawn band.
        {
            float wdir = vortex_data[3 * i + 2].x;
            float plat = asin(clamp(p.y, -1.0, 1.0));
            float plon = atan(p.z, p.x);
            float vlon = atan(a.z, a.x);
            float dlon = mod(plon - vlon + 3.0 * PI, 2.0 * PI) - PI;
            float asp = vortex_data[3 * i + 2].y;
            float xe = dlon * wdir / max(asp, 1.0);        // + = downstream
            float yn = plat - vlat;
            float az = atan(yn, xe);                       // 0 = downstream
            // FLANK-ONLY modulation: |cos(az)| is 1 on the east/west arcs
            // (the radius-locked "painted wrap ride" the reviews flagged)
            // and exactly 0 at the north/south apexes — the load-bearing
            // bow (the >=0.8 r_core boundary deflection the test pins) is
            // never weakened. Downstream flank sheds into the wake (up to
            // 0.6); upstream flank gets seeded raggedness (up to 0.35).
            float flank = abs(cos(az));
            float downw = smoothstep(0.2, 0.9,
                                     xe / max(length(vec2(xe, yn)), 1e-5));
            vec3 bph = u_hero_noise_offset * 11.7;
            float lobes = 0.5 + 0.5 * (0.6 * sin(2.0 * az + bph.x)
                                     + 0.4 * sin(3.0 * az + bph.y));
            bw *= 1.0 - flank * (0.6 * downw + 0.35 * lobes * (1.0 - downw));
        }
        float pull = u_hero_emergence * 0.75 * gate * bw * (lat - vlat);
        float cap = 1.1 * a.w;
        lat_s -= clamp(pull, -cap, cap);
    }
    return lat_s;
}
#endif  // HERO_EMERGENCE
