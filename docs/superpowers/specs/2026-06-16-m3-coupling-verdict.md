# M3-coupling — Render Gate Verdict

**Date:** 2026-06-16. **Branch:** `v2-m3-baroclinic`.
**Status:** <PASS | PARTIAL | FALLBACK>  (finalize after the authoritative RES=2048 run)

## What replaced the old gate
The T9 direct-render gate (top-layer eddy Rossby) is **closed** -- falsified by two
adversarial passes (see `2026-06-16-m3-coupling-design.md`): the baroclinic field is
the wrong direct render driver. This gate measures the COUPLED render instead.

## Gate results
Smoke run (RES=512, gain=1.2):
- Source dominant zonal m = 8  (coherence gate <= 15): OK
- Latitude concentration: baseline 0.959 -> coupled 0.929  (ratio 0.969, gate >1.05): FAIL
- Texture ratio highfreq(coupled)/highfreq(baseline) = 0.997  (gate 0.5..2.0): OK

Note: the smoke-run cadence (`update_every=32, baro_steps_per_update=400`, warmup 9000)
differs from the GPU pytest gate (`update_every=4, baro_steps_per_update=200`,
warmup 6000), which PASSED `coupled_conc >= base_conc`. The smoke run shows a slight
*de*-concentration at the wider script cadence, so the >1.05 strict-script threshold
is not met at RES=512; see "Honest read" + R1 fallback note below.

Authoritative run (RES=2048, gain=<G>): <PLACEHOLDER -- controller fills>
- Blind panel: out/audit/m3/gate/gate_{baseline,source,coupled}.png

## Gain calibration
<PLACEHOLDER -- swept gain in {0.8, 1.2, 1.5}; chosen <G> because <reason>.>

## Cadence benchmark + residency decision
- baro 20.7s + upload 0.09s vs v1.6 1.1s over 22 updates (OUTCROPPED) at RES=512.
- Recommendation: consider-residency (rule: overhead/v1.6 > 0.25).
- Decision: <PLACEHOLDER -- controller fills>.

## Honest read
<PLACEHOLDER -- 2-4 sentences: did the evolving source give a strong physically-grounded
imprint while keeping natural texture? If R1 bit (weak modulation), state the fallback.>
The GPU pytest gate (relative `coupled_conc >= base_conc`) passes, but the stricter
script ratio (>1.05) is not met at the slower script cadence/RES=512 smoke -- the
modulation is present but weak, which is consistent with the R1 risk (bottom-trapped
mode, weak top-layer imprint). The controller should run the authoritative RES=2048
gate and, if R1 bites, invoke the gain sweep / R1 fallback before finalizing the verdict.
