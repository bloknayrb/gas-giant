"""DETAIL_FX variant: longitudinal intermittency gate (and, from the
hero-spiral commit, the GRS internal lanes).

Cross-variant comparisons use atol, never byte-equality (different
binaries may reschedule FP in shared expressions); byte-equality is only
asserted within one program."""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams
from gasgiant.validate import validate_arrays

pytestmark = pytest.mark.gpu


def _quick_params(**detail) -> PlanetParams:
    p = PlanetParams(seed=42)
    p.sim.resolution = 512
    p.sim.dev_steps = 0
    p.detail.intensity = 0.8
    p.detail.striation_amount = 0.6
    for key, value in detail.items():
        setattr(p.detail, key, value)
    return p


def test_explicit_zero_routes_to_default_program(gpu):
    base = Simulation(_quick_params(), gpu).render_maps(256)["color"]
    zero = Simulation(_quick_params(intermittency=0.0), gpu).render_maps(256)["color"]
    np.testing.assert_array_equal(base, zero)


def test_forced_fx_variant_is_noop_at_epsilon(gpu):
    """intermittency=1e-6 forces the DETAIL_FX program while the gate stays
    1 +- 1.5e-6 — the test that actually exercises the variant."""
    base = Simulation(_quick_params(), gpu).render_maps(256)["color"]
    fx = Simulation(_quick_params(intermittency=1e-6), gpu).render_maps(256)["color"]
    assert np.allclose(base, fx, atol=1e-3), np.abs(base - fx).max()


def _synth_detail_field(gpu, params, size=(1024, 512)) -> np.ndarray:
    """Run DetailSynth directly (the composed color map mixes in the
    ungated cells and the T2-tracer term, which drown the gated filament
    signal in a ratio statistic)."""
    sim = Simulation(params, gpu)
    s = sim.solver
    out = gpu.texture2d(size, 1, "f4", linear=True)
    sim.detail_synth.synthesize(
        params.seed, s.equirect.vel_tex, s.equirect.tracers.cur,
        sim.profile_dyn, out, params.detail,
    )
    field = gpu.read_texture(out)[..., 0]
    out.release()
    return field


def test_intermittency_adds_longitudinal_patchiness_not_tone(gpu):
    base = _synth_detail_field(gpu, _quick_params())
    gated = _synth_detail_field(gpu, _quick_params(intermittency=0.9))
    assert not np.array_equal(base, gated)
    assert np.all(np.isfinite(gated))
    assert abs(float(gated.mean() - base.mean())) < 0.02  # patchiness, not tone

    # Intermittent modulation: per-block RATIO of filament energy
    # gated/base (16x16 blocks over band latitudes). Without the gate every
    # ratio is exactly 1; the gate multiplies the filament weight by
    # ~[0.32, 1.6] in a low-frequency pattern, so the ratios must SPREAD
    # around ~1 (some blocks calmed, some boosted) — patchiness, directly,
    # independent of how much longitudinal structure the base already had.
    def block_energy(field: np.ndarray) -> np.ndarray:
        h = field.shape[0]
        band = field[int(0.25 * h): int(0.75 * h)] - 0.5
        bh, bw = band.shape[0] // 16, band.shape[1] // 16
        blocks = band[: bh * 16, : bw * 16].reshape(bh, 16, bw, 16)
        return np.sqrt((blocks ** 2).mean(axis=(1, 3)))

    e_base = block_energy(base)
    e_gated = block_energy(gated)
    active = e_base > np.quantile(e_base, 0.5)   # blocks with real texture
    ratio = e_gated[active] / e_base[active]
    assert 0.5 < float(ratio.mean()) < 1.5       # modulation, not a tone shift
    assert float(ratio.std()) > 0.1, ratio.std()  # ...and it is SPATIALLY patchy
    assert float(ratio.min()) < 0.7               # some calm runs
    assert float(ratio.max()) > 1.2               # some busy patches


def test_intermittency_render_is_seam_clean(gpu):
    sim = Simulation(_quick_params(intermittency=0.9), gpu)
    maps = sim.render_maps(512)
    report = validate_arrays({"color": maps["color"], "height": maps["height"]})
    assert report.ok, report.problems
