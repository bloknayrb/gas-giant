"""GPU: the emission map — neutral-default identity, component behavior,
seam validation, determinism."""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams
from gasgiant.validate.seams import validate_arrays

pytestmark = pytest.mark.gpu


def _params(seed: int = 17) -> PlanetParams:
    p = PlanetParams(seed=seed)
    p.sim.resolution = 512
    p.sim.dev_steps = 60  # waves/storms develop enough to gate emission
    return p


def _emission_params(seed: int = 17) -> PlanetParams:
    p = _params(seed)
    p.emission.thermal_strength = 0.6
    p.emission.lightning_strength = 0.4
    p.emission.aurora_strength = 0.5
    return p


def test_defaults_produce_no_emission_key(gpu):
    maps = Simulation(_params(), gpu).render_maps(512)
    assert set(maps) == {"color", "height"}


def test_off_identity_across_program_switch(gpu):
    """Color/height must be byte-identical whether or not the emission
    program runs alongside — the neutral-defaults guard for bindings 0/1."""
    base = Simulation(_params(), gpu).render_maps(512)
    p = _params()
    p.emission.thermal_strength = 0.6
    p.emission.lightning_strength = 0.4
    p.emission.aurora_strength = 0.5
    on = Simulation(p, gpu).render_maps(512)
    np.testing.assert_array_equal(base["color"], on["color"])
    np.testing.assert_array_equal(base["height"], on["height"])
    assert on["emission"].shape == (256, 512, 4)


def test_thermal_anticorrelates_with_cloud_tops(gpu):
    p = _params()
    p.waves.festoon_strength = 1.2  # a strong hotspot chain
    p.waves.hotspot_depth = 0.7
    p.emission.thermal_strength = 1.0
    sim = Simulation(p, gpu)
    maps = sim.render_maps(512)
    em = maps["emission"][..., :3].sum(axis=-1)
    assert np.isfinite(em).all() and (em >= 0.0).all()
    assert em.max() > 0.0
    # The brightest emission sits where cloud tops are LOW.
    height = maps["height"]
    hot = em >= np.percentile(em[em > 0], 99)
    assert height[hot].mean() < height.mean()


def test_thermal_energy_concentrates_in_hotspots_not_belts(gpu):
    """The anomaly gate's reason to exist: with DEFAULT wave params the
    festoon-trough hot spots must fire the HDR term while undisturbed belt
    floors stay at the faint deck term — a >5:1 peak-to-belt ratio, not the
    1.5:1 wash an absolute threshold gave."""
    p = _params()
    p.emission.thermal_strength = 1.0
    maps = Simulation(p, gpu).render_maps(512)
    em = maps["emission"][..., :3].sum(axis=-1)
    lit = em[em > 1e-4]
    assert lit.size > 0
    assert float(em.max()) > 5.0 * float(np.median(lit))


def test_lightning_sparse_hdr(gpu):
    p = _params()
    p.emission.lightning_strength = 1.0
    maps = Simulation(p, gpu).render_maps(512)
    em = maps["emission"][..., :3].sum(axis=-1)
    assert np.isfinite(em).all() and (em >= 0.0).all()
    coverage = float((em > 1.0).mean())
    assert 1e-5 < coverage < 0.02  # sparse cells, not a wash and not nothing
    assert float(em.max()) > 5.0  # HDR cores exist


def test_aurora_polar_and_alpha_only(gpu):
    p = _params()
    p.emission.aurora_strength = 1.0
    maps = Simulation(p, gpu).render_maps(512)
    em = maps["emission"]
    assert np.allclose(em[..., :3], 0.0)  # aurora lives in alpha only
    alpha = em[..., 3]
    assert alpha.max() > 0.3
    h = alpha.shape[0]
    polar = np.concatenate([alpha[: h // 4], alpha[3 * h // 4 :]])
    assert polar.sum() > 0.95 * alpha.sum()  # poleward of 45 deg


def test_emission_validates_clean(gpu):
    p = _params()
    p.emission.thermal_strength = 0.6
    p.emission.lightning_strength = 0.5
    p.emission.aurora_strength = 0.5
    maps = Simulation(p, gpu).render_maps(512)
    em = maps["emission"]
    # HDR cores are sparse spikes: the wrap statistic runs on log1p; the raw
    # array is separately checked finite/non-negative above.
    validate_arrays({
        "emission_rgb_log": np.log1p(em[..., :3]),
        "emission_aurora": em[..., 3],
    })


# -- Phase 8: preview-path emission (its own scratch textures + dirty flag) -----


def _read_preview_color(sim, width=512):
    tex, _ = sim.ensure_preview(width)
    return sim.gpu.read_texture(tex).copy()


@pytest.mark.parametrize("make_params", [_params, _emission_params])
def test_preview_color_byte_identical_across_emission_path(gpu, make_params):
    """M8 guard: the Color the GUI shows via ``ensure_preview`` is byte-identical
    whether or not ``ensure_preview_emission`` is called before/after/interleaved
    -- the preview-path analog of ``test_off_identity_across_program_switch``.
    Checked on both an emission-disabled and an emission-enabled params set."""
    base = _read_preview_color(Simulation(make_params(), gpu))

    sim = Simulation(make_params(), gpu)
    sim.ensure_preview_emission(512)  # before the first color derive
    before = _read_preview_color(sim)
    sim.ensure_preview_emission(512)  # interleaved with a repeat color read
    after = _read_preview_color(sim)

    np.testing.assert_array_equal(base, before)
    np.testing.assert_array_equal(base, after)


def test_emission_preview_dirty_flag_independent_of_color(gpu):
    """The emission dirty flag is separate from the color flags: an
    ``ensure_preview`` that clears ``_post_dirty``/``_tracers_changed`` must NOT
    leave a later ``ensure_preview_emission`` thinking it is clean (else stale
    glow after a POST edit). A tick re-dirties both paths."""
    sim = Simulation(_emission_params(), gpu)
    sim.ensure_preview(512)  # clears the color flags only

    _, rerendered = sim.ensure_preview_emission(512)
    assert rerendered is True, "emission derives despite color flags being cleared"
    _, rerendered2 = sim.ensure_preview_emission(512)
    assert rerendered2 is False, "second call with no state change is cached"

    sim.tick(2)  # advancing the sim re-dirties both preview paths
    _, color_re = sim.ensure_preview(512)
    _, em_re = sim.ensure_preview_emission(512)
    assert color_re is True and em_re is True


def test_emission_preview_zero_when_disabled_nonzero_when_enabled(gpu):
    """Documents the "disabled still derives to all-zero" choice: with emission
    off the returned texture is a valid all-zero map (the GUI shows the
    "emission disabled" note off ``params.emission.enabled``); with it on the
    RGB glow is present. Aurora (alpha) is not asserted here -- it is invisible
    in the RGB preview by design."""
    off = Simulation(_params(), gpu)
    off.run_to_completion()
    tex_off, _ = off.ensure_preview_emission(512)
    assert np.allclose(off.gpu.read_texture(tex_off)[..., :3], 0.0)

    on = Simulation(_emission_params(), gpu)
    on.run_to_completion()
    tex_on, _ = on.ensure_preview_emission(512)
    assert float(on.gpu.read_texture(tex_on)[..., :3].max()) > 0.0


def test_emission_deterministic(gpu):
    def run():
        p = _params(seed=23)
        p.emission.thermal_strength = 0.5
        p.emission.lightning_strength = 0.5
        p.emission.aurora_strength = 0.5
        return Simulation(p, gpu).render_maps(512)["emission"]

    np.testing.assert_array_equal(run(), run())
