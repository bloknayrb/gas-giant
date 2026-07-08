"""T2 storm cast list: art-directed storms placed by hand (storms.cast).

Each StormOverride is stamped verbatim after the seeded populations, carries
origin="cast", and is exempt from the population cap and runtime mergers so a
director's storm survives the whole development run where it was placed. The
overriding invariant: an EMPTY cast (the default) is byte-identical to the
seeded-only field -- proven here field-by-field (test_empty_cast_identical).
Cast entries are deterministic (no RNG) and carry no ``rand`` metadata."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from gasgiant.params.model import (
    CastKind,
    PlanetParams,
    StormOverride,
    StormsParams,
)
from gasgiant.sim.bands import generate_bands
from gasgiant.sim.profiles import LatProfiles, build_profiles
from gasgiant.sim.solver import compute_dt
from gasgiant.sim.vortices import (
    KIND_OVAL,
    MAX_VORTICES,
    Vortex,
    VortexRegistry,
    generate_vortices,
    resolve_mergers,
)

# ---------------------------------------------------------------- helpers


def _profiles(seed: int, p: PlanetParams):
    bands = generate_bands(seed, p.bands)
    profiles = build_profiles(seed, bands, p.bands, p.jets)
    return bands, profiles


def _dt(p: PlanetParams, profiles) -> float:
    return compute_dt(p.sim.resolution, p.sim.dt_scale, profiles.max_speed)


def _fields(reg: VortexRegistry):
    return [
        (v.lat, v.lon, v.r_core, v.strength, v.kind, v.tint, v.brightness,
         v.wake_dir, v.wake_lat_off, v.aspect, v.origin)
        for v in reg.vortices
    ]


def _synth_profiles(shear: float = 1.0) -> LatProfiles:
    """u = shear * lat: monotone differential drift everywhere (clone of the
    test_mergers synthetic profile so a converging pair actually closes)."""
    n = 512
    lat = np.linspace(np.pi / 2.0, -np.pi / 2.0, n)
    u = shear * lat
    z = np.zeros(n)
    return LatProfiles(lat=lat, u=u, psi=z, shear_norm=z, belt_mask=z,
                       t0_stamp=z, t1_stamp=z, max_speed=float(np.abs(u).max()))


def _merge_storms(rate: float = 1.0):
    p = PlanetParams()
    p.storms.merge_rate = rate
    return p.storms


# ---------------------------------------------------------------- byte-identity


def test_empty_cast_identical():
    """Default vs an explicit empty cast=[] => registry field-by-field identical
    (the overriding byte-identity invariant)."""
    p_default = PlanetParams()
    p_explicit = PlanetParams()
    p_explicit.storms.cast = []
    assert p_default.storms.cast == []

    seed = 7
    bands, prof = _profiles(seed, p_default)
    dt = _dt(p_default, prof)
    reg_d = generate_vortices(
        seed, bands, prof, p_default.storms, p_default.poles,
        dt=dt, dev_steps=p_default.sim.dev_steps,
    )
    reg_e = generate_vortices(
        seed, bands, prof, p_explicit.storms, p_explicit.poles,
        dt=dt, dev_steps=p_explicit.sim.dev_steps,
    )

    assert len(reg_d.vortices) == len(reg_e.vortices)
    assert _fields(reg_d) == _fields(reg_e)
    # And every seeded vortex carries the default origin marker.
    assert all(v.origin == "seeded" for v in reg_d.vortices)


# ---------------------------------------------------------------- determinism


def test_cast_two_run_determinism():
    """A non-empty cast places verbatim, deterministically -- two runs match."""
    p = PlanetParams()
    p.storms.cast = [
        StormOverride(kind=CastKind.HERO, lat_deg=-20.0, lon_deg=45.0, radius=0.1),
        StormOverride(kind=CastKind.PEARL, lat_deg=30.0, lon_deg=-60.0),
    ]
    seed = 13
    bands, prof = _profiles(seed, p)
    dt = _dt(p, prof)
    r1 = generate_vortices(seed, bands, prof, p.storms, None, dt=dt,
                           dev_steps=p.sim.dev_steps)
    r2 = generate_vortices(seed, bands, prof, p.storms, None, dt=dt,
                           dev_steps=p.sim.dev_steps)
    assert _fields(r1) == _fields(r2)
    assert sum(1 for v in r1.vortices if v.origin == "cast") == 2


# ---------------------------------------------------------------- validators


def test_validator_rejects_over_cap_latitude():
    with pytest.raises(ValidationError, match="exchange band"):
        StormsParams(cast=[StormOverride(lat_deg=67.0, radius=0.15)])


def test_validator_rejects_too_many_entries():
    with pytest.raises(ValidationError, match="the cap is 16"):
        StormsParams(cast=[StormOverride() for _ in range(17)])


def test_validator_rejects_bad_kind():
    with pytest.raises(ValidationError):
        StormOverride(kind="banana")


def test_validator_accepts_full_cap():
    """Exactly 16 entries at legal latitudes validate (boundary)."""
    StormsParams(cast=[StormOverride(lat_deg=10.0) for _ in range(16)])


# ---------------------------------------------------------------- trim priority


def test_cast_trim_priority():
    """Near MAX_VORTICES the cast storm survives and a NEWEST non-cast entry is
    dropped to make room (cast entries are never trimmed)."""
    p = PlanetParams()
    p.bands.count = 40
    p.storms.oval_density = 3.0
    p.storms.small_density = 3.0
    seed = 7
    bands, prof = _profiles(seed, p)
    dt = _dt(p, prof)

    reg0 = generate_vortices(seed, bands, prof, p.storms, None, dt=dt,
                             dev_steps=p.sim.dev_steps)
    assert len(reg0.vortices) == MAX_VORTICES  # seeded population already at cap

    p.storms.cast = [StormOverride(kind=CastKind.OVAL, lat_deg=5.0, lon_deg=10.0)]
    reg1 = generate_vortices(seed, bands, prof, p.storms, None, dt=dt,
                             dev_steps=p.sim.dev_steps)
    # Still capped, but now a cast storm is present -- a non-cast was displaced.
    assert len(reg1.vortices) == MAX_VORTICES
    casts = [v for v in reg1.vortices if v.origin == "cast"]
    assert len(casts) == 1
    assert np.isclose(casts[0].lat, np.deg2rad(5.0))


# ---------------------------------------------------------------- merger exemption


def test_cast_merger_exemption():
    """A cast oval converging on a seeded oval at merge_rate>0 does NOT merge
    (cast is exempt); the identical all-seeded geometry DOES merge."""
    a = Vortex(0.30, 0.00, 0.03, 0.012, KIND_OVAL, tint=0.1, brightness=0.2,
               origin="cast")
    b = Vortex(0.33, -0.05, 0.03, 0.010, KIND_OVAL, tint=0.3, brightness=0.3,
               origin="seeded")
    reg = VortexRegistry([a, b])
    resolved = resolve_mergers(reg, _synth_profiles(), _merge_storms())
    assert resolved == []
    assert len(reg.vortices) == 2

    # Control: same geometry, both seeded -> the merge fires.
    a2 = Vortex(0.30, 0.00, 0.03, 0.012, KIND_OVAL)
    b2 = Vortex(0.33, -0.05, 0.03, 0.010, KIND_OVAL)
    reg2 = VortexRegistry([a2, b2])
    assert len(resolve_mergers(reg2, _synth_profiles(), _merge_storms())) == 1


# ---------------------------------------------------------------- preset round-trip


def test_preset_roundtrip_with_cast(tmp_path):
    from gasgiant.params.presets import load_preset, save_preset

    p = PlanetParams()
    p.storms.cast = [
        StormOverride(kind=CastKind.HERO, lat_deg=-20.0, lon_deg=45.0, radius=0.10),
        StormOverride(kind=CastKind.BARGE, lat_deg=15.0, lon_deg=-30.0, tint=0.5),
        StormOverride(kind=CastKind.OVAL, lat_deg=0.0, lon_deg=0.0, brightness=-0.3,
                      aspect=2.0),
    ]
    path = tmp_path / "cast.json"
    save_preset(p, path)
    loaded = load_preset(path)
    assert loaded.storms.cast == p.storms.cast

    # Empty cast round-trips as empty.
    p2 = PlanetParams()
    save_preset(p2, path)
    assert load_preset(path).storms.cast == []


# ---------------------------------------------------------------- checkpoint


def test_checkpoint_rejects_version_6(tmp_path):
    """A version-6 checkpoint (pre-cast registry) is rejected: its tracers would
    pair with a cast-unaware registry."""
    from gasgiant.engine.checkpoint import load_checkpoint

    path = tmp_path / "v6.npz"
    np.savez(path, generation_version=6)
    with pytest.raises(ValueError, match="generation_version"):
        load_checkpoint(path)


@pytest.mark.gpu
def test_cast_origin_survives_checkpoint(gpu, tmp_path):
    """The CPU-side origin marker round-trips through a checkpoint save/load."""
    from gasgiant.engine import Simulation
    from gasgiant.engine.checkpoint import load_checkpoint, save_checkpoint

    p = PlanetParams(seed=9)
    p.sim.resolution = 512
    p.sim.dev_steps = 20
    p.storms.cast = [StormOverride(kind=CastKind.OVAL, lat_deg=5.0, lon_deg=10.0)]
    sim = Simulation(p, gpu)
    path = tmp_path / "cast.npz"
    save_checkpoint(sim, path)
    restored = load_checkpoint(path, gpu)

    before = [v.origin for v in sim.solver.vortices.vortices]
    after = [v.origin for v in restored.solver.vortices.vortices]
    assert before == after
    assert after.count("cast") == 1


# ---------------------------------------------------------------- panels


def test_leaf_kind_classifies_cast_as_model_list():
    panels = pytest.importorskip("gasgiant.app.panels")
    info = StormsParams.model_fields["cast"]
    # Classifies from the annotation, so the EMPTY default still resolves.
    assert panels.leaf_kind("cast", info, []) == "model_list"


@pytest.fixture
def imgui_ctx():
    imgui = pytest.importorskip("imgui_bundle.imgui")
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(800.0, 600.0)
    io.delta_time = 1.0 / 60.0
    io.set_ini_filename(None)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    yield imgui
    imgui.destroy_context(ctx)


def test_draw_cast_list_add_row(imgui_ctx, monkeypatch):
    """Clicking "add storm" appends one valid StormOverride dict."""
    panels = pytest.importorskip("gasgiant.app.panels")
    imgui = imgui_ctx

    rows: list = []
    monkeypatch.setattr(panels.imgui, "small_button", lambda label: True)

    imgui.new_frame()
    imgui.begin("cast_test", None, 0)
    changed, committed = panels._draw_cast_list("cast", rows)
    imgui.end()
    imgui.end_frame()

    assert (changed, committed) == (True, True)
    assert len(rows) == 1
    StormOverride.model_validate(rows[0])  # the appended dict is a valid entry
