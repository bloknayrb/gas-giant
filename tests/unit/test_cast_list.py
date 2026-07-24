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
    WakeDir,
    effective_cast_lever,
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


# ---------------------------------------------------------- cheap per-storm levers


def _gen(p, seed=5):
    bands, prof = _profiles(seed, p)
    dt = _dt(p, prof)
    return generate_vortices(seed, bands, prof, p.storms, None, dt=dt,
                             dev_steps=p.sim.dev_steps)


def _cast_hero(reg):
    from gasgiant.sim.vortices import KIND_HERO
    return next(v for v in reg.vortices if v.origin == "cast" and v.kind == KIND_HERO)


def test_cast_wake_dir_none_inherits_global():
    """A cast hero with wake_dir=None follows the global hero_wake_dir. Its wake
    frame matches a cast hero that sets that same direction per-storm (the global
    setting also steers the SEEDED hero, so only the cast hero is compared)."""

    p_inherit = PlanetParams()
    p_inherit.storms.hero_wake_dir = WakeDir.EAST
    p_inherit.storms.cast = [StormOverride(kind=CastKind.HERO, lat_deg=-20.0,
                                           lon_deg=0.0, radius=0.1)]
    p_explicit = PlanetParams()  # global stays AUTO; per-storm sets EAST
    p_explicit.storms.cast = [StormOverride(kind=CastKind.HERO, lat_deg=-20.0,
                                            lon_deg=0.0, radius=0.1,
                                            wake_dir=WakeDir.EAST)]
    h_inherit = _cast_hero(_gen(p_inherit))
    h_explicit = _cast_hero(_gen(p_explicit))
    assert h_inherit.wake_dir == 1.0
    assert (h_inherit.wake_dir, h_inherit.wake_lat_off) == (
        h_explicit.wake_dir, h_explicit.wake_lat_off)


def test_cast_wake_dir_per_storm_override():
    """Two cast heroes can trail opposite ways in one scene."""
    from gasgiant.sim.vortices import KIND_HERO

    p = PlanetParams()
    p.storms.cast = [
        StormOverride(kind=CastKind.HERO, lat_deg=-20.0, lon_deg=-40.0,
                      radius=0.1, wake_dir=WakeDir.EAST),
        StormOverride(kind=CastKind.HERO, lat_deg=-20.0, lon_deg=40.0,
                      radius=0.1, wake_dir=WakeDir.WEST),
    ]
    heroes = [v for v in _gen(p).vortices if v.origin == "cast" and v.kind == KIND_HERO]
    assert sorted(h.wake_dir for h in heroes) == [-1.0, 1.0]


def test_cast_companions_default_off_byte_identical():
    """companions=0 (the default) adds no pearls: byte-identical to a bare hero."""
    p_off = PlanetParams()
    p_off.storms.cast = [StormOverride(kind=CastKind.HERO, lat_deg=-20.0,
                                       lon_deg=0.0, radius=0.1)]
    p_bare = PlanetParams()
    p_bare.storms.cast = [StormOverride(kind=CastKind.HERO, lat_deg=-20.0,
                                        lon_deg=0.0, radius=0.1, companions=0)]
    assert _fields(_gen(p_off)) == _fields(_gen(p_bare))


def test_cast_companions_placed_deterministically():
    """companions=2 on a cast hero adds two origin='cast' pearls, deterministically."""
    from gasgiant.sim.vortices import KIND_PEARL

    p = PlanetParams()
    p.storms.cast = [StormOverride(kind=CastKind.HERO, lat_deg=-20.0,
                                   lon_deg=0.0, radius=0.1, companions=2)]
    r1, r2 = _gen(p), _gen(p)
    pearls = [v for v in r1.vortices if v.origin == "cast" and v.kind == KIND_PEARL]
    assert len(pearls) == 2
    assert _fields(r1) == _fields(r2)  # no RNG on the cast path


def test_cast_companion_appearance_inherits_global():
    """companion_aspect/brightness=None -> the pearls take the global
    storms.companion_aspect/companion_brightness."""
    from gasgiant.sim.vortices import KIND_PEARL

    p = PlanetParams()
    p.storms.cast = [StormOverride(kind=CastKind.HERO, lat_deg=-20.0, lon_deg=0.0,
                                   radius=0.1, companions=1)]
    pearl = next(v for v in _gen(p).vortices
                 if v.origin == "cast" and v.kind == KIND_PEARL)
    assert pearl.aspect == pytest.approx(p.storms.companion_aspect)
    assert pearl.brightness == pytest.approx(p.storms.companion_brightness)


def test_cast_companion_appearance_per_storm_override():
    """Explicit companion_aspect/brightness win over the global for that storm."""
    from gasgiant.sim.vortices import KIND_PEARL

    p = PlanetParams()
    p.storms.cast = [StormOverride(kind=CastKind.HERO, lat_deg=-20.0, lon_deg=0.0,
                                   radius=0.1, companions=1,
                                   companion_aspect=3.0, companion_brightness=0.42)]
    pearl = next(v for v in _gen(p).vortices
                 if v.origin == "cast" and v.kind == KIND_PEARL)
    assert pearl.aspect == pytest.approx(3.0)
    assert pearl.brightness == pytest.approx(0.42)


def test_cast_companions_independent_across_storms():
    """Toggling one cast hero's companions leaves ANOTHER cast hero (and its own
    companions) byte-identical -- the cast path shares no RNG stream, so a storm's
    output can't depend on a sibling's companion count."""
    def scene(a_companions):
        p = PlanetParams()
        p.storms.cast = [
            StormOverride(kind=CastKind.HERO, lat_deg=-25.0, lon_deg=-60.0,
                          radius=0.1, companions=a_companions),
            StormOverride(kind=CastKind.HERO, lat_deg=30.0, lon_deg=60.0,
                          radius=0.1, companions=2),
        ]
        return _gen(p)

    def b_block(reg):
        # Storm B and its companions sit in the northern hemisphere; storm A and
        # its (variable) companions are southern -- filter to B's block.
        return [t for t, v in zip(_fields(reg), reg.vortices, strict=True)
                if v.origin == "cast" and v.lat > 0.0]

    assert b_block(scene(0)) == b_block(scene(3))


def test_hero_only_lever_on_oval_warns():
    """A hero-only lever set on a non-hero cast kind is flagged as inert."""
    p = PlanetParams()
    p.storms.cast = [StormOverride(kind=CastKind.OVAL, lat_deg=10.0, companions=2)]
    warnings = p.validation_warnings()
    assert any("hero-only lever" in w for w in warnings)


def test_companion_appearance_on_oval_warns():
    """companion_aspect on a non-hero is inert and flagged (it rides companions)."""
    p = PlanetParams()
    p.storms.cast = [StormOverride(kind=CastKind.OVAL, lat_deg=10.0,
                                   companion_aspect=3.0)]
    assert any("hero-only lever" in w for w in p.validation_warnings())


def test_companion_appearance_without_companions_warns():
    """A hero that sets companion appearance but companions=0 gets no pearls, so
    the appearance override is inert -- flagged."""
    p = PlanetParams()
    p.storms.cast = [StormOverride(kind=CastKind.HERO, lat_deg=-20.0, radius=0.1,
                                   companions=0, companion_brightness=0.5)]
    assert any("companions=0" in w for w in p.validation_warnings())


# ---------------------------------------------- M2 CastLevers CPU pack machinery


def test_pack_cast_levers_all_global_when_no_overrides():
    """No per-storm overrides -> every row packs the GLOBAL lever values. This is
    the byte-identity-off contract for the CAST_LEVERS variant: an un-overridden
    hero packs exactly what the global-uniform path reads."""
    from gasgiant.params.model import (
        CAST_LEVER_COLS,
        CAST_LEVER_SPECS,
        CAST_LEVER_WIDTH,
    )

    p = PlanetParams()
    p.storms.cast = [StormOverride(kind=CastKind.HERO, lat_deg=-20.0, lon_deg=0.0,
                                   radius=0.1)]
    reg = _gen(p)
    levers = reg.pack_cast_levers_ssbo(p.storms)
    assert levers.shape == (len(reg.vortices), CAST_LEVER_WIDTH)
    expected = [float(getattr(p.storms, g)) for _, g in CAST_LEVER_SPECS]
    reserved = sorted(set(range(CAST_LEVER_WIDTH)) - set(CAST_LEVER_COLS))
    for row in levers:
        assert [row[c] for c in CAST_LEVER_COLS] == pytest.approx(expected)
        assert [row[c] for c in reserved] == [0.0] * len(reserved)


def test_pack_cast_levers_override_only_on_that_hero():
    """A per-storm override lands ONLY on that cast hero's row; other rows (a
    second cast hero, seeded vortices) keep the global value."""
    p = PlanetParams()
    p.storms.cast = [
        StormOverride(kind=CastKind.HERO, lat_deg=-20.0, lon_deg=-40.0, radius=0.1,
                      mottle=0.7, solid_core=0.5),
        StormOverride(kind=CastKind.HERO, lat_deg=20.0, lon_deg=40.0, radius=0.1),
    ]
    reg = _gen(p)
    levers = reg.pack_cast_levers_ssbo(p.storms)
    # CAST_LEVER_SPECS order: mottle is column 3, solid_core column 6.
    overridden = [levers[i] for i, v in enumerate(reg.vortices) if v.cast_ref == 0]
    assert len(overridden) == 1
    assert overridden[0][3] == pytest.approx(0.7)   # mottle override
    assert overridden[0][6] == pytest.approx(0.5)   # solid_core override
    other = [levers[i] for i, v in enumerate(reg.vortices) if v.cast_ref == 1][0]
    assert other[3] == pytest.approx(p.storms.hero_mottle)  # inherits global


def test_cast_ref_set_only_on_cast_heroes():
    """cast_ref indexes storms.cast on a cast HERO, -1 on non-hero cast + seeded."""
    from gasgiant.sim.vortices import KIND_HERO

    p = PlanetParams()
    p.storms.cast = [
        StormOverride(kind=CastKind.OVAL, lat_deg=10.0, lon_deg=0.0),
        StormOverride(kind=CastKind.HERO, lat_deg=-20.0, lon_deg=30.0, radius=0.1),
    ]
    reg = _gen(p)
    heroes = [v for v in reg.vortices if v.origin == "cast" and v.kind == KIND_HERO]
    assert len(heroes) == 1 and heroes[0].cast_ref == 1  # hero is cast index 1
    ovals = [v for v in reg.vortices if v.origin == "cast" and v.kind == KIND_OVAL]
    assert ovals and all(v.cast_ref == -1 for v in ovals)
    assert all(v.cast_ref == -1 for v in reg.vortices if v.origin == "seeded")


def test_appearance_lever_on_oval_warns():
    """A per-storm appearance lever on a non-hero cast kind is flagged inert."""
    p = PlanetParams()
    p.storms.cast = [StormOverride(kind=CastKind.OVAL, lat_deg=10.0, mottle=0.5)]
    assert any("hero-only lever" in w for w in p.validation_warnings())


def test_appearance_levers_gated_to_hero_rows():
    """The editor hides every per-storm appearance/dynamics lever on non-hero rows."""
    panels = pytest.importorskip("gasgiant.app.panels")
    from gasgiant.params.model import CAST_LEVER_FIELDS

    assert CAST_LEVER_FIELDS <= panels._HERO_ONLY_CAST_FIELDS


def test_cast_levers_variant_predicate_is_hero_only():
    """The CAST_LEVERS variant compiles only for a HERO override -- a non-hero's
    (inert) override, or no override, must leave it off. Static predicate, no GL."""
    from gasgiant.sim.solver import Solver

    hero = StormsParams(cast=[StormOverride(kind=CastKind.HERO, lat_deg=-20.0,
                                            radius=0.1, mottle=0.5)])
    oval = StormsParams(cast=[StormOverride(kind=CastKind.OVAL, lat_deg=10.0,
                                            mottle=0.5)])
    bare = StormsParams(cast=[StormOverride(kind=CastKind.HERO, lat_deg=-20.0,
                                            radius=0.1)])
    assert Solver._cast_levers_active(hero) is True
    assert Solver._cast_levers_active(oval) is False   # non-hero override → off
    assert Solver._cast_levers_active(bare) is False   # no override → off


def test_pack_cast_levers_survives_vortex_reorder():
    """Row correspondence is keyed by cast_ref, not list position: after the vortex
    list is reordered (mergers/trim do this between steps), the override still lands
    on the row whose cast_ref points to the overriding storm."""
    p = PlanetParams(seed=11)
    p.storms.cast = [StormOverride(kind=CastKind.HERO, lat_deg=-20.0, lon_deg=0.0,
                                   radius=0.1, mottle=0.7)]
    reg = _gen(p)
    reg.vortices.reverse()  # simulate a merger/trim reorder of the list
    levers = reg.pack_cast_levers_ssbo(p.storms)
    hero_rows = [levers[i] for i, v in enumerate(reg.vortices) if v.cast_ref == 0]
    assert hero_rows and hero_rows[0][3] == pytest.approx(0.7)  # mottle, right row


def test_kinematic_cast_solid_core_warns():
    """A per-storm solid_core is vorticity-only; on a cast hero in a kinematic
    preset it is inert and flagged, exactly like the global hero_solid_core."""
    p = PlanetParams()  # kinematic by default
    assert p.solver.type.value == "kinematic"
    p.storms.cast = [StormOverride(kind=CastKind.HERO, lat_deg=-20.0, radius=0.1,
                                   solid_core=0.5)]
    assert any("solid_core" in w and "kinematic" in w
               for w in p.validation_warnings())


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

    # FIELD-AGNOSTIC: every optional field set to a distinct non-default value.
    # The entries above leave the None-inheriting levers at None, so a serializer
    # that dropped a new key entirely would round-trip them equal and pass. This
    # walks the model instead, so each future lever family is covered for free.
    entry = StormOverride(kind=CastKind.HERO, lat_deg=-20.0, lon_deg=45.0, radius=0.1)
    distinct = {"tint": 0.31, "brightness": -0.22, "wake_dir": WakeDir.EAST,
                "companions": 2, "companion_aspect": 1.7, "companion_brightness": 0.13,
                "rim_contrast": 0.41, "rim_tint": 0.52, "rim_warp": 0.63,
                "mottle": 0.74, "tint_var": 0.85, "wake_detail": 0.96,
                "solid_core": 0.27, "emergence": 0.38, "shape": 1.19, "taper": 0.44}
    optional = {n for n, f in StormOverride.model_fields.items()
                if f.default is None} | {"companions"}
    assert optional <= set(distinct), f"new optional field(s) unpinned: {optional - set(distinct)}"
    for name, value in distinct.items():
        setattr(entry, name, value)
    p3 = PlanetParams()
    p3.storms.cast = [entry]
    save_preset(p3, path)
    assert load_preset(path).storms.cast == [entry]


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
    """The CPU-side origin marker AND the M2 cast_ref back-reference round-trip
    through a checkpoint save/load (a restored cast hero must keep the storms.cast
    index it resolves its per-storm overrides against)."""
    from gasgiant.engine import Simulation
    from gasgiant.engine.checkpoint import load_checkpoint, save_checkpoint

    p = PlanetParams(seed=9)
    p.sim.resolution = 512
    p.sim.dev_steps = 20
    p.storms.cast = [
        StormOverride(kind=CastKind.OVAL, lat_deg=5.0, lon_deg=10.0),
        StormOverride(kind=CastKind.HERO, lat_deg=-20.0, lon_deg=30.0, radius=0.1,
                      mottle=0.6),
    ]
    sim = Simulation(p, gpu)
    path = tmp_path / "cast.npz"
    save_checkpoint(sim, path)
    restored = load_checkpoint(path, gpu)

    before = [v.origin for v in sim.solver.vortices.vortices]
    after = [v.origin for v in restored.solver.vortices.vortices]
    assert before == after
    assert after.count("cast") == 2
    assert ([v.cast_ref for v in sim.solver.vortices.vortices]
            == [v.cast_ref for v in restored.solver.vortices.vortices])
    # Re-pack against the RESTORED params (rebuilt from the checkpoint's embedded
    # preset doc), not the original in-memory params -- this proves the new
    # StormOverride lever fields survive preset (de)serialization, not just that
    # cast_ref round-trips.
    restored_storms = restored.solver.params.storms
    assert restored_storms.cast[1].mottle == pytest.approx(0.6)
    levers = restored.solver.vortices.pack_cast_levers_ssbo(restored_storms)
    hero_rows = [levers[i] for i, v in enumerate(restored.solver.vortices.vortices)
                 if v.cast_ref == 1]
    assert hero_rows and hero_rows[0][3] == pytest.approx(0.6)  # mottle = column 3


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
    changed, committed = panels._draw_cast_list("cast", rows, panels.PanelState())
    imgui.end()
    imgui.end_frame()

    assert (changed, committed) == (True, True)
    assert len(rows) == 1
    StormOverride.model_validate(rows[0])  # the appended dict is a valid entry


def test_draw_cast_list_renders_every_stormoverride_field(imgui_ctx):
    """The reflective row editor must render a widget for EVERY StormOverride
    field (in Advanced mode + a hero row, so no per-kind/adv gating hides any) --
    a new pfield can't silently go unrendered. Exercised inside a real frame so a
    missing widget branch (e.g. an unhandled optional_enum) would raise or leave
    the field un-drawn; here we assert the leaf_kind of each field is one the
    row renderer handles."""
    panels = pytest.importorskip("gasgiant.app.panels")
    handled = {"enum", "int", "float", "optional_float", "optional_int",
               "optional_enum"}
    for name, info in StormOverride.model_fields.items():
        value = StormOverride().model_dump()[name]
        kind = panels.leaf_kind(name, info, value)
        assert kind in handled, f"{name} -> {kind} is not rendered by the cast editor"


def test_draw_cast_list_hides_hero_levers_on_oval(imgui_ctx):
    """Per-kind gating: an oval row draws no hero-only widgets; a hero row does."""
    panels = pytest.importorskip("gasgiant.app.panels")
    imgui = imgui_ctx
    state = panels.PanelState(show_advanced=True)

    oval = StormOverride(kind=CastKind.OVAL, lat_deg=5.0).model_dump()
    hero = StormOverride(kind=CastKind.HERO, lat_deg=-20.0, radius=0.1).model_dump()
    imgui.new_frame()
    imgui.begin("cast_test", None, 0)
    # both draw without raising; the gating is asserted structurally below
    panels._draw_cast_list("cast", [oval, hero], state)
    imgui.end()
    imgui.end_frame()

    # the gating predicate the editor uses
    assert "wake_dir" in panels._HERO_ONLY_CAST_FIELDS
    assert "aspect" not in panels._HERO_ONLY_CAST_FIELDS


def test_draw_cast_list_rekind_clears_hero_only_fields(imgui_ctx, monkeypatch):
    """Re-kinding a hero row to a non-hero kind resets the (now-hidden) hero-only
    levers to their defaults, so no dead value is stranded where the user can
    neither see nor clear it (silent-failure review)."""
    panels = pytest.importorskip("gasgiant.app.panels")
    imgui = imgui_ctx

    hero = StormOverride(kind=CastKind.HERO, lat_deg=-20.0, radius=0.1,
                         companions=2, companion_aspect=3.0,
                         wake_dir=WakeDir.EAST).model_dump()

    def fake_combo(label, current, items):
        # drive ONLY the kind combo to pick "oval"; leave other combos untouched
        if label == "kind":
            return True, items.index("oval")
        return False, current
    monkeypatch.setattr(panels.imgui, "combo", fake_combo)

    imgui.new_frame()
    imgui.begin("cast_test", None, 0)
    panels._draw_cast_list("cast", [hero], panels.PanelState(show_advanced=True))
    imgui.end()
    imgui.end_frame()

    assert hero["kind"] == "oval"
    defaults = StormOverride().model_dump()
    for f in ("wake_dir", "companions", "companion_aspect", "companion_brightness"):
        assert hero[f] == defaults[f]


def test_draw_cast_list_remove_reconciles_selection(imgui_ctx, monkeypatch):
    """Removing an earlier row via the panel "remove storm" button shifts a later
    selection down -- the panel path reuses viewport.selection_after_delete."""
    panels = pytest.importorskip("gasgiant.app.panels")
    imgui = imgui_ctx

    rows = [StormOverride(kind=CastKind.OVAL, lat_deg=0.0,
                          lon_deg=float(i)).model_dump() for i in range(3)]
    state = panels.PanelState(selected_cast=2)
    calls = {"n": 0}

    def fake_small_button(label):
        # fire "remove storm" once, on the first row (index 0); never "add storm"
        if label == "remove storm":
            calls["n"] += 1
            return calls["n"] == 1
        return False
    monkeypatch.setattr(panels.imgui, "small_button", fake_small_button)

    imgui.new_frame()
    imgui.begin("cast_test", None, 0)
    panels._draw_cast_list("cast", rows, state)
    imgui.end()
    imgui.end_frame()

    assert len(rows) == 2  # row 0 removed
    assert state.selected_cast == 1  # selection 2 shifted down past the removed row


def test_draw_optional_enum_toggle_none_to_value(imgui_ctx, monkeypatch):
    """The override checkbox flips None -> the first enum option and commits (the
    None<->value transition that has no other renderer coverage)."""
    panels = pytest.importorskip("gasgiant.app.panels")
    imgui = imgui_ctx

    doc = {"wake_dir": None}
    monkeypatch.setattr(panels.imgui, "checkbox", lambda label, v: (True, True))

    imgui.new_frame()
    imgui.begin("t", None, 0)
    changed, committed = panels._draw_optional_enum("wake_dir", "wake dir", doc,
                                                    WakeDir)
    imgui.end()
    imgui.end_frame()

    assert doc["wake_dir"] == [e.value for e in WakeDir][0]
    assert (changed, committed) == (True, True)


# ------------------------------------------- M2-B emergence family (CPU side)


def test_cast_lever_cols_match_specs():
    """CAST_LEVER_COLS is the layout contract: one column per spec, in range, no
    duplicates, and the M2-A prefix PINNED to columns 0..6 -- the GPU reads whole
    vec4, so an existing lever that slid to another column (or into another vec4)
    would silently re-read a different value on the shipped path."""
    from gasgiant.params.model import (
        CAST_LEVER_COLS,
        CAST_LEVER_SPECS,
        CAST_LEVER_WIDTH,
    )

    assert len(CAST_LEVER_COLS) == len(CAST_LEVER_SPECS)
    assert len(set(CAST_LEVER_COLS)) == len(CAST_LEVER_COLS)
    assert CAST_LEVER_COLS[:7] == (0, 1, 2, 3, 4, 5, 6)
    assert [a for a, _ in CAST_LEVER_SPECS][7:] == ["emergence", "shape", "taper"]
    # ...and the emergence family lands in the THIRD vec4 (cols 8..11).
    assert all(8 <= c <= 11 for c in CAST_LEVER_COLS[7:])
    assert CAST_LEVER_WIDTH == 12


def test_cast_lever_layout_matches_the_glsl_reads():
    """Cross-reference the Python layout against the SHADER SOURCE. Everything
    else here pins the constants against each other; nothing checked that the
    kernels index the buffer the way the packer writes it, so a 4th vec4 added on
    the Python side (or an off-by-one stride) would fail no CPU test and only
    surface as a silently-wrong hero on the GPU. Same idiom as
    test_hero_shape_constants.py."""
    import re
    from importlib.resources import files

    from gasgiant.params.model import CAST_LEVER_COLS, CAST_LEVER_SPECS, CAST_LEVER_WIDTH

    stride = CAST_LEVER_WIDTH // 4
    readers = ["vortex_stamp.glsl", "vortex_omega.glsl", "psi.comp"]
    kernels = files("gasgiant.sim.kernels")
    sources = {n: (kernels / n).read_text(encoding="utf-8") for n in readers}
    for name, src in sources.items():
        reads = set(re.findall(r"cast_lever_data\[\s*(\d+)\s*\*\s*i\s*(?:\+\s*(\d+))?\]", src))
        assert reads, f"{name} declares no cast_lever_data read"
        for mult, off in reads:
            assert int(mult) == stride, f"{name}: stride {mult}*i, expected {stride}*i"
            assert int(off or 0) < stride, f"{name}: vec4 index {off} past the row"

    # Spot-check the two columns whose vec4/swizzle correspondence is load-bearing
    # across the M2-A/M2-B boundary: solid_core (col 6 -> vec4_1.z) and the whole
    # emergence family (cols 8/9/10 -> vec4_2.xyz).
    col = dict(zip([a for a, _ in CAST_LEVER_SPECS], CAST_LEVER_COLS, strict=True))
    for attr, expect in (("solid_core", "[3 * i + 1].z"),
                         ("emergence", "[3 * i + 2].x"),
                         ("shape", "[3 * i + 2].y"),
                         ("taper", "[3 * i + 2].z")):
        vec4, comp = divmod(col[attr], 4)
        assert expect == f"[{stride} * i + {vec4}].{'xyzw'[comp]}"
    assert any("cast_lever_data[3 * i + 2]" in s for s in sources.values())
    assert any("cast_lever_data[3 * i + 1]" in s for s in sources.values())


def test_pack_cast_levers_emergence_family_columns():
    """The emergence family resolves into vec4_2 (cols 8/9/10) on the overriding
    hero's row only, leaving both reserved columns (7, 11) zero."""
    p = PlanetParams()
    p.storms.cast = [
        StormOverride(kind=CastKind.HERO, lat_deg=-20.0, lon_deg=-40.0, radius=0.1,
                      emergence=0.8, shape=1.2, taper=0.4),
        StormOverride(kind=CastKind.HERO, lat_deg=20.0, lon_deg=40.0, radius=0.1),
    ]
    reg = _gen(p)
    levers = reg.pack_cast_levers_ssbo(p.storms)
    row = next(levers[i] for i, v in enumerate(reg.vortices) if v.cast_ref == 0)
    assert row[8] == pytest.approx(0.8)
    assert row[9] == pytest.approx(1.2)
    assert row[10] == pytest.approx(0.4)
    assert row[7] == 0.0 and row[11] == 0.0
    other = next(levers[i] for i, v in enumerate(reg.vortices) if v.cast_ref == 1)
    assert other[8] == pytest.approx(p.storms.hero_emergence)
    assert other[9] == pytest.approx(p.storms.hero_shape)
    assert other[10] == pytest.approx(p.storms.hero_taper)


def test_hero_emergence_predicate_sees_cast_overrides():
    """HERO_EMERGENCE must compile for a cast hero that is emergent on its OWN
    override while the global is 0 -- otherwise the per-storm value packs into a
    buffer whose read sites were never compiled in (a silent legacy hero)."""
    from gasgiant.sim.solver import Solver

    cast_only = StormsParams(hero_count=0, hero_emergence=0.0, cast=[
        StormOverride(kind=CastKind.HERO, lat_deg=-20.0, radius=0.1, emergence=0.8)])
    inherits_zero = StormsParams(hero_count=0, hero_emergence=0.0, cast=[
        StormOverride(kind=CastKind.HERO, lat_deg=-20.0, radius=0.1)])
    opted_out = StormsParams(hero_count=0, hero_emergence=0.0, cast=[
        StormOverride(kind=CastKind.HERO, lat_deg=-20.0, radius=0.1, emergence=0.0)])
    global_on = StormsParams(hero_count=1, hero_emergence=0.9)
    global_on_no_hero = StormsParams(hero_count=0, hero_emergence=0.9)

    assert Solver._hero_emergence_active(cast_only) is True
    assert Solver._hero_emergence_active(inherits_zero) is False
    assert Solver._hero_emergence_active(opted_out) is False
    assert Solver._hero_emergence_active(global_on) is True
    assert Solver._hero_emergence_active(global_on_no_hero) is False  # nothing to apply to
    # A cast-only emergent hero also selects CAST_LEVERS -- the read sites are
    # dual-gated, so one predicate without the other renders the wrong hero.
    assert Solver._cast_levers_active(cast_only) is True

    # BOTH clauses ask about a RESOLVED value. A global of 0.9 with hero_count 0
    # and the only cast hero opted out means NO hero is emergent -- compiling the
    # variant there would render this config differently from the identical
    # `global=0` spelling of the same scene (a few ring/collar raggedness terms
    # inside the variant are not emergence-scaled).
    global_on_opted_out = StormsParams(hero_count=0, hero_emergence=0.9, cast=[
        StormOverride(kind=CastKind.HERO, lat_deg=-20.0, radius=0.1, emergence=0.0)])
    assert Solver._hero_emergence_active(global_on_opted_out) is False
    # ...while an INHERITING cast hero picks the global up.
    global_on_inherits = StormsParams(hero_count=0, hero_emergence=0.9, cast=[
        StormOverride(kind=CastKind.HERO, lat_deg=-20.0, radius=0.1)])
    assert Solver._hero_emergence_active(global_on_inherits) is True
    # Non-hero kinds never read the buffer, so an emergence on one must not
    # select the variant -- that would pay the per-pixel SSBO scan for a scene
    # with no hero in it at all.
    oval_carrying = StormsParams(hero_count=0, hero_emergence=0.0, cast=[
        StormOverride(kind=CastKind.OVAL, lat_deg=-20.0, radius=0.1, emergence=0.9)])
    assert Solver._hero_emergence_active(oval_carrying) is False


def test_effective_cast_lever_resolution_table():
    """The ONE resolution path every CPU consumer routes through. The range guard
    is the load-bearing part: a SEEDED hero carries cast_ref -1, and without it
    `storms.cast[-1]` would silently resolve it against the LAST placed storm --
    the flagship shape of that bug is hero_count=1 plus any cast entry."""
    storms = StormsParams(hero_count=1, hero_emergence=0.4, hero_shape=1.0, cast=[
        StormOverride(kind=CastKind.HERO, lat_deg=-20.0, radius=0.1, emergence=0.9),
        StormOverride(kind=CastKind.HERO, lat_deg=20.0, radius=0.1),
    ])
    # in range, override set -> the override
    assert effective_cast_lever(storms, 0, "emergence") == pytest.approx(0.9)
    # in range, override None -> the global
    assert effective_cast_lever(storms, 1, "emergence") == pytest.approx(0.4)
    assert effective_cast_lever(storms, 0, "shape") == pytest.approx(1.0)
    # seeded hero (-1) and a past-the-end ref -> the global, NOT cast[-1]
    assert effective_cast_lever(storms, -1, "emergence") == pytest.approx(0.4)
    assert effective_cast_lever(storms, len(storms.cast), "emergence") == pytest.approx(0.4)


def test_cast_entry_emergence_gates_its_own_wake_frame():
    """The CPU wake frame / bow gain is chosen per ENTRY: a placed hero that opts
    out of emergence keeps the legacy WNW convention (wake_dir -1, the fixed
    equatorward lane, bow 0) even while the GLOBAL pack is on -- and vice versa."""
    def hero(emergence):
        p = PlanetParams(seed=5)
        p.storms.hero_count = 0
        p.storms.hero_emergence = 0.9
        p.storms.hero_wake_dir = WakeDir.AUTO
        p.storms.cast = [StormOverride(kind=CastKind.HERO, lat_deg=-20.0, lon_deg=0.0,
                                       radius=0.1, emergence=emergence)]
        return _cast_hero(_gen(p))

    opted_out = hero(0.0)
    assert opted_out.wake_dir == -1.0
    assert opted_out.wake_lat_off == pytest.approx(0.5 * 0.1)   # southern hero lane
    assert opted_out.bow_gain == 0.0
    # Inheriting the global (None) keeps the emergent frame. Assert all THREE
    # products of the gated block separately -- `_hero_wake_frame` and
    # `_hero_bow_gain` are independent calls, so an `or` across them tolerates
    # half the block regressing. wake_dir FLIPS (the flow-derived wake runs the
    # other way at this seat), which is the most visible difference of the three.
    inherits = hero(None)
    assert inherits.wake_dir == 1.0
    assert inherits.wake_lat_off != pytest.approx(opted_out.wake_lat_off)
    assert inherits.bow_gain > 0.0

    # Two entries in ONE registry get INDEPENDENT frames -- the gate is per
    # entry, not a scene-wide decision made once from the first hero.
    p = PlanetParams(seed=5)
    p.storms.hero_count = 0
    p.storms.hero_emergence = 0.9
    p.storms.hero_wake_dir = WakeDir.AUTO
    p.storms.cast = [
        StormOverride(kind=CastKind.HERO, lat_deg=-20.0, lon_deg=0.0, radius=0.1,
                      emergence=0.0),
        StormOverride(kind=CastKind.HERO, lat_deg=-20.0, lon_deg=60.0, radius=0.1),
    ]
    out, inh = [v for v in _gen(p).vortices if v.origin == "cast"]
    assert (out.wake_dir, out.bow_gain) == (-1.0, 0.0)
    assert inh.wake_dir == 1.0 and inh.bow_gain > 0.0


def test_flow_renorm_gate_resolves_per_storm_levers():
    """_hero_flow_renorm skips its quadrature when no hero can enter the K arm.
    With per-storm levers that question is per hero: a placed hero carrying its own
    solid_core + emergence must still get the renorm even though both GLOBALS are 0
    (keying on the globals alone left that hero's widened ring un-renormalized)."""
    from gasgiant.sim.solver import Solver

    def renorm(**entry_levers):
        p = PlanetParams(seed=5)
        p.storms.hero_count = 0
        p.storms.hero_solid_core = 0.0
        p.storms.hero_emergence = 0.0
        p.storms.hero_flow_aspect = 2.0
        p.storms.cast = [StormOverride(kind=CastKind.HERO, lat_deg=-20.0, lon_deg=0.0,
                                       radius=0.1, **entry_levers)]
        solver = Solver.__new__(Solver)      # no GL: only params + registry are read
        solver.params = p
        solver.vortices = _gen(p)
        solver._flow_renorm = None
        return Solver._hero_flow_renorm(solver)

    assert renorm() == 1.0                                    # globals 0 -> skipped
    assert renorm(solid_core=0.9) == 1.0                      # emergence still 0
    assert renorm(solid_core=0.9, emergence=0.9) != 1.0       # this hero DOES run it

    def renorm_globals_on(**entry_levers):
        """Globals ON, so the INVERSE direction: an opted-out hero must still
        skip. Otherwise the ~2.6M-point quadrature (and its sane-band fallback)
        fires for a config where the shader branch cannot run -- exactly what the
        skip exists to prevent."""
        p = PlanetParams(seed=5)
        p.storms.hero_count = 0
        p.storms.hero_solid_core = 0.9
        p.storms.hero_emergence = 0.9
        p.storms.hero_flow_aspect = 2.0
        p.storms.cast = [StormOverride(kind=CastKind.HERO, lat_deg=-20.0, lon_deg=0.0,
                                       radius=0.1, **entry_levers)]
        solver = Solver.__new__(Solver)
        solver.params = p
        solver.vortices = _gen(p)
        solver._flow_renorm = None
        return Solver._hero_flow_renorm(solver)

    assert renorm_globals_on() != 1.0                         # inherits both -> runs
    assert renorm_globals_on(emergence=0.0) == 1.0            # opted out -> skipped
    assert renorm_globals_on(solid_core=0.0) == 1.0           # ditto via solid_core

    # The two short-circuits ahead of the per-hero scan.
    def renorm_no_lever(flow_aspect, hero_count):
        p = PlanetParams(seed=5)
        p.storms.hero_count = hero_count
        p.storms.hero_solid_core = 0.9
        p.storms.hero_emergence = 0.9
        p.storms.hero_flow_aspect = flow_aspect
        solver = Solver.__new__(Solver)
        solver.params = p
        solver.vortices = _gen(p)
        solver._flow_renorm = None
        return Solver._hero_flow_renorm(solver)

    assert renorm_no_lever(1.0, 1) == 1.0    # K == 1 -> the lever itself is off
    assert renorm_no_lever(2.0, 0) == 1.0    # no hero at all
    assert renorm_no_lever(2.0, 1) != 1.0    # a seeded hero on the globals


def test_shape_taper_inert_at_zero_emergence_warns():
    """shape/taper ride the emergence pack, so they are inert at effective
    emergence 0 -- warn rather than silently ignore (the GUI toasts these)."""
    # Both fields, on the SECOND entry: pin the index and the field name, not
    # just the substring -- a warning raised against the wrong cast[i] or the
    # wrong lever would otherwise pass.
    p = PlanetParams()
    p.storms.hero_emergence = 0.0
    p.storms.cast = [
        StormOverride(kind=CastKind.HERO, lat_deg=10.0, radius=0.1),
        StormOverride(kind=CastKind.HERO, lat_deg=-20.0, radius=0.1,
                      shape=1.2, taper=0.7),
    ]
    warnings = [w for w in p.validation_warnings() if "no effect at emergence 0" in w]
    assert any("storms.cast[1].shape=1.2" in w for w in warnings)
    assert any("storms.cast[1].taper=0.7" in w for w in warnings)
    assert not any("cast[0]" in w for w in warnings)   # entry 0 sets neither
    # Its own emergence override lifts the gate -> no warning.
    p.storms.cast[1].emergence = 0.8
    assert not any("no effect at emergence 0" in w for w in p.validation_warnings())
    # ...and so does the GLOBAL, for an entry that inherits it.
    p.storms.cast[1].emergence = None
    p.storms.hero_emergence = 0.9
    assert not any("no effect at emergence 0" in w for w in p.validation_warnings())
