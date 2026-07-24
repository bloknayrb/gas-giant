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
    from gasgiant.params.model import WakeDir

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
    from gasgiant.params.model import WakeDir
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
    from gasgiant.params.model import CAST_LEVER_SPECS

    p = PlanetParams()
    p.storms.cast = [StormOverride(kind=CastKind.HERO, lat_deg=-20.0, lon_deg=0.0,
                                   radius=0.1)]
    reg = _gen(p)
    levers = reg.pack_cast_levers_ssbo(p.storms)
    assert levers.shape == (len(reg.vortices), 8)
    expected = [float(getattr(p.storms, g)) for _, g in CAST_LEVER_SPECS]
    for row in levers:
        assert list(row[:7]) == pytest.approx(expected)
        assert row[7] == 0.0  # reserved column


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
    from gasgiant.params.model import WakeDir

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
    from gasgiant.params.model import WakeDir

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
