"""B1-7/B4-3: aurora rides the exported emission map's alpha channel. The
viewport composites it (alpha x aurora_color) into the Emission channel
preview, but it stays invisible in the Color preview -- the UI must say so
where the aurora controls live, and the disclosure must point at the Emission
channel now that it actually shows the aurora."""

from __future__ import annotations

import pytest

from gasgiant.params.model import EmissionParams, PlanetParams

panels = pytest.importorskip("gasgiant.app.panels")
imgui = pytest.importorskip("imgui_bundle.imgui")


@pytest.fixture
def imgui_ctx():
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(800.0, 600.0)
    io.delta_time = 1.0 / 60.0
    io.set_ini_filename(None)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    yield imgui
    imgui.destroy_context(ctx)


def test_aurora_strength_description_discloses_preview_limit():
    """The tooltip (pydantic description, rendered verbatim) must warn that
    the Color preview will not move AND point at the Emission channel, which
    now composites the aurora (B4-3)."""
    desc = EmissionParams.model_fields["aurora_strength"].description or ""
    assert "Color preview" in desc
    assert "emission.exr" in desc
    assert "Emission channel" in desc


def test_emission_section_blurb_discloses_preview_limit():
    blurb = panels._SECTION_BLURBS["emission"]
    assert "Color preview" in blurb
    assert "Emission channel" in blurb


def _note_calls(monkeypatch) -> list[str]:
    calls: list[str] = []
    original = imgui.text_colored

    def spy(color, text):
        calls.append(text)
        return original(color, text)

    monkeypatch.setattr(panels.imgui, "text_colored", spy)
    return calls


def test_aurora_note_drawn_when_aurora_on(imgui_ctx, monkeypatch):
    calls = _note_calls(monkeypatch)
    doc = PlanetParams().model_dump()["emission"]
    doc["aurora_strength"] = 1.0
    imgui.new_frame()
    imgui.begin("w", None, 0)
    panels._draw_emission_aurora_note(doc)
    imgui.end()
    imgui.end_frame()
    assert calls == [panels._AURORA_PREVIEW_NOTE]
    assert "Emission channel" in calls[0], "the note points at where aurora IS visible"
    assert "Color preview" in calls[0], "...and at where it is not"


def test_aurora_note_absent_when_aurora_off(imgui_ctx, monkeypatch):
    calls = _note_calls(monkeypatch)
    doc = PlanetParams().model_dump()["emission"]
    assert doc["aurora_strength"] == 0.0
    imgui.new_frame()
    imgui.begin("w", None, 0)
    panels._draw_emission_aurora_note(doc)
    imgui.end()
    imgui.end_frame()
    assert calls == []


def test_full_panel_draw_reaches_the_note(imgui_ctx, monkeypatch):
    """The note is wired into _draw_model's emission branch (not just an
    orphan helper): a full panel draw with aurora on emits it."""
    calls = _note_calls(monkeypatch)
    params = PlanetParams()
    params.emission.aurora_strength = 1.0
    state = panels.PanelState()
    imgui.new_frame()
    imgui.begin("w", None, 0)
    panels.draw_params_panel(params, state)
    imgui.end()
    imgui.end_frame()
    assert panels._AURORA_PREVIEW_NOTE in calls


# -- B4-3: the Emission-channel composite selector (pure logic) ------------------


def test_emission_composite_off_by_default():
    """Aurora off -> plain .rgb blit (u_channel 0) and a zero aurora color:
    the pre-B4-3 Emission preview is byte-identical."""
    viewport = pytest.importorskip("gasgiant.app.viewport")
    params = PlanetParams()
    assert params.emission.aurora_strength == 0.0
    assert viewport._emission_composite(params) == (0, (0.0, 0.0, 0.0))


def test_emission_composite_on_uses_aurora_color():
    viewport = pytest.importorskip("gasgiant.app.viewport")
    params = PlanetParams()
    params.emission.aurora_strength = 0.7
    u_ch, aurora = viewport._emission_composite(params)
    assert u_ch == viewport._AURORA_COMPOSITE_CHANNEL
    assert aurora == tuple(params.emission.aurora_color)


def test_channel_list_unchanged_by_composite():
    """The composite is a blit-mode switch, not a new channel entry -- the
    user-facing channel list stays exactly the PR #13 seven."""
    viewport = pytest.importorskip("gasgiant.app.viewport")
    assert viewport.CHANNELS == (
        "Color",
        "Height (cloud-top)",
        "Emission",
        "T0 color-idx",
        "T1 thickness",
        "T2 detail",
        "T3 storm-tint",
    )
