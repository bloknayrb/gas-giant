"""B1-7: aurora affects only the exported emission map (alpha channel) and is
invisible in the Color preview -- the UI must say so where the aurora
controls live, instead of a zero-feedback slider drag."""

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
    the Color preview will not move."""
    desc = EmissionParams.model_fields["aurora_strength"].description or ""
    assert "Color preview" in desc
    assert "emission.exr" in desc


def test_emission_section_blurb_discloses_preview_limit():
    blurb = panels._SECTION_BLURBS["emission"]
    assert "Color preview" in blurb


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
    assert "not the Color preview" in calls[0]


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
