"""B1-3: Help (F1) explains the development-run/playback model and the
dirty/unsaved preset indicators.

Content is pinned on the module constants (the same text draw_help renders);
the draw itself gets a headless-imgui smoke frame, the pattern
test_phase8_ui.py uses for the export modal.
"""

from __future__ import annotations

import pytest

main = pytest.importorskip("gasgiant.app.main")
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


def test_help_explains_the_development_run():
    text = main._HELP_DEV_RUN
    assert "development run" in text
    assert "developing N/M" in text
    assert "Playback" in text
    # the tier consequences, in the same glyph vocabulary the badges use
    assert "RESTART" in text and "VELOCITY" in text and "POST" in text


def test_help_explains_dirty_and_unsaved():
    text = main._HELP_DIRTY
    assert "'*'" in text
    assert "'unsaved'" in text
    assert "restored" in text  # the session-restore origin of "unsaved"


def test_draw_help_renders_new_sections(imgui_ctx):
    app = main.StudioApp.__new__(main.StudioApp)
    app._show_help = True
    imgui.new_frame()
    app.draw_help()
    imgui.end_frame()
    assert app._show_help is True  # not self-closed by drawing


def test_help_documents_undo_scope_and_exclusions():
    """B4-6: Help must disclose what undo covers, the exclusions (export
    settings are now IN scope after B4-4; Restart dev, locks, and on-disk
    preset file operations are out), and the 64-entry cap."""
    text = main._HELP_UNDO
    # in scope
    assert "export settings" in text
    assert "preset loads" in text
    assert "one step" in text  # drag coalescing
    # exclusions
    assert "Restart dev" in text
    assert "locks" in text
    assert "deleting preset files" in text
    # the cap
    assert "64" in text


def test_help_undo_wording_matches_the_b44_policy():
    """The Help copy must not claim export settings are excluded -- B4-4
    folded them into history (guards against the copy regressing if the
    policy ever changes without updating Help)."""
    text = main._HELP_UNDO
    assert "NOT undoable" in text
    exclusions = text.split("NOT undoable")[1]
    assert "resolution" not in exclusions
    assert "PNG compression" not in exclusions
