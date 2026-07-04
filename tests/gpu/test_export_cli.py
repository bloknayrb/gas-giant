from __future__ import annotations

import pytest

from gasgiant.cli import main

pytestmark = pytest.mark.gpu


@pytest.fixture
def _require_gl():
    """Skip if a headless GL context can't be created (e.g. CI llvmpipe with no
    display). These tests drive the CLI, which builds its OWN context, so they
    can't take the session `gpu` fixture (it would corrupt context currency, per
    conftest). This guard mirrors that fixture's skip so they don't hard-error
    where every other GPU test already skips."""
    from gasgiant.gl import GpuContext

    try:
        ctx = GpuContext.headless()
    except Exception as exc:  # noqa: BLE001 - any context failure means skip
        pytest.skip(f"no OpenGL context available: {exc}")
    ctx.release()


def test_export_then_validate_cli(_require_gl, tmp_path):
    out = tmp_path / "mapset"
    rc = main(["export", "--preset", "jupiter_like", "--res", "512", "--out", str(out)])
    assert rc == 0
    assert (out / "mapset.json").is_file()
    assert (out / "color.png").is_file()
    assert (out / "height.exr").is_file()

    rc = main(["validate", str(out)])
    assert rc == 0


def test_export_seed_override_changes_output(_require_gl, tmp_path):
    from gasgiant.export.writers import read_png16

    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    assert main(["export", "--res", "512", "--seed", "1", "--out", str(a_dir)]) == 0
    assert main(["export", "--res", "512", "--seed", "2", "--out", str(b_dir)]) == 0
    a = read_png16(a_dir / "color.png")
    b = read_png16(b_dir / "color.png")
    assert (a != b).any()


def test_export_unknown_preset_fails_cleanly(tmp_path):
    rc = main(["export", "--preset", "not_a_preset", "--out", str(tmp_path / "x")])
    assert rc == 2


def test_export_dev_steps_override_changes_output(_require_gl, tmp_path):
    from gasgiant.export.writers import read_png16

    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    assert main(["export", "--res", "512", "--dev-steps", "5", "--out", str(a_dir)]) == 0
    assert main(["export", "--res", "512", "--dev-steps", "60", "--out", str(b_dir)]) == 0
    a = read_png16(a_dir / "color.png")
    b = read_png16(b_dir / "color.png")
    assert (a != b).any()
