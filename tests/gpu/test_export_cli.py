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


@pytest.mark.parametrize("seq", [False, True])
def test_cli_reports_memoryerror_instead_of_traceback(_require_gl, tmp_path, monkeypatch, capsys, seq):
    """cli.py wrapped neither export entry point, so an out-of-memory export
    exited 1 with a raw traceback. numpy's own message names the exact shape and
    dtype it could not allocate -- better than any estimate we could print.

    GPU-tier because the CLI builds a real context and Simulation before it ever
    reaches the export call this patches."""
    import gasgiant.export.exporter as exporter

    boom = MemoryError(
        "Unable to allocate 8.00 GiB for an array with shape (16384, 32768, 4) "
        "and data type float32"
    )

    def _raise(*a, **k):
        raise boom

    monkeypatch.setattr(exporter, "run_export", _raise)
    monkeypatch.setattr(exporter, "run_export_sequence", _raise)

    argv = ["export", "--preset", "jupiter_like", "--res", "512", "--out", str(tmp_path / "o")]
    if seq:
        argv += ["--frames", "3", "--steps-per-frame", "2"]

    rc = main(argv)
    err = capsys.readouterr().err
    assert rc == 2, f"expected exit 2, got {rc}"
    assert "out of memory" in err
    assert "8.00 GiB" in err, "numpy's own detail must survive to the user"
