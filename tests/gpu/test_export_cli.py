from __future__ import annotations

import pytest

from gasgiant.cli import main

pytestmark = pytest.mark.gpu


def test_export_then_validate_cli(tmp_path):
    out = tmp_path / "mapset"
    rc = main(["export", "--preset", "jupiter_like", "--res", "512", "--out", str(out)])
    assert rc == 0
    assert (out / "mapset.json").is_file()
    assert (out / "color.png").is_file()
    assert (out / "height.exr").is_file()

    rc = main(["validate", str(out)])
    assert rc == 0


def test_export_seed_override_changes_output(tmp_path):
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
