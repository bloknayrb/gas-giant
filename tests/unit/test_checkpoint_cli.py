"""T3: CLI arg-validation for the checkpoint wiring (`gasgiant checkpoint`
and `gasgiant export --resume`). These exercise the pure argument/error paths
that run BEFORE any GL build, so they need no GPU. The GL-backed round-trip
(checkpoint -> resume -> export byte-identity) lives in
tests/gpu/test_checkpoint_resume.py."""

from __future__ import annotations

import numpy as np

from gasgiant.cli import main


def test_export_preset_and_resume_are_mutually_exclusive(tmp_path, capsys):
    rc = main([
        "export", "--preset", "jupiter_like",
        "--resume", str(tmp_path / "state.npz"),
        "--out", str(tmp_path / "out"),
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "mutually exclusive" in err


def test_export_recipe_and_resume_are_mutually_exclusive(tmp_path, capsys):
    rc = main([
        "export", "--recipe", "faded_seb",
        "--resume", str(tmp_path / "state.npz"),
        "--out", str(tmp_path / "out"),
    ])
    assert rc == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_export_resume_missing_file_errors_clearly(tmp_path, capsys):
    missing = tmp_path / "nope.npz"
    rc = main(["export", "--resume", str(missing), "--out", str(tmp_path / "out")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "checkpoint not found" in err
    assert str(missing) in err


def test_export_resume_generation_mismatch_surfaces_message(tmp_path, capsys):
    """A stale checkpoint (wrong generation_version) must produce the engine's
    clear re-create message, not a raw traceback. Built as a bare .npz so this
    stays a no-GL test (load_checkpoint checks the version BEFORE any GL work)."""
    stale = tmp_path / "stale.npz"
    np.savez_compressed(stale, generation_version=1)  # != current GENERATION_VERSION
    rc = main(["export", "--resume", str(stale), "--out", str(tmp_path / "out")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "generation_version" in err
    assert "Re-create the checkpoint" in err


def test_checkpoint_subcommand_parses_and_resolves_preset(tmp_path, capsys):
    """The `checkpoint` subcommand parses and routes through preset resolution;
    an unknown preset errors cleanly BEFORE any GL build (so this needs no GPU)."""
    rc = main([
        "checkpoint", "--preset", "definitely_not_a_real_preset_xyz",
        "--out", str(tmp_path / "c.npz"),
    ])
    assert rc == 2
    assert "error:" in capsys.readouterr().err


def test_checkpoint_subcommand_requires_out(capsys):
    """--out is required; argparse rejects its absence with exit code 2."""
    import pytest

    with pytest.raises(SystemExit) as exc:
        main(["checkpoint", "--preset", "jupiter_like"])
    assert exc.value.code == 2


def test_export_resume_rejects_seed_and_dev_steps(tmp_path, capsys):
    """--seed/--dev-steps define the DEVELOPED run; silently ignoring them on
    --resume would export something other than what the user asked for. Pure
    arg validation -- runs before the checkpoint is even opened."""
    rc = main([
        "export", "--resume", str(tmp_path / "state.npz"),
        "--seed", "5", "--out", str(tmp_path / "out"),
    ])
    assert rc == 2
    assert "resumed checkpoint" in capsys.readouterr().err

    rc = main([
        "export", "--resume", str(tmp_path / "state.npz"),
        "--dev-steps", "100", "--out", str(tmp_path / "out"),
    ])
    assert rc == 2
    assert "resumed checkpoint" in capsys.readouterr().err
