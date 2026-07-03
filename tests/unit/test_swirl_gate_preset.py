"""Unit tests for scripts/swirl_gate.py --preset resolution (W9 Task 1).

The gate's ship-config (--raw) mode must accept BOTH a factory preset name and
a path to a .json preset file (so candidate re-tunes can be gated before they
are promoted to factory presets). Pure config-building — NO gpu mark.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from gasgiant.params.presets import PresetError, load_factory_preset, save_preset

# scripts/ is not a package; add it to sys.path so we can import directly.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from swirl_gate import build_cfg  # noqa: E402


def test_build_cfg_accepts_factory_name():
    p = build_cfg(seed=4201, drag=0.0, width=512, raw=True, preset="jupiter_vorticity")
    assert p.seed == 4201
    assert p.sim.resolution == 512  # ship-config mode pins resolution to width
    assert p.solver.vort_drag == 0.0  # the swept axis is set explicitly


def test_build_cfg_accepts_json_path(tmp_path):
    src = load_factory_preset("jupiter_vorticity")
    tweaked = src.model_copy(deep=True)
    tweaked.appearance.chroma_aging = 0.123  # marker proving the FILE was loaded
    path = tmp_path / "candidate.json"
    save_preset(tweaked, path, name="candidate")

    p = build_cfg(seed=7, drag=0.0, width=512, raw=True, preset=str(path))
    assert p.seed == 7
    assert p.appearance.chroma_aging == pytest.approx(0.123)


def test_build_cfg_unknown_name_still_errors():
    with pytest.raises(PresetError):
        build_cfg(seed=1, drag=0.0, width=512, raw=True, preset="no_such_preset")
