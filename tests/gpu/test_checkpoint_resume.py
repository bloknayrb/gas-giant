"""T3: checkpoint -> resume -> export round-trip and the facade ``release()``.

The headline guarantee: in KINEMATIC mode a develop -> save_checkpoint ->
load_checkpoint -> render is BYTE-FOR-BYTE identical to a direct develop ->
render of the same preset (the kinematic path is byte-exact; the render-hash
gate pins it). The vorticity variant asserts within the documented SOR LSB
floor (_VORT_SOR_ATOL) rather than byte-equality. ``release()`` is exercised for
idempotence + full teardown.

The kinematic test function name carries ``identical`` so CI's gpu-smoke job
(`-k "identical or noop or no_op"`) selects it on every PR.
"""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.engine.checkpoint import load_checkpoint, save_checkpoint
from gasgiant.params.model import PlanetParams, SolverType

pytestmark = pytest.mark.gpu

# Documented vorticity SOR LSB noise floor; the kinematic path is exact. Same
# value the other vorticity gpu tests use (tests/gpu/test_checkpoint.py defines
# it inline; kept in sync here).
_VORT_SOR_ATOL = 1e-3


def _kinematic_params() -> PlanetParams:
    p = PlanetParams(seed=91)
    p.sim.resolution = 512
    p.sim.dev_steps = 40
    p.export.width = 512  # 512 is the ExportParams.width minimum
    return p


def test_kinematic_resume_export_is_byte_identical_to_direct(gpu, tmp_path):
    """checkpoint -> resume -> render == direct develop -> render, byte-for-byte,
    in kinematic mode. This is the guarantee CI gpu-smoke enforces."""
    # Direct path: develop fully and render.
    direct = Simulation(_kinematic_params(), gpu)
    direct_maps = direct.render_maps()
    direct.release()

    # Checkpoint path: develop fully, save, load into a fresh sim, render.
    src = Simulation(_kinematic_params(), gpu)
    src.run_to_completion()
    path = tmp_path / "kin.npz"
    save_checkpoint(src, path)
    src.release()

    resumed = load_checkpoint(path, gpu)
    assert resumed.steps_done == 40
    assert resumed.is_developed
    resumed_maps = resumed.render_maps()

    np.testing.assert_array_equal(
        resumed_maps["color"], direct_maps["color"],
        err_msg="kinematic resume->export color must be byte-identical to direct",
    )
    np.testing.assert_array_equal(
        resumed_maps["height"], direct_maps["height"],
        err_msg="kinematic resume->export height must be byte-identical to direct",
    )
    resumed.release()


def test_vorticity_resume_export_matches_within_sor_floor(gpu, tmp_path):
    """Vorticity mode is NOT byte-exact across instances (SOR Poisson LSB
    noise). Two guarantees here:

    - resume -> render reproduces the SOURCE sim's render byte-for-byte, because
      the checkpoint byte-restores the renderable state and a developed-run
      render does no further SOR solve (deterministic derive).
    - resume -> render matches an INDEPENDENT direct develop -> render only
      within the documented SOR floor (_VORT_SOR_ATOL), where cross-instance
      LSB noise lives."""
    def _vort_params() -> PlanetParams:
        p = PlanetParams(seed=91)
        p.solver.type = SolverType.VORTICITY
        p.sim.resolution = 512
        p.sim.dev_steps = 40
        p.export.width = 512  # 512 is the ExportParams.width minimum
        return p

    direct = Simulation(_vort_params(), gpu)
    direct_maps = direct.render_maps()
    direct.release()

    src = Simulation(_vort_params(), gpu)
    src.run_to_completion()
    src_maps = src.render_maps()
    path = tmp_path / "vort.npz"
    save_checkpoint(src, path)
    src.release()

    resumed = load_checkpoint(path, gpu)
    resumed_maps = resumed.render_maps()
    # Against the source it was saved from: the restore is byte-exact and the
    # developed-run derive does no further solve, so this is byte-identical by
    # construction -- but CLAUDE.md categorically bans byte-exact assertions on
    # vorticity output (GL session-context LSB noise can perturb even a pure
    # re-derive on some machines), so we assert within the documented floor.
    # The byte-exactness of the RESTORE mechanism itself is proven by the
    # kinematic test above, which exercises the same load_checkpoint path.
    np.testing.assert_allclose(
        resumed_maps["color"], src_maps["color"], atol=_VORT_SOR_ATOL,
        err_msg="vorticity resume->render must reproduce the saved sim's render within the floor",
    )
    # Within the SOR noise floor against an independent direct render.
    np.testing.assert_allclose(
        resumed_maps["color"], direct_maps["color"], atol=_VORT_SOR_ATOL,
        err_msg="vorticity resume->export must match direct within the SOR noise floor",
    )
    resumed.release()


def test_release_is_idempotent_and_frees_owned_textures(gpu):
    """release() frees every owned GPU resource and is safe to call twice."""
    sim = Simulation(_kinematic_params(), gpu)
    # Populate the lazily-allocated resources release() must also cover.
    sim.ensure_preview(256)
    sim.render_maps()

    assert sim.solver is not None
    assert sim.profile_dyn is not None
    assert sim._preview_color is not None

    sim.release()

    # Handles nulled (idempotence guards).
    assert sim.solver is None
    assert sim.profile_dyn is None
    assert sim.profile_stamp is None
    assert sim.profile_omega is None
    assert sim._preview_color is None
    assert sim._preview_height is None
    assert sim._detail_tex is None
    assert sim.deriver._palette_tex is None

    # Second call must not raise (double-release safe).
    sim.release()
