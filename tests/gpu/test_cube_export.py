"""T17 cube-map export (GPU tier).

Three contracts:
  (a) identical: the default (equirect) export path is byte-identical to a plain
      single-pass equirect derive -- the cube variant/cache-key widening leaves
      the default program untouched (kinematic mode is byte-exact);
  (b) cross-sample: sampling the equirect map at a direction's (lat, lon) agrees
      with the value the corresponding cube face+uv produces -- i.e. the cube
      projection maps to the sphere correctly;
  (c) continuity: an exported cube set passes the validator's 12-edge
      face-continuity branch (kinematic), and the manifest is the v2 faces form.
"""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.export.exporter import run_export
from gasgiant.export.manifest import CUBE_FACE_NAMES, read_manifest
from gasgiant.params.model import PlanetParams, ProjectionKind, SolverType

pytestmark = pytest.mark.gpu


def _params(width: int = 1024) -> PlanetParams:
    p = PlanetParams(seed=91)
    p.solver.type = SolverType.KINEMATIC  # byte-exact path
    p.sim.resolution = 512
    p.sim.dev_steps = 12
    p.detail.intensity = 0.0  # isolate projection (synth detail is equirect-only)
    p.export.width = width
    return p


def _cube_dir(face: int, uc: np.ndarray, vc: np.ndarray):
    if face == 0:
        return (np.ones_like(uc), -vc, -uc)
    if face == 1:
        return (-np.ones_like(uc), -vc, uc)
    if face == 2:
        return (uc, np.ones_like(uc), vc)
    if face == 3:
        return (uc, -np.ones_like(uc), -vc)
    if face == 4:
        return (uc, -vc, np.ones_like(uc))
    return (-uc, -vc, -np.ones_like(uc))


def _sample_equirect(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear sample of an (H, W, C) equirect image at uv (u wraps, v clamps)."""
    h, w = img.shape[:2]
    fx = u * w - 0.5
    fy = v * h - 0.5
    x0 = np.floor(fx).astype(int)
    y0 = np.floor(fy).astype(int)
    tx = (fx - x0)[..., None]
    ty = (fy - y0)[..., None]
    x0m, x1m = x0 % w, (x0 + 1) % w
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    top = img[y0c, x0m] * (1 - tx) + img[y0c, x1m] * tx
    bot = img[y1c, x0m] * (1 - tx) + img[y1c, x1m] * tx
    return top * (1 - ty) + bot * ty


def _derive_equirect(sim, snap, params, w, h, gpu) -> np.ndarray:
    """Single-pass equirect color, detail/lanes off (float RGB in [0,1])."""
    tex_c = gpu.texture2d((w, h), 4, "f4")
    tex_hh = gpu.texture2d((w, h), 1, "f4")
    try:
        sim.deriver.derive(
            snap.tracers_eq, snap.tracers_n, snap.tracers_s,
            snap.patch_rho_max, snap.blend_band, tex_c, tex_hh, params.appearance,
            detail_tex=None, detail_intensity=0.0, origin=(0, 0), full_size=(w, h),
            lanes=[], warp=snap.warp, seed=params.seed,
        )
        return gpu.read_texture(tex_c)[..., :3].copy()
    finally:
        tex_c.release()
        tex_hh.release()


def test_default_export_identical(gpu, tmp_path):
    """(a) The default equirect export's color map equals a plain single-pass
    equirect derive, byte for byte -- the cube code path is fully inert when off.
    (Both use lanes=[]; the run is asserted lane-free so the reference matches.)"""
    from gasgiant.export.writers import read_png16

    sim = Simulation(_params(1024), gpu)
    sim.run_to_completion()
    snap = sim.create_snapshot()
    params = snap.params
    w, h = params.export.width, params.export.width // 2
    try:
        assert len(snap.lanes) == 0  # reference derive uses lanes=[]
        sim.deriver.update_palettes(params.appearance)
        ref = _derive_equirect(sim, snap, params, w, h, gpu)
        ref_u16 = (np.clip(ref, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    finally:
        snap.release()

    out = tmp_path / "eq"
    run_export(Simulation(_params(1024), gpu), out)
    assert {p.name for p in out.iterdir()} == {"color.png", "height.exr", "mapset.json"}
    # read_png16 round-trips the stored uint16 back through /65535; compare in
    # that same normalized space so the check is byte-exact on the stored PNG.
    ref_norm = ref_u16.astype(np.float32) / np.float32(65535.0)
    np.testing.assert_array_equal(read_png16(out / "color.png"), ref_norm)
    assert read_manifest(out)["schema_version"] == 1
    assert read_manifest(out)["projection"] == "equirectangular"


def test_equirect_cube_cross_sample(gpu):
    """(b) For every cube face, the face pixels' (lat, lon) sampled from the
    equirect color agree with the cube face render -- the projection is correct."""
    sim = Simulation(_params(1024), gpu)
    sim.run_to_completion()
    snap = sim.create_snapshot()
    params = snap.params
    w, h = 1024, 512
    face_size = 256
    try:
        sim.deriver.update_palettes(params.appearance)
        eq = _derive_equirect(sim, snap, params, w, h, gpu)

        face_tex = gpu.texture2d((face_size, face_size), 4, "f4")
        face_h = gpu.texture2d((face_size, face_size), 1, "f4")
        c = (np.arange(face_size) + 0.5) / face_size * 2.0 - 1.0
        uc = np.broadcast_to(c[None, :], (face_size, face_size))
        vc = np.broadcast_to(c[:, None], (face_size, face_size))
        worst_mean = 0.0
        try:
            for face in range(6):
                sim.deriver.derive(
                    snap.tracers_eq, snap.tracers_n, snap.tracers_s,
                    snap.patch_rho_max, snap.blend_band, face_tex, face_h,
                    params.appearance, detail_tex=None, detail_intensity=0.0,
                    origin=(0, 0), full_size=(face_size, face_size),
                    lanes=[], warp=snap.warp, seed=params.seed,
                    projection_cube=True, cube_face=face,
                )
                cube_rgb = gpu.read_texture(face_tex)[..., :3]
                dx, dy, dz = _cube_dir(face, uc, vc)
                n = np.sqrt(dx * dx + dy * dy + dz * dz)
                dx, dy, dz = dx / n, dy / n, dz / n
                lat = np.arcsin(np.clip(dy, -1.0, 1.0))
                lon = np.arctan2(dz, dx)
                u = (lon + np.pi) / (2.0 * np.pi)
                v = (0.5 * np.pi - lat) / np.pi
                ref = _sample_equirect(eq, u, v)
                diff = np.abs(cube_rgb - ref)
                worst_mean = max(worst_mean, float(diff.mean()))
                # A wrong face convention yields mean ~0.15-0.3; correct mapping
                # leaves only bilinear-resample error at band edges.
                assert diff.mean() < 0.02, f"face {face} mean diff {diff.mean():.3e}"
                assert np.percentile(diff, 99.0) < 0.08, (
                    f"face {face} p99 diff {np.percentile(diff, 99.0):.3e}"
                )
        finally:
            face_tex.release()
            face_h.release()
    finally:
        snap.release()
    assert worst_mean < 0.02


def test_cube_export_round_trip_and_continuity(gpu, tmp_path):
    """(c) A cube export writes 6 faces per map + a v2 faces-manifest, and passes
    the validator's 12-edge face-continuity branch (kinematic)."""
    from gasgiant.validate import validate_mapset

    p = _params(1024)  # face_size = 256
    p.export.projection = ProjectionKind.CUBE
    sim = Simulation(p, gpu)
    out = tmp_path / "cube"
    run_export(sim, out)

    manifest = read_manifest(out)
    assert manifest["schema_version"] == 2
    assert manifest["projection"] == "cube"
    assert manifest["resolution"] == [256, 256]
    for name in ("color", "height"):
        entry = manifest["maps"][name]
        assert set(entry["faces"]) == set(CUBE_FACE_NAMES)
    # Every declared face file exists.
    for fn in CUBE_FACE_NAMES:
        assert (out / f"color_{fn}.png").is_file()
        assert (out / f"height_{fn}.exr").is_file()

    report = validate_mapset(out)
    assert report.ok, report.summary()
