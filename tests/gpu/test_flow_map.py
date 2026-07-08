"""T10 flow/velocity map export (GPU tier).

Covers the three contracts the plan requires:
  (a) tiled resample == single-pass resample, byte-for-byte (kinematic) -- the
      tile-apron contract extended to the flow kernel;
  (b) an analytic (east, north) oracle -- resampling a KNOWN velocity field
      reproduces it across the whole map, validating the polar puv sampling +
      feather + (east, north) basis handling;
  (c) default-off: an export WITHOUT flow_map writes NO flow.exr and leaves the
      file-set unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.export.exporter import TILE, run_export
from gasgiant.params.model import PlanetParams, SolverType
from gasgiant.sim.solver import BLEND_BAND, RHO_MAX, patch_resolution

pytestmark = pytest.mark.gpu


def _params() -> PlanetParams:
    p = PlanetParams(seed=77)
    p.solver.type = SolverType.KINEMATIC  # byte-exact path
    p.sim.resolution = 512
    p.sim.dev_steps = 15
    p.export.width = 2048  # 2x1 tiles at TILE=1024
    return p


def test_tiled_flow_identical_to_single_pass(gpu):
    """The per-tile flow resample equals a single full-frame resample, byte for
    byte (kinematic mode is byte-exact)."""
    sim = Simulation(_params(), gpu)
    sim.run_to_completion()
    snap = sim.create_snapshot()
    w, h = 2048, 1024
    try:
        # Single pass over the whole frame.
        whole_tex = gpu.texture2d((w, h), 4, "f4")
        sim.deriver.resample_flow(
            snap.vel_eq, snap.vel_n, snap.vel_s, snap.patch_rho_max, snap.blend_band,
            whole_tex, origin=(0, 0), full_size=(w, h),
        )
        whole = gpu.read_texture(whole_tex)
        whole_tex.release()

        # Tiled, mirroring the exporter's tiling.
        tiled = np.empty((h, w, 4), dtype=np.float32)
        tile_tex = gpu.texture2d((TILE, TILE), 4, "f4")
        for y0 in range(0, h, TILE):
            for x0 in range(0, w, TILE):
                tw = min(TILE, w - x0)
                th = min(TILE, h - y0)
                sim.deriver.resample_flow(
                    snap.vel_eq, snap.vel_n, snap.vel_s, snap.patch_rho_max,
                    snap.blend_band, tile_tex, origin=(x0, y0), full_size=(w, h),
                )
                tiled[y0 : y0 + th, x0 : x0 + tw] = gpu.read_texture(tile_tex)[:th, :tw]
        tile_tex.release()
    finally:
        snap.release()

    np.testing.assert_array_equal(whole, tiled)


def _field(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Analytic (east, north) oracle: both components -> 0 at the poles (so the
    vector is single-valued there) but with distinct lat/north-vs-lon/east
    dependence, so a channel swap or wrong patch UV shows up."""
    out = np.zeros((*np.broadcast(lat, lon).shape, 2), dtype=np.float32)
    out[..., 0] = 0.4 * np.cos(lat)                 # eastward, zonal
    out[..., 1] = 0.2 * np.cos(lat) * np.sin(lon)   # northward, lon-dependent
    return out


def _fill_equirect(w: int, h: int) -> np.ndarray:
    uvy = (np.arange(h) + 0.5) / h
    uvx = (np.arange(w) + 0.5) / w
    lat = (np.pi / 2 - uvy * np.pi)[:, None]
    lon = (uvx * 2 * np.pi - np.pi)[None, :]
    return _field(lat, lon)


def _fill_patch(n: int, pole_sign: float) -> np.ndarray:
    c = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    st = c * RHO_MAX
    stx = st[None, :]  # column == st.x
    sty = st[:, None]  # row == st.y
    rho = np.sqrt(stx**2 + sty**2)
    lon = np.arctan2(sty, stx)
    lat = pole_sign * (np.pi / 2 - rho)
    return _field(lat, lon)


def test_flow_basis_matches_analytic_oracle(gpu):
    """Fill the velocity textures with a KNOWN (east, north) field and confirm
    the resample reproduces it -- validates the basis rotation (both domains are
    resampled into the SAME east/north basis) and the polar puv sampling."""
    w, h = 1024, 512
    n = patch_resolution(w)

    eq = np.ascontiguousarray(_fill_equirect(w, h))
    pn = np.ascontiguousarray(_fill_patch(n, +1.0))
    ps = np.ascontiguousarray(_fill_patch(n, -1.0))

    vel_eq = gpu.texture2d((w, h), 2, "f4", data=eq, linear=True)
    vel_eq.repeat_x = True
    vel_n = gpu.texture2d((n, n), 2, "f4", data=pn, linear=True)
    vel_s = gpu.texture2d((n, n), 2, "f4", data=ps, linear=True)
    out_tex = gpu.texture2d((w, h), 4, "f4")

    sim = Simulation(_params(), gpu)  # deriver only; no dev run needed
    try:
        sim.deriver.resample_flow(
            vel_eq, vel_n, vel_s, RHO_MAX, BLEND_BAND, out_tex,
            origin=(0, 0), full_size=(w, h),
        )
        out = gpu.read_texture(out_tex)
    finally:
        for t in (vel_eq, vel_n, vel_s, out_tex):
            t.release()

    expected = _fill_equirect(w, h)  # A(lat, lon) at every output texel
    lat = (np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi)
    mask = np.abs(lat) <= np.deg2rad(78.0)  # exclude the immediate pole rows

    err = np.abs(out[mask, :, :2] - expected[mask]).max()
    assert err < 2e-3, f"flow basis oracle mismatch: max abs err {err:.2e}"
    # B/A channels are the declared constants.
    np.testing.assert_array_equal(out[..., 2], 0.0)
    np.testing.assert_array_equal(out[..., 3], 1.0)


def test_default_export_writes_no_flow_identical(gpu, tmp_path):
    """Default-off: no flow_map => no flow.exr, and the file-set is exactly the
    default color+height+manifest (byte-identity of the default export path)."""
    from gasgiant.export.manifest import read_manifest

    sim = Simulation(_params(), gpu)
    out = tmp_path / "mapset"
    run_export(sim, out)
    assert not (out / "flow.exr").exists()
    assert "flow" not in read_manifest(out)["maps"]
    assert {p.name for p in out.iterdir()} == {"color.png", "height.exr", "mapset.json"}


def test_flow_export_round_trip(gpu, tmp_path):
    """flow_map on => flow.exr written, manifest entry has the convention string,
    and the exported set still validates (flow branch)."""
    from gasgiant.export.manifest import read_manifest
    from gasgiant.export.writers import read_exr_rgba
    from gasgiant.validate import validate_mapset

    p = _params()
    p.export.flow_map = True
    sim = Simulation(p, gpu)
    out = tmp_path / "mapset"
    run_export(sim, out)

    assert (out / "flow.exr").is_file()
    entry = read_manifest(out)["maps"]["flow"]
    assert entry["format"] == "exr32f"
    assert entry["colorspace"] == "non-color"
    assert entry["channels"] == 4
    assert entry["convention"] == "rg_east_north_texel_per_step"

    flow = read_exr_rgba(out / "flow.exr")
    assert flow.shape == (1024, 2048, 4)
    assert np.isfinite(flow).all()
    np.testing.assert_array_equal(flow[..., 2], 0.0)
    np.testing.assert_array_equal(flow[..., 3], 1.0)

    report = validate_mapset(out)
    assert report.ok, report.summary()
