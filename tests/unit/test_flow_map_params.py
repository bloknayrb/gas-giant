"""T10 flow-map: param metadata + the validate/seams flow branch (pure numpy)."""

from __future__ import annotations

import numpy as np

from gasgiant.params.model import ExportParams, PlanetParams, Tier
from gasgiant.validate.seams import validate_arrays


def test_flow_map_default_off():
    """Default export file-set is unchanged: flow_map is off by default."""
    assert PlanetParams().export.flow_map is False
    assert ExportParams().flow_map is False


def test_flow_map_is_post_tier_no_rand():
    """POST tier (re-derive only, no dev rerun) and NOT in the randomize draw
    (enabling a side-car map from randomize() would surprise the user)."""
    info = ExportParams.model_fields["flow_map"]
    extra = info.json_schema_extra
    assert extra["tier"] == Tier.POST.value
    assert extra["ui"] == "Export"
    assert "rand" not in extra


def _analytic_flow(h: int, w: int) -> np.ndarray:
    """A smooth (east, north) field that -> 0 at both poles (pole-safe: the
    vector is single-valued there) with distinct east/north dependence."""
    uvy = (np.arange(h) + 0.5) / h
    uvx = (np.arange(w) + 0.5) / w
    lat = (np.pi / 2 - uvy * np.pi)[:, None]
    lon = (uvx * 2 * np.pi - np.pi)[None, :]
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[..., 0] = 0.4 * np.cos(lat)                 # eastward, zonal, ->0 at poles
    flow[..., 1] = 0.2 * np.cos(lat) * np.sin(lon)   # northward, lon-dependent, ->0 at poles
    return flow


def test_validate_flow_branch_passes_on_smooth_field():
    flow = _analytic_flow(256, 512)
    report = validate_arrays({"flow": flow}, flow_names={"flow"})
    assert report.ok, report.summary()
    # The flow branch runs the SPEED pole check, not a per-component one.
    assert any("flow speed" in c.name for c in report.checks)
    assert any("finite" in c.name for c in report.checks)


def test_validate_flow_pole_speed_tolerates_component_rotation():
    """A solid polar rotation has +east/-east on opposite sides of the pole, so
    per-component near-constancy FAILS but the SPEED is constant -> the flow
    branch (speed check) must pass where the generic per-component check would
    not."""
    h, w = 256, 512
    uvy = (np.arange(h) + 0.5) / h
    uvx = (np.arange(w) + 0.5) / w
    lat = (np.pi / 2 - uvy * np.pi)[:, None]
    lon = (uvx * 2 * np.pi - np.pi)[None, :]
    # Solid rotation about the axis: constant speed on each latitude ring, but
    # east/north components swing with longitude near the pole.
    flow = np.zeros((h, w, 2), dtype=np.float32)
    speed = 0.3 * np.cos(lat)
    flow[..., 0] = speed * np.cos(lon)
    flow[..., 1] = speed * np.sin(lon)
    ok = validate_arrays({"flow": flow}, flow_names={"flow"})
    assert ok.ok, ok.summary()
    # The flow branch checks the SPEED magnitude at the pole (not per-component):
    # here the speed |v| = 0.3*cos(lat) is exactly axisymmetric, so the pole
    # tangential-variation check sees a ~flat ring.
    speed_checks = [c for c in ok.checks if "flow speed" in c.name and "pole" in c.name]
    assert speed_checks and all(c.ok for c in speed_checks)


def test_validate_flow_flags_nonfinite():
    flow = _analytic_flow(128, 256)
    flow[10, 20, 0] = np.nan
    report = validate_arrays({"flow": flow}, flow_names={"flow"})
    assert not report.ok
    assert any("finite" in c.name and not c.ok for c in report.checks)


def test_validate_flow_wrap_continuity_flags_seam():
    """A hard longitudinal discontinuity trips the wrap-continuity check."""
    flow = _analytic_flow(128, 256)
    flow[:, 0, 0] += 5.0  # break the seam column only
    report = validate_arrays({"flow": flow}, flow_names={"flow"})
    assert not report.ok
    assert any("wrap continuity" in c.name and not c.ok for c in report.checks)
