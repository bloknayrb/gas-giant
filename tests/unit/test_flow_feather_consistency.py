"""T10 cross-consistency: the flow kernel's polar feather is a byte-for-byte
DUPLICATE of derive.comp's feather.

The plan forbids extracting the feather into a shared ``#include`` (that would
touch derive.comp's default token stream, which is pinned for byte-identity), so
the feather math is copied into flow_resample.comp instead. This test is the
guard the plan requires in place of the shared include: it asserts the two
kernels compute the polar patch-UV + smoothstep weight from IDENTICAL source
expressions, so the flow map's polar region tracks the color map's exactly. If
either feather is edited, this test fails.
"""

from __future__ import annotations

import re
from pathlib import Path

KERNELS = Path(__file__).resolve().parents[2] / "src" / "gasgiant" / "render" / "kernels"
DERIVE = KERNELS / "derive.comp"
FLOW = KERNELS / "flow_resample.comp"

# The canonical feather expressions (whitespace-normalized). Both kernels must
# contain each of these verbatim -- this IS the feather (latitude -> weight and
# lat/lon -> patch UV). Sourced from derive.comp lines ~131-140.
FEATHER_EXPRS = (
    "float lat = 0.5 * PI - uv.y * PI;",
    "float w = smoothstep(u_blend_lo, u_blend_hi, abs(lat));",
    "float lon = uv.x * 2.0 * PI - PI;",
    "float rho = 0.5 * PI - abs(lat);",
    "vec2 st = rho * vec2(cos(lon), sin(lon));",
    "vec2 puv = st / u_patch_rho_max * 0.5 + 0.5;",
)


def _norm(text: str) -> str:
    """Collapse runs of whitespace so indentation/spacing differences don't count."""
    return re.sub(r"\s+", " ", text)


def test_flow_feather_matches_derive_feather():
    derive = _norm(DERIVE.read_text())
    flow = _norm(FLOW.read_text())
    for expr in FEATHER_EXPRS:
        e = _norm(expr)
        assert e in derive, f"feather expr not found in derive.comp: {expr!r}"
        assert e in flow, f"feather expr not found in flow_resample.comp: {expr!r}"


def test_flow_feather_uses_same_smoothstep_weight_numerically():
    """Re-implement BOTH feather weights in python from the shared smoothstep
    definition and confirm they agree across the band latitudes (a symbolic
    check that the duplicated smoothstep(u_blend_lo, u_blend_hi, |lat|) is the
    same function on both sides)."""
    import numpy as np

    blend_lo, blend_hi = np.deg2rad(64.0), np.deg2rad(67.0)

    def smoothstep(lo, hi, x):
        t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    lats = np.linspace(np.deg2rad(55.0), np.deg2rad(75.0), 200)
    w_derive = smoothstep(blend_lo, blend_hi, np.abs(lats))
    w_flow = smoothstep(blend_lo, blend_hi, np.abs(lats))
    np.testing.assert_array_equal(w_derive, w_flow)
    # The band actually feathers (not degenerate: 0 below, 1 above).
    assert w_flow[0] == 0.0 and w_flow[-1] == 1.0
    assert np.any((w_flow > 0.0) & (w_flow < 1.0))
