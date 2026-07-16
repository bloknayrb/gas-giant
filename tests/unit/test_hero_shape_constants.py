"""Cross-site consistency guard for the hero outline-deformation constants.

The R(theta) deformation (storms.hero_shape lobes + equatorward flatten,
storms.hero_taper upstream wedge) is deliberately DUPLICATED at three GLSL
sites — the stamp anatomy and heroRelaxWeight in vortex_stamp.glsl, and the
vorticity ring/skirt in vortex_omega.glsl — plus a numpy mirror in the moat
GPU test that probes the deformed frame. Matched constants are the contract:
the relaxation must not fight the stamp, the flow must carry the same outline
as the tracer target, and the test mirror must probe the geometry the shader
actually draws.

This is not hypothetical: the moat mirror shipped STALE (0.05/0.04 vs the
shader's raised 0.075/0.055) for several commits — it only places probe
windows, so nothing failed. A hash pin cannot catch cross-FILE divergence;
this test extracts the constants with tightly ANCHORED patterns (the seeded
sph/sphs phase bank and the u_hero_taper factor chain — loose float sweeps
would hit prose comments, e.g. the stamp comment quoting the OLD 0.05/0.04)
and requires the expected match COUNT per file, so zero matches is a loud
failure, never a vacuous pass.
"""
from __future__ import annotations

import importlib.resources as ir
import re
from pathlib import Path

_KERNELS = "gasgiant.sim.kernels"
_MIRROR = Path(__file__).parents[1] / "gpu" / "test_hero_emergence.py"

# hero_shape: equatorward flatten + seeded m=2/3 lobes.
_FLATTEN = re.compile(r"([\d.]+)\s*\*\s*neqs?\s*\*\s*neqs?")
_LOBE_M2 = re.compile(r"([\d.]+)\s*\*\s*(?:np\.)?sin\(\s*2\.0\s*\*\s*\w+\s*\+\s*sphs?(?:\.x|\[0\])\s*\)")
_LOBE_M3 = re.compile(r"([\d.]+)\s*\*\s*(?:np\.)?sin\(\s*3\.0\s*\*\s*\w+\s*\+\s*sphs?(?:\.y|\[1\])\s*\)")
# hero_taper: amplitude (leading factor of the u_hero_taper chain), wedge
# normalization (leading factor of the c^4(1-c^2) window), and the Rr floor.
_TAPER_AMP = re.compile(r"([\d.]+)\s*\*\s*u_hero_taper\s*\*\s*u_hero_emergence\s*\*\s*tw")
_TAPER_NORM = re.compile(r"tw\s*=\s*([\d.]+)\s*\*\s*tc2\s*\*\s*tc2\s*\*\s*\(1\.0\s*-\s*tc2\)")
_TAPER_FLOOR = re.compile(r"=\s*max\(Rrs?,\s*([\d.]+)\)")

# (pattern, expected values keyed by source: file -> (value, expected count)).
# vortex_stamp has TWO deformation sites (stamp anatomy + heroRelaxWeight);
# vortex_omega and the numpy mirror have one each. The mirror runs taper=0
# (model default) and correctly omits the wedge — no dead code forced in.
_EXPECT = [
    (_FLATTEN, "0.11", {"vortex_stamp.glsl": 2, "vortex_omega.glsl": 1, "mirror": 1}),
    (_LOBE_M2, "0.075", {"vortex_stamp.glsl": 2, "vortex_omega.glsl": 1, "mirror": 1}),
    (_LOBE_M3, "0.055", {"vortex_stamp.glsl": 2, "vortex_omega.glsl": 1, "mirror": 1}),
    (_TAPER_AMP, "0.25", {"vortex_stamp.glsl": 2, "vortex_omega.glsl": 1}),
    # THREE wedge windows in vortex_stamp: stamp anatomy, heroRelaxWeight,
    # and the heroBandDeflect convergence (the bow's outer recovery pulls in
    # on the wedge arc — the stagnation-point closure).
    (_TAPER_NORM, "6.75", {"vortex_stamp.glsl": 3, "vortex_omega.glsl": 1}),
    (_TAPER_FLOOR, "0.4", {"vortex_stamp.glsl": 2, "vortex_omega.glsl": 1}),
]


def _source(name: str) -> str:
    if name == "mirror":
        return _MIRROR.read_text(encoding="utf-8")
    return ir.files(_KERNELS).joinpath(name).read_text(encoding="utf-8")


def test_hero_shape_blocks_agree():
    problems: list[str] = []
    for pattern, value, sites in _EXPECT:
        for name, count in sites.items():
            found = pattern.findall(_source(name))
            if len(found) != count:
                problems.append(
                    f"{name}: {pattern.pattern!r} matched {len(found)}x, expected {count}"
                )
            for got in found:
                if got != value:
                    problems.append(
                        f"{name}: {pattern.pattern!r} -> {got}, expected {value}"
                    )
    assert not problems, (
        "hero outline-deformation constants diverged across sites (the stamp, "
        "heroRelaxWeight, the omega ring/skirt and the moat-test mirror must "
        "carry ONE R(theta)):\n  " + "\n  ".join(problems)
    )


# ------------------------------------------------- flow_renorm profile mirror

# The emergence ring/skirt smoothstep windows + amplitudes, as written in
# vortex_omega.glsl: `<amp> * scale * (smoothstep(i0, i1, qh) -
# smoothstep(o0, o1, qh))`. Anchored on `scale` and `qh` so prose comments
# (which quote several of these numbers) cannot match.
_RING_SKIRT = re.compile(
    r"(-?[\d.]+)\s*\*\s*scale\s*\*\s*\(smoothstep\((-?[\d.]+),\s*(-?[\d.]+),"
    r"\s*qh\)\s*-\s*smoothstep\((-?[\d.]+),\s*(-?[\d.]+),\s*qh\)\)"
)


def test_flow_renorm_mirrors_ring_skirt_windows():
    """sim/flow_renorm.py computes u_hero_flow_renorm (the hero_flow_aspect
    net-circulation renorm) by quadrature over a NUMPY MIRROR of the omega
    ring/skirt profile. A retune of the GLSL windows or amplitudes without the
    mirror silently mis-normalizes the widened ring's circulation — the exact
    stale-mirror class this file exists for. Exactly two profile terms must
    exist (ring, then skirt) and both must equal the mirror tuples."""
    from gasgiant.sim.flow_renorm import RING_WINDOW, SKIRT_WINDOW

    found = _RING_SKIRT.findall(_source("vortex_omega.glsl"))
    assert len(found) == 2, (
        f"expected exactly 2 ring/skirt profile terms in vortex_omega.glsl, "
        f"matched {len(found)} — the anchored pattern or the kernel drifted"
    )
    for (amp, i0, i1, o0, o1), expect, label in zip(
        found, (RING_WINDOW, SKIRT_WINDOW), ("ring", "skirt"), strict=True
    ):
        got = (float(i0), float(i1), float(o0), float(o1), float(amp))
        assert got == expect, (
            f"{label} profile diverged: vortex_omega.glsl has {got}, "
            f"flow_renorm.py mirror has {expect}"
        )
