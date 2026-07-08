"""Seam, pole, and continuity invariants for equirect maps.

Conventions match core.domain.EquirectGrid texel centers: there is no
duplicated 0/360 column, so the wrap check tests CONTINUITY (the seam
column-pair difference must look like an interior column-pair difference),
never column identity. Pole rows sit at ~+/-89.99 degrees, not the poles
themselves, so they are checked for NEAR-constancy relative to mid rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# The seam pair may differ from the mean interior pair by at most this factor.
WRAP_FACTOR = 3.0
# Tangential variation may not GROW toward the pole by more than this factor.
# Texel circles shrink poleward so smooth content varies less — but a polar
# vortex's spiral arms wind TIGHTER toward its center, legitimately raising
# variation ~2x; the pinch artifacts this guards against show 10-30x.
POLE_TANGENTIAL_FACTOR = 3.0
# The pole row may not jump away from its neighbor row by more than this
# factor of the next row-pair difference.
POLE_VERTICAL_FACTOR = 3.0
# Ignore variation below this (essentially flat images).
ABS_FLOOR = 1e-3


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


@dataclass
class Report:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.checks.append(CheckResult(name, ok, detail))

    def summary(self) -> str:
        lines = [f"[{'PASS' if c.ok else 'FAIL'}] {c.name}: {c.detail}" for c in self.checks]
        lines.append(f"=> {'OK' if self.ok else 'FAILED'}")
        return "\n".join(lines)


def _flat(arr: np.ndarray) -> np.ndarray:
    """(H, W) or (H, W, C) -> (H, W, C) float32 (16K maps in float64 would
    need multi-GB temporaries)."""
    a = np.asarray(arr, dtype=np.float32)
    return a[..., None] if a.ndim == 2 else a


def check_wrap_continuity(arr: np.ndarray, name: str, report: Report) -> None:
    a = _flat(arr)
    seam = float(np.abs(a[:, 0] - a[:, -1]).mean())
    # Interior reference from a column subsample — same statistics, no
    # full-image diff temporary at 16K.
    w = a.shape[1]
    stride = max(w // 1024, 1)
    cols = np.arange(0, w - 1, stride)
    interior = float(np.abs(a[:, cols + 1] - a[:, cols]).mean())
    limit = max(WRAP_FACTOR * interior, ABS_FLOOR)
    report.add(
        f"{name}: wrap continuity",
        bool(seam <= limit),
        f"seam diff {seam:.3e} vs interior {interior:.3e} (limit {limit:.3e})",
    )


def check_pole_rows(arr: np.ndarray, name: str, report: Report) -> None:
    """Pole continuity, two invariants per pole.

    High-frequency content legitimately still varies along the near-pole row
    (it sits at ~89.x degrees, not at the pole), so we do NOT require
    constancy. We require what smooth content on a sphere guarantees:
    tangential variation shrinks toward the pole, and the pole row does not
    jump away from its neighbor.
    """
    a = _flat(arr)
    for label, r0, r1, r2 in (
        ("north", a[0], a[1], a[2]),
        ("south", a[-1], a[-2], a[-3]),
    ):
        var0 = float(r0.std(axis=0).mean())
        var1 = float(r1.std(axis=0).mean())
        limit_t = max(POLE_TANGENTIAL_FACTOR * var1, ABS_FLOOR)
        report.add(
            f"{name}: {label} pole tangential variation",
            bool(var0 <= limit_t),
            f"pole-row std {var0:.3e} vs neighbor {var1:.3e} (limit {limit_t:.3e})",
        )
        jump = float(np.abs(r0 - r1).mean())
        step = float(np.abs(r1 - r2).mean())
        limit_v = max(POLE_VERTICAL_FACTOR * step, ABS_FLOOR)
        report.add(
            f"{name}: {label} pole vertical continuity",
            bool(jump <= limit_v),
            f"pole-row jump {jump:.3e} vs next pair {step:.3e} (limit {limit_v:.3e})",
        )


def check_finite(arr: np.ndarray, name: str, report: Report) -> None:
    bad = int(np.size(arr) - np.isfinite(arr).sum())
    report.add(f"{name}: finite", bad == 0, f"{bad} non-finite values")


def check_pole_speed(arr: np.ndarray, name: str, report: Report) -> None:
    """Pole continuity for a VECTOR (flow) map, checked on the SPEED magnitude.

    The (east, north) components of a flow map rotate through the tangent basis
    as you go around a near-pole ring — a solid polar vortex is +east on one
    side of the pole and -east on the other — so per-component near-constancy
    (``check_pole_rows``) is the wrong invariant and would false-positive. The
    physical invariant that survives the basis rotation is the SPEED
    |v| = sqrt(vE^2 + vN^2): it is near-axisymmetric around the pole. So reduce
    RG to speed and reuse the pole-row tangential/vertical continuity checks on
    that scalar."""
    a = _flat(arr)
    speed = np.sqrt((a[..., :2].astype(np.float32) ** 2).sum(axis=-1))
    check_pole_rows(speed[..., None], f"{name} speed", report)


# A processing seam (detail route switch, domain feather bug) is a row-pair
# jump that is UNIFORM across longitude; legitimate sharp content in the same
# band (band edges) meanders and varies along the row. A row is flagged only
# when its jump is both large relative to the band (size gate) and more
# uniform along the row than content ever is (uniformity gate).
# Tuned against real content: a meandering-but-steep band edge in the height
# map measures uniformity ~3-4; hard processing cliffs measure 5+.
BAND_SEAM_SIZE_FACTOR = 10.0      # mean diff vs band median
BAND_SEAM_UNIFORMITY = 5.0        # mean diff vs along-row std of the diff
_BAND_LO_DEG = 55.0
_BAND_HI_DEG = 70.0


def check_latitude_band_continuity(arr: np.ndarray, name: str, report: Report) -> None:
    """No horizontal seam across the polar routing / domain blend band."""
    a = _flat(arr)
    h = a.shape[0]
    lats = 90.0 - (np.arange(h) + 0.5) / h * 180.0
    for label, sel in (
        ("north", (lats > _BAND_LO_DEG) & (lats < _BAND_HI_DEG)),
        ("south", (lats < -_BAND_LO_DEG) & (lats > -_BAND_HI_DEG)),
    ):
        rows = np.where(sel)[0]
        if rows.size < 8:
            continue
        # Column-subsampled row-pair diffs (16K-safe).
        w = a.shape[1]
        cols = np.arange(0, w, max(w // 1024, 1))
        band = a[rows[0] : rows[-1] + 2, cols]
        d = np.abs(np.diff(band, axis=0)).mean(axis=2)  # (rows, cols)
        mean_r = d.mean(axis=1)
        std_r = d.std(axis=1)
        med = float(np.median(mean_r))
        size_limit = max(BAND_SEAM_SIZE_FACTOR * med, ABS_FLOOR)
        uniformity = mean_r / (std_r + ABS_FLOOR)
        seam_rows = (mean_r > size_limit) & (uniformity > BAND_SEAM_UNIFORMITY)
        worst = int(np.argmax(mean_r * (uniformity > BAND_SEAM_UNIFORMITY)))
        report.add(
            f"{name}: {label} blend-band continuity",
            bool(not seam_rows.any()),
            f"{int(seam_rows.sum())} uniform-jump rows (worst mean "
            f"{mean_r[worst]:.3e}, uniformity {uniformity[worst]:.1f}, "
            f"band median {med:.3e})",
        )


# --- Cube-map face-continuity validation (T17) ------------------------------
# Face order matches export.manifest.CUBE_FACE_NAMES and the GL cube-map axis
# convention used by derive.comp's PROJECTION_CUBE branch (+X,-X,+Y,-Y,+Z,-Z).
CUBE_FACE_NAMES = ("px", "nx", "py", "ny", "pz", "nz")
_CUBE_EDGES = ("top", "bottom", "left", "right")
# A shared cube edge's two-face border pixels sample nearly the same directions
# (offset by ~1 texel across the seam), so the seam diff must look like an
# interior adjacent-pixel step, never a cliff.
CUBE_EDGE_FACTOR = 6.0


def _cube_dir(face: int, uc: float, vc: float) -> tuple[float, float, float]:
    """Unnormalized cube-surface direction for face-local coords (uc, vc) in
    [-1, 1]. MUST match derive.comp's PROJECTION_CUBE branch exactly."""
    if face == 0:
        return (1.0, -vc, -uc)   # +X
    if face == 1:
        return (-1.0, -vc, uc)   # -X
    if face == 2:
        return (uc, 1.0, vc)     # +Y
    if face == 3:
        return (uc, -1.0, -vc)   # -Y
    if face == 4:
        return (uc, -vc, 1.0)    # +Z
    return (-uc, -vc, -1.0)      # -Z


def _cube_edge_uc_vc(edge: str, s: float) -> tuple[float, float]:
    """(uc, vc) at parameter ``s`` in [0, 1] along a face edge. left/right run
    down the rows (v, top->bottom); top/bottom run across the columns
    (u, left->right) -- matching ``_cube_edge_line`` pixel ordering."""
    t = 2.0 * s - 1.0
    if edge == "left":
        return (-1.0, t)
    if edge == "right":
        return (1.0, t)
    if edge == "top":
        return (t, -1.0)
    return (t, 1.0)  # bottom


def _cube_edge_corners(face: int, edge: str) -> tuple[tuple, tuple]:
    """The two corner directions (s=0, s=1) of a face edge, rounded to ints so
    physical edges compare exactly (cube corners have integer +/-1 coords)."""
    def corner(s: float) -> tuple:
        uc, vc = _cube_edge_uc_vc(edge, s)
        return tuple(int(round(c)) for c in _cube_dir(face, uc, vc))
    return corner(0.0), corner(1.0)


def _build_cube_edge_table() -> tuple:
    """The 12 shared cube edges as ``(faceA, edgeA, faceB, edgeB, reversed)``,
    derived geometrically (never hand-typed): two face edges are the SAME
    physical edge when their corner-direction SETS match; ``reversed`` is True
    when the two parameterizations run opposite along it. ``faceA < faceB``, each
    physical edge listed once."""
    faces_edges = [(f, e) for f in range(6) for e in _CUBE_EDGES]
    table: list[tuple] = []
    seen: set = set()
    for fa, ea in faces_edges:
        a0, a1 = _cube_edge_corners(fa, ea)
        for fb, eb in faces_edges:
            if fb <= fa:
                continue
            b0, b1 = _cube_edge_corners(fb, eb)
            if {a0, a1} != {b0, b1}:
                continue
            key = frozenset((a0, a1))
            if key in seen:
                continue
            seen.add(key)
            table.append((fa, ea, fb, eb, a0 == b1 and a1 == b0))
    return tuple(table)


CUBE_EDGE_TABLE = _build_cube_edge_table()


def _cube_edge_line(a: np.ndarray, edge: str) -> np.ndarray:
    """The 1-texel border line of face image ``a`` (H, W, C) along ``edge``,
    ordered by s in [0, 1] (left/right: rows top->bottom; top/bottom: cols
    left->right)."""
    if edge == "left":
        return a[:, 0, :]
    if edge == "right":
        return a[:, -1, :]
    if edge == "top":
        return a[0, :, :]
    return a[-1, :, :]  # bottom


def check_cube_face_continuity(
    faces: dict[str, np.ndarray], name: str, report: Report
) -> None:
    """Continuity across all 12 shared cube-face edges.

    ``faces`` maps each face name (px..nz) to its (H, W[, C]) array. For every
    shared edge the two faces' border pixels are compared (reversing the second
    when the edge parameterizations run opposite, per ``CUBE_EDGE_TABLE``), and
    the worst seam diff is checked against the interior adjacent-pixel gradient."""
    arrs = {k: _flat(v) for k, v in faces.items()}
    grads: list[float] = []
    for a in arrs.values():
        stride = max(a.shape[1] // 256, 1)
        cols = np.arange(0, a.shape[1] - 1, stride)
        grads.append(float(np.abs(a[:, cols + 1] - a[:, cols]).mean()))
        rows = np.arange(0, a.shape[0] - 1, stride)
        grads.append(float(np.abs(a[rows + 1, :] - a[rows, :]).mean()))
    interior = float(np.mean(grads)) if grads else 0.0
    limit = max(CUBE_EDGE_FACTOR * interior, ABS_FLOOR)
    worst = 0.0
    worst_edge = "-"
    for fa, ea, fb, eb, rev in CUBE_EDGE_TABLE:
        la = _cube_edge_line(arrs[CUBE_FACE_NAMES[fa]], ea)
        lb = _cube_edge_line(arrs[CUBE_FACE_NAMES[fb]], eb)
        if rev:
            lb = lb[::-1]
        d = float(np.abs(la - lb).mean())
        if d > worst:
            worst = d
            worst_edge = f"{CUBE_FACE_NAMES[fa]}.{ea}|{CUBE_FACE_NAMES[fb]}.{eb}"
    report.add(
        f"{name}: cube face continuity",
        bool(worst <= limit),
        f"worst edge {worst_edge} diff {worst:.3e} vs interior {interior:.3e} "
        f"(limit {limit:.3e}, {len(CUBE_EDGE_TABLE)} edges)",
    )


def validate_cube_arrays(face_maps: dict[str, dict[str, np.ndarray]]) -> Report:
    """Run finiteness + 12-edge face-continuity checks on a cube map set.

    ``face_maps`` maps each map name (color/height/emission) to its 6-face dict
    (face name -> array)."""
    report = Report()
    for name, faces in face_maps.items():
        for fn, arr in faces.items():
            check_finite(arr, f"{name} {fn}", report)
        check_cube_face_continuity(faces, name, report)
    return report


def validate_arrays(
    maps: dict[str, np.ndarray], flow_names: frozenset[str] | set[str] = frozenset()
) -> Report:
    """Run the seam/pole/continuity checks on each named map.

    ``flow_names`` marks maps whose RG channels are an (east, north) VELOCITY
    field: they wrap-check on the components but take the pole check on the SPEED
    magnitude (``check_pole_speed``) rather than per-component, because the
    components rotate through the basis around the pole (see ``check_pole_speed``)."""
    report = Report()
    for name, arr in maps.items():
        is_flow = name in flow_names
        check_finite(arr, name, report)
        check_wrap_continuity(arr, name, report)
        if is_flow:
            check_pole_speed(arr, name, report)
        else:
            check_pole_rows(arr, name, report)
            check_latitude_band_continuity(arr, name, report)
    return report


def validate_mapset(mapset_dir: Path) -> Report:
    """Load an exported map set via its manifest and run all checks."""
    from gasgiant.export.manifest import read_manifest
    from gasgiant.export.writers import read_exr_gray, read_exr_rgba, read_png16

    manifest = read_manifest(mapset_dir)
    if manifest.get("projection") == "cube":
        # Cube map set (T17): each maps entry carries a 6-face ``faces`` block.
        face_maps: dict[str, dict[str, np.ndarray]] = {}
        for name, entry in manifest["maps"].items():
            arrs: dict[str, np.ndarray] = {}
            for fn, rel in entry["faces"].items():
                path = mapset_dir / rel
                if entry["format"] == "png16":
                    arrs[fn] = read_png16(path)
                elif entry["format"] == "exr32f":
                    if entry.get("channels", 1) >= 3:
                        # HDR emission-class map: continuity in log space (sparse
                        # radiance cores make the raw statistic flaky), NaN/Inf
                        # survive log1p's monotonicity for the finiteness check.
                        arrs[fn] = np.log1p(np.maximum(read_exr_rgba(path)[..., :3], 0.0))
                    else:
                        arrs[fn] = read_exr_gray(path)
            face_maps[name] = arrs
        return validate_cube_arrays(face_maps)

    maps: dict[str, np.ndarray] = {}
    flow_names: set[str] = set()
    for name, entry in manifest["maps"].items():
        path = mapset_dir / entry["file"]
        if entry["format"] == "png16":
            maps[name] = read_png16(path)
        elif entry["format"] == "exr32f":
            if name == "flow":
                # Flow/velocity map: keep the RG (east, north) channels; the pole
                # check runs on speed magnitude, not per-component (see
                # check_pole_speed). Must precede the channels>=3 emission branch.
                maps[name] = read_exr_rgba(path)[..., :2]
                flow_names.add(name)
            elif entry.get("channels", 1) >= 3:
                # Emission-class HDR map. Sparse radiance-10+ cores make the
                # raw wrap statistic flaky, so continuity runs in log space;
                # finiteness/range still covers the raw values via log1p's
                # monotonicity (NaN/Inf survive it).
                arr = read_exr_rgba(path)
                maps[f"{name}_rgb_log"] = np.log1p(np.maximum(arr[..., :3], 0.0))
                maps[f"{name}_alpha"] = arr[..., 3]
            else:
                maps[name] = read_exr_gray(path)
    return validate_arrays(maps, flow_names=flow_names)
