"""T17: the cube-map face-continuity validator (orientation table + continuity
check) on synthetic faces -- pure numpy, no GL.

The orientation table (which edge of face A meets which edge of face B, and
whether the shared-edge parameter runs forward or reversed) is the fiddly part.
These tests pin it two ways: (1) geometrically -- the border DIRECTIONS of the
two faces of each of the 12 edges agree once the reversed flag is applied; and
(2) end to end -- a smooth field sampled per face passes continuity, while a
per-face offset (a seam) fails.
"""

from __future__ import annotations

import numpy as np

from gasgiant.validate.seams import (
    CUBE_EDGE_TABLE,
    CUBE_FACE_NAMES,
    _cube_dir,
    _cube_edge_line,
    check_cube_face_continuity,
    validate_cube_arrays,
)


def _face_dir_image(face: int, n: int) -> np.ndarray:
    """(n, n, 3) unit directions for every texel of ``face`` (texel centers)."""
    c = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    uc = c[None, :] * np.ones((n, 1))   # column -> uc
    vc = c[:, None] * np.ones((1, n))   # row -> vc
    dx, dy, dz = np.broadcast_arrays(*_cube_dir(face, uc, vc))
    d = np.stack([dx, dy, dz], axis=-1).astype(np.float64)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def test_edge_table_has_twelve_unique_edges():
    assert len(CUBE_EDGE_TABLE) == 12
    # Each physical edge appears once (faceA < faceB, no duplicate corner set).
    for fa, _ea, fb, _eb, _rev in CUBE_EDGE_TABLE:
        assert fa < fb
    # Every face contributes exactly 4 edges across the whole table.
    counts = {i: 0 for i in range(6)}
    for fa, _ea, fb, _eb, _rev in CUBE_EDGE_TABLE:
        counts[fa] += 1
        counts[fb] += 1
    assert all(c == 4 for c in counts.values())


def test_edge_table_orientation_matches_directions():
    """For each table edge, the two faces' border directions coincide (to within
    ~1 texel) once the reversed flag is applied -- proving the edge pairing AND
    the forward/reversed orientation are geometrically correct."""
    n = 64
    dirs = [_face_dir_image(f, n) for f in range(6)]
    for fa, ea, fb, eb, rev in CUBE_EDGE_TABLE:
        la = _cube_edge_line(dirs[fa], ea)
        lb = _cube_edge_line(dirs[fb], eb)
        if rev:
            lb = lb[::-1]
        # Border texels of the two faces sit ~1 texel apart across the seam.
        assert np.abs(la - lb).max() < 3.0 / n, (
            f"edge {CUBE_FACE_NAMES[fa]}.{ea}|{CUBE_FACE_NAMES[fb]}.{eb} "
            f"orientation mismatch"
        )


def _smooth_field(face: int, n: int) -> np.ndarray:
    """A smooth scalar field of direction -> continuous across every seam."""
    d = _face_dir_image(face, n)
    return (0.5 + 0.3 * d[..., 0] + 0.2 * d[..., 1] - 0.15 * d[..., 2]).astype(np.float32)


def test_smooth_field_passes_continuity():
    n = 96
    faces = {CUBE_FACE_NAMES[f]: _smooth_field(f, n) for f in range(6)}
    from gasgiant.validate.seams import Report

    report = Report()
    check_cube_face_continuity(faces, "color", report)
    assert report.ok, report.summary()


def test_per_face_offset_fails_continuity():
    """A constant offset on one face is a seam on all four of its edges -> the
    continuity check must catch it."""
    n = 96
    faces = {CUBE_FACE_NAMES[f]: _smooth_field(f, n) for f in range(6)}
    faces["pz"] = faces["pz"] + 0.5  # break the +Z face away from its neighbors
    from gasgiant.validate.seams import Report

    report = Report()
    check_cube_face_continuity(faces, "color", report)
    assert not report.ok


def test_validate_cube_arrays_finite_and_continuous():
    n = 48
    faces = {CUBE_FACE_NAMES[f]: _smooth_field(f, n) for f in range(6)}
    report = validate_cube_arrays({"color": faces})
    assert report.ok, report.summary()
    # A non-finite value trips the finiteness check.
    bad = {k: v.copy() for k, v in faces.items()}
    bad["px"][0, 0] = np.nan
    assert not validate_cube_arrays({"color": bad}).ok
