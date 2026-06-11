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


def validate_arrays(maps: dict[str, np.ndarray]) -> Report:
    report = Report()
    for name, arr in maps.items():
        check_finite(arr, name, report)
        check_wrap_continuity(arr, name, report)
        check_pole_rows(arr, name, report)
        check_latitude_band_continuity(arr, name, report)
    return report


def validate_mapset(mapset_dir: Path) -> Report:
    """Load an exported map set via its manifest and run all checks."""
    from gasgiant.export.manifest import read_manifest
    from gasgiant.export.writers import read_exr_gray, read_png16

    manifest = read_manifest(mapset_dir)
    maps: dict[str, np.ndarray] = {}
    for name, entry in manifest["maps"].items():
        path = mapset_dir / entry["file"]
        if entry["format"] == "png16":
            maps[name] = read_png16(path)
        elif entry["format"] == "exr32f":
            maps[name] = read_exr_gray(path)
    return validate_arrays(maps)
