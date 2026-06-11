"""Grid topology and projections.

Conventions (must match the GLSL kernels and validate/seams.py exactly):

- Equirect texel centers: pixel (i, j) on a WxH grid samples
  lon = ((i + 0.5) / W) * 2*pi - pi          (-pi..pi, periodic)
  lat =  pi/2 - ((j + 0.5) / H) * pi          (+pi/2 at row 0 down to -pi/2)
  There is no duplicated 0/360 column; wrap continuity is a property of
  adjacent texels, not column identity.

Polar azimuthal-equidistant patches arrive in Phase 3b.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EquirectGrid:
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width != 2 * self.height:
            raise ValueError(f"equirect grid must be 2:1, got {self.width}x{self.height}")

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def lonlat(self) -> tuple[np.ndarray, np.ndarray]:
        """(lon, lat) arrays in radians at texel centers; shapes (H, W) and (H,)."""
        i = np.arange(self.width, dtype=np.float64)
        j = np.arange(self.height, dtype=np.float64)
        lon = ((i + 0.5) / self.width) * (2.0 * np.pi) - np.pi
        lat = np.pi / 2.0 - ((j + 0.5) / self.height) * np.pi
        return np.broadcast_to(lon, (self.height, self.width)), lat

    def sphere_points(self) -> np.ndarray:
        """(H, W, 3) float32 unit-sphere positions at texel centers (y = north)."""
        lon, lat = self.lonlat()
        lat_col = lat[:, None]
        cos_lat = np.cos(lat_col)
        pts = np.empty((self.height, self.width, 3), dtype=np.float32)
        pts[..., 0] = cos_lat * np.cos(lon)
        pts[..., 1] = np.sin(lat_col)
        pts[..., 2] = cos_lat * np.sin(lon)
        return pts
