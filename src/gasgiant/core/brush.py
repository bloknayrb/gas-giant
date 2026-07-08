"""Great-circle mask brush: pure-numpy painting into an equirect buffer.

The buffer is a single-channel ``(H, W)`` float32 array on the same equirect
grid the mask texture uses: column ``i`` maps longitude ``-180..+180`` left to
right, row ``j`` maps latitude ``+90..-90`` top to bottom (texel centers, per
``core.domain.EquirectGrid``). A stamp deposits (or, when erasing, removes) a
smooth radial blob measured in GREAT-CIRCLE degrees -- not euclidean pixels --
so a brush keeps its true angular size everywhere and does not smear into an
ellipse near the poles.

Two seamlessness properties fall out of using the great-circle metric over the
full longitude row of the affected latitude band:

- Dateline wrap: ``cos(lon - lon0)`` is periodic, so a stamp centered near
  +-180 paints continuously across the +-180 seam (both image edges rise).
- Pole coverage: near a pole ``cos(lat)`` -> 0, so the angular distance from
  the stamp center barely depends on longitude and a stamp at the pole paints
  the whole top/bottom latitude band -- no singular longitude and no NaN.

This module imports numpy only (it lives in the ``core`` layer, below ``gl``);
it never touches a GL context or any GUI code.
"""

from __future__ import annotations

import math

import numpy as np

# Canonical CPU paint buffer resolution (2:1 equirect). The GUI allocates its
# buffer at this size; brush.stamp works on any (H, W) 2:1 array though.
BUFFER_H = 512
BUFFER_W = 1024


def new_buffer(height: int = BUFFER_H, width: int = BUFFER_W) -> np.ndarray:
    """A zeroed ``(height, width)`` float32 paint buffer."""
    return np.zeros((height, width), dtype=np.float32)


def stamp(
    buffer: np.ndarray,
    lon_deg: float,
    lat_deg: float,
    radius_deg: float,
    strength: float,
    erase: bool = False,
) -> None:
    """Deposit a smooth brush stamp into ``buffer`` IN PLACE.

    ``(lon_deg, lat_deg)`` is the stamp center in degrees. ``radius_deg`` is the
    great-circle radius (degrees) at which the contribution falls to zero;
    ``strength`` is the peak deposit at the center. With ``erase=True`` the same
    profile is SUBTRACTED (painting toward 0 instead of toward 1). The result is
    clamped to ``[0, 1]``. Paint then erase at the same spot with the same
    radius/strength is an exact inverse (up to clamping), because the deposited
    profile is identical.

    The falloff is a smoothstep in ``d / radius_deg`` (weight 1 at the center,
    0 at and beyond ``radius_deg``), so the painted set -- the texels that
    actually change -- is exactly the texels strictly within ``radius_deg``
    great-circle distance of the center.

    Only the latitude band ``|lat - lat_deg| <= radius_deg`` is touched (a hard
    bound, since great-circle distance is at least the latitude difference); the
    full longitude row within that band is evaluated so the dateline-wrap and
    pole-coverage properties hold without special cases.
    """
    if buffer.ndim != 2:
        raise ValueError(f"paint buffer must be (H, W), got {buffer.shape}")
    if radius_deg <= 0.0 or strength == 0.0:
        return
    h, w = buffer.shape

    # Row-center latitudes (deg): +90 at row 0 down to -90 at the last row.
    row_lat = 90.0 - (np.arange(h, dtype=np.float64) + 0.5) / h * 180.0
    band = np.abs(row_lat - lat_deg) <= radius_deg
    if not band.any():
        return
    # row_lat is monotonic, so the band is a contiguous slice -> a real in-place
    # view of the buffer (fancy indexing would copy).
    rows = np.nonzero(band)[0]
    r0, r1 = int(rows[0]), int(rows[-1]) + 1

    # Column-center longitudes (deg): -180 at col 0 to +180 at the last col.
    col_lon = -180.0 + (np.arange(w, dtype=np.float64) + 0.5) / w * 360.0

    lat0 = math.radians(lat_deg)
    lon0 = math.radians(lon_deg)
    rlat = np.radians(row_lat[r0:r1])[:, None]  # (R, 1)
    rlon = np.radians(col_lon)[None, :]         # (1, W)

    # Great-circle angle via the spherical law of cosines; clamp guards arccos
    # against float overshoot just past +-1 (a pole-centered stamp lands there).
    cos_d = (
        math.sin(lat0) * np.sin(rlat)
        + math.cos(lat0) * np.cos(rlat) * np.cos(rlon - lon0)
    )
    np.clip(cos_d, -1.0, 1.0, out=cos_d)
    d = np.degrees(np.arccos(cos_d))  # (R, W) great-circle degrees

    t = np.clip(d / radius_deg, 0.0, 1.0)
    weight = 1.0 - t * t * (3.0 - 2.0 * t)  # smoothstep: 1 at center -> 0 at edge
    delta = (float(strength) * weight).astype(buffer.dtype)

    sub = buffer[r0:r1, :]  # contiguous view
    if erase:
        np.subtract(sub, delta, out=sub)
    else:
        np.add(sub, delta, out=sub)
    np.clip(sub, 0.0, 1.0, out=sub)
