"""Image writers: 16-bit PNG via OpenCV, float32 EXR via OpenEXR.

All writers take float32 arrays in [0, 1] (or unbounded for EXR) and own the
format conversion. Isolated here so a library swap touches one file.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import OpenEXR


def write_png16_rgb(path: Path, rgb: np.ndarray, compression: int = 2) -> None:
    """(H, W, 3) float32 0..1 -> 16-bit RGB PNG. Values are clipped."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"expected (H, W, 3+) array, got {rgb.shape}")
    u16 = np.clip(rgb[..., :3], 0.0, 1.0)
    u16 = (u16 * 65535.0 + 0.5).astype(np.uint16)
    bgr = u16[..., ::-1]  # OpenCV writes BGR order
    ok = cv2.imwrite(str(path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, int(compression)])
    if not ok:
        raise OSError(f"cv2.imwrite failed for {path}")


def write_png16_rgb_u16(path: Path, rgb_u16: np.ndarray, compression: int = 2) -> None:
    """(H, W, 3) uint16 (already converted) -> 16-bit RGB PNG, no extra copy."""
    if rgb_u16.dtype != np.uint16 or rgb_u16.ndim != 3:
        raise ValueError(f"expected (H, W, 3) uint16, got {rgb_u16.dtype} {rgb_u16.shape}")
    ok = cv2.imwrite(str(path), rgb_u16[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, int(compression)])
    if not ok:
        raise OSError(f"cv2.imwrite failed for {path}")


def write_png16_gray(path: Path, gray: np.ndarray, compression: int = 2) -> None:
    """(H, W) float32 0..1 -> 16-bit grayscale PNG."""
    if gray.ndim != 2:
        raise ValueError(f"expected (H, W) array, got {gray.shape}")
    u16 = (np.clip(gray, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    ok = cv2.imwrite(str(path), u16, [cv2.IMWRITE_PNG_COMPRESSION, int(compression)])
    if not ok:
        raise OSError(f"cv2.imwrite failed for {path}")


def write_exr_gray(path: Path, gray: np.ndarray) -> None:
    """(H, W) float32 -> single-channel ("Y") ZIP-compressed scanline EXR."""
    if gray.ndim != 2:
        raise ValueError(f"expected (H, W) array, got {gray.shape}")
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    channels = {"Y": np.ascontiguousarray(gray, dtype=np.float32)}
    with OpenEXR.File(header, channels) as f:
        f.write(str(path))


def read_png16(path: Path) -> np.ndarray:
    """16-bit PNG -> float32 0..1; RGB order for color images, (H, W) for gray."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f"cv2.imread failed for {path}")
    arr = img.astype(np.float32) / np.float32(65535.0)
    if arr.ndim == 3:
        arr = arr[..., ::-1].copy()  # BGR -> RGB
    return arr


def read_exr_gray(path: Path) -> np.ndarray:
    """Single-channel EXR -> (H, W) float32."""
    with OpenEXR.File(str(path)) as f:
        channels = f.channels()
        name = "Y" if "Y" in channels else next(iter(channels))
        return np.asarray(channels[name].pixels, dtype=np.float32)
