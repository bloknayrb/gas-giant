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


def write_exr_rgba(path: Path, rgba: np.ndarray) -> None:
    """(H, W, 4) float32 (unbounded HDR) -> RGBA ZIP-compressed scanline EXR.
    Grouped "RGBA" channels produce standard R/G/B/A planes on disk —
    exactly what Blender's Image Texture expects."""
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"expected (H, W, 4) array, got {rgba.shape}")
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    channels = {"RGBA": np.ascontiguousarray(rgba, dtype=np.float32)}
    with OpenEXR.File(header, channels) as f:
        f.write(str(path))


def decode_image(path: Path) -> np.ndarray:
    """Read an equirect grayscale mask PNG -> (H, W) float32 in [0, 1].

    Single-channel: a color image is converted to luminance. The mask MUST be a
    2:1 equirect (width == 2*height) -- anything else is a clear error, not a
    silent stretch. A missing/unreadable file raises OSError."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f"cv2.imread failed for {path} (missing or unreadable image)")
    if img.ndim == 3:
        # BGR(A) -> single-channel luminance (drop any alpha first).
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    if w != 2 * h:
        raise ValueError(
            f"{path}: mask must be a 2:1 equirect (width == 2*height), got {w}x{h}"
        )
    maxv = np.float32(65535.0 if img.dtype == np.uint16 else 255.0)
    return np.ascontiguousarray(img.astype(np.float32) / maxv)


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
        arr = np.asarray(channels[name].pixels, dtype=np.float32)
    if arr.ndim != 2:
        # The first-channel fallback would otherwise silently return an
        # (H, W, C) plane group for a multi-channel file.
        raise ValueError(f"{path}: expected a single-channel EXR, got {arr.shape}")
    return arr


def read_exr_rgba(path: Path) -> np.ndarray:
    """RGBA EXR -> (H, W, 4) float32. Handles both a grouped "RGBA" plane
    (our writer) and separate R/G/B/A channels (third-party files)."""
    with OpenEXR.File(str(path)) as f:
        channels = f.channels()
        if "RGBA" in channels:
            arr = np.asarray(channels["RGBA"].pixels, dtype=np.float32)
        elif "RGB" in channels:
            rgb = np.asarray(channels["RGB"].pixels, dtype=np.float32)
            a = (np.asarray(channels["A"].pixels, dtype=np.float32)
                 if "A" in channels else np.ones(rgb.shape[:2], np.float32))
            arr = np.concatenate([rgb, a[..., None]], axis=-1)
        else:
            planes = [np.asarray(channels[c].pixels, dtype=np.float32)
                      for c in ("R", "G", "B") if c in channels]
            if len(planes) != 3:
                raise ValueError(f"{path}: no RGB(A) channels found ({list(channels)})")
            a = (np.asarray(channels["A"].pixels, dtype=np.float32)
                 if "A" in channels else np.ones(planes[0].shape, np.float32))
            arr = np.stack([*planes, a], axis=-1)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"{path}: expected (H, W, 4), got {arr.shape}")
    return arr
