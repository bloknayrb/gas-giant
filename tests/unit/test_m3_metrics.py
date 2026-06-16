"""Honest M3 render-gate metrics: latitude concentration and high-frequency
texture energy."""
from __future__ import annotations

import numpy as np

from gasgiant.render.m3_metrics import (
    banded_coherent_fraction,
    highfreq_energy,
    latitude_concentration,
)


def _banded_image(H=256, W=512, band_deg=(20.0, 55.0), amp=1.0, rng_seed=0):
    """Luminance image whose eddy variance is concentrated in a latitude band."""
    rng = np.random.default_rng(rng_seed)
    lat = 90.0 - (np.arange(H) + 0.5) / H * 180.0
    in_band = (np.abs(lat) >= band_deg[0]) & (np.abs(lat) <= band_deg[1])
    img = 0.5 + 0.001 * rng.standard_normal((H, W))
    img[in_band] += amp * rng.standard_normal((in_band.sum(), W))
    return np.clip(img, 0, 1).astype(np.float32)


def test_latitude_concentration_banded_vs_flat():
    flat = (0.5 + 0.001 * np.random.default_rng(1).standard_normal((256, 512))).astype(np.float32)
    banded = _banded_image(amp=0.2)
    assert latitude_concentration(flat) < 1.5
    assert latitude_concentration(banded) > 3.0


def test_banded_coherent_fraction_wave_vs_noise():
    """A coherent low-m zonal wave in the active band carries a far higher
    coherent fraction than broadband noise in the same band."""
    H, W = 256, 512
    lat = 90.0 - (np.arange(H) + 0.5) / H * 180.0
    in_band = (np.abs(lat) >= 20.0) & (np.abs(lat) <= 55.0)
    lam = np.arange(W) / W * 2 * np.pi

    rng = np.random.default_rng(3)
    wave = 0.5 + 0.001 * rng.standard_normal((H, W))
    wave[in_band] += 0.1 * np.cos(8 * lam)[None, :]          # coherent m=8
    wave = np.clip(wave, 0, 1).astype(np.float32)

    noise = (0.5 + 0.05 * rng.standard_normal((H, W))).astype(np.float32)

    f_wave = banded_coherent_fraction(wave)
    f_noise = banded_coherent_fraction(noise)
    assert f_wave > 0.8          # nearly all energy in m=1..12
    assert f_wave > 3.0 * f_noise


def test_highfreq_energy_monotonic():
    rng = np.random.default_rng(2)
    smooth = np.full((128, 256), 0.5, dtype=np.float32)
    noisy = (0.5 + 0.1 * rng.standard_normal((128, 256))).astype(np.float32)
    assert highfreq_energy(noisy) > highfreq_energy(smooth)
