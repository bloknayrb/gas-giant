"""P0.5 (v1.5): GPU baseline-hash gate.

Captures SHA1s of Simulation(params, gpu).render_maps(width) FLOAT32 arrays
(color + height + emission when present) for default params and jupiter_like at
a few resolutions x seeds, on THIS machine/driver. Hashes are over raw float32
bytes (never PNG -- quantization would mask exactly the FP drift this gate
exists to catch).

This replaces the falsified "pinned GPU fingerprints" premise: there are no
committed GPU fingerprints, only CPU SHA1s. Default-path stability across kernel
recompiles is verified EMPIRICALLY here -- run before P1, then re-assert after
every sim-side packet (P1, P3b, P3c) and at P6. A diff means a "default-off"
feature changed the default output: investigate (could be legitimate FP reorder
on recompile -- record the atol -- or a real bug).

dev_steps stays > 0 deliberately: the quick-test dev_steps=0 path never executes
advect.comp, so P1's kernel would be invisible to a dev_steps=0 hash.

Usage:
    uv run python scripts/p05_baseline_hash.py            # print current hashes
    uv run python scripts/p05_baseline_hash.py --check     # compare to baseline json
The first run writes out/audit/p05_baseline.json if absent; --check asserts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gasgiant.engine.facade import Simulation
from gasgiant.gl import GpuContext
from gasgiant.params.model import PlanetParams
from gasgiant.params.presets import load_factory_preset

BASELINE = Path("out/audit/p05_baseline.json")
SEEDS = [0, 1, 2]
# (label, preset_or_none, width). dev_steps forced > 0 below.
CONFIGS = [
    ("default@512", None, 512),
    ("jupiter@512", "jupiter_like", 512),
    ("jupiter@1024", "jupiter_like", 1024),
]
DEV_STEPS = 60  # > 0 so advect.comp runs; small enough to stay quick


def _params(preset: str | None) -> PlanetParams:
    if preset is None:
        return PlanetParams()
    return load_factory_preset(preset)


def _hash_maps(maps: dict[str, np.ndarray]) -> dict[str, str]:
    out = {}
    for key in sorted(maps):
        arr = np.ascontiguousarray(maps[key], dtype=np.float32)
        out[key] = hashlib.sha1(arr.tobytes()).hexdigest()
    return out


def capture(gpu: GpuContext) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for label, preset, width in CONFIGS:
        for seed in SEEDS:
            p = _params(preset)
            p.seed = seed
            p.sim.resolution = 512
            p.sim.dev_steps = DEV_STEPS
            sim = Simulation(p, gpu)
            maps = sim.render_maps(width)
            result[f"{label}:seed{seed}"] = _hash_maps(maps)
            sim._release_sim()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="assert against committed baseline")
    args = ap.parse_args()

    gpu = GpuContext.headless()
    gpu.make_current()
    current = capture(gpu)

    for key in sorted(current):
        print(key, current[key])

    if args.check:
        if not BASELINE.exists():
            raise SystemExit(f"--check requested but {BASELINE} does not exist")
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        drift = {k: (baseline.get(k), current[k]) for k in current if baseline.get(k) != current[k]}
        if drift:
            print("\nHASH DRIFT DETECTED:")
            for k, (b, c) in drift.items():
                print(f"  {k}\n    baseline={b}\n    current ={c}")
            raise SystemExit(1)
        print(f"\nOK: all {len(current)} hashes match baseline.")
    else:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        if BASELINE.exists():
            print(f"\n{BASELINE} exists -- not overwriting. Use --check to compare.")
        else:
            BASELINE.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
            print(f"\nWrote baseline -> {BASELINE}")


if __name__ == "__main__":
    main()
