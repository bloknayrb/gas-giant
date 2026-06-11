"""Checkpoint save/restore: the complete simulation state as a compressed
.npz + the generating params. Restoring rebuilds the sim from the saved
params (velocity/profiles/registry are deterministic from the seed) and
overwrites the tracer textures — this is the stepping-stone for the future
animation exporter (restore -> step k -> export per frame)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from gasgiant.params.presets import load_preset_doc, to_preset_doc

if TYPE_CHECKING:
    from gasgiant.engine.facade import Simulation

# Bumped whenever the seeded GENERATION algorithms change (band layout,
# vortex populations, profiles): a checkpoint pairs saved tracer textures
# with a registry/profile rebuild replayed from the seed, so a checkpoint
# from a different generation algorithm would load without error and render
# subtly wrong. 2 = v1.1 (small-storm field, polar field, width tail, ...).
GENERATION_VERSION = 2


def save_checkpoint(sim: Simulation, path: Path) -> None:
    s = sim.solver
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        preset=json.dumps(to_preset_doc(sim.params)),
        generation_version=GENERATION_VERSION,
        step_index=s.step_index,
        tracers_eq=sim.gpu.read_texture(s.equirect.tracers.cur),
        tracers_n=sim.gpu.read_texture(s.north.tracers.cur),
        tracers_s=sim.gpu.read_texture(s.south.tracers.cur),
    )


def load_checkpoint(path: Path, gpu=None) -> Simulation:
    from gasgiant.engine.facade import Simulation

    data = np.load(path, allow_pickle=False)
    found = int(data["generation_version"]) if "generation_version" in data else 1
    if found != GENERATION_VERSION:
        raise ValueError(
            f"{path}: checkpoint generation_version {found} != "
            f"{GENERATION_VERSION}; the seeded generation algorithms changed, "
            "so its tracers would pair with a different vortex registry. "
            "Re-create the checkpoint."
        )
    params = load_preset_doc(json.loads(str(data["preset"])), source=str(path))
    sim = Simulation(params, gpu)
    s = sim.solver
    for tex, key in (
        (s.equirect.tracers.cur, "tracers_eq"),
        (s.north.tracers.cur, "tracers_n"),
        (s.south.tracers.cur, "tracers_s"),
    ):
        tex.write(np.ascontiguousarray(data[key], dtype=np.float32).tobytes())
    s.step_index = int(data["step_index"])
    # Vortex drift is deterministic: replay it so the registry matches the
    # tracer state's step.
    for _ in range(s.step_index):
        s.vortices.drift(s.profiles, s.dt)
    sim._tracers_changed = True
    return sim
