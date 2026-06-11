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


def save_checkpoint(sim: Simulation, path: Path) -> None:
    s = sim.solver
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        preset=json.dumps(to_preset_doc(sim.params)),
        step_index=s.step_index,
        tracers_eq=sim.gpu.read_texture(s.equirect.tracers.cur),
        tracers_n=sim.gpu.read_texture(s.north.tracers.cur),
        tracers_s=sim.gpu.read_texture(s.south.tracers.cur),
    )


def load_checkpoint(path: Path, gpu=None) -> Simulation:
    from gasgiant.engine.facade import Simulation

    data = np.load(path, allow_pickle=False)
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
