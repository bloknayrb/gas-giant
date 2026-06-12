"""Checkpoint save/restore: the complete simulation state as a compressed
.npz + the generating params. Restoring rebuilds the sim from the saved
params (velocity/profiles/jets are deterministic from the seed) and overwrites
the tracer textures AND the vortex registry — the registry is serialized, not
replayed, because live registry evolution (events, and later mergers) is not a
pure function of (seed, step) once mid-run param edits enter the picture.
This is the stepping-stone for the future animation exporter
(restore -> step k -> export per frame)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from gasgiant.params.presets import load_preset_doc, to_preset_doc
from gasgiant.sim.vortices import Vortex

if TYPE_CHECKING:
    from gasgiant.engine.facade import Simulation

# Bumped whenever the seeded GENERATION algorithms or the checkpoint format
# change: a checkpoint pairs saved tracer textures with profiles/jets rebuilt
# from the seed, so a mismatch would load without error and render subtly
# wrong. 2 = v1.1 (small-storm field, polar field, width tail, ...).
# 3 = v1.2: the vortex registry is serialized into the checkpoint (older
# checkpoints carried no registry state and relied on drift-only replay,
# which mis-restored outbreak-bearing or mid-edit runs).
GENERATION_VERSION = 3

# Registry scalar fields serialized per vortex. float64: the "restored
# registry is identical" guarantee is exact-round-trip, and pack_ssbo computes
# cos/sin in float64 before its f32 cast — f32-quantized lat/lon would flip
# SSBO last bits. Trailing fields added later (cooldown/ttl) load tolerantly
# by npz key presence.
_REG_FIELDS = ("lat", "lon", "r_core", "strength", "kind", "tint", "brightness", "wake_dir")


def save_checkpoint(sim: Simulation, path: Path) -> None:
    s = sim.solver
    path.parent.mkdir(parents=True, exist_ok=True)
    vortices = s.vortices.vortices
    reg = {
        f"reg_{name}": np.array([getattr(v, name) for v in vortices], dtype=np.float64)
        for name in _REG_FIELDS
    }
    reg["reg_cooldown"] = np.array([v.cooldown for v in vortices], dtype=np.int32)
    outbreak_links = np.full(len(s.events.outbreaks) if s.events else 0, -1, dtype=np.int32)
    if s.events is not None:
        index_of = {id(v): i for i, v in enumerate(vortices)}
        for j, ob in enumerate(s.events.outbreaks):
            if ob.vortex is not None:
                outbreak_links[j] = index_of.get(id(ob.vortex), -1)
    np.savez_compressed(
        path,
        preset=json.dumps(to_preset_doc(sim.params)),
        generation_version=GENERATION_VERSION,
        step_index=s.step_index,
        extra_steps=np.int32(sim.steps_target - sim.params.sim.dev_steps),
        tracers_eq=sim.gpu.read_texture(s.equirect.tracers.cur),
        tracers_n=sim.gpu.read_texture(s.north.tracers.cur),
        tracers_s=sim.gpu.read_texture(s.south.tracers.cur),
        outbreak_links=outbreak_links,
        **reg,
    )


def load_checkpoint(path: Path, gpu=None) -> Simulation:
    from gasgiant.engine.facade import Simulation

    data = np.load(path, allow_pickle=False)
    found = int(data["generation_version"]) if "generation_version" in data else 1
    if found != GENERATION_VERSION:
        raise ValueError(
            f"{path}: checkpoint generation_version {found} != "
            f"{GENERATION_VERSION}; the generation algorithms or checkpoint "
            "format changed, so its tracers would pair with a different "
            "vortex registry. Re-create the checkpoint."
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
    sim._extra_steps = int(data["extra_steps"]) if "extra_steps" in data else 0

    # The registry was serialized at save time; the freshly generated one is
    # discarded (bands/profiles/jets/schedule do not depend on it).
    n = int(data["reg_lat"].shape[0])
    cols = {name: data[f"reg_{name}"] for name in _REG_FIELDS}
    cooldown = data["reg_cooldown"] if "reg_cooldown" in data else np.zeros(n, np.int32)
    s.vortices.vortices = [
        Vortex(
            **{name: float(cols[name][i]) for name in _REG_FIELDS},
            cooldown=int(cooldown[i]),
        )
        for i in range(n)
    ]
    if s.events is not None and "outbreak_links" in data:
        links = data["outbreak_links"]
        for j, ob in enumerate(s.events.outbreaks):
            idx = int(links[j]) if j < len(links) else -1
            ob.vortex = s.vortices.vortices[idx] if 0 <= idx < n else None
    sim._tracers_changed = True
    return sim
