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

from gasgiant.params.model import SolverType
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
# 4 = v1.6: vorticity-solver prognostic state (omega field + warm-start psi)
# serialized for vorticity-mode round-trip (equirect only — P4 incomplete).
# 5 = v1.6 Phase B P8a: all three domains' omega+psi state serialized so
# polar-patch vorticity round-trips are byte-exact.
# 6 = placement-chirality fixes (review F12/F06, 2026-07-03): festoon edge
# selection and hero wake geometry changed, so old checkpoints' tracers were
# developed under different generation; registry gains wake_lat_off and the
# previously-unserialized aspect (restored heroes silently reset to round on
# aspect!=1 presets before this).
# 7 = storm cast list (origin-marked vortices): the registry gains a CPU-side
# origin marker ("seeded"/"cast"); cast storms are exempt from the population
# cap and runtime mergers, so a resumed checkpoint must restore which entries
# were cast for the merger/trim exemptions to hold.
# 8 = the frozen velocity texture is now saved in KINEMATIC mode too (not just
# vorticity). The detail-synth render pass flow-stretches through the velocity,
# so a resumed sim that never steps again needs it to reproduce the render;
# without it, resume->export was not byte-identical when detail was enabled.
# Pre-v8 kinematic checkpoints lack the velocity, so they are refused rather
# than silently rendering a wrong detail layer.
# 9 = GRS hero-interaction pass (2026-07-16): generation output changed for
# every emergence-on registry (jet-derived wake frame flips warm's wake_dir
# east, bow_gain gates the belt bow, the accent roots hero-relative), the
# same class as the v6 bump. A v8 warm checkpoint resuming here would mix
# old-generation tracers with new-generation stamps/registry silently.
# 10 = vortex chirality fix (2026-07-17): every seeded/cast storm's `strength`
# sign flipped (co-rotate with ambient shear instead of counter-rotating;
# polar caps flipped to genuinely cyclonic), plus the warm-preset bake (a new
# local zonal jet term + hero_latitude retune). A v9 checkpoint's registry
# strengths and tracer textures were generated under the OLD (wrong-class)
# sign convention, so resuming here would silently mix old-sign storms with
# new-sign profiles/detail winding.
GENERATION_VERSION = 10

# Registry scalar fields serialized per vortex. float64: the "restored
# registry is identical" guarantee is exact-round-trip, and pack_ssbo computes
# cos/sin in float64 before its f32 cast — f32-quantized lat/lon would flip
# SSBO last bits. Trailing fields added later (cooldown/ttl) load tolerantly
# by npz key presence.
_REG_FIELDS = (
    "lat", "lon", "r_core", "strength", "kind", "tint", "brightness",
    "wake_dir", "wake_lat_off", "aspect",
    # bow_gain (added 2026-07-15, belt-bow gate) rode the v9 version bump, so
    # every loadable checkpoint carries it — loads are STRICT (a missing
    # registry column is data corruption, not a schema skew, and must raise).
    "bow_gain",
)


def save_checkpoint(sim: Simulation, path: Path) -> None:
    s = sim.solver
    path.parent.mkdir(parents=True, exist_ok=True)
    vortices = s.vortices.vortices
    reg = {
        f"reg_{name}": np.array([getattr(v, name) for v in vortices], dtype=np.float64)
        for name in _REG_FIELDS
    }
    reg["reg_cooldown"] = np.array([v.cooldown for v in vortices], dtype=np.int32)
    reg["reg_ttl"] = np.array([v.ttl for v in vortices], dtype=np.int32)
    # Cast-list back-reference (M2 CastLevers): a restored cast hero must keep the
    # storms.cast index it resolves its per-storm overrides against. Tolerant on
    # load (default -1) so pre-M2 checkpoints resume unchanged.
    reg["reg_cast_ref"] = np.array([v.cast_ref for v in vortices], dtype=np.int32)
    # Provenance marker (int-coded): 1 = cast-list storm, 0 = seeded. Restored
    # so the merger/trim cast exemptions survive a checkpoint round-trip.
    reg["reg_origin"] = np.array(
        [1 if v.origin == "cast" else 0 for v in vortices], dtype=np.int32
    )
    outbreak_links = np.full(len(s.events.outbreaks) if s.events else 0, -1, dtype=np.int32)
    if s.events is not None:
        index_of = {id(v): i for i, v in enumerate(vortices)}
        for j, ob in enumerate(s.events.outbreaks):
            if ob.vortex is not None:
                outbreak_links[j] = index_of.get(id(ob.vortex), -1)
    # The frozen velocity texture (dom.vel_tex, written from psi each step) is
    # saved in BOTH solver modes: the detail-synth render pass flow-stretches
    # its filaments through the velocity, so a resumed sim that never steps
    # again must carry the developed velocity or its detail layer diverges from
    # the original render (kinematic used to omit this, so resume->export was
    # NOT byte-identical when detail was on). Vorticity ALSO needs it because
    # _omega_step reads vel_tex to advect q on the first step after restore.
    vel_extra = {
        "vel_eq": sim.gpu.read_texture(s.equirect.vel_tex),
        "vel_n":  sim.gpu.read_texture(s.north.vel_tex),
        "vel_s":  sim.gpu.read_texture(s.south.vel_tex),
    }
    # Vorticity mode ALSO saves the prognostic ω field (state.cur, the advected
    # absolute-vorticity q) and the warm-start ψ (dom.psi_tex, last
    # solved+feathered) for ALL three domains. These keys are absent in
    # kinematic checkpoints (absent-key-tolerant on load).
    vort_extra: dict = {}
    if sim.params.solver.type == SolverType.VORTICITY and s._omega_states is not None:
        from gasgiant.sim.solver import DOMAIN_EQUIRECT, DOMAIN_NORTH, DOMAIN_SOUTH
        vort_extra["omega_eq"] = sim.gpu.read_texture(s._omega_states[DOMAIN_EQUIRECT].cur)
        vort_extra["psi_eq"]   = sim.gpu.read_texture(s.equirect.psi_tex)
        vort_extra["omega_n"]  = sim.gpu.read_texture(s._omega_states[DOMAIN_NORTH].cur)
        vort_extra["psi_n"]    = sim.gpu.read_texture(s.north.psi_tex)
        vort_extra["omega_s"]  = sim.gpu.read_texture(s._omega_states[DOMAIN_SOUTH].cur)
        vort_extra["psi_s"]    = sim.gpu.read_texture(s.south.psi_tex)
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
        **vel_extra,
        **vort_extra,
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
    # STRICT: the version gate above guarantees every _REG_FIELDS column is
    # present in a loadable checkpoint; a missing one is a truncated/corrupt
    # npz and must raise, not resume as zero-strength/zero-radius storms.
    cols = {name: data[f"reg_{name}"] for name in _REG_FIELDS}
    cooldown = data["reg_cooldown"] if "reg_cooldown" in data else np.zeros(n, np.int32)
    ttl = data["reg_ttl"] if "reg_ttl" in data else np.full(n, -1, np.int32)
    origin_arr = data["reg_origin"] if "reg_origin" in data else np.zeros(n, np.int32)
    cast_ref = data["reg_cast_ref"] if "reg_cast_ref" in data else np.full(n, -1, np.int32)
    s.vortices.vortices = [
        Vortex(
            **{name: float(cols[name][i]) for name in _REG_FIELDS},
            cooldown=int(cooldown[i]),
            ttl=int(ttl[i]),
            origin=("cast" if int(origin_arr[i]) == 1 else "seeded"),
            cast_ref=int(cast_ref[i]),
        )
        for i in range(n)
    ]
    if s.events is not None and "outbreak_links" in data:
        links = data["outbreak_links"]
        for j, ob in enumerate(s.events.outbreaks):
            idx = int(links[j]) if j < len(links) else -1
            ob.vortex = s.vortices.vortices[idx] if 0 <= idx < n else None

    # Restore the frozen velocity texture in BOTH solver modes: the detail-synth
    # render pass reads it (a resumed sim never steps again, so without it the
    # detail layer renders from a stale/step-0 velocity and resume->export is
    # not byte-identical), and vorticity's _omega_step reads it on the first
    # step after resume. Absent in pre-v8 kinematic checkpoints — tolerate.
    for dom, vel_key in ((s.equirect, "vel_eq"), (s.north, "vel_n"), (s.south, "vel_s")):
        if vel_key in data:
            dom.vel_tex.write(
                np.ascontiguousarray(data[vel_key], dtype=np.float32).tobytes()
            )

    # Vorticity mode ALSO restores the prognostic ω field and warm-start ψ for
    # all three domains. Keys are absent in kinematic checkpoints — tolerate.
    if params.solver.type == SolverType.VORTICITY and s._omega_states is not None:
        from gasgiant.sim.solver import DOMAIN_EQUIRECT, DOMAIN_NORTH, DOMAIN_SOUTH
        _vort_pairs = [
            (DOMAIN_EQUIRECT, s.equirect, "omega_eq", "psi_eq"),
            (DOMAIN_NORTH,    s.north,    "omega_n",  "psi_n"),
            (DOMAIN_SOUTH,    s.south,    "omega_s",  "psi_s"),
        ]
        for kind, dom, omega_key, psi_key in _vort_pairs:
            if omega_key in data:
                s._omega_states[kind].cur.write(
                    np.ascontiguousarray(data[omega_key], dtype=np.float32).tobytes()
                )
            if psi_key in data:
                dom.psi_tex.write(
                    np.ascontiguousarray(data[psi_key], dtype=np.float32).tobytes()
                )

    sim._tracers_changed = True
    return sim
