# Field-Driven Detail Placement Implementation Plan

> **⚠️ SUPERSEDED (2026-07-07).** This plan was executed (strain-driven placement,
> commits `0d67929..83c02b8`) and then reverted the same day: visual calibration
> found strain-selective density read patchy, so the feature became **uniform
> detail coverage** (`detail.spread`). See the banner in the paired spec and
> `docs/roadmap.md`. Kept as an execution record.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive detail-FX *placement* from local flow (eddy strain `|∇v|` + vorticity) instead of latitude LUTs, as a default-off byte-identical `FIELD_DRIVE` shader variant.

**Architecture:** A new sim-res "activity" pass computes raw strain + vorticity from the baked equirect velocity gradient tensor. A CPU readback reduces it to a per-latitude-row mean (for eddy = strain − rowmean) and two global scalar means. `detail.comp` gains a `#ifdef FIELD_DRIVE` block that, at sample time, normalizes eddy-strain against the mean with an absolute floor, partitions it into cell/lace/fold flavors, and blends each existing latitude gate toward the flow-driven placement by `field_drive`. Default text (variant undefined) is byte-for-byte today's kernel.

**Tech Stack:** Python 3.13, moderngl (GLSL 430 compute), pydantic params, pytest. `uv` for all commands.

## Global Constraints

- **Layering** (import-linter): `params|palette -> gl -> core -> sim -> render -> jobs -> export -> engine -> app|cli`. `render/activity.py` sits in `render`; it may import `gl`, `params`. No GUI imports below `app`. Run `uv run lint-imports` after any new import.
- **Byte-identity when off** is via preprocessor variant, NOT a runtime branch. `FIELD_DRIVE` undefined ⇒ post-preprocess detail.comp text is byte-for-byte today's. The `#else` arm of every wrapped base-path line is a VERBATIM copy of the current line — a stray space breaks identity.
- **Determinism:** kinematic path byte-exact; vorticity path within documented floors (`GPU_NOISE_ATOL = 1e-2`, SOR LSB noise ~0.004 cross-session). NEVER write a byte-exact assertion against vorticity-mode output.
- **preview==export:** the activity field + its reduction must be computed identically in the facade preview path and the `ExportSnapshot` export path. The reduction is numpy (identical code both paths).
- **New opt-in lever checklist** (CLAUDE.md): pfield (tier POST, default no-op) → shader uniform + preprocessor block → variant predicate → `_set` wiring → dedicated behavior test + forced-variant no-op test → `docs/sliders.md` entry.
- **Line length 100** (`uv run ruff check .`). E701/E702 off.
- **Fast test loop:** `uv run pytest -m "not gpu and not slow" -q` (~40s). GPU tier needs a GL 4.3 context (skips cleanly without one).
- **Establish a baseline before editing:** run `uv run python scripts/p05_baseline_hash.py --check` and the target test subset BEFORE the first code change; byte-identity gates fail on ANY tracked default-output move, including someone else's uncommitted work.

---

## File Structure

- **Create** `src/gasgiant/render/kernels/activity.comp` — velocity-gradient → raw strain/vort, RG32F, sim res.
- **Create** `src/gasgiant/render/activity.py` — `ActivitySynth` (programs, `build`, `release`), `ActivityMeans` dataclass (scalar means + per-row-mean 1-D texture).
- **Create** `tests/unit/test_field_drive_metadata.py` — CPU metadata/predicate/tripwire pins.
- **Create** `tests/gpu/test_field_drive.py` — GPU routing byte-identity, forced-variant no-op, activity finiteness, behavior, seam.
- **Create** `tests/unit/test_field_drive_golden_hash.py` — preprocessed non-variant detail.comp text hash (`#else`-drift guard).
- **Modify** `src/gasgiant/params/model.py` — `pfield` gains `field_drive` kwarg; `DetailParams` gains 3 pfields.
- **Modify** `src/gasgiant/render/detail.py` — `field_drive_enabled` predicate, `_assert_field_drive_uniforms` tripwire, `_FIELD_DRIVE_PARAMS`, cache key `(fx, field_drive)`, `synthesize` gains `activity`/`means` args + FIELD_DRIVE uniform block.
- **Modify** `src/gasgiant/render/kernels/detail.comp` — `#ifdef FIELD_DRIVE` driver block, per-site `mix` blends, re-keyed guards, base-path `#ifdef/#else`.
- **Modify** `src/gasgiant/engine/facade.py` — facade-owned `ActivitySynth` + `_activity`/`_activity_means`; build in `_derive` when enabled; release+null in `_release_sim`.
- **Modify** `src/gasgiant/engine/snapshot.py` — `activity_eq` + `activity_means` on `ExportSnapshot`, gated build in `capture()`, release in `release()`.
- **Modify** `src/gasgiant/export/exporter.py` — build activity per frame from snapshot velocity in both loops; `_derive_tile` gains `activity`/`means`.
- **Modify** `docs/sliders.md`, `docs/architecture.md`, `docs/roadmap.md`.

---

## Task 1: Params, predicate, and metadata pin

**Files:**
- Modify: `src/gasgiant/params/model.py` (`pfield` signature ~L45-90; `DetailParams` ~L835 end)
- Modify: `src/gasgiant/render/detail.py` (add predicate + param-name helper)
- Test: `tests/unit/test_field_drive_metadata.py` (create)

**Interfaces:**
- Produces: `pfield(..., field_drive: bool = False)` stores `extra["field_drive"] = True`. `DetailParams.field_drive: float`, `.field_scale: float`, `.field_vort_influence: float`. `detail_mod.field_drive_enabled(params: DetailParams) -> bool`; `detail_mod._FIELD_DRIVE_PARAMS: tuple[str, ...]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_field_drive_metadata.py
"""Field-driven detail: the FIELD_DRIVE variant selector is METADATA (pfield
`field_drive=True`), mirroring the DETAIL_FX `fx=True` machinery. Only
`field_drive` selects the variant; `field_scale`/`field_vort_influence` are
plain sample-time tunables and must NOT be selectors (M5)."""
from __future__ import annotations

from gasgiant.params.model import DetailParams
from gasgiant.render import detail as detail_mod

EXPECTED_FIELD_DRIVE_LEVERS = {"field_drive"}


def test_field_drive_metadata_matches_selector():
    assert set(detail_mod._FIELD_DRIVE_PARAMS) == EXPECTED_FIELD_DRIVE_LEVERS


def test_field_drive_flag_lives_on_exactly_field_drive():
    flagged = {
        name
        for name, info in DetailParams.model_fields.items()
        if isinstance(info.json_schema_extra, dict)
        and info.json_schema_extra.get("field_drive")
    }
    assert flagged == EXPECTED_FIELD_DRIVE_LEVERS


def test_predicate_off_by_default_and_only_field_drive_selects():
    assert detail_mod.field_drive_enabled(DetailParams()) is False
    assert detail_mod.field_drive_enabled(DetailParams(field_drive=1e-6)) is True
    # field_scale / field_vort_influence alone must NOT select the variant
    assert detail_mod.field_drive_enabled(DetailParams(field_scale=4.0)) is False
    assert detail_mod.field_drive_enabled(
        DetailParams(field_vort_influence=1.0)
    ) is False


def test_new_levers_are_post_tier_and_no_rand():
    for name in ("field_drive", "field_scale", "field_vort_influence"):
        extra = DetailParams.model_fields[name].json_schema_extra
        assert extra["tier"] == "post", name
        assert "rand" not in extra, f"{name} rand draw would reorder randomize"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_field_drive_metadata.py -q`
Expected: FAIL — `AttributeError: module 'gasgiant.render.detail' has no attribute '_FIELD_DRIVE_PARAMS'` / `DetailParams` has no field `field_drive`.

- [ ] **Step 3a: Add the `field_drive` kwarg to `pfield`**

In `src/gasgiant/params/model.py`, add `field_drive: bool = False` to the `pfield` signature (next to `fx`), and after the `if fx:` block:

```python
    if fx:
        extra["fx"] = True
    if field_drive:
        extra["field_drive"] = True
```

- [ ] **Step 3b: Add the three pfields to `DetailParams`**

Append to `DetailParams` (after `polar_filaments`, ~L835):

```python
    field_drive: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=1.0, adv=True, ui="Detail",
        field_drive=True,
        description="Place detail-FX texture by LOCAL FLOW (eddy strain |grad v|) "
                    "instead of the latitude band LUT: at full drive, folds land "
                    "on jet edges/vortex rims/fold zones and quiescent interiors "
                    "clear, so band structure emerges from the flow. 0 = pure "
                    "latitude gating (byte-identical). Vorticity presets first",
    )
    field_scale: float = pfield(
        1.0, tier=Tier.POST, lo=0.25, hi=4.0, adv=True, ui="Detail",
        description="Normalization scale k in strain/(k*mean): raise to require "
                    "stronger-than-average strain before texture appears (cleaner "
                    "interiors), lower to spread texture onto weaker structure. "
                    "Sample-time only; not a variant selector",
    )
    field_vort_influence: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=1.0, adv=True, ui="Detail",
        description="Add lace inside vortex cores (high |vorticity|, low strain) "
                    "where the strain driver alone leaves them bare. Only bites "
                    "when field_drive>0; not a variant selector",
    )
```

- [ ] **Step 3c: Add the predicate + param helper to `render/detail.py`**

After `_FX_PARAMS` (~L41):

```python
def _field_drive_param_names() -> tuple[str, ...]:
    """FIELD_DRIVE selector lever(s) from pfield `field_drive=True` metadata —
    ONLY `field_drive` (M5): `field_scale`/`field_vort_influence` are sample-time
    tunables that do nothing at drive=0, so they are not variant selectors."""
    return tuple(
        name
        for name, info in DetailParams.model_fields.items()
        if isinstance(info.json_schema_extra, dict)
        and info.json_schema_extra.get("field_drive")
    )


_FIELD_DRIVE_PARAMS: tuple[str, ...] = _field_drive_param_names()


def field_drive_enabled(params: DetailParams) -> bool:
    """True when field_drive>0 -> select the FIELD_DRIVE program variant. Exact
    zero keeps the non-variant program (byte-identical by construction)."""
    return any(getattr(params, name) > 0.0 for name in _FIELD_DRIVE_PARAMS)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_field_drive_metadata.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Guard the existing metadata + randomize goldens**

Run: `uv run pytest tests/unit/test_detail_fx_metadata.py tests/unit/test_randomize.py -q`
Expected: PASS — `field_drive`/`field_scale`/`field_vort_influence` are NOT `fx=True` (so `EXPECTED_FX_LEVERS` unchanged) and carry no `rand` (so the randomize draw order golden is unchanged).

- [ ] **Step 6: Lint + commit**

Run: `uv run ruff check src/gasgiant/params/model.py src/gasgiant/render/detail.py && uv run lint-imports`
```bash
git add src/gasgiant/params/model.py src/gasgiant/render/detail.py tests/unit/test_field_drive_metadata.py
git commit -m "feat(params): field_drive/field_scale/field_vort_influence pfields + FIELD_DRIVE predicate"
```

---

## Task 2: Activity pass — `activity.comp` + `ActivitySynth`

**Files:**
- Create: `src/gasgiant/render/kernels/activity.comp`
- Create: `src/gasgiant/render/activity.py`
- Test: `tests/gpu/test_field_drive.py` (create; activity portion)

**Interfaces:**
- Consumes: `GpuContext.compute`, `GpuContext.texture2d`, `GpuContext.read_texture`, `GpuContext.lut_texture`.
- Produces:
  - `ActivityMeans` dataclass: `mean_eddy: float`, `mean_vort: float`, `rowmean_tex: moderngl.Texture` (sim-H × 1, `.r` = per-row mean strain); method `.release()`.
  - `ActivitySynth(gpu).build(vel_tex, out_tex) -> ActivityMeans` — writes RG32F raw strain/vort into caller-supplied `out_tex` (sim res, mipmap-capable), builds its mipmaps, reads it back once, computes the `|lat|<66°`-masked global eddy mean, `|vort|` mean, and per-row mean strain (numpy), uploads the per-row mean as a fresh 1-D texture, returns `ActivityMeans`.
  - `ActivitySynth.release()`.
  - `ActivitySynth.SIM_MASK_DEG = 66.0` (band edge; aligned with detail.comp ROUTE_LO per M13).

- [ ] **Step 1: Write the failing GPU test (finiteness + mean sanity)**

```python
# tests/gpu/test_field_drive.py
"""GPU tier: field-driven detail activity pass + FIELD_DRIVE variant."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

from gasgiant.render.activity import ActivitySynth


def _shear_velocity(gpu, w, h):
    """A pure zonal jet u=sin(2*lat) (strong midlat shear, zero vorticity-free
    check not needed) as an RG f4 equirect velocity, repeat_x wrapped."""
    lat = (0.5 - (np.arange(h) + 0.5) / h) * np.pi  # +pi/2..-pi/2
    u = np.sin(2.0 * lat)[:, None] * np.ones((1, w))
    vel = np.zeros((h, w, 2), np.float32)
    vel[:, :, 0] = u
    tex = gpu.texture2d((w, h), 2, "f4", data=vel, linear=True)
    return tex


def test_activity_is_finite_including_poles(gpu):
    w, h = 256, 128
    synth = ActivitySynth(gpu)
    vel = _shear_velocity(gpu, w, h)
    act = gpu.texture2d((w, h), 2, "f4", linear=True)
    means = synth.build(vel, act)
    arr = gpu.read_texture(act)
    assert np.all(np.isfinite(arr)), "activity has NaN/Inf (pole 1/cos blowup?)"
    assert means.mean_eddy >= 0.0
    means.release()
    synth.release()


def test_activity_strain_peaks_at_jet_shear(gpu):
    w, h = 256, 128
    synth = ActivitySynth(gpu)
    vel = _shear_velocity(gpu, w, h)
    act = gpu.texture2d((w, h), 2, "f4", linear=True)
    synth.build(vel, act)
    strain = gpu.read_texture(act)[:, :, 0]
    # du/dlat = 2cos(2lat): max at lat=0 (equator row), min at lat=+-45deg.
    eq_row = h // 2
    q_row = h // 4  # ~+45deg
    assert strain[eq_row].mean() > strain[q_row].mean()
    synth.release()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/gpu/test_field_drive.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gasgiant.render.activity'` (or skip if no GL context; run on a GL box).

- [ ] **Step 3a: Write `activity.comp`**

```glsl
// src/gasgiant/render/kernels/activity.comp
// Local flow "activity" for field-driven detail placement: raw strain rate and
// vorticity from the baked equirect velocity gradient tensor. NO normalization
// here (the mean divide + absolute floor happen at detail sample time so the
// map stays cache-valid across a field_scale POST edit). Output RG32F sim-res:
//   R = strain = sqrt((du_dlam' - dv_dphi)^2 + (du_dphi + dv_dlam')^2)
//   G = vort   = dv_dlam' - du_dphi
// where the prime denotes the 1/cos(phi) metric factor on the longitude deriv.
#version 430
layout(local_size_x = 16, local_size_y = 16) in;
layout(rg32f, binding = 0) uniform image2D u_out;

uniform sampler2D u_vel;          // equirect velocity (rg), repeat_x
uniform ivec2 u_size;             // sim resolution
uniform vec2 u_texel;             // (1/W, 1/H)

const float PI = 3.14159265358979;
const float COS_FLOOR = 0.30;     // raised floor: cap the 1/cos pole blowup
const float ACT_CLAMP = 32.0;     // bound strain/vort before write

vec2 velAt(vec2 uv) { return texture(u_vel, uv).rg; }

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    if (px.x >= u_size.x || px.y >= u_size.y) return;
    vec2 uv = (vec2(px) + 0.5) * u_texel;
    float lat = 0.5 * PI - uv.y * PI;           // +pi/2 .. -pi/2
    // Central differences over +-1 texel via normalized coords (texture(),
    // NOT texelFetch) so the +-180 x-seam wraps (u_vel is repeat_x).
    vec2 vxp = velAt(uv + vec2(u_texel.x, 0.0));
    vec2 vxm = velAt(uv - vec2(u_texel.x, 0.0));
    vec2 vyp = velAt(uv + vec2(0.0, u_texel.y));
    vec2 vym = velAt(uv - vec2(0.0, u_texel.y));
    // d/dlam (x) and d/dphi (y). uv.y increases southward => phi decreases,
    // so d/dphi = -(d/duv.y). Metric: divide the longitude deriv by cos(phi).
    float inv2dx = 1.0 / (2.0 * u_texel.x * 2.0 * PI); // duv.x -> dlam (radians)
    float inv2dy = 1.0 / (2.0 * u_texel.y * PI);        // duv.y -> dphi magnitude
    float cphi = max(cos(lat), COS_FLOOR);
    float du_dlam = (vxp.x - vxm.x) * inv2dx / cphi;
    float dv_dlam = (vxp.y - vxm.y) * inv2dx / cphi;
    float du_dphi = -(vyp.x - vym.x) * inv2dy;
    float dv_dphi = -(vyp.y - vym.y) * inv2dy;
    float sh = du_dlam - dv_dphi;
    float st = du_dphi + dv_dlam;
    float strain = sqrt(sh * sh + st * st);
    float vort = dv_dlam - du_dphi;
    // Finite-scrub + clamp (pole rows / seam can still spike).
    if (isnan(strain) || isinf(strain)) strain = 0.0;
    if (isnan(vort)   || isinf(vort))   vort = 0.0;
    strain = clamp(strain, 0.0, ACT_CLAMP);
    vort = clamp(vort, -ACT_CLAMP, ACT_CLAMP);
    imageStore(u_out, px, vec4(strain, vort, 0.0, 0.0));
}
```

- [ ] **Step 3b: Write `render/activity.py`**

```python
# src/gasgiant/render/activity.py
"""Activity pass: local strain + vorticity from the baked equirect velocity,
plus the CPU reduction (per-row mean strain + global eddy/vort means) that the
FIELD_DRIVE detail variant normalizes against. The reduction is numpy so the
preview and export paths divide by an identical value (preview==export)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gasgiant.gl import GpuContext

if TYPE_CHECKING:
    import moderngl

_KERNELS = "gasgiant.render.kernels"
_GROUP = 16


@dataclass
class ActivityMeans:
    """Snapshot/caller-scoped reduction outputs. rowmean_tex is a sim-H x 1 LUT
    (.r = masked per-row mean strain) sampled by latitude in detail.comp; the
    caller owns it and must release() it when the activity texture is released."""

    mean_eddy: float
    mean_vort: float
    rowmean_tex: moderngl.Texture

    def release(self) -> None:
        self.rowmean_tex.release()


class ActivitySynth:
    SIM_MASK_DEG = 66.0  # |lat|<66 band for the mean (aligned to ROUTE_LO, M13)

    def __init__(self, gpu: GpuContext) -> None:
        self.gpu = gpu
        self.prog = gpu.compute(_KERNELS, "activity.comp")

    def build(self, vel_tex: moderngl.Texture, out_tex: moderngl.Texture) -> ActivityMeans:
        """Fill out_tex (RG32F, sim res, mipmap-capable) with raw strain/vort,
        build its mipmaps, and reduce to per-row + global means."""
        w, h = out_tex.size
        vel_tex.use(location=0)
        self.prog["u_vel"].value = 0
        self.prog["u_size"].value = (w, h)
        self.prog["u_texel"].value = (1.0 / w, 1.0 / h)
        out_tex.bind_to_image(0, read=False, write=True)
        gx = (w + _GROUP - 1) // _GROUP
        gy = (h + _GROUP - 1) // _GROUP
        self.prog.run(gx, gy, 1)
        self.gpu.ctx.memory_barrier()
        out_tex.build_mipmaps()  # M9: fill sample uses textureLod
        # CPU reduction (deterministic, exact, per-row for eddy strain, M3).
        arr = self.gpu.read_texture(out_tex)            # (h, w, 2)
        strain = arr[:, :, 0]
        vort = np.abs(arr[:, :, 1])
        lat_deg = (0.5 - (np.arange(h) + 0.5) / h) * 180.0
        band = np.abs(lat_deg) < self.SIM_MASK_DEG      # (h,)
        rowmean = strain.mean(axis=1)                   # (h,) — full row (all lon)
        eddy = np.clip(strain - rowmean[:, None], 0.0, None)
        mean_eddy = float(eddy[band].mean()) if band.any() else 0.0
        mean_vort = float(vort[band].mean()) if band.any() else 0.0
        lut = np.zeros((h, 4), dtype=np.float32)
        lut[:, 0] = rowmean
        rowmean_tex = self.gpu.lut_texture(lut)         # h x 1, linear, clamped
        return ActivityMeans(mean_eddy, mean_vort, rowmean_tex)

    def release(self) -> None:
        # Program lives in the GpuContext compute cache; nothing else owned.
        pass
```

Note: `out_tex` must be created mipmap-capable with a mip filter. In callers (Task 5/6), allocate it via `gpu.texture2d(size, 2, "f4", linear=True)` then set `tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)` before the first `build`. Encapsulate that in a helper (Step 3c).

- [ ] **Step 3c: Add a mip-capable allocator helper**

Append to `render/activity.py`:

```python
def new_activity_texture(gpu: GpuContext, size: tuple[int, int]) -> moderngl.Texture:
    """RG32F, sim-res, mip-capable activity target (textureLod fill sample)."""
    import moderngl
    tex = gpu.texture2d(size, 2, "f4", linear=True)
    tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    return tex
```

Update the two GPU tests in Step 1 to allocate via `new_activity_texture(gpu, (w, h))` instead of `gpu.texture2d(..., 2, ...)`.

- [ ] **Step 4: Run to verify the activity tests pass**

Run: `uv run pytest tests/gpu/test_field_drive.py -q`
Expected: PASS (2 tests) on a GL box; skip cleanly without a context.

- [ ] **Step 5: Lint + commit**

Run: `uv run ruff check src/gasgiant/render/activity.py && uv run lint-imports`
```bash
git add src/gasgiant/render/kernels/activity.comp src/gasgiant/render/activity.py tests/gpu/test_field_drive.py
git commit -m "feat(render): activity pass (raw strain/vort + CPU eddy/row-mean reduction)"
```

---

## Task 3: `detail.comp` FIELD_DRIVE variant (shader)

**Files:**
- Modify: `src/gasgiant/render/kernels/detail.comp` (main body ~L193-263 + new block)
- Test: `tests/unit/test_field_drive_golden_hash.py` (create)

**Interfaces:**
- Consumes: new sampler `u_activity` (unit 7), `u_rowmean` (a spare unit, use 8), uniforms `u_field_drive`, `u_field_scale`, `u_field_vort`, `u_mean_eddy`, `u_mean_vort`, `u_act_texel`. All set by Task 4.
- Produces: FIELD_DRIVE-variant placement identical in shape to the spec §"detail.comp — new FIELD_DRIVE variant".

**Cut-point constants** (compile-time `const` in the `#ifdef` block; calibrated on renders in the separate rollout, first-guess here):
`A = 0.35`, `M = 0.75`, `D = 1.30` (strain-normalized), `S_REF_ABS = 0.02`, `W_REF_ABS = 0.02`, `MEAN_LO = 0.01`, `MEAN_HI = 0.05`, `SHEAR_HI = 1.5`, `FILL_W = 0.5`, `FILL_LOD = 2.0`.

- [ ] **Step 1: Write the failing golden-hash unit test**

```python
# tests/unit/test_field_drive_golden_hash.py
"""The non-FIELD_DRIVE preprocessed detail.comp text must be byte-stable: the
base-path #ifdef/#else wrapping (Task 3) puts today's w_streak/w_cell lines in
the #else arm, and a stray edit there would silently move default output. This
pins the flattened (include-expanded, no-defines) source hash. Textual — no GL
context needed (gl/context flattening is pure string ops)."""
from __future__ import annotations

import hashlib

from gasgiant.gl.context import _load_flattened


def test_non_variant_detail_source_hash_is_stable():
    source, _ = _load_flattened("gasgiant.render.kernels", "detail.comp", {})
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    # Regenerate deliberately (paste the printed value) ONLY when a real base-
    # path change is intended; a diff here on a FIELD_DRIVE-only edit is a bug.
    assert digest == "PLACEHOLDER_FILL_FROM_FIRST_RUN", digest
```

- [ ] **Step 2: Capture the current baseline hash (pre-edit)**

Run: `uv run pytest tests/unit/test_field_drive_golden_hash.py -q`
Expected: FAIL printing the current digest. Paste that digest into the assert, replacing `PLACEHOLDER_FILL_FROM_FIRST_RUN`. Re-run → PASS. **This locks today's text before any shader edit.**

- [ ] **Step 3a: Declare the new uniforms (before any `#include` that uses them)**

Near the existing uniform declarations at the top of `detail.comp`, add (unconditionally — declaring an unused uniform is harmless and the tripwire wants them in the variant; but to preserve byte-identity of the DEFAULT text, wrap in `#ifdef FIELD_DRIVE`):

```glsl
#ifdef FIELD_DRIVE
uniform sampler2D u_activity;   // unit 7: RG32F raw strain(.r)/vort(.g), sim res
uniform sampler2D u_rowmean;    // unit 8: per-row mean strain LUT (.r)
uniform float u_field_drive;
uniform float u_field_scale;
uniform float u_field_vort;
uniform float u_mean_eddy;
uniform float u_mean_vort;
#endif
```

- [ ] **Step 3b: Insert the driver block in `main()` after `belt` is read (~L208), before `w_streak`/`w_cell`**

Add, guarded, computing the placement axis and per-site weights (mirrors spec pseudocode; `eqUV(ll)` and `ll` are already in scope; `latV` = the same latitude coord used for `u_profile_dyn`):

```glsl
#ifdef FIELD_DRIVE
    float fd_latV = clamp((0.5 * PI - ll.y) / PI, 0.0, 1.0);
    const float A = 0.35, M_CUT = 0.75, D = 1.30;
    const float S_REF_ABS = 0.02, W_REF_ABS = 0.02;
    const float MEAN_LO = 0.01, MEAN_HI = 0.05, SHEAR_HI = 1.5;
    const float FILL_W = 0.5, FILL_LOD = 2.0;
    float fdS     = texture(u_activity, eqUV(ll)).r;
    float fdSfill = textureLod(u_activity, eqUV(ll), FILL_LOD).r;
    float fdW     = texture(u_activity, eqUV(ll)).g;
    float fdRow   = texture(u_rowmean, vec2(fd_latV, 0.5)).r;
    float fdSe    = max(fdS     - fdRow, 0.0);
    float fdSfe   = max(fdSfill - fdRow, 0.0);
    float fdME    = max(u_mean_eddy, 1e-6);
    float strain_n = fdSe  / max(u_field_scale * fdME, S_REF_ABS);
    float fill_n   = fdSfe / max(u_field_scale * fdME, S_REF_ABS);
    float wn       = abs(fdW) / max(u_field_scale * u_mean_vort, W_REF_ABS);
    float drive_eff = u_field_drive
                    * smoothstep(MEAN_LO, MEAN_HI, fdME)   // self-disable quiet fields
                    * (1.0 - routeW);                      // pole fade (M13)
    float place    = max(strain_n, FILL_W * fill_n);
    float fd_cell  = 1.0 - smoothstep(A, M_CUT, place);
    float fd_lace  = smoothstep(A, M_CUT, place) * (1.0 - smoothstep(M_CUT, D, place));
    float fd_fold  = smoothstep(M_CUT, D, place);
    fd_lace += u_field_vort * wn * (1.0 - fd_fold);        // vortex core-fill
    float shear_drv = smoothstep(0.0, SHEAR_HI, place);
    float fold_place = fd_fold;
#endif
```

- [ ] **Step 3c: Wrap the base-path `w_streak`/`w_cell` (current L260-263) in `#ifdef/#else`**

Replace the current two assignments with (the `#else` arm is a VERBATIM copy of today's lines):

```glsl
#ifdef FIELD_DRIVE
    float fd_sh = mix(shearN, shear_drv, drive_eff);
    w_streak = clamp(0.2 + 0.8 * (fd_sh + speedN), 0.0, 1.0) * (0.4 + 0.6 * tr.b)
             * (1.0 + 1.4 * hero);
    w_cell = u_cell_amount * mix(1.0 - belt, fd_cell, drive_eff) * (1.0 - speedN)
           * (1.0 - fd_sh) * (1.0 - 0.6 * routeW);
#else
    w_streak = clamp(0.2 + 0.8 * (shearN + speedN), 0.0, 1.0) * (0.4 + 0.6 * tr.b)
             * (1.0 + 1.4 * hero);
    w_cell = u_cell_amount * (1.0 - belt) * (1.0 - speedN) * (1.0 - shearN)
           * (1.0 - 0.6 * routeW);  // the caps are not quiet zone interiors
#endif
```

- [ ] **Step 3d: Blend the DETAIL_FX belt/mottle sites + re-key guards**

Inside the existing `#ifdef DETAIL_FX` region, at each belt-placement site, gate by `fold_place`/`drive_eff` when FIELD_DRIVE is also defined. Use nested `#ifdef FIELD_DRIVE` so a DETAIL_FX-only build is unchanged. For the belt floor (current L288):

```glsl
#if defined(FIELD_DRIVE)
    float belt_place = mix(belt, fold_place, drive_eff);
    w_streak += u_belt_texture * 0.45 * belt_place * gate * (1.0 - routeW);
#else
    w_streak += u_belt_texture * 0.45 * belt * gate * (1.0 - routeW);
#endif
```

Apply the analogous `belt_place`/`aw' = mix(aw, fd_lace, drive_eff)` substitution and the re-keyed early-out guards (`max(belt, fold_place*drive_eff) > 0.02`, `belt < 0.98` similarly, mottle `if (aw > 0.0 || fd_lace*drive_eff > 0.0)`) at the `belt_texture`, `belt_texture_fine`, striation belt-gate, `zone_texture`, and `mottle` sites per the spec §"Then blend each existing gate". Each guarded with `#if defined(FIELD_DRIVE)` so the DETAIL_FX-only text is byte-stable.

- [ ] **Step 4: Confirm the golden hash still passes (default text unchanged)**

Run: `uv run pytest tests/unit/test_field_drive_golden_hash.py -q`
Expected: PASS — the flattened NO-defines source is unchanged (every edit is inside `#ifdef FIELD_DRIVE`/`#else` with verbatim else-arms). **If this fails, a base-path line drifted — fix before continuing.**

- [ ] **Step 5: Commit (shader compiles via Task 4's program build; behavior verified there)**

```bash
git add src/gasgiant/render/kernels/detail.comp tests/unit/test_field_drive_golden_hash.py
git commit -m "feat(shader): detail.comp FIELD_DRIVE variant (eddy-strain placement, verbatim #else)"
```

---

## Task 4: Wire `DetailSynth.synthesize` + tripwire + cache key

**Files:**
- Modify: `src/gasgiant/render/detail.py`
- Test: `tests/unit/test_field_drive_metadata.py` (add dispatch cross-ref), `tests/gpu/test_field_drive.py` (add routing byte-identity + forced-variant no-op)

**Interfaces:**
- Consumes: `ActivityMeans` (Task 2), `_FIELD_DRIVE_PARAMS`/`field_drive_enabled` (Task 1).
- Produces: `DetailSynth.synthesize(..., activity=None, means=None)` — when `field_drive_enabled(params)`, selects the `(fx, field_drive=True)` program, binds `u_activity`@7 + `u_rowmean`@8, sets `u_field_drive/u_field_scale/u_field_vort/u_mean_eddy/u_mean_vort`. `_assert_field_drive_uniforms(prog)`.

- [ ] **Step 1: Write the failing GPU routing byte-identity test**

Add to `tests/gpu/test_field_drive.py`:

```python
from gasgiant.params.model import DetailParams
from gasgiant.render.detail import DetailSynth


def _synth_detail(gpu, params, activity=None, means=None):
    # minimal 1x1-hero-free equirect inputs at output res
    w, h = 128, 64
    vel = gpu.texture2d((w, h), 2, "f4", linear=True)
    tracers = gpu.texture2d((w, h), 4, "f4", linear=True)
    prof = gpu.lut_texture(np.zeros((h, 4), np.float32))
    out = gpu.texture2d((w, h), 1, "f4", linear=True)
    DetailSynth(gpu).synthesize(
        7, vel, tracers, prof, out, params, activity=activity, means=means,
    )
    return gpu.read_texture(out)[:, :, 0]


def test_field_drive_zero_is_byte_identical_to_default(gpu):
    base = _synth_detail(gpu, DetailParams(intensity=0.55))
    off = _synth_detail(gpu, DetailParams(intensity=0.55, field_drive=0.0))
    np.testing.assert_array_equal(base, off)  # non-variant program selected
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/gpu/test_field_drive.py::test_field_drive_zero_is_byte_identical_to_default -q`
Expected: FAIL — `synthesize()` has no `activity`/`means` kwargs.

- [ ] **Step 3a: Add the tripwire + FIELD_DRIVE program cache**

In `render/detail.py`, add after `_assert_fx_uniforms`:

```python
_FIELD_DRIVE_UNIFORMS = ("u_field_drive", "u_field_vort", "u_activity")


def _assert_field_drive_uniforms(prog) -> None:
    """Tripwire mirroring _assert_fx_uniforms: the FIELD_DRIVE variant must
    expose the placement uniforms, else the KeyError-suppressing _set silently
    no-ops the whole effect."""
    missing = [u for u in _FIELD_DRIVE_UNIFORMS if _absent(prog, u)]
    if missing:
        raise RuntimeError(
            f"detail.comp FIELD_DRIVE variant missing uniform(s) {missing}: "
            f"placement would silently no-op."
        )
```

- [ ] **Step 3b: Re-key the program cache to `(fx, field_drive)`**

Change `self._progs: dict[bool, ...]` to `dict[tuple[bool, bool], ...]` and `_program`:

```python
    def _program(self, fx: bool, field_drive: bool = False):
        key = (fx, field_drive)
        if key not in self._progs:
            defines = {}
            if fx:
                defines["DETAIL_FX"] = "1"
            if field_drive:
                defines["FIELD_DRIVE"] = "1"
            prog = self.gpu.compute(_KERNELS, "detail.comp", defines=defines or None)
            if fx:
                _assert_fx_uniforms(prog)
            if field_drive:
                _assert_field_drive_uniforms(prog)
            self._progs[key] = prog
        return self._progs[key]
```

Update `__init__`: `self.prog = self._program(fx=False)`.

- [ ] **Step 3c: Extend `synthesize`**

Add `activity: moderngl.Texture | None = None` and `means: ActivityMeans | None = None` params (import `ActivityMeans` under `TYPE_CHECKING`). Replace `fx_on = ...; prog = self._program(fx=fx_on)` with:

```python
        fx_on = detail_fx_enabled(params)
        fd_on = field_drive_enabled(params)
        prog = self._program(fx=fx_on, field_drive=fd_on)
```

After the polar binding block (which uses units 3-6), before `prog["u_origin"]`, add the FIELD_DRIVE binding:

```python
        if fd_on:
            if activity is None or means is None:
                raise ValueError(
                    "field_drive>0 but no activity/means supplied to synthesize"
                )
            activity.use(location=7)
            prog["u_activity"].value = 7
            means.rowmean_tex.use(location=8)
            prog["u_rowmean"].value = 8
            _set(prog, "u_field_drive", params.field_drive)
            _set(prog, "u_field_scale", params.field_scale)
            _set(prog, "u_field_vort", params.field_vort_influence)
            _set(prog, "u_mean_eddy", means.mean_eddy)
            _set(prog, "u_mean_vort", means.mean_vort)
```

- [ ] **Step 4: Run the routing test**

Run: `uv run pytest tests/gpu/test_field_drive.py::test_field_drive_zero_is_byte_identical_to_default -q`
Expected: PASS — `field_drive=0` selects the non-variant program; output identical.

- [ ] **Step 5: Add the forced-variant no-op + dispatch cross-ref tests**

Add to `tests/gpu/test_field_drive.py` (forced variant needs real activity/means):

```python
def test_field_drive_forced_variant_is_near_default(gpu):
    from gasgiant.render.activity import ActivitySynth, new_activity_texture
    w, h = 128, 64
    velfield = gpu.texture2d((w, h), 2, "f4", linear=True)
    act = new_activity_texture(gpu, (w, h))
    means = ActivitySynth(gpu).build(velfield, act)
    base = _synth_detail(gpu, DetailParams(intensity=0.55))
    tiny = _synth_detail(
        gpu, DetailParams(intensity=0.55, field_drive=1e-6), activity=act, means=means,
    )
    np.testing.assert_allclose(base, tiny, atol=1e-3)  # cross-binary FP reschedule
    means.release()
```

Add to `tests/unit/test_field_drive_metadata.py` a dispatch cross-ref mirroring `test_fx_levers_cross_reference_the_synthesize_dispatch_block`: assert every `_FIELD_DRIVE_UNIFORMS` name that maps to a lever is uploaded, and `field_drive` is read + uploaded in the `if fd_on:` block. (Copy the AST-walk helper, keyed to `fd_on`.)

Run: `uv run pytest tests/gpu/test_field_drive.py tests/unit/test_field_drive_metadata.py -q`
Expected: PASS.

- [ ] **Step 6: Lint + commit**

Run: `uv run ruff check src/gasgiant/render/detail.py && uv run lint-imports`
```bash
git add src/gasgiant/render/detail.py tests/gpu/test_field_drive.py tests/unit/test_field_drive_metadata.py
git commit -m "feat(render): wire FIELD_DRIVE variant in DetailSynth.synthesize + tripwire"
```

---

## Task 5: Facade — build activity in preview `_derive`, invalidation, release

**Files:**
- Modify: `src/gasgiant/engine/facade.py` (`__init__` ~L45-58, `_release_sim` ~L106-113, `_derive` ~L339-388)
- Test: `tests/gpu/test_field_drive.py` (facade preview integration)

**Interfaces:**
- Consumes: `ActivitySynth`, `new_activity_texture`, `ActivityMeans`.
- Produces: `self.activity_synth`, `self._activity: moderngl.Texture | None`, `self._activity_means: ActivityMeans | None`; helper `self._ensure_activity()` returning `(activity_tex, means)` sized to `s.equirect.vel_tex.size`, rebuilt whenever `_derive` runs the detail synth (tied to the same `_post_dirty`/`_tracers_changed` re-derive signal — M1).

- [ ] **Step 1: Write the failing integration test**

```python
def test_facade_preview_field_drive_builds_and_differs(gpu):
    from gasgiant.engine.facade import Simulation
    from gasgiant.params.presets import load_preset_doc
    import json, pathlib
    doc = json.loads(pathlib.Path("src/gasgiant/presets/gas_giant_warm.json").read_text())
    params = load_preset_doc(doc, "test").model_copy(deep=True)
    sim = Simulation(params, gpu=gpu)
    sim.run_to_completion(1)
    base_color, _ = sim.ensure_preview(256)
    base = gpu.read_texture(base_color).copy()
    sim.params.detail.field_drive = 1.0  # POST edit
    sim.update_params(sim.params)        # marks _post_dirty
    fd_color, _ = sim.ensure_preview(256)
    fd = gpu.read_texture(fd_color)
    assert not np.allclose(base, fd, atol=1e-2), "field_drive=1 did not change preview"
```

(Adjust `update_params`/`run_to_completion` call names to the real facade API if they differ — verify against `facade.py`.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/gpu/test_field_drive.py::test_facade_preview_field_drive_builds_and_differs -q`
Expected: FAIL — no activity built; `field_drive=1` currently raises `ValueError` in synthesize (no activity supplied) OR is a no-op.

- [ ] **Step 3a: `__init__` — construct synth + null state**

After `self.detail_synth = DetailSynth(self.gpu)` (L45):

```python
        from gasgiant.render.activity import ActivitySynth
        self.activity_synth = ActivitySynth(self.gpu)
        self._activity: moderngl.Texture | None = None
        self._activity_means = None  # ActivityMeans | None
```

- [ ] **Step 3b: `_release_sim` — release + null (M7)**

Add to `_release_sim` (after the profile releases):

```python
        if self._activity is not None:
            self._activity.release()
            self._activity = None
        if self._activity_means is not None:
            self._activity_means.release()
            self._activity_means = None
```

- [ ] **Step 3c: `_derive` — build activity when field_drive on**

Replace the detail-synth block in `_derive` (L356-369):

```python
        activity = None
        means = None
        if p.detail.intensity > 0.0:
            from gasgiant.engine.snapshot import hero_centers
            from gasgiant.render.activity import new_activity_texture
            from gasgiant.render.detail import PolarRoute, field_drive_enabled

            detail_tex = self._get_detail_tex(color_tex.size)
            if field_drive_enabled(p.detail):
                vsize = s.equirect.vel_tex.size
                if self._activity is None or self._activity.size != vsize:
                    if self._activity is not None:
                        self._activity.release()
                    self._activity = new_activity_texture(self.gpu, vsize)
                # M1: rebuild every derive — velocity re-bakes on every sim step,
                # and _derive only runs on _post_dirty/_tracers_changed anyway,
                # so this is exactly as fresh as the detail it feeds.
                if self._activity_means is not None:
                    self._activity_means.release()
                self._activity_means = self.activity_synth.build(
                    s.equirect.vel_tex, self._activity
                )
                activity, means = self._activity, self._activity_means
            self.detail_synth.synthesize(
                p.seed, s.equirect.vel_tex, s.equirect.tracers.cur,
                self.profile_dyn, detail_tex, p.detail,
                heroes=hero_centers(self.vortices),
                polar=PolarRoute(
                    s.north.vel_tex, s.south.vel_tex,
                    s.north.tracers.cur, s.south.tracers.cur, RHO_MAX,
                ),
                activity=activity, means=means,
            )
```

(Keep the existing `detail_tex = None` initializer above the `if`.)

- [ ] **Step 4: Run the integration test + preview byte-identity guard**

Run: `uv run pytest tests/gpu/test_field_drive.py -q`
Expected: PASS. Then confirm default preview unchanged:

Run: `uv run python scripts/p05_baseline_hash.py --check`
Expected: PASS (field_drive defaults 0 ⇒ no activity built ⇒ default render hash unmoved).

- [ ] **Step 5: Lint + commit**

Run: `uv run ruff check src/gasgiant/engine/facade.py && uv run lint-imports`
```bash
git add src/gasgiant/engine/facade.py tests/gpu/test_field_drive.py
git commit -m "feat(engine): build activity for FIELD_DRIVE in preview _derive; release+null on restart"
```

---

## Task 6: Snapshot — activity_eq + means (export, snapshot-scoped)

**Files:**
- Modify: `src/gasgiant/engine/snapshot.py`
- Test: `tests/gpu/test_field_drive.py` (snapshot capture/release)

**Interfaces:**
- Consumes: `ActivitySynth`, `new_activity_texture`.
- Produces: `ExportSnapshot.activity_eq: moderngl.Texture | None`, `ExportSnapshot.activity_means: ActivityMeans | None` — built in `capture()` ONLY when `field_drive_enabled(sim.params.detail)` (M8), released in `release()`. Snapshot-scoped exactly like `vel_eq`.

- [ ] **Step 1: Write the failing test**

```python
def test_snapshot_builds_activity_only_when_field_drive_on(gpu):
    from gasgiant.engine.facade import Simulation
    from gasgiant.params.presets import load_preset_doc
    import json, pathlib
    doc = json.loads(pathlib.Path("src/gasgiant/presets/gas_giant_warm.json").read_text())
    params = load_preset_doc(doc, "t").model_copy(deep=True)
    sim = Simulation(params, gpu=gpu)
    sim.run_to_completion(1)
    snap_off = sim.create_snapshot()
    assert snap_off.activity_eq is None
    snap_off.release()
    sim.params.detail.field_drive = 1.0
    snap_on = sim.create_snapshot()
    assert snap_on.activity_eq is not None
    assert snap_on.activity_means is not None
    snap_on.release()  # must not raise
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/gpu/test_field_drive.py::test_snapshot_builds_activity_only_when_field_drive_on -q`
Expected: FAIL — `ExportSnapshot` has no `activity_eq`.

- [ ] **Step 3a: Add fields + gated build**

Add to the `ExportSnapshot` dataclass (after `warp`):

```python
    activity_eq: moderngl.Texture = None  # type: ignore[assignment]
    activity_means: object = None  # ActivityMeans | None (avoid render import cycle)
```

In `capture()`, before the `return`, build gated:

```python
        from gasgiant.render.detail import field_drive_enabled
        activity_eq = None
        activity_means = None
        if field_drive_enabled(sim.params.detail):
            from gasgiant.render.activity import ActivitySynth, new_activity_texture
            activity_eq = new_activity_texture(gpu, s.equirect.vel_tex.size)
            # Build from the CLONED snapshot velocity so it matches the frozen tiles.
            vel_clone = gpu.clone_texture(s.equirect.vel_tex)
            try:
                activity_means = ActivitySynth(gpu).build(vel_clone, activity_eq)
            finally:
                vel_clone.release()
```

Pass `activity_eq=activity_eq, activity_means=activity_means` into the `cls(...)` call. (Note: `capture()` already clones `vel_eq`; reuse `vel_eq` instead of a second clone — build from the just-created `vel_eq` field to avoid a redundant clone. Prefer: build activity from `vel_eq` after it is created.)

Refined: create `vel_eq = gpu.clone_texture(s.equirect.vel_tex)` as a local first, pass it to `cls`, AND feed it to `ActivitySynth.build` — one clone, both uses.

- [ ] **Step 3b: Release in `release()`**

```python
    def release(self) -> None:
        for tex in (self.tracers_eq, self.tracers_n, self.tracers_s,
                    self.vel_eq, self.vel_n, self.vel_s,
                    self.profile_dyn, self.profile_stamp):
            tex.release()
        if self.activity_eq is not None:
            self.activity_eq.release()
        if self.activity_means is not None:
            self.activity_means.release()
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/gpu/test_field_drive.py::test_snapshot_builds_activity_only_when_field_drive_on -q`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

Run: `uv run ruff check src/gasgiant/engine/snapshot.py && uv run lint-imports`
```bash
git add src/gasgiant/engine/snapshot.py tests/gpu/test_field_drive.py
git commit -m "feat(engine): snapshot-scoped activity_eq + means, gated on field_drive"
```

---

## Task 7: Exporter — per-frame activity in both loops

**Files:**
- Modify: `src/gasgiant/export/exporter.py` (`_derive_tile` L46-88; both loops already pass `snap`)
- Test: `tests/gpu/test_field_drive.py` (export seam parity at field_drive>0)

**Interfaces:**
- Consumes: `snap.activity_eq`, `snap.activity_means`.
- Produces: `_derive_tile` passes `activity=snap.activity_eq, means=snap.activity_means` into `synthesize`. No signature change needed — `snap` already carries them.

- [ ] **Step 1: Write the failing seam test**

```python
def test_export_tiled_matches_full_at_field_drive(gpu):
    """Tiled export (per-tile origin/full_size) must equal a single-tile render
    at field_drive>0 (activity sampled by absolute lat/lon, seam-safe)."""
    from gasgiant.engine.facade import Simulation
    from gasgiant.export.exporter import _derive_tile
    from gasgiant.params.presets import load_preset_doc
    import json, pathlib
    doc = json.loads(pathlib.Path("src/gasgiant/presets/gas_giant_warm.json").read_text())
    params = load_preset_doc(doc, "t").model_copy(deep=True)
    params.detail.field_drive = 1.0
    sim = Simulation(params, gpu=gpu)
    sim.run_to_completion(1)
    snap = sim.create_snapshot()
    w, h = 256, 128
    # full render (one tile covering the map)
    full_c = gpu.texture2d((w, h), 4, "f4"); full_hh = gpu.texture2d((w, h), 1, "f4")
    full_d = gpu.texture2d((w, h), 1, "f4", linear=True)
    _derive_tile(sim, snap, snap.params, 0, 0, w, h, full_c, full_hh, full_d, None)
    full = gpu.read_texture(full_c).copy()
    # two vertically-split tiles
    tile_c = gpu.texture2d((w, h // 2), 4, "f4"); tile_hh = gpu.texture2d((w, h // 2), 1, "f4")
    tile_d = gpu.texture2d((w, h // 2), 1, "f4", linear=True)
    _derive_tile(sim, snap, snap.params, 0, 0, w, h, tile_c, tile_hh, tile_d, None)
    top = gpu.read_texture(tile_c)[: h // 2].copy()
    np.testing.assert_allclose(full[: h // 2], top, atol=1e-3)
    snap.release()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/gpu/test_field_drive.py::test_export_tiled_matches_full_at_field_drive -q`
Expected: FAIL — `_derive_tile` passes no `activity` ⇒ `synthesize` raises `ValueError` at field_drive>0.

- [ ] **Step 3: Pass activity through `_derive_tile`**

In `_derive_tile`, change the `synthesize` call (L66-74) to add:

```python
        sim.detail_synth.synthesize(
            params.seed, snap.vel_eq, snap.tracers_eq, snap.profile_dyn,
            tile_detail, params.detail, origin=(x0, y0), full_size=(w, h),
            heroes=snap.heroes,
            polar=PolarRoute(
                snap.vel_n, snap.vel_s, snap.tracers_n, snap.tracers_s,
                snap.patch_rho_max,
            ),
            activity=snap.activity_eq, means=snap.activity_means,
        )
```

Both `export_job` and `export_sequence_job` already build a fresh `snap` per frame (`sim.create_snapshot()`), so per-frame activity is automatic via Task 6's gated `capture()`.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/gpu/test_field_drive.py::test_export_tiled_matches_full_at_field_drive -q`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

Run: `uv run ruff check src/gasgiant/export/exporter.py && uv run lint-imports`
```bash
git add src/gasgiant/export/exporter.py tests/gpu/test_field_drive.py
git commit -m "feat(export): per-frame snapshot activity through _derive_tile (both loops)"
```

---

## Task 8: Full-suite green, 16K cost check, docs

**Files:**
- Modify: `docs/architecture.md`, `docs/roadmap.md`, `docs/sliders.md`
- Test: full tiers

- [ ] **Step 1: Full no-GPU tier**

Run: `uv run pytest -m "not gpu" -q`
Expected: PASS (~380 tests). Fix any fallout.

- [ ] **Step 2: GPU byte-identity smoke (the PR-blocking class)**

Run: `LP_NUM_THREADS=1 uv run pytest -m gpu -k "identical or noop or no_op or field_drive" -q`
(On the RTX box, establish a green/red baseline first — see CLAUDE.md machine-local flakiness protocol; CI llvmpipe is authoritative for byte-identity.)
Expected: routing byte-identity + p05 unchanged; field_drive tests green.

- [ ] **Step 3: 16K export cost re-verify (M10)**

Run a timed 16K export at `field_drive=1.0` on gas_giant_warm vs `field_drive=0.0`; confirm the re-keyed guards (`max(belt, fold_place*drive_eff) > 0.02`) still bound per-pixel backtrace cost (no runaway — the fill floor stays ~0 in genuinely quiet regions). Record the delta in the roadmap entry. If cost regresses materially, raise `S_REF_ABS`/`MEAN_LO` so quiet interiors early-out.

```bash
time uv run gasgiant export --preset gas_giant_warm --res 16384 --out out/fd_16k_off
# then with a field_drive=1 preset variant; compare wall time
```

- [ ] **Step 4: Docs**

- `docs/architecture.md`: add a short "field-driven detail placement" note under the detail/variants section (activity pass, eddy-strain normalization, default-off byte-identity, CPU reduction preview==export).
- `docs/roadmap.md`: mark the "separable companion win: drive amplitude masks from local 2-D sim fields" line as SHIPPED (default-off), with the 16K cost delta.
- `docs/sliders.md`: regenerate text: `uv run python scripts/render_slider_examples.py --no-render`; render the three new slider images per the script docstring (part of the calibration/rollout pass, not the default-off merge — note images pending if deferred). Confirm the drift gate: `uv run python scripts/render_slider_examples.py --check`.

- [ ] **Step 5: Final commit**

Run: `uv run ruff check . && uv run lint-imports && uv run python scripts/p05_baseline_hash.py --check`
```bash
git add docs/architecture.md docs/roadmap.md docs/sliders.md
git commit -m "docs: field-driven detail placement (shipped default-off; 16K cost noted)"
```

---

## Rollout (SEPARATE, gated on user visual sign-off — NOT part of this plan's merge)

Default-off merge lands via the tasks above (byte-identical, full matrix green). A follow-up visual pass then, on **gas_giant_warm / jupiter_vorticity first**: measure eddy-mean per preset, calibrate `A/M/D`, `S_REF_ABS`, `MEAN_LO/HI`, `field_scale`, `field_vort_influence`, `FILL_W/FILL_LOD` on renders; user signs off on montages; deliberate p05 re-baseline + preset JSON regen. `jupiter_like` only if a skeptical A/B shows a win; `saturn_pale`/`ice_giant` likely stay latitude-gated.

## Self-Review Notes

- **Spec coverage:** M1 (Task 5 rebuild tied to `_derive`), M2 (Task 6 snapshot-scoped means), M3 (Task 2 per-row eddy), M4 (Task 3 `place = max(strain_n, FILL_W*fill_n)` + low FILL_LOD), M5 (Task 1 predicate = `{field_drive}`), M6 (Task 3d mottle guard re-key), M7 (Task 5 release+null), M8 (Task 6 gated capture), M9 (Task 2 `build_mipmaps` + `new_activity_texture` mip filter), M10 (Task 8 16K re-verify), M11 (Task 3 A<M<D, sum=1), M13 (Task 2 `SIM_MASK_DEG=66` aligned ROUTE_LO), M14 (Task 2 CPU means → `u_rowmean` LUT + scalar uniforms, no GPU→CPU→GPU per-pixel stall), M15 (Task 1 `field_scale` sample-time, excluded from selector). O2r **deviation:** GPU two-pass tree reduction replaced by a CPU numpy readback reduction — simpler, exactly deterministic, per-row/eddy for free, preview==export by identical numpy; single readback stall per build (acceptable; `_derive` already reads textures). Documented in Task 2 rationale + `activity.py` docstring.
- **Byte-identity:** golden-hash (Task 3), routing array_equal (Task 4), p05 --check (Tasks 5, 8).
- **Type consistency:** `ActivityMeans(mean_eddy, mean_vort, rowmean_tex)`, `ActivitySynth.build(vel, out) -> ActivityMeans`, `new_activity_texture(gpu, size)`, `synthesize(..., activity=, means=)` used identically in facade (5), snapshot (6), exporter (7).
