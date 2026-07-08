"""Generate the visual slider reference: docs/sliders.md + docs/img/sliders/*.jpg.

Renders an equirectangular Jupiter-like map for the low / preset / high value of
every numeric slider in the GUI, so the visual effect of each knob is obvious.

Sliders are enumerated live from ``PlanetParams`` via the same ``panels`` helpers
the GUI uses, so this doc tracks the real UI -- add a slider in params/model.py
and it shows up here on the next run.

Rendering is tier-aware: POST-tier sliders (color/detail/emission/physical) reuse
one developed simulation and only re-derive (seconds each); VELOCITY/RESTART
sliders need a fresh simulation per value (~2 min each under software GL). The
batch is resumable -- existing images are skipped.

Usage::

    # list every detected slider (no GL, fast sanity check)
    uv run python scripts/render_slider_examples.py --list

    # render one group (POST groups are fast)
    xvfb-run -a uv run python scripts/render_slider_examples.py --group Appearance

    # full batch (multi-hour under software GL), then write the markdown
    xvfb-run -a uv run python scripts/render_slider_examples.py

    # regenerate markdown only, from images already on disk
    uv run python scripts/render_slider_examples.py --no-render

    # CI doc-drift gate: fail if docs/sliders.md is stale vs the params model
    # (text-only comparison -- no GL, no image rendering)
    uv run python scripts/render_slider_examples.py --check
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import annotated_types
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from gasgiant.params.model import PlanetParams, Tier
from gasgiant.params.presets import load_factory_preset


# Mirrors of the two pure helpers in gasgiant.app.panels. Inlined (not imported)
# because panels.py imports imgui_bundle at module load, which the headless/CLI
# env does not have. Kept byte-faithful so the slider set matches the GUI.
def _bounds(info: FieldInfo) -> tuple[float | None, float | None]:
    lo = hi = None
    for m in info.metadata:
        if isinstance(m, annotated_types.Ge):
            lo = float(m.ge)
        elif isinstance(m, annotated_types.Le):
            hi = float(m.le)
    return lo, hi


def _is_color_field(name: str, value: Any) -> bool:
    return (
        isinstance(value, (tuple, list))
        and len(value) == 3
        and all(isinstance(v, (int, float)) for v in value)
        and "color" in name
    )


def leaf_kind(name: str, info: FieldInfo, value: Any) -> str | None:
    import types
    from enum import StrEnum
    from typing import Union, get_args, get_origin

    ann = info.annotation
    if get_origin(ann) in (Union, types.UnionType):
        inner = [a for a in get_args(ann) if a is not type(None)]
        if len(inner) == 1 and isinstance(inner[0], type) and issubclass(inner[0], BaseModel):
            return "optional_model"
        if len(inner) == 1 and inner[0] is float:
            return "optional_float"
        if len(inner) == 1 and inner[0] is int:
            return "optional_int"
        if len(inner) == 1 and inner[0] is str:
            return "optional_str"
    if isinstance(ann, type) and issubclass(ann, StrEnum):
        return "enum"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if _is_color_field(name, value):
        return "color"
    if isinstance(value, str):
        return "str"
    # Annotation-keyed list-of-model editor (storms.cast); classifies the EMPTY
    # default. Palette/stops lists are excluded (bespoke value-shape editors).
    if get_origin(ann) is list:
        args = get_args(ann)
        if len(args) == 1 and isinstance(args[0], type) and issubclass(args[0], BaseModel):
            from gasgiant.params.model import GradientStop, PaletteRow
            if args[0] not in (GradientStop, PaletteRow):
                return "model_list"
    if isinstance(value, list) and value and isinstance(value[0], dict) and "pos" in value[0]:
        return "stops"
    if isinstance(value, list) and value and isinstance(value[0], dict) and "stops" in value[0]:
        return "palette_rows"
    return None

REPO = Path(__file__).resolve().parent.parent
IMG_DIR = REPO / "docs" / "img" / "sliders"
DOC = REPO / "docs" / "sliders.md"

# Render settings -- small + deterministic so every difference is the slider's.
# dev_steps/resolution are deliberately below the jupiter_like preset (500/2048):
# software-GL render time is ~linear in dev_steps, and every variant shares these
# settings, so the *comparison* stays fair while the batch fits one session.
SEED = 4201            # jupiter_like seed; fixed across all variants
SIM_RES = 768          # sim grid width (2:1)
DEV_STEPS = 150        # development steps before snapshot
OUT_WIDTH = 768        # equirect image width (height = 384)
IMG_EXT = "jpg"        # JPEG keeps the ~190 example images to ~20 MB in-repo
JPEG_QUALITY = 92      # high enough that cloud banding shows no visible artifacts

# Emission-map demo levels: the emission baseline turns all three glows on so
# each emission slider's effect on the night-side map is visible.
EMISSION_DEMO = {"thermal_strength": 1.0, "lightning_strength": 1.0,
                 "aurora_strength": 1.0}

# Free-input fields (rendered as input_int in the GUI, not sliders) -- skip.
SLIDER_INPUT_INT_LIMIT = 1_000_000

# GUI group order (matches the collapsing headers top to bottom).
GROUP_ORDER = [
    "Sim", "Solver", "Bands", "Jets", "Turbulence", "Storms",
    "Waves", "Poles", "Appearance", "Detail", "Mask", "Emission", "Physical", "Export",
]

# Groups whose sliders only do anything under the vorticity solver.
VORTICITY_FIELDS = {
    "solver.poisson_iters", "solver.sor_omega", "solver.vort_relax_tau",
    "solver.vort_hypervisc", "solver.coriolis_f0", "solver.vort_inject",
    "solver.vort_inject_scale", "solver.vort_drag",
}
BAROCLINIC_PREFIX = "solver.baroclinic."

# Sliders that do not change the rendered texture at all: Blender-import scale
# hints, output file controls, and the baroclinic internal cadence knobs
# (ui="" hidden, "fixed cadence" per the model). Documented as text, never
# rendered.
NON_VISUAL_FIELDS = {
    "physical.radius_km", "physical.height_scale", "physical.height_midlevel",
    "export.width", "export.png_compression",
    "solver.baroclinic.warmup_steps",
    "solver.baroclinic.baro_steps_per_update",
    "solver.baroclinic.update_every",
    # Quality / numerical-convergence knobs, not a distinct look (and their
    # extremes -- res 8192, 512 SOR sweeps/step -- are impractical to render
    # under software GL). Documented as text.
    "sim.resolution",
    "solver.poisson_iters",
    "solver.sor_omega",
}

# Endpoints whose true max is too slow/large to render: render a tractable value
# instead (labelled honestly with the value actually used).
SAMPLE_OVERRIDES = {
    "sim.dev_steps": (0.0, 1000.0),  # true hi=3000 (~17 min/render); 1000 suffices
}


@dataclass
class Slider:
    path: str           # dotted path, e.g. "storms.hero_radius"
    group: str          # top-level group, e.g. "Storms"
    label: str          # display label, e.g. "hero radius"
    lo: float
    hi: float
    default: float
    is_int: bool
    log: bool
    tier: str
    description: str
    baseline: str = "kinematic"  # which baseline preset to render against
    channel: str = "color"       # which map to show: "color" | "emission"
    visual: bool = True          # whether to render images
    preset_value: float | None = None  # value in the baseline (the middle column)
    # StrEnum dropdowns (B3-6): documented as text entries, never rendered.
    enum_values: tuple[str, ...] | None = None
    enum_default: str | None = None
    # Optional numerics (pin checkbox + slider in the GUI, None = auto):
    # documented as text entries, never rendered (None defaults can't anchor
    # a fair lo/preset/hi image row, and pinned extremes can be
    # validator-coupled to sibling fields).
    optional: bool = False
    # Non-numeric widgets documented as text (never rendered): bool checkboxes
    # and list-of-model editors (storms.cast). ``text_widget`` selects the
    # markdown blurb; ``text_default`` is the default-value repr shown.
    text_only: bool = False
    text_widget: str = ""       # "bool" | "model_list"
    text_default: str = ""


def _walk(model: type[BaseModel], doc: dict[str, Any], prefix: str,
          group: str) -> list[Slider]:
    out: list[Slider] = []
    for name, info in model.model_fields.items():
        ann = info.annotation
        path = f"{prefix}{name}"
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            sub_group = group or name.capitalize()
            out.extend(_walk(ann, doc[name], f"{path}.", sub_group))
            continue
        value = doc[name]
        kind = leaf_kind(name, info, value)
        extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
        if kind == "enum":
            # StrEnum dropdowns (B3-6: e.g. solver.vort_inject_mask) get a
            # text-only entry -- values + default + description, no images.
            out.append(Slider(
                path=path, group=group or "Global", label=name.replace("_", " "),
                lo=0.0, hi=0.0, default=0.0, is_int=False, log=False,
                tier=str(extra.get("tier", "")),
                description=info.description or "",
                visual=False,
                enum_values=tuple(m.value for m in ann),
                enum_default=str(value),
            ))
            continue
        if kind in ("optional_float", "optional_int"):
            # Pin-checkbox + slider widgets (None = auto), e.g.
            # storms.hero_latitude, bands.faded_band_index: text-only entry.
            lo, hi = _bounds(info)
            if lo is None or hi is None:
                continue
            out.append(Slider(
                path=path, group=group or "Global", label=name.replace("_", " "),
                lo=float(lo), hi=float(hi), default=0.0,
                is_int=(kind == "optional_int"), log=bool(extra.get("log")),
                tier=str(extra.get("tier", "")),
                description=info.description or "",
                visual=False, optional=True,
            ))
            continue
        if kind == "optional_str":
            # Optional string path (mask.file): text-entry + Browse button in the
            # GUI. Text-only entry, no images (a path can't anchor a lo/hi row).
            out.append(Slider(
                path=path, group=group or "Global", label=name.replace("_", " "),
                lo=0.0, hi=0.0, default=0.0, is_int=False, log=False,
                tier=str(extra.get("tier", "")),
                description=info.description or "",
                visual=False, text_only=True, text_widget="path",
                text_default="None",
            ))
            continue
        if kind == "model_list":
            # List-of-model editor (storms.cast): text-only entry, no images.
            out.append(Slider(
                path=path, group=group or "Global", label=name.replace("_", " "),
                lo=0.0, hi=0.0, default=0.0, is_int=False, log=False,
                tier=str(extra.get("tier", "")),
                description=info.description or "",
                visual=False, text_only=True, text_widget="model_list",
                text_default="empty list",
            ))
            continue
        if kind == "bool":
            # Boolean toggle (GUI checkbox): text-only entry, no images.
            out.append(Slider(
                path=path, group=group or "Global", label=name.replace("_", " "),
                lo=0.0, hi=0.0, default=0.0, is_int=False, log=False,
                tier=str(extra.get("tier", "")),
                description=info.description or "",
                visual=False, text_only=True, text_widget="bool",
                text_default=str(value),
            ))
            continue
        if kind not in ("int", "float"):
            continue
        lo, hi = _bounds(info)
        if lo is None or hi is None:
            continue
        if kind == "int" and (int(hi) - int(lo)) > SLIDER_INPUT_INT_LIMIT:
            continue  # seed etc. -- free input, not a slider
        out.append(Slider(
            path=path,
            group=group or "Global",
            label=name.replace("_", " "),
            lo=float(lo), hi=float(hi), default=float(value),
            is_int=(kind == "int"),
            log=bool(extra.get("log")),
            tier=str(extra.get("tier", "")),
            description=info.description or "",
        ))
    return out


def enumerate_sliders() -> list[Slider]:
    """Every numeric slider in the GUI, in group order."""
    p = PlanetParams()
    doc = p.model_dump()
    sliders = _walk(PlanetParams, doc, "", "")
    for s in sliders:
        if s.path in VORTICITY_FIELDS:
            s.baseline = "vorticity"
        elif s.path.startswith(BAROCLINIC_PREFIX):
            s.baseline = "baroclinic"
        if s.group == "Emission":
            s.baseline = "emission"
            s.channel = "emission"
        if s.path in NON_VISUAL_FIELDS:
            s.visual = False

    # The middle ("preset") column shows the value the baseline preset actually
    # uses, which can differ from the bare model default. Read it per slider so
    # labels and the lo/hi dedup are correct. Emission keeps None (demo baseline).
    preset_cache: dict[str, dict[str, Any]] = {}
    for s in sliders:
        if (s.channel == "emission" or s.enum_values is not None or s.optional
                or s.text_only):
            continue  # emission uses a demo baseline; enums/optionals/text are text-only
        dump = preset_cache.get(s.baseline)
        if dump is None:
            dump = _baseline_params(s.baseline).model_dump()
            preset_cache[s.baseline] = dump
        node: Any = dump
        for k in s.path.split("."):
            node = node[k]
        s.preset_value = float(node)

    order = {g: i for i, g in enumerate(GROUP_ORDER)}
    sliders.sort(key=lambda s: (order.get(s.group, 99), s.path))
    return sliders


def _fmt(v: float, is_int: bool) -> str:
    if is_int:
        return str(int(round(v)))
    return f"{v:g}"


def sample_values(s: Slider) -> list[tuple[str, float]]:
    """(edge_label, value) pairs to render. For color sliders the baseline image
    already shows the preset/default value, so an endpoint equal to the default
    is skipped. Emission uses a demo baseline (all glows on) rather than the
    model default, so both endpoints are always rendered."""
    out: list[tuple[str, float]] = []
    skip = s.preset_value if s.channel != "emission" else None
    lo, hi = SAMPLE_OVERRIDES.get(s.path, (s.lo, s.hi))
    for edge, v in (("lo", lo), ("hi", hi)):
        if skip is not None and abs(v - skip) < 1e-9:
            continue
        if any(abs(v - ov) < 1e-9 for _, ov in out):
            continue
        out.append((edge, v))
    return out


def _set_path(d: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


# ---------------------------------------------------------------------------
# Rendering (imports GL lazily so --list / --no-render stay GL-free)
# ---------------------------------------------------------------------------

def _baseline_params(kind: str) -> PlanetParams:
    if kind in ("kinematic", "emission"):
        p = load_factory_preset("jupiter_like")
        if kind == "emission":
            for fld, val in EMISSION_DEMO.items():
                setattr(p.emission, fld, val)
    elif kind == "vorticity":
        p = load_factory_preset("jupiter_vorticity")
    elif kind == "baroclinic":
        # jupiter_baroclinic was DROPPED 2026-06-28 (the coupling engine stays);
        # demo the baroclinic sliders on the vorticity preset with coupling on.
        p = load_factory_preset("jupiter_vorticity")
        p.solver.baroclinic.enabled = True
        # The preset spins the baroclinic CPU solver up for 8000 steps; that is
        # minutes per render under software GL. Only baroclinic.gain is shown
        # here, and its effect is visible after a shorter warmup -- cap it so the
        # doc batch stays tractable.
        p.solver.baroclinic.warmup_steps = min(p.solver.baroclinic.warmup_steps, 2000)
    else:
        raise ValueError(kind)
    p.seed = SEED
    p.sim.resolution = SIM_RES
    p.sim.dev_steps = DEV_STEPS
    return p


def _baseline_channel(kind: str) -> str:
    return "emission" if kind == "emission" else "color"


def _write_rgb(rgb: Any, path: Path) -> None:
    import cv2
    import numpy as np

    h, w = rgb.shape[:2]
    if w != OUT_WIDTH:
        new_h = max(1, round(h * OUT_WIDTH / w))
        rgb = cv2.resize(rgb, (OUT_WIDTH, new_h), interpolation=cv2.INTER_AREA)
    u8 = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), u8[..., ::-1],  # cv2 wants BGR
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def _encode_color(maps: dict, path: Path) -> None:
    """Color map is float 0..1 sRGB already (what the exporter/viewport show)."""
    import numpy as np

    _write_rgb(np.clip(maps["color"][..., :3], 0.0, 1.0), path)


# Aurora display tint (matches the default aurora_color, pink-magenta).
_AURORA_TINT = (0.85, 0.35, 0.60)


def _encode_emission(maps: dict, path: Path) -> None:
    """Emission map is HDR float radiance: RGB = thermal+lightning, A = aurora.
    Reinhard tonemap + sRGB gamma so deep hot-spots stay readable; the aurora
    alpha is composited back as a pink tint so its sliders are visible."""
    import numpy as np

    em = maps.get("emission")
    if em is None:
        _write_rgb(np.zeros((OUT_WIDTH // 2, OUT_WIDTH, 3), np.float32), path)
        return
    rgb = np.asarray(em[..., :3], np.float32)
    disp = (rgb / (1.0 + rgb)) ** (1.0 / 2.2)
    if em.shape[-1] >= 4:
        a = np.asarray(em[..., 3], np.float32)[..., None]
        disp = disp + a * np.array(_AURORA_TINT, np.float32)
    _write_rgb(np.clip(disp, 0.0, 1.0), path)


def _encode(channel: str, maps: dict, path: Path) -> None:
    (_encode_emission if channel == "emission" else _encode_color)(maps, path)


def baseline_png(kind: str) -> Path:
    return IMG_DIR / f"_baseline_{kind}.{IMG_EXT}"


def slider_png(s: Slider, edge: str) -> Path:
    safe = s.path.replace(".", "__")
    return IMG_DIR / f"{safe}__{edge}.{IMG_EXT}"


def _order_key(s: Slider) -> tuple:
    # Cheapest first: POST sliders re-derive off a developed sim (seconds);
    # then kinematic fresh sims; vorticity/baroclinic (slow solver) last. This
    # way an interrupted batch still produced the high-value images.
    base_rank = {"kinematic": 1, "emission": 1, "vorticity": 2, "baroclinic": 3}
    return (0 if s.tier == Tier.POST.value else 1, base_rank.get(s.baseline, 9), s.path)


def render_batch(sliders: list[Slider], only_group: str | None) -> None:
    from gasgiant.engine import Simulation
    from gasgiant.gl import GpuContext

    gpu = GpuContext.headless()
    todo = [s for s in sliders if s.visual and (only_group is None or s.group == only_group)]
    todo.sort(key=_order_key)
    kinds_needed = {s.baseline for s in todo}

    # One developed sim per baseline -- reused for POST sliders and the baseline image.
    dev_sims: dict[str, Simulation] = {}
    base_params: dict[str, PlanetParams] = {}
    for kind in kinds_needed:
        bp = _baseline_params(kind)
        base_params[kind] = bp
        out = baseline_png(kind)
        if kind not in dev_sims:
            print(f"[baseline:{kind}] building developed sim...", flush=True)
            dev_sims[kind] = Simulation(bp.model_copy(deep=True), gpu=gpu)
        if not out.exists():
            print(f"[baseline:{kind}] rendering {out.name}", flush=True)
            maps = dev_sims[kind].render_maps(OUT_WIDTH)
            _encode(_baseline_channel(kind), maps, out)

    total = sum(len(sample_values(s)) for s in todo)
    done = 0
    for s in todo:
        for edge, value in sample_values(s):
            out = slider_png(s, edge)
            done += 1
            if out.exists():
                print(f"({done}/{total}) skip {out.name} (exists)", flush=True)
                continue
            bp = base_params[s.baseline]
            variant = bp.model_dump()
            _set_path(variant, s.path, int(round(value)) if s.is_int else value)
            # A large hero_radius tightens the hero_latitude validator cap below
            # the preset's pinned latitude; unpin it so the geometry slider can
            # reach its max (the storm just takes seeded placement).
            if s.path == "storms.hero_radius":
                variant["storms"]["hero_latitude"] = None
            # One bad (slider, value) combo must never kill the whole batch.
            try:
                params = PlanetParams.model_validate(variant)
                print(f"({done}/{total}) render {out.name}  [{s.tier}/{s.baseline}]", flush=True)
                if s.tier == Tier.POST.value:
                    sim = dev_sims[s.baseline]
                    sim.update_params(params)
                    maps = sim.render_maps(OUT_WIDTH)
                    _encode(s.channel, maps, out)
                    sim.update_params(bp.model_copy(deep=True))  # restore baseline
                else:
                    sim = Simulation(params, gpu=gpu)
                    maps = sim.render_maps(OUT_WIDTH)
                    _encode(s.channel, maps, out)
            except Exception as exc:  # noqa: BLE001 -- log and continue
                print(f"({done}/{total}) SKIP {out.name}: {type(exc).__name__}: {exc}",
                      flush=True)


# ---------------------------------------------------------------------------
# Markdown assembly
# ---------------------------------------------------------------------------

def _rel(path: Path) -> str:
    return path.relative_to(DOC.parent).as_posix()


def _img_cell(path: Path, caption: str) -> str:
    if path.exists():
        return f'<td align="center"><img src="{_rel(path)}" width="320"><br><sub>{caption}</sub></td>'
    return f'<td align="center"><sub>{caption}<br>(not rendered)</sub></td>'


def build_markdown(sliders: list[Slider]) -> str:
    lines: list[str] = []
    lines.append("# Slider reference\n")
    lines.append(
        "What every slider in the live-preview GUI (`uv run gasgiant-studio`) "
        "actually does, shown on the planet. Each row renders the **low**, "
        "**preset**, and **high** value of one slider; everything else is held "
        "at the `jupiter_like` preset (seed "
        f"{SEED}, sim resolution {SIM_RES}, {DEV_STEPS} development steps). "
        "Images are the raw equirectangular color map -- the same texture the "
        "exporter writes and the viewport's *Color* channel shows (under the "
        "*Standard* view transform).\n")
    lines.append(
        "> The panels are auto-generated from `PlanetParams` "
        "(`src/gasgiant/params/model.py`): every `int`/`float` field becomes a "
        "slider, every `StrEnum` field becomes a dropdown, and every optional "
        "numeric field becomes a pin-checkbox + slider (dropdowns and optional "
        "fields are documented here as text entries). This document is "
        "generated from the same model by "
        "`scripts/render_slider_examples.py`, so it tracks the real UI "
        "(CI runs it with `--check` and fails when this file is stale).\n")
    lines.append(
        "> **Tier** is what the engine recomputes when you move the slider: "
        "`post` re-derives the maps only (instant), `velocity` rebuilds the "
        "flow field, `restart` re-runs the development from step 0.\n")

    # Table of contents.
    groups = [g for g in GROUP_ORDER if any(s.group == g for s in sliders)]
    extra_groups = sorted({s.group for s in sliders} - set(GROUP_ORDER))
    groups += extra_groups
    lines.append("## Contents\n")
    for g in groups:
        anchor = g.lower().replace(" ", "-")
        lines.append(f"- [{g}](#{anchor})")
    lines.append("")

    for g in groups:
        lines.append(f"\n## {g}\n")
        for s in [s for s in sliders if s.group == g]:
            lines.append(f"### {s.label}\n")
            if s.enum_values is not None:
                meta = (
                    f"`{s.path}` &mdash; dropdown, one of "
                    + " / ".join(f"`{v}`" for v in s.enum_values)
                    + f", default **`{s.enum_default}`**, tier `{s.tier}`")
                lines.append(meta + ".\n")
                if s.description:
                    lines.append(f"{s.description}\n")
                lines.append(
                    "_Choice field (GUI dropdown) &mdash; documented as text; "
                    "no rendered example._\n")
                continue
            if s.text_only:
                if s.text_widget == "bool":
                    meta = (
                        f"`{s.path}` &mdash; toggle (on/off), default "
                        f"**`{s.text_default}`**, tier `{s.tier}`")
                elif s.text_widget == "path":
                    meta = (
                        f"`{s.path}` &mdash; file path, default "
                        f"**{s.text_default}**, tier `{s.tier}`")
                else:  # model_list
                    meta = (
                        f"`{s.path}` &mdash; list editor, default "
                        f"**{s.text_default}**, tier `{s.tier}`")
                lines.append(meta + ".\n")
                if s.description:
                    lines.append(f"{s.description}\n")
                if s.text_widget == "bool":
                    lines.append(
                        "_Boolean toggle (GUI checkbox) &mdash; documented as "
                        "text; no rendered example._\n")
                elif s.text_widget == "path":
                    lines.append(
                        "_File-path field: the GUI shows a text entry + **Browse...** "
                        "button (empty = None). Documented as text; no rendered "
                        "example._\n")
                else:
                    lines.append(
                        "_List of hand-placed sub-records edited in a dedicated "
                        "GUI panel &mdash; documented as text; no rendered "
                        "example._\n")
                continue
            if s.optional:
                meta = (
                    f"`{s.path}` &mdash; optional; pin range "
                    f"**{_fmt(s.lo, s.is_int)} to {_fmt(s.hi, s.is_int)}**, "
                    f"default **None (auto)**, tier `{s.tier}`"
                    + (", log scale" if s.log else ""))
                lines.append(meta + ".\n")
                if s.description:
                    lines.append(f"{s.description}\n")
                lines.append(
                    "_Optional field: the GUI shows a **pin** checkbox &mdash; "
                    "unpinned (None) keeps the automatic/seeded behavior, pinned "
                    "uses the slider value verbatim. Documented as text; no "
                    "rendered example._\n")
                continue
            meta = (
                f"`{s.path}` &mdash; range **{_fmt(s.lo, s.is_int)} "
                f"to {_fmt(s.hi, s.is_int)}**, default "
                f"**{_fmt(s.default, s.is_int)}**, tier `{s.tier}`"
                + (", log scale" if s.log else ""))
            lines.append(meta + ".\n")
            if s.description:
                lines.append(f"{s.description}\n")
            if not s.visual:
                lines.append(
                    "_Passed to the Blender importer / controls the output file, "
                    "not the texture appearance &mdash; no visual example._\n")
                continue
            if s.channel == "emission":
                lines.append(
                    "_Shown on the **emission map** (night-side glow) with all "
                    "three glows enabled; tonemapped for display. The color map "
                    "is unchanged by emission sliders._\n")
            elif s.baseline != "kinematic":
                lines.append(
                    f"_Rendered against the `{s.baseline}` solver baseline "
                    "(inert under the default kinematic solver)._\n")
            if s.path in SAMPLE_OVERRIDES:
                lines.append(
                    "_High example capped below the slider maximum so it renders "
                    "in reasonable time; the column label shows the value used._\n")

            samples = sample_values(s)
            los = [sv for sv in samples if sv[0] == "lo"]
            his = [sv for sv in samples if sv[0] == "hi"]
            if s.channel == "emission":
                mid_label = "demo &middot; all glows on"
            else:
                mid_label = f"preset &middot; {_fmt(s.preset_value, s.is_int)}"
            cells: list[str] = []
            if los:
                cells.append(_img_cell(slider_png(s, "lo"),
                                       f"low &middot; {_fmt(los[0][1], s.is_int)}"))
            cells.append(_img_cell(baseline_png(s.baseline), mid_label))
            if his:
                cells.append(_img_cell(slider_png(s, "hi"),
                                       f"high &middot; {_fmt(his[0][1], s.is_int)}"))
            lines.append("<table><tr>")
            lines.append("".join(cells))
            lines.append("</tr></table>\n")

    return "\n".join(lines) + "\n"


def write_markdown(sliders: list[Slider]) -> None:
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(build_markdown(sliders), encoding="utf-8", newline="\n")
    print(f"wrote {DOC.relative_to(REPO)}")


def check_markdown(sliders: list[Slider]) -> int:
    """Text-only doc-drift gate (no GL, no rendering): regenerate the markdown
    in memory and diff it against docs/sliders.md. Exit 1 when stale. Image
    cells only depend on which files EXIST under docs/img/sliders (committed),
    so this is deterministic on a clean checkout."""
    import difflib

    expected = build_markdown(sliders)
    actual = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    if actual == expected:
        print(f"{DOC.relative_to(REPO)} is up to date ({len(sliders)} entries)")
        return 0
    diff = list(difflib.unified_diff(
        actual.splitlines(), expected.splitlines(),
        "docs/sliders.md (on disk)", "generated from params model", lineterm=""))
    head = diff[:120]
    print("\n".join(head))
    if len(diff) > len(head):
        print(f"... ({len(diff) - len(head)} more diff lines)")
    print(
        "\ndocs/sliders.md is STALE relative to the params model. Regenerate the "
        "text with `uv run python scripts/render_slider_examples.py --no-render` "
        "and commit; if the diff shows '(not rendered)' cells, also render the "
        "missing images (see this script's docstring for the render commands).")
    return 1


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render the slider reference doc")
    ap.add_argument("--list", action="store_true", help="list sliders and exit (no GL)")
    ap.add_argument("--group", default=None, help="render only this group")
    ap.add_argument("--no-render", action="store_true",
                    help="(re)write markdown from existing PNGs, no rendering")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if docs/sliders.md is stale vs the params model "
                         "(text-only; no GL, no image rendering)")
    args = ap.parse_args(argv)

    sliders = enumerate_sliders()

    if args.check:
        return check_markdown(sliders)

    if args.list:
        for g in GROUP_ORDER + sorted({s.group for s in sliders} - set(GROUP_ORDER)):
            gs = [s for s in sliders if s.group == g]
            if not gs:
                continue
            print(f"\n{g} ({len(gs)})")
            for s in gs:
                if s.enum_values is not None:
                    print(f"  {s.path:36s} enum: {'|'.join(s.enum_values)} "
                          f"def={s.enum_default}  {s.tier}  [dropdown]")
                    continue
                if s.optional:
                    print(f"  {s.path:36s} {_fmt(s.lo, s.is_int):>8} .. "
                          f"{_fmt(s.hi, s.is_int):<8} def=None     "
                          f"{s.tier}  [optional]")
                    continue
                if s.text_only:
                    print(f"  {s.path:36s} def={s.text_default:<12} "
                          f"{s.tier}  [{s.text_widget}]")
                    continue
                flag = "" if s.visual else "  [non-visual]"
                base = "" if s.baseline == "kinematic" else f"  [{s.baseline}]"
                print(f"  {s.path:36s} {_fmt(s.lo, s.is_int):>8} .. "
                      f"{_fmt(s.hi, s.is_int):<8} def={_fmt(s.default, s.is_int):<8} "
                      f"{s.tier}{base}{flag}")
        print(f"\ntotal sliders: {len(sliders)}")
        return 0

    if not args.no_render:
        render_batch(sliders, args.group)
    write_markdown(sliders)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
