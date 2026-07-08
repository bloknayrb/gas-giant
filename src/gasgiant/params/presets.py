"""Preset files: a versioned JSON envelope around PlanetParams.

Policy (deliberate, see docs/presets.md): presets are STRICT — unknown keys are
errors, because a hand-edited typo silently falling back to a default is data
loss. (The mapset manifest read by the Blender add-on has the opposite,
tolerant policy.)
"""

from __future__ import annotations

import json
import shutil
from enum import StrEnum
from importlib import resources
from pathlib import Path

from pydantic import ValidationError

import gasgiant
from gasgiant.params.migrations import CURRENT_PRESET_FORMAT, migrate
from gasgiant.params.model import PlanetParams

_FACTORY_PACKAGE = "gasgiant.presets"
# Epoch recipes (T15) live as raw-overlay JSONs in a subdir of the factory
# preset package, so they ship as package data by the same mechanism (hatchling
# includes every tracked file under the package tree).
_RECIPE_SUBDIR = "recipes"

# User presets live alongside the session file under ~/.gasgiant (matching
# SESSION_PATH in app.main). Kept in the params layer (no GUI import) so the CLI
# and tests can enumerate/load user presets too.
USER_PRESET_DIR = Path.home() / ".gasgiant" / "presets"


class PresetError(ValueError):
    pass


def resolve_mask_path(params: PlanetParams, base_dir: Path) -> None:
    """In-place: make ``params.mask.file`` ABSOLUTE by resolving a relative path
    against ``base_dir`` (a loaded preset's own folder). None stays None; an
    already-absolute path is normalized. The MODEL never knows its source -- this
    resolution runs only through the preset/session/CLI I/O boundary, so the
    in-memory params carry an absolute path the engine can decode directly (a
    missing file is handled downstream: the CLI errors, the engine warns+disables)."""
    f = params.mask.file
    if not f:
        return
    p = Path(f)
    if not p.is_absolute():
        p = base_dir / p
    params.mask.file = str(p.resolve())


def _relativized_for_save(params: PlanetParams, dest_dir: Path) -> PlanetParams:
    """Return a copy of ``params`` whose ``mask.file`` is re-relativized to just
    the sidecar filename next to ``dest_dir`` (a portable saved preset), copying
    the mask PNG into ``dest_dir`` when it lives elsewhere. None stays None. The
    original (in-memory, absolute) params are left untouched."""
    f = params.mask.file
    if not f:
        return params
    src = Path(f)
    out = params.model_copy(deep=True)
    dest = dest_dir / src.name
    if src.is_file() and src.resolve() != dest.resolve():
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    out.mask.file = src.name  # sits next to the saved JSON -> portable
    return out


def to_preset_doc(params: PlanetParams, name: str | None = None) -> dict:
    return {
        "preset_format": CURRENT_PRESET_FORMAT,
        "app_version": gasgiant.__version__,
        "name": name or params.name,
        "params": json.loads(params.to_json()),
    }


def save_preset(
    params: PlanetParams, path: Path, name: str | None = None,
    relativize_mask: bool = True,
) -> None:
    """Write ``params`` as a preset envelope at ``path``.

    ``relativize_mask`` (default) re-relativizes ``mask.file`` to a sidecar next
    to ``path`` and copies the mask PNG there, so a saved preset is portable. The
    session autosave passes ``relativize_mask=False`` to keep the ABSOLUTE path
    (the session is machine-local state, not a portable artifact)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc_params = _relativized_for_save(params, path.parent) if relativize_mask else params
    path.write_text(json.dumps(to_preset_doc(doc_params, name), indent=2), encoding="utf-8")


def _version_tuple(version: str) -> tuple[int, ...]:
    """Best-effort numeric version key for app_version comparisons: leading
    digits per dot-segment, non-numeric tails ignored ("0.2.0rc1" -> (0, 2, 0)),
    a fully non-numeric string collapses to (0,). Deliberately tiny -- this
    only decides which ERROR MESSAGE a failed load gets, never load behavior."""
    parts: list[int] = []
    for piece in str(version).split("."):
        digits = ""
        for ch in piece:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) or (0,)


def load_preset_doc(doc: dict, source: str = "<preset>") -> PlanetParams:
    if not isinstance(doc, dict) or "params" not in doc:
        raise PresetError(f"{source}: not a preset file (missing 'params')")
    fmt = doc.get("preset_format", 1)
    if fmt > CURRENT_PRESET_FORMAT:
        raise PresetError(
            f"{source}: preset_format {fmt} is newer than this app understands "
            f"({CURRENT_PRESET_FORMAT}); upgrade gasgiant"
        )
    doc = migrate(doc, fmt)
    try:
        return PlanetParams.model_validate(doc["params"])
    except ValidationError as exc:
        # B4-5: strict models reject unknown keys, so a preset saved by a
        # NEWER app (an additive field this version doesn't know) would read
        # exactly like a typo. app_version travels in every envelope for this
        # moment -- consult it and blame the version gap when it applies.
        saved_by = doc.get("app_version")
        if saved_by is not None and _version_tuple(str(saved_by)) > _version_tuple(
            gasgiant.__version__
        ):
            raise PresetError(
                f"{source}: this preset was saved by gasgiant {saved_by}, which is "
                f"newer than this app ({gasgiant.__version__}) — it may use fields "
                f"this version doesn't know. Upgrade gasgiant to load it as-is. "
                f"(details: {_summarize(exc)})"
            ) from exc
        raise PresetError(f"{source}: {_summarize(exc)}") from exc


def load_preset(path: Path) -> PlanetParams:
    doc = json.loads(path.read_text(encoding="utf-8"))
    params = load_preset_doc(doc, source=str(path))
    # Resolve a relative mask sidecar against the preset's OWN folder so the
    # in-memory params carry an absolute path (app + CLI file case).
    resolve_mask_path(params, path.parent)
    return params


def factory_preset_names() -> list[str]:
    files = resources.files(_FACTORY_PACKAGE)
    return sorted(p.name.removesuffix(".json") for p in files.iterdir() if p.name.endswith(".json"))


def load_factory_preset(name: str) -> PlanetParams:
    ref = resources.files(_FACTORY_PACKAGE) / f"{name}.json"
    if not ref.is_file():
        known = ", ".join(factory_preset_names())
        raise PresetError(f"unknown factory preset {name!r} (available: {known})")
    return load_preset_doc(json.loads(ref.read_text(encoding="utf-8")), source=f"preset:{name}")


def user_preset_names() -> list[str]:
    """Filename stems of JSON files in USER_PRESET_DIR.

    Enumeration NEVER opens or parses a file: a corrupt user preset must not
    crash the dropdown just by being listed. It only fails when the user
    actually loads it (via ``load_user_preset``)."""
    if not USER_PRESET_DIR.is_dir():
        return []
    return sorted(p.stem for p in USER_PRESET_DIR.glob("*.json") if p.is_file())


def load_user_preset(name: str) -> PlanetParams:
    """Load ``USER_PRESET_DIR/<name>.json`` via the strict envelope path. Any
    failure (missing file, unreadable/corrupt JSON, or a validation error)
    surfaces as ``PresetError`` so the GUI can toast it exactly like the
    factory-preset path -- raised only here at load/click time, never during
    enumeration."""
    path = USER_PRESET_DIR / f"{name}.json"
    try:
        return load_preset(path)
    except PresetError:
        raise
    except (OSError, ValueError) as exc:  # missing file, bad JSON, etc.
        raise PresetError(f"{path}: {exc}") from exc


def import_preset(path: Path) -> tuple[str, PlanetParams]:
    """B4-5 "Import preset...": validate ``path`` through the strict envelope
    (including the version-aware rejection messages), then install it as a
    durable USER preset -- ``USER_PRESET_DIR/<stem>.json`` -- so it appears in
    the merged dropdown, unlike Load's transient FILE identity. Re-saved via
    ``save_preset`` (not byte-copied) so an old-format import lands migrated
    in the current envelope. Returns ``(name, params)``. Never clobbers an
    existing user preset; never writes anything if validation fails."""
    try:
        params = load_preset(path)
    except PresetError:
        raise
    except (OSError, ValueError) as exc:  # missing file, bad JSON, etc.
        raise PresetError(f"{path}: {exc}") from exc
    name = path.stem
    dest = USER_PRESET_DIR / f"{name}.json"
    if dest.exists():
        raise PresetError(
            f"user preset '{name}' already exists ({dest}); delete it first or "
            f"rename the file being imported"
        )
    save_preset(params, dest, name=name)
    return name, params


class PresetSource(StrEnum):
    """Which namespace an active preset identity came from. A ``StrEnum`` so it
    stays ``==``-comparable with the bare strings it replaced (and renders as the
    plain value in f-strings), but gives callers a typed, misspell-proof vocab
    instead of scattering ``"factory"``/``"user"``/``"file"`` literals."""

    FACTORY = "factory"
    USER = "user"
    FILE = "file"


def available_presets() -> list[tuple[str, PresetSource]]:
    """(name, source) pairs for the merged dropdown: factory presets first
    (source ``FACTORY``), then user presets (source ``USER``). The source
    discriminates the two namespaces so the GUI can show ``user/<name>`` and so
    a user preset can't collide with a same-named factory preset."""
    factory = [(name, PresetSource.FACTORY) for name in factory_preset_names()]
    user = [(name, PresetSource.USER) for name in user_preset_names()]
    return factory + user


def resolve_preset(name_or_path: str) -> PlanetParams:
    """A CLI/GUI convenience: a path if it exists on disk, else a factory name."""
    p = Path(name_or_path)
    if p.suffix == ".json" and p.exists():
        return load_preset(p)
    return load_factory_preset(name_or_path)


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursive dict-merge: where BOTH sides map a key to a dict, RECURSE into
    them; otherwise the overlay leaf REPLACES the base value. Returns a new dict
    (the inputs are left untouched). This is the crux of ``apply_overlay`` -- a
    shallow ``dict.update`` would drop a whole nested group (and reset its
    untouched siblings to their model defaults) whenever the overlay touches a
    single leaf of that group."""
    out = dict(base)
    for key, value in overlay.items():
        existing = out.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            out[key] = _deep_merge(existing, value)
        else:
            out[key] = value
    return out


def apply_overlay(params: PlanetParams, overlay: dict) -> PlanetParams:
    """Deep-merge a recipe ``overlay`` onto ``params`` and re-validate.

    The overlay is a sparse nested dict naming only the fields it changes. It is
    recursively merged into ``params.model_dump()`` (siblings of any group the
    overlay enters are preserved -- see ``_deep_merge``), then the merged dict is
    re-validated through the strict model. Strict validation is deliberate: a
    typo'd or unknown overlay key raises (``ValidationError``) rather than
    silently falling back to a default, exactly like a hand-edited preset. The
    result is a fresh ``PlanetParams``; ``params`` is not mutated."""
    return PlanetParams.model_validate(_deep_merge(params.model_dump(), overlay))


def _recipe_dir():
    return resources.files(_FACTORY_PACKAGE) / _RECIPE_SUBDIR


def available_recipes() -> list[str]:
    """Filename stems of the packaged epoch-recipe JSONs. Mirrors
    ``factory_preset_names`` (enumerate-only; never opens a file), so a malformed
    recipe still lists and only errors when actually loaded via ``load_recipe``."""
    d = _recipe_dir()
    if not d.is_dir():
        return []
    return sorted(p.name.removesuffix(".json") for p in d.iterdir() if p.name.endswith(".json"))


def load_recipe(name: str) -> tuple[str, dict, dict]:
    """Load the packaged epoch recipe ``name``. Returns
    ``(base_preset_name, overlay_dict, meta)``.

    A recipe file is a RAW overlay (NOT a preset envelope): a JSON object with
    ``base`` (the factory preset the overlay is designed for), ``overlay`` (the
    sparse nested dict of fields to merge), and optional ``name``/``description``
    metadata. ``meta`` is ``{"name": ..., "description": ...}`` (name falls back
    to the file stem, description to ""). Raises ``PresetError`` on an unknown
    recipe or a structurally-invalid file."""
    ref = _recipe_dir() / f"{name}.json"
    if not ref.is_file():
        known = ", ".join(available_recipes())
        raise PresetError(f"unknown recipe {name!r} (available: {known})")
    try:
        doc = json.loads(ref.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PresetError(f"recipe:{name}: {exc}") from exc
    if not isinstance(doc, dict):
        raise PresetError(f"recipe:{name}: not a recipe object")
    base = doc.get("base")
    if not isinstance(base, str) or not base:
        raise PresetError(f"recipe:{name}: missing/invalid 'base' preset name")
    overlay = doc.get("overlay")
    if not isinstance(overlay, dict):
        raise PresetError(f"recipe:{name}: missing/invalid 'overlay' object")
    meta = {"name": doc.get("name") or name, "description": doc.get("description") or ""}
    return base, overlay, meta


def _summarize(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err["loc"])
        lines.append(f"{loc}: {err['msg']}")
    return "; ".join(lines)
