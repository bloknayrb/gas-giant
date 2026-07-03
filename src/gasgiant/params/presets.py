"""Preset files: a versioned JSON envelope around PlanetParams.

Policy (deliberate, see docs/presets.md): presets are STRICT — unknown keys are
errors, because a hand-edited typo silently falling back to a default is data
loss. (The mapset manifest read by the Blender add-on has the opposite,
tolerant policy.)
"""

from __future__ import annotations

import json
from enum import StrEnum
from importlib import resources
from pathlib import Path

from pydantic import ValidationError

import gasgiant
from gasgiant.params.migrations import CURRENT_PRESET_FORMAT, migrate
from gasgiant.params.model import PlanetParams

_FACTORY_PACKAGE = "gasgiant.presets"

# User presets live alongside the session file under ~/.gasgiant (matching
# SESSION_PATH in app.main). Kept in the params layer (no GUI import) so the CLI
# and tests can enumerate/load user presets too.
USER_PRESET_DIR = Path.home() / ".gasgiant" / "presets"


class PresetError(ValueError):
    pass


def to_preset_doc(params: PlanetParams, name: str | None = None) -> dict:
    return {
        "preset_format": CURRENT_PRESET_FORMAT,
        "app_version": gasgiant.__version__,
        "name": name or params.name,
        "params": json.loads(params.to_json()),
    }


def save_preset(params: PlanetParams, path: Path, name: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_preset_doc(params, name), indent=2), encoding="utf-8")


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
    return load_preset_doc(doc, source=str(path))


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


def _summarize(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err["loc"])
        lines.append(f"{loc}: {err['msg']}")
    return "; ".join(lines)
