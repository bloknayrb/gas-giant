"""Preset files: a versioned JSON envelope around PlanetParams.

Policy (deliberate, see docs/presets.md): presets are STRICT — unknown keys are
errors, because a hand-edited typo silently falling back to a default is data
loss. (The mapset manifest read by the Blender add-on has the opposite,
tolerant policy.)
"""

from __future__ import annotations

import json
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


def available_presets() -> list[tuple[str, str]]:
    """(name, source) pairs for the merged dropdown: factory presets first
    (source ``"factory"``), then user presets (source ``"user"``). The source
    discriminates the two namespaces so the GUI can show ``user/<name>`` and so
    a user preset can't collide with a same-named factory preset."""
    factory = [(name, "factory") for name in factory_preset_names()]
    user = [(name, "user") for name in user_preset_names()]
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
