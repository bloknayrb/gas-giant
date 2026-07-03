"""Console entry point for ``gasgiant-studio`` (B1-2).

The real GUI module (``gasgiant.app.main``) imports imgui_bundle at module
scope, so a plain ``uv sync`` (no GUI extra) used to die with a bare
ImportError traceback the moment the entry point imported it -- the first
minute of the first-run journey. This launcher owns the entry point instead:
it try-imports the GUI module and translates a *missing GUI extra* into one
actionable message, while any OTHER ImportError (a broken install, a typo in
our own modules) still surfaces as its original traceback. No imgui import
happens at this module's import time, so it is importable (and unit-testable)
in a GUI-less environment.
"""

from __future__ import annotations

import sys

# The distribution extra is "gui"; the module it installs is imgui_bundle.
_GUI_MODULE = "imgui_bundle"

_MISSING_GUI_MESSAGE = (
    "gasgiant-studio needs the GUI extra (imgui-bundle), which is not installed.\n"
    "Install it with:\n"
    "\n"
    "    uv sync --all-extras\n"
    "\n"
    "(or `pip install 'gasgiant[gui]'` for a pip install)."
)


def missing_gui_message(exc: ImportError) -> str | None:
    """The friendly message for a missing-GUI-extra ImportError, or None when
    the ImportError is about anything else (those must keep their traceback --
    swallowing an unrelated import bug behind an 'install the extra' hint
    would send the user chasing the wrong fix)."""
    name = exc.name or ""
    if name == _GUI_MODULE or name.startswith(_GUI_MODULE + "."):
        return f"{_MISSING_GUI_MESSAGE}\n(missing module: {name})"
    return None


def _import_studio_main():
    """Deferred import of the real GUI entry point -- the only place the GUI
    extra is touched. A separate function so tests can monkeypatch the import
    failure without uninstalling imgui_bundle."""
    from gasgiant.app.main import main as studio_main

    return studio_main


def main() -> int:
    try:
        studio_main = _import_studio_main()
    except ImportError as exc:
        message = missing_gui_message(exc)
        if message is None:
            raise
        print(message, file=sys.stderr)
        return 1
    return studio_main()


if __name__ == "__main__":
    raise SystemExit(main())
