"""Build the Blender extension zip.

A plain zip with blender_manifest.toml at the archive root is exactly what
`blender --command extension build` produces for a wheel-free add-on, and it
installs by drag-and-drop into Blender 4.2+.

Usage: python scripts/build_addon.py [output_dir]
"""

from __future__ import annotations

import sys
import tomllib
import zipfile
from pathlib import Path

ADDON_DIR = Path(__file__).resolve().parent.parent / "blender_addon" / "gasgiant_importer"


def build(out_dir: Path) -> Path:
    manifest = tomllib.loads((ADDON_DIR / "blender_manifest.toml").read_text(encoding="utf-8"))
    name = f"{manifest['id']}-{manifest['version']}.zip"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(ADDON_DIR.rglob("*")):
            if file.is_dir() or "__pycache__" in file.parts:
                continue
            zf.write(file, file.relative_to(ADDON_DIR))
    print(f"built {out_path}")
    return out_path


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dist")
    build(target)
