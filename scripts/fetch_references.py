"""Download NASA reference imagery used to calibrate and review presets.

Sources are the NASA Image and Video Library asset CDN
(images-assets.nasa.gov) — the Photojournal (photojournal.jpl.nasa.gov)
blocks scripted clients. Each ID is tried at the ~medium rendition first,
falling back to ~small (not every item publishes a medium).

Usage: python scripts/fetch_references.py [dest_dir]   (default: refs/)
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

REFERENCES: dict[str, str] = {
    # Cassini true-color cylindrical map of Jupiter — the calibration ground
    # truth (same equirect projection as our exports).
    "PIA07782": "Cassini Jupiter cylindrical map (true color)",
    "PIA21775": "Juno Great Red Spot (true color)",
    "PIA21641": "Juno south polar cyclone cluster",
    "PIA11141": "Cassini Saturn global (natural color)",
    "PIA21611": "Cassini Saturn hexagon (two color epochs)",
}

_RENDITIONS = ("medium", "small")
_URL = "https://images-assets.nasa.gov/image/{id}/{id}~{rendition}.jpg"


def _fetch(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
    except (urllib.error.URLError, TimeoutError):
        return None
    # Missing renditions come back as an XML error document, not a 404.
    return data if data[:3] == b"\xff\xd8\xff" else None


def fetch_all(dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    failures = 0
    for pia_id, title in REFERENCES.items():
        out = dest / f"{pia_id}.jpg"
        if out.exists():
            print(f"  {pia_id}: already present")
            continue
        for rendition in _RENDITIONS:
            data = _fetch(_URL.format(id=pia_id, rendition=rendition))
            if data is not None:
                out.write_bytes(data)
                print(f"  {pia_id}: {title} ({rendition}, {len(data) // 1024} KB)")
                break
        else:
            print(f"  {pia_id}: FAILED — {title}", file=sys.stderr)
            failures += 1
    return failures


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("refs")
    raise SystemExit(1 if fetch_all(target) else 0)
