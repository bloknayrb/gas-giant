"""gasgiant-studio entry point. The live-preview GUI lands in Phase 2."""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "gasgiant-studio: the live-preview GUI arrives in Phase 2.\n"
        "Use the headless CLI meanwhile:  gasgiant export --preset jupiter_like --out out/planet",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
