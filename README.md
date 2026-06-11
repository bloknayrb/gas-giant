# Gas Giant Studio

Procedural gas giant texture map generator. A GPU "sim-advected procedural" engine —
a physically motivated velocity field (alternating zonal jets, injected storm vortices,
shear-driven turbulence) through which cloud tracer fields are advected — produces
seamless equirectangular map sets (color, height, optional normal/emission) for wrapping
on a sphere, plus a thin Blender extension that imports a map set as a ready-to-render
planet.

Modeled on the visible cloud formations of Jupiter and Saturn: zones and belts, the
Great Red Spot class of anticyclones, white ovals, brown barges, turbulent wakes,
festoons and hot spots, vortex streets, Kelvin–Helmholtz billows, convective outbreaks,
Jupiter's polar cyclone clusters, and Saturn's polar hexagon. See `docs/formations.md`.

## Components

- `gasgiant` — headless CLI: render and validate map sets.
- `gasgiant-studio` — live-preview GUI: watch the simulation evolve, tweak parameters,
  export when it looks right.
- `blender_addon/gasgiant_importer` — Blender 4.2+/5.x extension that imports an
  exported map set and builds the planet material, atmosphere shell, and demo scene.

## Requirements

- Python 3.13+
- A GPU with OpenGL 4.3 (development target: NVIDIA RTX 3070, Windows 11)
- [uv](https://docs.astral.sh/uv/)

## Quick start

```sh
uv sync --all-extras
uv run gasgiant export --preset jupiter_like --res 2048 --out out/test1
uv run gasgiant validate out/test1
uv run gasgiant-studio
```

## Status

Early development. Build phases and architecture: `docs/architecture.md`.
