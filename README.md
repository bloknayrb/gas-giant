# Gas Giant Studio

<img width="3439" height="1394" alt="image" src="https://github.com/user-attachments/assets/df57fc9f-c6e8-44ac-8505-daea23e35201" />

Procedural gas giant texture map generator. A GPU "sim-advected procedural"
engine — a physically motivated velocity field (alternating zonal jets,
injected storm vortices, shear-driven turbulence) through which cloud tracer
fields are advected — produces seamless equirectangular map sets (16-bit
color + float height, plus an optional HDR emission map: thermal hot-spot
glow, lightning, aurora) for wrapping on a sphere, plus a Blender extension
that imports a map set as a ready-to-render planet with material, atmosphere,
and demo scene.

Modeled on the visible cloud formations of Jupiter and Saturn: zones and
belts with meandering boundaries, alternating jets, GRS-class anticyclones
with turbulent wakes, white ovals, brown barges, strings of pearls,
Kelvin–Helmholtz billows, festoons and hot spots, convective outbreaks,
vortex mergers with debris collars, GRS internal spiral lanes, intermittent
belt turbulence, Saturn's ribbon, Jupiter's polar cyclone clusters, and
Saturn's polar hexagon. The full catalog and how each is implemented:
`docs/formations.md`. Color and texture are calibrated against NASA
reference maps with chroma-aware per-latitude metrics: `docs/realism.md`.

## Requirements

- Python 3.13+, [uv](https://docs.astral.sh/uv/)
- A GPU with OpenGL 4.3 (developed on an RTX 3070, Windows 11)
- Blender 4.2+ for the importer (verified on 5.1.2)

## Quick start

```sh
uv sync --all-extras

# Live-preview GUI: watch the simulation evolve, tweak, export.
uv run gasgiant-studio
# First launch grows the default planet through a ~700-step development run
# (up to ~15 min at the default speed on a midrange GPU) — watch the
# "developing N/M (~Xm left)" progress in the Playback pane and over the
# viewport; the image is not final until it completes.
# Panels are searchable and auto-generated from the parameters, with
# per-slider help, undo/redo, and playback (pause / step / extend) controls.
# What each slider does, shown on the planet: docs/sliders.md

# Headless: render a map set (factory presets: gas_giant_warm [default],
# jupiter_like, jupiter_vorticity, saturn_pale, ice_giant, neptune,
# ember_dwarf, cobalt_gale, green_giant)
uv run gasgiant export --preset jupiter_like --res 4096 --out out/jove
uv run gasgiant validate out/jove        # seam/pole invariants

# Override how long the planet develops before the snapshot:
uv run gasgiant export --preset saturn_pale --dev-steps 1000 --out out/saturn

# Big one: 16384x8192 (about half a minute on an RTX 3070)
uv run gasgiant export --preset jupiter_like --res 16384 --out out/jove16k

# Animate: export an N-frame color sequence, advancing the sim between frames
uv run gasgiant export --preset jupiter_vorticity --frames 120 \
    --steps-per-frame 4 --out out/jove_seq
```

## Presets

Nine factory presets ship in `src/gasgiant/presets/`. Each image below is that
preset developed for 1,000 steps, exported as the raw equirectangular color map
— the same texture that wraps onto the sphere. Parameter details and the
manifest contract: `docs/presets.md`.

**`gas_giant_warm`** — flagship, and the GUI startup default. Vorticity solver:
high-contrast warm bands, a Great-Red-Spot-class hero storm with a turbulent
wake, and flowing eddies.

![gas_giant_warm developed for 1,000 steps](docs/img/presets/gas_giant_warm.png)

**`jupiter_like`** — kinematic (v1.5), calibrated against Cassini reference
maps: orange belts, white zones, red/white ovals, and dark polar hoods.

![jupiter_like developed for 1,000 steps](docs/img/presets/jupiter_like.png)

**`jupiter_vorticity`** — the prognostic vorticity solver (v1.6): stronger
barotropic instability and eddy-shedding give a more turbulent, filament-rich
Jupiter with spiral-laned storms.

![jupiter_vorticity developed for 1,000 steps](docs/img/presets/jupiter_vorticity.png)

**`saturn_pale`** — Saturn's muted gold-and-cream palette: soft, low-contrast
bands and pale ovals.

![saturn_pale developed for 1,000 steps](docs/img/presets/saturn_pale.png)

**`ice_giant`** — a Uranus-like ice giant: cool, pale blue banding with a
discrete dark vortex.

![ice_giant developed for 1,000 steps](docs/img/presets/ice_giant.png)

**`neptune`** — a deep methane-blue Neptune: smooth broad zones (laminar, no
belt churn), a dark Great-Dark-Spot anticyclone with bright companion clouds,
and wind-sheared cirrus streaks.

![neptune developed for 1,000 steps](docs/img/presets/neptune.png)

**`ember_dwarf`** — not a planet: a cloudy L/T-transition brown dwarf, and the
one preset here that is lit from *below*. Broad sodium and potassium absorption
eats the green out of a brown dwarf's spectrum, so the cloud deck is a dim
magenta-plum; the L/T transition is the regime where that silicate deck breaks
apart, and through the tears you see down to hotter gas. Bright means a hole,
not a cloud — so the ramp tops out in ember orange and gold instead of white
cirrus, the fire is made by convective excursions rather than painted on by band
values, and the hero storm is a giant glowing clearing ringed by a dark moat.
Fast rotation (2–5 h) sets the rest: high Coriolis, small deformation radius,
many narrow bands.

![ember_dwarf developed for 1,000 steps](docs/img/presets/ember_dwarf.png)

> `ember_dwarf`'s image is rendered at its shipped `sim.resolution` 4096 rather
> than the reduced grid (it is one of two presets pinned this way — see
> `green_giant` below): its tears come from convective excursions, so
> the fraction of the disc reaching ember grows by about half again between sim-res
> 1024 and 4096 (13% → 20% of the disc), and the reduced grid understates it. The
> generator enforces this per-preset, so a plain regen cannot overwrite it.

**`cobalt_gale`** — a tidally locked hot Jupiter, after HD 189733b. Blue for the
opposite reason to `neptune`: not methane *absorbing* the red end, but Rayleigh
scattering off a high silicate haze, which throws back blue as λ⁻⁴. That inverts
the tonal structure, because the haze is the only reflector — bright means thick
cloud, and dark means clearer air seen down into a hot, near-black,
alkali-absorbing depth. At ~1200 K it glows in the infrared, not the visible, so
unlike `ember_dwarf` every bright pixel here is reflected light and emission is
zero. It is also the only preset whose subject is a **jet** rather than a storm:
tidally locked giants develop equatorial superrotation, one broad prograde jet
instead of Jupiter's many alternating bands, so there is no hero at all and the
composition is an authored 9-band skeleton with a ±13° equatorial zone. Its
rotation is tidally locked to a ~2.2-day orbit — five times slower than Jupiter —
which sets the rest, and makes it the mirror of `ember_dwarf`: low Coriolis, a
large deformation radius, few very wide bands.

![cobalt_gale developed for 1,000 steps](docs/img/presets/cobalt_gale.png)

**`green_giant`** — an enriched, sulfur-rich giant, and the one preset here that
is still a Jupiter-class planet: the separation is carried by chemistry rather
than by moving the object somewhere exotic. H₂S outstrips NH₃ in the visible
atmosphere (real — Uranus's deck *is* H₂S), UV photolysis polymerises it into
yellow S₈/polysulfide chromophores, and residual methane eats the red end, so
yellow pigment under a red-absorbing atmosphere reads olive to chartreuse. The
same enrichment drives the weather: a higher mean molecular weight means a
smaller scale height and a smaller deformation radius, so the field is finer and
busier than Jupiter's at the *same* rotation rate — the premise moves
`deformation_radius` (0.13) and deliberately leaves `coriolis_f0` at Jupiter's
3.0. Its composition is a fifth kind: not a hero, not a jet, but an
**alternation** — glassy chartreuse zones studded with pale cream ovals against
churned olive belts streaked with dark cigars. It is the first factory preset to
use `solver.vort_inject_mask = belts`, which confines eddy churn to the cyclonic
lanes and leaves the zones laminar.

![green_giant developed for 1,000 steps](docs/img/presets/green_giant.png)

> `green_giant`'s image is rendered at its shipped `sim.resolution` 4096, for a
> different reason than `ember_dwarf`'s: the ratio alone (16.6% → 21.6% of the
> disc above the zone knee, ~1.30×) would not have justified the pin. The render
> does. Its subject is a seeded vortex population whose members must be
> individually resolved to read as members, so at 1024 the pale zone ovals never
> separate and the planet is a marbled ball rather than banded.

> Generated with `scripts/render_readme_examples.py` (a reduced sim grid keeps
> the set tractable under software GL; the shipped presets develop at
> `sim.resolution` 2048–4096, so a full-quality render carries finer detail).

## Into Blender

```sh
uv run python scripts/build_addon.py     # -> dist/gasgiant_importer-1.1.0.zip
```

Drag the zip into Blender, then *File → Import → Gas Giant Map Set (.json)*
and pick `out/jove/mapset.json`. Enable "Create demo scene" for a framed,
sun-lit, AgX-graded first render. Details and options: `docs/blender_addon.md`.

## How it works

Four cloud tracers (color index, cloud-top height, detail, storm tint) are
advected by a semi-Lagrangian MacCormack solver through a streamfunction-
built velocity field, on an equirect grid plus two azimuthal-equidistant
polar patches slaved by a per-step nesting exchange. Relaxation forcing
toward the analytic band/storm stamps keeps structure alive indefinitely;
export-time advected-coordinate noise adds flow-stretched filament detail at
any output resolution. Architecture: `docs/architecture.md`.

Everything is deterministic from one seed. Presets: `docs/presets.md`.

## Development

```sh
uv run pytest            # unit + GPU tests (llvmpipe works)
uv run ruff check .
uv run lint-imports      # layer contracts
```

The Blender import test runs inside Blender:
`blender --background --factory-startup --python tests/blender/test_import.py -- <mapset_dir>`
(writes `tests/blender/result.json`).
