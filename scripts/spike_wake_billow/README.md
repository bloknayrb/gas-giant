# Wake-billow crux spike (measurement-only research harness)

Throwaway CPU pseudo-spectral 2D barotropic (screened-QG) vorticity solver built
to answer ONE question and then be archived: does the hero-wake "dense small
billow chain" look (reference NASA PIA07782) require reducing the production
solver's numerical dissipation (SOR ψ-solve + MacCormack limiter), a cheaper
lever, or is it unreachable in the flow?

**Verdict:** `docs/superpowers/specs/2026-07-19-wake-billow-crux-VERDICT.md`
(dissipation rewrite DISFAVORED; pursue via render synthesis). Design + gates:
`docs/superpowers/specs/2026-07-19-wake-billow-crux-spike-design.md`. Plan:
`docs/superpowers/plans/2026-07-19-wake-billow-crux-spike.md`.

This is a **measurement-only record**, committed per the `spike_detail_character.py`
precedent. It imports `gasgiant` READ-ONLY (for the warm preset constants) and
touches **no `src/gasgiant/**`**. It is outside `testpaths` (pytest never collects
it). Run from the repo root with `uv run python scripts/spike_wake_billow/<x>.py`.

NOT committed (regenerable / large): the 531 MB `gate1_runs/` scan npz, the 79 MB
`discriminator_runs/` npz, and the reference photo / production-render assets the
metrics self-test reads. The three committed PNGs are the key evidence panels.

## Files
- `config.py` — single-source nondim constants (production cell size Δ=π/2048, DT
  from the warm profiles, L_d 0.18, β, the re-derived south-flank `Y_SHEET`).
- `solver.py` — the pseudo-spectral Box (RK4 for ω + tracer, screened inversion,
  Nyquist-only ν₈ + 2/3 dealias, CFL substep guard, energy/enstrophy gate).
- `environment.py` — forced spatially-developing config (Dirichlet inflow strip,
  mean-hold, sponges, deficit/tanh sheet forms, flank guard).
- `metrics.py` — FROZEN Gate-1 criteria (self-test: real status-quo wake FAILS, a
  reference-true chain PASSES). Reference measurement convention lives here.
- `run_gate0a.py` — Gate 0a harness validation (textbook KH tanh layer; PASS).
- `mock_billows.py` — Gate 0b visual acceptance mock (accepted).
- `run_gate1.py` — the 50-run pre-registered scan (A×δ×form×seed + shear-off).
- `healthy_window.py` — strictly-stable-window re-adjudication of the scan.
- `discriminator_decay.py` — the unforced sheet-decay discriminator (removes the
  Dirichlet pump; the run that resolved the forced blowup as a harness artifact).
- `render_healthy.py`, `render_discriminator.py`, `probe_geometry.py` — diagnostics.

## Key evidence panels
- `gate0a_panel.png` — harness validation (clean KH rollup, growth within 2.7%).
- `healthy_window_montage.png` — forced scan: laminar/large-rolls → CFL blowup.
- `discriminator_terminal.png` — unforced decay: stable, but 2D turbulence (not an
  ordered chain); the panel behind the final verdict.
