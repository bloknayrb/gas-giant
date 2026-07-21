# Stratified-KH wake-billow feasibility spike — VERDICT (2026-07-21)

**Question.** Can the sim produce the reference GRS wake's *ordered, arrested billow chain* —
a persistent downstream row of many co-scaled Kelvin–Helmholtz billows that do **not** merge
(NASA PIA07782)? Four prior approaches were exhausted (passive-tracer stamp; frozen-field flow
injection; solver-dissipation rewrite — FALSIFIED by the 2026-07-20 crux spike; render-time
synthesis — user-rejected as decoration, PR #48 closed). All shared one wall: in a strictly-2D
**barotropic** vorticity solver (the production engine), a row of like-signed vortices is
unconditionally unstable to subharmonic pairing → merger is inevitable, so "a few large rolls or
2D turbulence" is the correct 2D behaviour, not a bug. This spike tested the one remaining
physically-motivated escape route: **explicit stratification** (the finite-Richardson pairing-
saturation mechanism), which a barotropic solver structurally cannot represent because it has no
buoyancy.

**Instrument (throwaway; measurement-only; nothing in `src/`).** A new vertical-plane 2D
**Boussinesq** box (vorticity–streamfunction + buoyancy), Fourier-x / finite-difference-z, anomaly
formulation about a `tanh` shear layer with stable stratification `N²z`. Corrected baroclinic-torque
sign `Dζ/Dt = −∂b/∂x` (a stratified rest state must oscillate, not grow — the sign gate; the CFD
design review caught the flipped sign). Biharmonic hyperviscosity for grid-scale control. Two
adversarial design reviews (CFD/numerics + falsifiability) validated the direction and re-scoped it
to a multi-day careful build. Scripts: `scratchpad/spike3/` (throwaway).

## What is genuinely established (verified, adversarially defended)

1. **The instrument is valid.** An unstratified layer (Ri=0) rolls up into cat's-eye KH billows at
   the linear wavelength (λ≈16.8·δ_tanh) and then **pairs** (6 billows → 3), with and without
   hyperviscosity.
2. **Stratification genuinely arrests subharmonic pairing / upscale transfer — in 2D.** At bulk
   Richardson number Ri≈0.10 the layer holds a co-scaled billow chain that refuses to coarsen,
   confirmed **four independent ways**: direct field renders; modal spectra; seed robustness (2
   seeds); and the decisive arrest-vs-coarsen contrast at *identical* hyperviscosity — unstratified
   drains the KH mode and piles energy at the largest scales (modal centroid n8→n4.3, or n7→n3/n6),
   while stratified shows ~zero low-mode (upscale) energy.
3. **This is stratification, not a numerical artifact.** An adversarial result-review refuted the
   killer objection (that the hyperviscosity was suppressing the pairing mode) quantitatively: the
   biharmonic damping timescale at the subharmonic wavenumber is ~10³× the run length; halving the
   hyperviscosity leaves the arrest identical; and the *same* hyperviscosity lets the Ri=0 control
   pair normally. Regime is physical (Ri<0.25, Miles–Howard; matches the stratified-KH pairing-
   suppression literature, Klaassen & Peltier 1985; Smyth & Peltier).

**Net positive finding:** buoyancy **is** the missing ingredient the 2D barotropic production solver
lacks. This *refutes the flat "impossible in our solver class"* framing — the arrest mechanism is
real and reachable *in principle*.

## Why this is still a dead end for a texture-solver feature

1. **The 2D ordered chain is itself a 2D artifact.** The same stratified-KH literature shows that at
   this Richardson number the primary billows are unstable to **3D secondary instabilities** and
   break down into turbulence. A strictly-2D simulation preserves the tidy chain only by *forbidding*
   that breakdown route. The confound-closing even-box run showed this directly: in that
   configuration the stratified layer did **not** hold a clean chain — it roughened toward small
   scales (the 2D shadow of the 3D turbulent breakdown), while still refusing to coarsen. The clean
   persistent chain of the first run was configuration-dependent.
2. **Idealisation inflates durability.** The frozen analytic base (shear + stratification never mixed
   away) and the temporal periodic box overstate how long the arrest lasts; a spatially-developing
   (convective) wake case was never validly run.
3. **The real route is a different engine.** A physical ordered chain requires a full **3D**
   (or carefully-designed 2.5D) **stratified** solver — a major project, a foreign architecture to
   the equirect single-2D-field texture pipeline, and one the literature says still is **not
   guaranteed** to yield a clean chain rather than turbulence.

## Verdict

**NO-GO for a production feature; mechanism understood.** The ordered wake billow chain is
intrinsically a **3D, finite-stratification** phenomenon living in a narrow, turbulence-prone
window. The 2D barotropic production solver cannot host it (no buoyancy). A 2D **stratified** model
recovers the arrest *mechanism* but only as a fragile artifact that does not robustly hold a clean
chain and would break down in 3D. Therefore the phenomenon is **beyond a tractable texture solver**
(2D barotropic, 2D stratified, or 2.5D). Do not re-attempt the ordered wake chain via any 2D solver
change, render synthesis, or dissipation tuning. The only genuine route is a full 3D stratified
engine — treat that as an independent major project, not a wake feature, and only if broadly
justified (it would not be guaranteed to succeed).

Five approaches now exhausted: tracer-stamp · flow-injection · dissipation-rewrite · render-synthesis
· 2D-stratified-spike. This spike is the terminal, mechanistically-grounded, adversarially-verified
close of the emergent-wake-chain line.

*(Measurement-only research; nothing was written to `src/gasgiant/**`. Spike harness is throwaway
scratch under `scratchpad/spike3` + `scratchpad/spike3_review`.)*
