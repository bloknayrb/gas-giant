"""Nondim table + FROZEN environment constants. SINGLE SOURCE. All rates per-step."""
import numpy as np
from gasgiant.params.presets import load_factory_preset
from gasgiant.sim.profiles import build_profiles
from gasgiant.sim.bands import generate_bands
from gasgiant.sim.solver import compute_dt

P = load_factory_preset("gas_giant_warm")

RC   = P.storms.hero_radius                 # 0.108
LAT0 = np.radians(P.storms.hero_latitude)   # -24 deg
DX   = np.pi / (P.sim.resolution // 2)      # pi/2048 (production dphi = dlam)
NX, NY = 1024, 512
LX, LY = NX * DX, NY * DX
LD   = P.solver.deformation_radius; INV_LD2 = 1.0 / (LD * LD)
BETA = P.solver.coriolis_f0 * np.cos(LAT0)  # ~2.74

_bands    = generate_bands(P.seed, P.bands)
_profiles = build_profiles(P.seed, _bands, P.bands, P.jets,
                           hero_lat_deg=P.storms.hero_latitude, hero_r_core=RC)
DT = compute_dt(P.sim.resolution, P.sim.dt_scale, _profiles.max_speed)

TAU_STEPS = P.solver.vort_relax_tau         # 600; per-step nudge fraction = 1/TAU_STEPS
OMEGA_CEILING = 60.0                        # omega_force.comp:30
HYPERVISC = P.solver.vort_hypervisc
PSI_DRAG  = P.solver.vort_psi_drag          # per-step fraction (omega_force.comp:225)
WAKE_TURB_AMP = 0.6 * P.storms.wake_turbulence * P.storms.hero_emergence  # :195
WAKE_FREQ = 0.9 / RC
EVOLUTION_RATE = P.turbulence.evolution_rate  # fbm time axis increment/step (solver.py:878)
DEV_STEPS = P.sim.dev_steps                 # 700 from the preset, not hard-coded

# ---- FROZEN environment constants (Task 4 sign-off locks these) -------------
# Re-derived per Task-0 STOP (probe_geometry.py): LY/2 is the bracket SEAT (u~0.036,
# band-fill 39365 steps >> 8000 cap). 0.45*LY = lat -26.24, the south shear flank:
# u=+1.08 eastward (+x downstream, no frame flip), |du/dy|=26.3 (the strain
# environment), strip->band 356 / band-fill 1311 steps. Ambient vorticity is
# uniform-sign (~-26) across the whole bracket window so the flanks are
# mirror-equivalent for sheet dynamics. PENDING USER RATIFICATION at the Gate-0b
# checkpoint; frozen at Task-4 sign-off.
Y_SHEET   = 0.45 * LY
STRIP_X   = (0.5 * RC, 0.5 * RC + 4 * DX)   # Dirichlet inflow strip (hard-set)
SEED_AMP_FRAC = 0.02                        # fresh strip noise, fraction of A, k < pi/(4 DX)
TR_STEP_W = 0.5 * RC                        # tracer belt/zone step width
SPONGE_X  = 1.5 * RC                        # cosine-ramped, per-step fraction 1/20
SPONGE_RATE = 1.0 / 20.0
MEANHOLD_RATE = 1.0 / TAU_STEPS             # zonal-mean ambient hold; OFF when nudge flag on
BAND_X    = (3 * RC, 12 * RC)               # scoring band (x)
BAND_HW   = 1.2 * RC                        # scoring band half-width (y, about Y_SHEET)
FLANK_W   = 1.0 * RC                        # flank-guard region at y-boundaries
FLANK_MAX = 0.10                            # (deviation from spec's 'boundary third': recorded)
# Task-4 CONTROLLER DECISION (pending checkpoint ratification): y-flank EDDY
# sponges — domain-openness infrastructure, NOT production machinery. The
# doubly-periodic box retains beta-radiated waves and fed turbulence that an
# open domain (and production, via its everywhere-nudge) sheds; the flanks are
# declared inert buffers, so sponging their EDDY components ((w - zonal_mean),
# (tr - tr_amb)) enforces the declared geometry. Zonal MEAN stays governed by
# the mean-hold. Cosine-ramped 0 at the interior edge -> full rate at the wall.
FLANK_SPONGE_W    = FLANK_W                 # width of each y-edge eddy sponge
FLANK_SPONGE_RATE = SPONGE_RATE             # peak per-step fraction (1/20)
EDGE_BLEND = NY // 8                        # Tukey edge-match (deviation from spec 'mirrored': recorded)

SEEDS = {"gate1": 11, "gate1_replicate": 12, "seed_noise": 13, "fbm": 14,
         "gate15": 15, "gate2": 16, "collateral": 17}

def warm_profile_window():
    y = (np.arange(NY) + 0.5) * DX
    lat = LAT0 + (y - LY / 2.0)
    u = np.interp(lat, _profiles.lat[::-1], _profiles.u[::-1])
    edge = EDGE_BLEND
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(edge) / edge))
    mean_uv = 0.5 * (u[:edge].mean() + u[-edge:].mean())
    u[:edge]  = mean_uv + (u[:edge]  - mean_uv) * ramp
    u[-edge:] = mean_uv + (u[-edge:] - mean_uv) * ramp[::-1]
    return y, u, -np.gradient(u, DX)

def transit_report():
    """u at the sheet line and the advective clocks every gate is checked against."""
    y, u, _ = warm_profile_window()
    u_sheet = abs(np.interp(Y_SHEET, y, u))
    strip_to_band = (BAND_X[0] - STRIP_X[1]) / max(u_sheet, 1e-6) / DT   # steps
    band_fill     = (BAND_X[1] - BAND_X[0]) / max(u_sheet, 1e-6) / DT    # steps
    return u_sheet, strip_to_band, band_fill

if __name__ == "__main__":
    assert 5e-4 < DT < 1e-3, DT
    us, s2b, bf = transit_report()
    print(f"DT={DT:.3e} TAU_PHYS={TAU_STEPS*DT:.3f} BETA={BETA:.2f} "
          f"box={LX/RC:.1f}x{LY/RC:.1f} rc")
    print(f"u(sheet)={us:.3f}  strip->band={s2b:.0f} steps  band-fill={bf:.0f} steps")
    _, _, om = warm_profile_window()
    print(f"ambient max|du/dy|={np.abs(om).max():.1f} (expect ~22)")
