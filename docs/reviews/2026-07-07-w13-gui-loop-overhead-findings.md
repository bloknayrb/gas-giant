# W13 — GUI-loop per-step overhead: root-cause by elimination

*Closes the last item of the 2026-07-02 comprehensive-review remediation program.
Investigates the open perf question W10a exposed (see the `DEFAULT_STEPS_PER_FRAME`
comment in `src/gasgiant/app/main.py`): the dev-run preview steps at ~85 ms/step headless
but ~330 ms/step under the live GUI — a ~4× per-step gap that is where the first-launch
minutes actually go.*

## What W10a already established

- GUI dev-run throughput is **flat across steps-per-frame** (~3.0 steps/s at both 2 and
  8 spf; frame time grows 653 ms → 2379 ms to match). So the overhead is **per-step, not
  per-frame** — a per-frame cost (one vsync/present/derive per frame) would amortize across
  more steps and throughput would rise. It doesn't. This already refuted the original
  "vsync suspect" framing for a *per-frame* vsync cost.
- `Simulation.tick()` is chunk-invariant, so the gap is not a chunking artifact.

## Headless decomposition (this wave)

Measured on the primary dev box (RTX 3070), `gas_giant_warm` (vorticity, sim res 4096,
`PREVIEW_WIDTH` 2048), every stage forced to true GPU completion with `ctx.finish()`
(`_diag`/job-tmp harness, not committed):

| Stage | What it measures | Result |
|-------|------------------|--------|
| **A** amortized step | `step(40)` + one finish — the export path's deep-pipeline cost | **80.6 ms/step** |
| **C** step + per-frame finish | 20×[`step(2)`; finish] — finish every frame, no derive | 82.5 ms/step |
| **B** GUI-equivalent | 20×[`step(2)`; full 2048 derive; finish] — everything the GUI's GL work does per frame | **85.0 ms/step** |
| **D** full derive @2048 | one `ensure_preview(2048)` | 5.2 ms/frame |
| **D** full derive @1024 / @512 | " | 3.6 / 1.8 ms/frame |

**The entire headless GUI-equivalent GL path — sim step + per-frame GPU drain + full-res
derive — is ~85 ms/step, identical to the export path.** The three most-plausible in-code
culprits are each negligible:

- **GPU-pipeline draining per frame: 1.9 ms** (C − A). Forcing a `finish()` every frame does
  *not* remove meaningful cross-step pipelining — the vorticity steps are serialized on the
  GPU by their `memory_barrier` dependencies regardless, so there was little pipeline to lose.
- **Full-res preview derive: ~5 ms/frame**, 3 % of a 2-step frame. Cheap.
- **`memory_barrier()` calls** in the solver are GPU-side barriers (glMemoryBarrier), not
  CPU↔GPU syncs — they never drain the pipeline to the CPU.

## Conclusion

The ~245 ms/step of overhead is **not in the sim, the derive, or GPU synchronization** —
all of that is the 85 ms the export path also pays. It lives entirely in the **windowed
presentation path** the headless harness does not exercise: hello_imgui / GLFW frame
present under the desktop compositor, with the compute-heavy sim callback running on the
same GL context inside the vsync'd frame loop. `fps_idling.enable_idling` is already
`False` (`main.py:1657`), so power-save idling is **not** the cause; the runner uses
hello_imgui's default swap interval (vsync on) with no override.

### Two tempting fixes this rules OUT

- **Lowering `PREVIEW_WIDTH` (2048 → 512):** saves ~3.4 ms/frame. The derive was never the
  cost. Not worth the preview-quality loss.
- **Stepping more per frame:** W10a already measured this flat. The overhead is per-step.

### Recommended next step (needs a live display — not runnable headless)

The remaining suspect is only observable in a real window. The GUI already owns a
`RenderPerfMeter` (`app.render_perf`, the Playback/Performance pane). The one experiment
that would localize it:

1. In a live session, split the per-frame timing into **(a)** the sim+GL callback
   (`_draw_viewport`/`draw_sphere`) vs **(b)** the hello_imgui present/`swap_buffers`, and
   read the split off the perf pane during an active dev run.
2. If present/swap dominates (the expectation), test **vsync off during an active dev run**
   — the highest-probability cheap fix. hello_imgui doesn't expose a swap-interval knob on
   `RunnerParams` directly; the lever is a `glfwSwapInterval(0)` in `post_init`, gated to
   only apply while a dev run is in flight so idle UI still vsyncs (no tearing when static).

This is a one-line experiment but it **must be verified live** — a headless harness has no
swapchain, so it cannot confirm the fix. Left as a documented, scoped experiment rather
than an unverified code change to the GUI loop.

## Disposition

**W13 = INVESTIGATED / ROOT-CAUSED BY ELIMINATION.** The 4× is confirmed *not* an
engine/derive/pipeline cost (headless GUI-equivalent path = export path = 85 ms/step); it
is a windowed-present cost. The fix is a scoped live vsync experiment (above), not code
this wave can verify. This closes the 15-wave (W0–W14 + CI-fix) remediation program's
investigation obligations; the vsync experiment is the single carried-forward TODO.
