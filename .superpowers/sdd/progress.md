# UI/UX Roadmap — Progress Ledger

Plan: C:\Users\blokn\.claude\plans\plan-out-improving-the-streamed-valiant.md
Worktree: .claude/worktrees/ui-ux-roadmap (branch worktree-ui-ux-roadmap)
Baseline: 521 passed, 12 pre-existing GPU-noise failures (unrelated scope) — user approved proceeding 2026-06-30.

## Tasks
Task 0 (Phase 0): complete (commits df366a7..969c1bd, review clean — Approved, no Critical/Important).
  Minor (carry to final review): sphere_preview.py reset() bound to double-click (disclosed scope-add, not in brief); no test file for sphere_preview.py (pre-existing convention, no regression).
Task 1 (Phase 1): complete (commits 969c1bd..4d2cc2a, review clean — Approved, no Critical/Important).
  Minor (carry to final review): in-code comment at main.py:215 should note "Phase 2: push undo entry before this clear"; task-1-report.md overstates test count (10 actual vs 12 claimed, all non-vacuous) — cosmetic only.
  Removed _try_commit_draft entirely (verified zero remaining callers, no lost behavior).
Task 2 (Phase 2): complete (commits 4d2cc2a..210f6d3, review clean — Approved, no Critical/Important).
  Minor (carry to final review): test_undo_redo.py covers only the Load discrete-push site directly; preset-combo/Randomize/Reroll pushes share the same one-liner but lack dedicated tests.
Task 3 (Phase 3): complete (commits 210f6d3..ccfd16a, review clean — Approved, no Critical/Important).
  Minor (carry to final review): Reroll is a no-op when seed is locked (correct but undocumented consequence, main.py _randomize); composite palette/stops right-click popup only hits the last sub-widget drawn, not the whole row (acceptable rough edge, revisit when palette editor is overhauled); seed renders both in header and in the auto-generated Sim section (harmless redundancy); two _process_edit calls/frame when seed changes (benign, header commits before panel reads updated _live).
Task 4 (Phase 4): complete (commits ccfd16a..32fc6d3, review clean — Approved, no Critical/Important).
  Minor (carry to final review): docs/sliders.md not regenerated (scripts/render_slider_examples.py has a pre-existing, unrelated bug — dropped jupiter_baroclinic preset never removed from the generator; also the generator doesn't read ui/adv metadata at all, so a fix wouldn't even reflect Phase 4's grouping today — separate follow-up needed); "N advanced differ" hint only fires for sections with ZERO visible leaves, so a mixed Basic/Advanced section with a hidden non-default advanced field shows no hint (in-spec per brief's literal scope, but a real discoverability gap); PanelState.show_advanced default flipped True->False (justified/necessary for the feature, honestly test-documented, no Phase 3 test broken).
Task 5 (Phase 5): complete (commits 32fc6d3..3e7283d, review clean — Approved, no Critical/Important).
  Scope note (not a defect): export-gating expanded beyond the brief's two literal items (Restart-dev + POST commits) to cover all six discrete-action paths (preset/Load/Randomize/Reroll/Undo/Redo) since they all bypass _process_edit via direct _commit calls — reviewer confirmed this is the minimum needed to satisfy the brief's actual "all param commits blocked" goal, not scope creep. Same recurring pattern as Phase 1/2's "all six/four discrete paths."
  Minor (carry to final review): the begin_disabled(exporting) gate on the six discrete-action buttons is verified only by inspection (imgui disabled-state isn't unit-testable headless); MAX_STEPS_PER_FRAME=256 "Max" speed bound is sane but unbenchmarked.
Task 6 (Phase 6): complete (commits 3e7283d..975a0da, review clean — Approved, no Critical/Important).
  Notable: widened Phase 2's UndoRecord third field from bool|None to PlanetParams|None (full pristine snapshot) — reviewer confirmed necessary (no lighter alternative works) and control-flow-preserving (only _record's payload changed; Phase 2's push/pop/gesture logic and existing test assertions unmodified/still pass).
  Process note: implementer's report misattributed GPU test failures to "uncommitted shader/facade edits" it found mid-investigation -- that description matches the MAIN repo's pre-existing unrelated dirty state (view_transform.frag/viewport.py/facade.py/test_hero_wake.py), not this worktree. Controller independently confirmed the worktree stayed clean throughout and only Phase 6's 6 files were committed. Likely a stray cwd reset to the main checkout during investigation (same class of quirk Phase 1 hit and self-corrected). No actual cross-contamination occurred either time.
  Minor (carry to final review): test_no_preset_label_is_unsaved_never_starred name over-promises vs. real restored-session behavior (label shows "unsaved *" when dirty after a session restore, which is correct behavior but not what the test covers).
Controller-independent GPU re-run on 975a0da (Phase 6): 163 passed / 11 failed / 1 skipped -- same known-flaky physics family, membership shuffled again vs. prior runs (as expected for environmental noise). Confirms Phase 6 introduced no GPU regression.

Task 7 (Phase 7: help & shortcuts): complete (commits 975a0da..d7e7afc, review clean).
Reviewer approved. Two self-reported concerns independently adjudicated by reviewer:
- Pre-existing IM_ASSERT crash in _draw_seed_header_control confirmed genuinely pre-existing
  (reproduced identically on base commit 975a0da via isolated worktree) -- out of scope, carry
  to final review as a known blocker for live smoke runs.
- Ctrl+S behavior confirmed identical to the Save button (single code path via
  _open_save_dialog(), no divergence, no silent overwrite).
Minor (carry to final review): draw_help() window body has no automated test coverage (only
headless-verified manually by implementer and reviewer); a future regression there won't be
caught by CI.

Task 8 (Phase 8: deeper polish) [final phase]: complete (commits d7e7afc..a621968,
review clean). Reviewer independently confirmed: dirty-flag independence (grepped ALL
_post_dirty/_tracers_changed sites in facade.py, every one has a matching
_emission_preview_dirty assignment); scratch-texture isolation (no cross-write between
ensure_preview's _preview_color/_preview_height and ensure_preview_emission's separate
_preview_em_* trio) with a guard test whose interleaved-call ordering would actually catch
an aliasing regression, not just a call-once smoke test; export modal genuinely adds a
confirm step (old bare-click-opens-folder-picker behavior is gone) and preserves the
Phase 5 six-path export-gate; sun_azimuth/sun_elevation confirmed zero footprint in
params/model.py (grepped). Refuted the implementer's self-reported "untested GL tail"
concern as overstated -- the enabled-emission path IS exercised end-to-end on the real
headless gpu fixture; only the generic view_transform blit tail (shared by all channels,
pre-existing, not new to this phase) is GL-only.
Controller-independently re-ran the full GPU suite: 13 failed / 167 passed / 1 skipped
(membership: checkpoint, deformation_radius, eddy_drag, hero_solid_core, m3_ship family,
oval_solid_core, psi_drag, vort_inject_mask -- exactly the documented known-flaky physics
family; none touch facade.py's emission-preview code or the new viewport/sphere/main.py
export-modal code). Confirms Phase 8 introduced no GPU regression.

ALL 9 PHASES (0-8) COMPLETE. Proceeding to final whole-branch review.

## Final whole-branch review (df366a7..a621968)

Verdict: NOT APPROVED -- one Critical blocker. All 11 consolidated guard tests verified
present+passing; 478 unit tests green; ruff/lint-imports clean; cross-phase consistency
(six discrete-action paths) confirmed uniform; no dead code from _try_commit_draft removal;
panels.py reflection genericity holds end-to-end. All prior carried-forward Minors triaged
as genuinely Minor/ship-as-follow-up EXCEPT:

CRITICAL (reclassified from a Phase-7-carried Minor): IM_ASSERT(id != 0) crash on the FIRST
frame the Controls panel draws. Root cause pinned: imgui.input_int()'s default +/- step
buttons wrap the widget in BeginGroup/EndGroup; EndGroup finishes with ItemAdd(id=0), so
begin_popup_context_item() called with no explicit str_id (both main.py:597
_draw_seed_header_control and panels.py:576 _draw_leaf's seed leaf -- seed is the only
input_int leaf and is Basic-visible) hits the assert. CORRECTION TO PRIOR RECORDS: the
Phase 7 reviewer's "confirmed pre-existing on base commit 975a0da" was misleading --
975a0da is Phase 6's tip, already downstream of Phase 3 which INTRODUCED both
_draw_seed_header_control and the begin_popup_context_item call. Measured against
origin/master (df366a7, the actual merge-base), grep confirms NEITHER exists there. This
branch introduces a frame-1 crash in its own centerpiece feature -- not inherited scope.
Fix: explicit str_id on both begin_popup_context_item() calls (or step=0 on both input_int
calls). Dispatching one fix subagent now.

Fix (Critical, from final review): complete (commit dc808f1, on top of a621968).
Explicit str_id given to both begin_popup_context_item() calls (main.py:602
"seed_header_context"; panels.py:585 "context", safe within the leaf's existing
push_id(name) scope). Third-site grep across src/gasgiant/app/ found no other
grouped-widget+bare-popup combination. Crash reproduced pre-fix (headless imgui frame,
both sites), confirmed gone post-fix. Two regression tests added to test_shortcuts.py
that fail against unfixed source and pass against the fix. Full unit suite 480 passed,
ruff clean, lint-imports clean. GPU suite not re-run (fix scoped to app/main.py +
app/panels.py + a unit test only, no engine/facade/shader touch).
Controller-independently verified: worktree clean beyond the 3 committed files; main
repo's pre-existing unrelated dirty state unchanged (same files noted at session start).

Re-review of Critical fix (commit dc808f1): independently verified by a fresh reviewer
(fix correctness, push_id/pop_id balance traced, id-collision grep, crash reproduced
fixed AND reproduced broken via temporary revert, regression tests confirmed non-vacuous,
full suite re-run: 480 passed, ruff clean, lint-imports clean).

FINAL VERDICT: APPROVED FOR MERGE. All 9 phases + the one Critical fix complete and
reviewed clean. Proceeding to superpowers:finishing-a-development-branch.

## Post-PR review remediation (2026-07-01, commits ae03844 + ecc7823)

A fresh multi-agent review of the open PR (#13) — general bugs, silent failures,
test coverage, type design — surfaced findings the per-phase reviews missed
(cross-phase interactions). Each fix landed TDD (failing test verified red before
green); plan itself went through two adversarial reviews before implementation.

Behavioral regressions (branch-introduced): #1 paused mid-development read as a
hang (draw_perf now shows a static paused label; tick() no-ops once developed so
force-ticking was wrong); #2 randomize(seed) drift from the Phase-4 wake_turbulence
reorder (added a golden guard pinning current output + a StormsParams declaration-
order note); #4 stale-redo clobber on export-setting edits (new _commit_output_setting
clears redo); #5 committed out-of-range typed value silently swallowed (toast restored).

Silent failures: #6 pre-migration session-backup write failure logged (was a silent
pass); #7 export failure logs the traceback + non-empty toast.

Minors: no-op undo suppression, color list/tuple marker (_leaf_changed), F1/A
repeat=False, gradient add-copy. DECLINED: step-granularity (contradicts a tested
intended design). DEFERRED: section-name search (risks the guarded M9 separator
property).

#3 (input_int seed "never commits") REFUTED empirically: is_item_deactivated_after_edit
DOES fire for input_int (EndGroup forwards the deactivated flag); the id==0 quirk only
ever affected the popup crash. Regression test added.

Type refactors (F, behavior-preserving): PresetSource/DialogKind StrEnums, UndoRecord
NamedTuple, ExportJob dataclass, FieldMeta typed metadata, pfield factory= for the
AppearanceParams mutable-default lists.

Verification: unit 500 passed (was 480; +20 regression/guard tests); ruff + lint-imports
clean. GPU 171 passed / 1 skipped / 9 flaky byte-identity failures — VERIFIED PRE-EXISTING
by re-running the same tests on baseline dc808f1 (same failures there; the failing SET
shifts run-to-run = session-context LSB noise, not a code regression; no GPU/solver/GLSL
code was touched).
