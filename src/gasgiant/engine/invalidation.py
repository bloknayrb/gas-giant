"""Parameter-diff -> invalidation tiers (engine-layer re-export).

The tier-walk itself moved DOWN to ``gasgiant.params.tiers`` (params layer) so a
params-layer consumer -- the ramp validator in ``params.interp`` -- can use it
without importing this engine-layer module. This module re-exports the same
objects so every existing ``from gasgiant.engine.invalidation import diff_tiers``
(facade, app, and the test suite) keeps working unchanged.

- POST     re-derive maps from existing tracers (instant)
- VELOCITY rebuild the velocity field, sim continues under the new field
- RESTART  re-initialize the development run from step 0
"""

from __future__ import annotations

from gasgiant.params.tiers import (
    diff_tier_paths,
    diff_tiers,
    needs_repost,
    needs_restart,
    needs_velocity_rebuild,
)

__all__ = [
    "diff_tier_paths",
    "diff_tiers",
    "needs_repost",
    "needs_restart",
    "needs_velocity_rebuild",
]
