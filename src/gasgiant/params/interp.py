"""Parameter interpolation for ramp (state A -> state B) sequence exports.

A "ramp" morphs the look of a planet over an exported animation by interpolating
every leaf of the parameter tree from a base state ``a`` toward a target ``b``.
``lerp_params`` produces the frame's params; ``validate_ramp`` rejects a target
pairing that cannot be ramped (a mid-sequence dev-run restart, or a seed change
-- a ramp keeps the SAME developed world and only re-derives/re-velocity it).

Type dispatch (leaf-by-leaf over the two ``model_dump()`` trees):

- float ....................... linear lerp ``a + (b - a) * t``
- color (a 3-tuple, the model's only tuple type) ... linear-RGB per-component lerp
- int ......................... rounded lerp ``round(a + (b - a) * t)``
- bool / enum / str ........... must be EQUAL in a and b (cannot interpolate)
- ``None`` <-> value .......... rejected (an optional pin cannot appear/vanish)
- stop-list / palette-row shape mismatch (length or ``pos`` set) ... rejected

``t == 0.0`` returns ``a`` BIT-EXACT (a deep copy -- no float round-trip); the
non-endpoint result is re-validated through ``PlanetParams.model_validate``.
"""

from __future__ import annotations

from gasgiant.params.model import PlanetParams, Tier
from gasgiant.params.tiers import diff_tier_paths


class RampError(ValueError):
    """A ramp cannot be built from the given endpoints (non-interpolable leaf,
    shape mismatch) or is disallowed (RESTART-tier / seed diff)."""


def _lerp_scalar(x: float, y: float, t: float):
    """Lerp two numbers; INT+INT -> rounded int, otherwise float."""
    if isinstance(x, int) and isinstance(y, int):  # bools are handled before we get here
        return int(round(x + (y - x) * t))
    return x + (y - x) * t


def _lerp_node(va, vb, t: float, path: str):
    # None <-> value: an optional pin appearing/disappearing cannot interpolate.
    if va is None or vb is None:
        if va is None and vb is None:
            return None
        raise RampError(
            f"{path}: cannot interpolate an optional field that is set on one side "
            f"only (a={va!r}, b={vb!r})"
        )
    # bool BEFORE int (bool is an int subclass): booleans can't interpolate.
    if isinstance(va, bool) or isinstance(vb, bool):
        if va != vb:
            raise RampError(f"{path}: cannot interpolate boolean {va!r} -> {vb!r} (must be equal)")
        return va
    # dict: recurse per key (both trees share the model's key set).
    if isinstance(va, dict):
        return {
            k: _lerp_node(va[k], vb[k], t, f"{path}.{k}" if path else k)
            for k in va
        }
    # tuple: the model's ONLY tuple type is a 3-component linear-RGB color.
    if isinstance(va, tuple):
        if len(va) != len(vb):
            raise RampError(f"{path}: color length mismatch ({len(va)} vs {len(vb)})")
        return tuple(_lerp_scalar(x, y, t) for x, y in zip(va, vb, strict=True))
    # list: shape-checked (length + stop/palette ``pos`` sets), then element-wise.
    if isinstance(va, list):
        if len(va) != len(vb):
            raise RampError(
                f"{path}: list length mismatch ({len(va)} vs {len(vb)}); "
                f"cannot interpolate a reshaped stop-list / palette-row set"
            )
        _check_pos_sets(va, vb, path)
        return [
            _lerp_node(x, y, t, f"{path}[{i}]")
            for i, (x, y) in enumerate(zip(va, vb, strict=True))
        ]
    # numbers (float / int) -- bools already handled above.
    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
        return _lerp_scalar(va, vb, t)
    # str / enum (and anything else): must be equal.
    if va != vb:
        raise RampError(f"{path}: cannot interpolate {va!r} -> {vb!r} (must be equal)")
    return va


def _check_pos_sets(va: list, vb: list, path: str) -> None:
    """For a list of gradient STOPS (dicts carrying ``pos``), the anchor positions
    must match one-to-one: a reshaped gradient (stops added/removed/moved) cannot
    be interpolated stop-by-stop."""
    if va and isinstance(va[0], dict) and "pos" in va[0]:
        pos_a = [e.get("pos") for e in va]
        pos_b = [e.get("pos") for e in vb]
        if pos_a != pos_b:
            raise RampError(
                f"{path}: gradient stop positions differ (a={pos_a}, b={pos_b}); "
                f"cannot interpolate a reshaped gradient"
            )


def lerp_params(a: PlanetParams, b: PlanetParams, t: float) -> PlanetParams:
    """Interpolate the whole parameter tree from ``a`` (t=0) to ``b`` (t=1).

    ``t == 0.0`` returns a bit-exact deep copy of ``a`` (no float round-trip);
    ``t == 1.0`` a bit-exact deep copy of ``b``. In between, every leaf is
    dispatched by type (see module docstring) and the merged dict is re-validated
    through ``PlanetParams.model_validate``. Raises ``RampError`` on any leaf that
    cannot interpolate (differing bool/enum/str, optional appearing/vanishing,
    stop-list/palette-row shape mismatch)."""
    if not 0.0 <= t <= 1.0:
        raise RampError(f"ramp t must be in [0, 1], got {t}")
    if t == 0.0:
        return a.model_copy(deep=True)
    if t == 1.0:
        return b.model_copy(deep=True)
    merged = _lerp_node(a.model_dump(), b.model_dump(), t, "")
    return PlanetParams.model_validate(merged)


def validate_ramp(a: PlanetParams, b: PlanetParams) -> None:
    """Raise ``RampError`` if the ramp from ``a`` to ``b`` is disallowed.

    A ramp advances ONE developed world under changing params, so it can only
    touch POST-tier (re-derive) and VELOCITY-tier (rebuild jets, run continues)
    fields. A RESTART-tier diff would re-initialize the dev run mid-sequence, and
    a ``seed`` change would swap the world outright -- both are rejected, naming
    the offending field paths."""
    offenders = sorted(
        path for path, tier in diff_tier_paths(a, b) if tier == Tier.RESTART
    )
    if offenders:
        raise RampError(
            "cannot ramp: these changes require restarting the development run "
            "(RESTART-tier fields, including seed) and cannot be interpolated "
            "mid-sequence: " + ", ".join(offenders)
        )
    # Dry-run a NON-ENDPOINT lerp so any non-interpolable leaf — a differing
    # bool/enum/str, an optional pin set on one side only, a reshaped
    # stop-list — raises HERE (fail fast), not at frame 1 of a sequence export
    # after frame 0 was already fully rendered and written.
    _lerp_node(a.model_dump(), b.model_dump(), 0.5, "")
