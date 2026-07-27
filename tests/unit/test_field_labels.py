"""Authored widget captions (``pfield(label=...)``).

The caption is the only text an artist reads while SCANNING the panel -- the
description appears on hover. Derived from the field name, a solver lever reads
``vort psi drag``; authored, it reads "Swirl brake (large only)".

The risk this file guards is not a wrong caption, it is a caption that quietly
breaks something else: search, the docs, or an imgui id.
"""

from __future__ import annotations

import typing

import pytest
from pydantic import BaseModel

from gasgiant.params.model import (
    FieldMeta,
    PlanetParams,
    derived_label,
    field_label,
)

panels = pytest.importorskip("gasgiant.app.panels")


def _leaves():
    """``(dotted_path, field_name, FieldInfo)`` for every pfield leaf."""
    out, seen = [], set()

    def walk(model, prefix=""):
        for name, info in model.model_fields.items():
            for member in (info.annotation, *typing.get_args(info.annotation)):
                if isinstance(member, type) and issubclass(member, BaseModel):
                    walk(member, f"{prefix}{name}.")
            if "tier" in (info.json_schema_extra or {}):
                key = (model.__name__, name)
                if key not in seen:
                    seen.add(key)
                    out.append((f"{prefix}{name}", name, info))

    walk(PlanetParams)
    return out


def _labelled():
    return [(p, n, i) for p, n, i in _leaves() if FieldMeta.of(i).label]


def test_some_fields_are_labelled():
    """Sanity: the rest of this file is vacuous if the metadata stopped landing."""
    assert len(_labelled()) >= 20


def test_unlabelled_fields_fall_back_to_the_derived_caption():
    for _path, name, info in _leaves():
        if not FieldMeta.of(info).label:
            assert field_label(name, info) == derived_label(name)


def test_authored_captions_are_plain_text():
    """imgui eats everything after ``##`` in a widget caption (it is the id
    separator), and ``|`` breaks the docs/sliders.md tables the same string is
    rendered into."""
    for path, name, info in _labelled():
        label = field_label(name, info)
        assert label.strip() == label, f"{path}: padded caption {label!r}"
        assert "#" not in label, f"{path}: '#' is imgui id syntax"
        assert "|" not in label, f"{path}: '|' breaks the doc tables"
        assert "\n" not in label, f"{path}: captions are single-line"


def test_authored_captions_drop_the_engine_vocabulary():
    """A caption that still says 'vort' has not done its job -- that is the
    exact string the artist could not read in the first place."""
    banned = ("vort", "psi", "sor ", "coriolis", "poisson", "hypervisc", "wavenumber", "tau")
    for path, name, info in _labelled():
        low = field_label(name, info).lower()
        assert not [t for t in banned if t in low], f"{path}: caption still jargon: {low!r}"


def test_relabelled_fields_stay_findable_by_their_old_caption():
    """The regression an authored label invites: the search haystack carries the
    shown caption, so REPLACING the derived form silently un-finds every
    relabelled field. Searching "vort psi" must still reach vort_psi_drag.

    Parametrized over the authored set itself rather than a fixed example --
    the existing search tests all use fields that will never be relabelled, so
    they stay green through exactly this regression.
    """
    state = panels.PanelState(show_advanced=True)
    for path, name, info in _labelled():
        state.search = derived_label(name)
        assert panels._leaf_visible(name, info, {}, state), (
            f"{path} is no longer findable by its derived caption {state.search!r}"
        )


def test_relabelled_fields_are_findable_by_the_new_caption_too(  # noqa: D103
):
    state = panels.PanelState(show_advanced=True)
    for path, name, info in _labelled():
        state.search = field_label(name, info)
        assert panels._leaf_visible(name, info, {}, state), f"{path} not findable by its caption"


def test_field_names_are_untouched():
    """Captions are display-only. A preset, the JSON schema and every
    cross-reference in a description still address fields by name."""
    for path, name, _info in _labelled():
        assert path.endswith(name)
        assert " " not in name
