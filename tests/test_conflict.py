"""One conflict model: which font keeps the clean name when several share it, and when they are not duplicates at all."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest

from fontrenamer.conflict import (
    cluster_same_face,
    explain_winner,
    parse_version,
    rank,
    same_face,
)


@dataclass
class M:
    """A stand-in for the metadata both tools hold."""

    file_path: str
    font_revision: float = 1.0
    version_string: str = ""
    glyph_count: int = 500
    head_modified: Optional[float] = None
    head_created: Optional[float] = None
    weight_class: Optional[int] = 400
    is_italic: Optional[bool] = False
    is_variable: bool = False
    language_support: set = field(default_factory=set)
    opentype_features: set = field(default_factory=set)


def names(ms):
    return [m.file_path for m in ms]


# ---- version strings ---------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("Version 2.001", 2.001),
        ("Version 001.000", 1.0),
        ("2.35;PS 2.35;hotconv 1.0.116", 2.35),
        ("Version 1.0", 1.0),
        ("v3", 3.0),
        ("", None),
        ("no digits here", None),
        (None, None),
    ],
)
def test_parse_version(text, expected):
    assert parse_version(text) == expected


# ---- ranking: revision, version, coverage, then a confirmed date -----------------------------------------

def test_the_higher_revision_wins_whatever_else():
    old = M("old.ttf", font_revision=1.0, glyph_count=900, head_modified=9e9)
    new = M("new.ttf", font_revision=2.0, glyph_count=400, head_modified=1.0)
    assert names(rank([old, new])) == ["new.ttf", "old.ttf"]


def test_equal_revision_the_higher_version_string_wins():
    a = M("a.ttf", version_string="Version 2.000")
    b = M("b.ttf", version_string="Version 2.001")
    assert names(rank([a, b])) == ["b.ttf", "a.ttf"]


def test_a_newer_modification_date_does_not_beat_a_higher_version():
    # a WOFF made last week from an older font: newer date, older version
    woff = M("converted.woff", version_string="Version 1.000", head_modified=9e9)
    ttf = M("original.ttf", version_string="Version 1.100", head_modified=1e9)
    assert names(rank([woff, ttf])) == ["original.ttf", "converted.woff"]


def test_same_version_more_coverage_beats_a_newer_date():
    small = M("small.ttf", version_string="Version 1.000", glyph_count=300, head_modified=9e9)
    big = M("big.ttf", version_string="Version 1.000", glyph_count=600, head_modified=1e9,
            opentype_features={"kern", "liga"}, language_support={"latin-extended"})
    assert names(rank([small, big])) == ["big.ttf", "small.ttf"]


def test_the_modification_date_decides_only_when_the_version_is_confirmed_equal():
    a = M("a.ttf", version_string="Version 1.000", head_modified=1e9)
    b = M("b.ttf", version_string="Version 1.000", head_modified=2e9)
    assert names(rank([a, b])) == ["b.ttf", "a.ttf"]


def test_without_a_readable_version_the_date_is_not_trusted():
    a = M("a.ttf", version_string="", head_modified=2e9)
    b = M("b.ttf", version_string="", head_modified=1e9)
    # the newer date is NOT used: the file name keeps the order stable and predictable
    assert names(rank([b, a])) == ["a.ttf", "b.ttf"]
    assert names(rank([a, b])) == ["a.ttf", "b.ttf"]


def test_the_order_does_not_depend_on_the_input_order():
    ms = [M(f"{c}.ttf", font_revision=1.0 + i / 10) for i, c in enumerate("dcba")]
    assert names(rank(ms)) == names(rank(list(reversed(ms))))


def test_explain_says_which_rule_decided():
    hi, lo = M("hi.ttf", font_revision=2.0), M("lo.ttf", font_revision=1.0)
    assert "revision" in explain_winner(*rank([lo, hi])[:2])
    a = M("a.ttf", version_string="Version 2.000"); b = M("b.ttf", version_string="Version 2.001")
    assert "version" in explain_winner(*rank([a, b])[:2])
    c = M("c.ttf", version_string="Version 1.0", glyph_count=300); d = M("d.ttf", version_string="Version 1.0", glyph_count=700)
    assert "coverage" in explain_winner(*rank([c, d])[:2])
    e = M("e.ttf", version_string="Version 1.0", head_modified=1e9); f = M("f.ttf", version_string="Version 1.0", head_modified=2e9)
    assert "modification date" in explain_winner(*rank([e, f])[:2])
    g, h = M("g.ttf"), M("h.ttf")
    assert "file name" in explain_winner(*rank([g, h])[:2])


# ---- the same-face check ----------------------------------------------------------------------------------

def test_same_weight_and_slant_is_the_same_face():
    assert same_face(M("a.ttf"), M("b.ttf", glyph_count=520))


@pytest.mark.parametrize(
    "other",
    [
        M("b.ttf", weight_class=700),
        M("b.ttf", is_italic=True),
        M("b.ttf", glyph_count=900),
    ],
)
def test_different_weight_slant_or_size_is_a_different_face(other):
    assert not same_face(M("a.ttf"), other)


def test_unknown_weight_or_slant_is_not_evidence_against():
    assert same_face(M("a.ttf"), M("b.ttf", weight_class=None, is_italic=None))


def test_a_variable_font_may_pair_with_a_static_of_a_different_weight():
    assert same_face(M("static.ttf", weight_class=700), M("vf.ttf", weight_class=400, is_variable=True))


def test_clustering_splits_a_group_that_is_not_duplicates():
    thin, bold, bold2 = M("thin.ttf", weight_class=100), M("bold.ttf", weight_class=700), M("bold2.ttf", weight_class=700)
    clusters = cluster_same_face([thin, bold, bold2])
    assert sorted(len(c) for c in clusters) == [1, 2]


def test_clustering_keeps_true_duplicates_together():
    assert len(cluster_same_face([M("a.ttf"), M("b.ttf"), M("c.ttf")])) == 1


def test_the_organizer_ranks_with_the_same_model():
    import fontrenamer.conflict as conflict
    import fontrenamer.organizer as organizer

    assert organizer.rank_duplicates is conflict.rank
    assert not hasattr(organizer, "sort_by_version_priority")
