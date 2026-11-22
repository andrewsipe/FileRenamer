"""One conflict model, shared by the renamer and the organizers.

When several fonts want the same name, two questions decide what happens:

1. Are they really the same font (a duplicate, an older copy, a WOFF made from a TTF)? `cluster_same_face` says so.
   Fonts that share a name but differ in weight, slant or size are not duplicates: the shared name is bad data, so
   nothing is ranked and nothing is renamed to it.
2. Which copy keeps the clean name? `rank` orders them:
     1. the higher head revision;
     2. the higher version string (nameID 5), for fonts whose revision ties;
     3. more coverage: glyphs, languages and OpenType features;
     4. the newer modification date, but only when the version is read and equal. A file date can be newer because a tool
        converted or re-saved an older version of the font (a WOFF made last week from an old TTF), so the date is never
        trusted on its own;
     5. the file name, so the result never depends on the order the files were found in.

The creation date is not used: a conversion tool resets it, so it says nothing about which design is newer.
Our own tools never update head.modified (they keep the font's dates), so a change one of them makes does not show up as a
newer date. That is intended: the tools can make the change again whenever it is needed.
Works with any metadata object that has the fields below; a missing field counts as "unknown", never as evidence.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence, Tuple

# The same weights the renamer's quality score used for these three parts.
WEIGHT_LANGUAGE = 2.5
WEIGHT_FEATURES = 2.0
WEIGHT_GLYPHS = 100

_VALUABLE_FEATURES = {"kern", "liga", "dlig", "smcp", "c2sc", "onum", "lnum", "tnum", "frac", "sups", "subs"}
_VERSION_RE = re.compile(r"(\d+(?:\.\d+)?)")
GLYPH_COUNT_TOLERANCE = 0.10


def parse_version(text: Optional[str]) -> Optional[float]:
    """The first number in a version string ("Version 2.001; hotconv" -> 2.001), or None."""
    if not text:
        return None
    match = _VERSION_RE.search(str(text))
    return float(match.group(1)) if match else None


def _get(meta: Any, name: str, default: Any = None) -> Any:
    value = getattr(meta, name, default)
    return default if value is None else value


def coverage_score(meta: Any, group: Sequence[Any]) -> float:
    """Glyphs, languages and features: the parts of the old quality score that say how much font there is."""
    languages = _get(meta, "language_support", set())
    if "cyrillic" in languages and "latin-extended" in languages:
        lang = 100
    elif "cyrillic" in languages or "greek" in languages:
        lang = 50
    elif "latin-extended" in languages:
        lang = 25
    elif "vietnamese" in languages:
        lang = 20
    else:
        lang = 0
    features = _get(meta, "opentype_features", set())
    feature_points = len(features & _VALUABLE_FEATURES) * 10 + len(features - _VALUABLE_FEATURES) * 2
    score = lang * WEIGHT_LANGUAGE + min(feature_points, 200) * WEIGHT_FEATURES
    # more glyphs than the smallest copy in the group, by at least 10%, earns up to WEIGHT_GLYPHS
    counts = [c for c in (_get(m, "glyph_count", 0) for m in group) if c > 0]
    own = _get(meta, "glyph_count", 0)
    if len(counts) > 1 and own > 0:
        smallest = min(counts)
        if own >= smallest * 1.10:
            score += min(((own / smallest) - 1.0) * WEIGHT_GLYPHS, WEIGHT_GLYPHS)
    return score


def _name(meta: Any) -> str:
    return str(_get(meta, "file_path", ""))


def _key(meta: Any, group: Sequence[Any]) -> Tuple:
    revision = _get(meta, "font_revision", None)
    version = parse_version(_get(meta, "version_string", ""))
    modified = _get(meta, "head_modified", None)
    return (
        (0, -revision) if revision else (1, 0),  # a known revision first, higher first
        (0, -version) if version is not None else (1, 0),
        -coverage_score(meta, group),
        # a date only counts when the version it belongs to was read
        -modified if (modified and version is not None) else 0,
        _name(meta),
    )


def rank(metas: Sequence[Any]) -> List[Any]:
    """Best copy first (see the module docstring for the order)."""
    group = list(metas)
    return sorted(group, key=lambda m: _key(m, group))


def explain_winner(winner: Any, runner_up: Any, group: Optional[Sequence[Any]] = None) -> str:
    """One sentence: which rule put `winner` ahead of `runner_up`."""
    group = list(group) if group is not None else [winner, runner_up]
    w_rev, r_rev = _get(winner, "font_revision", 0) or 0, _get(runner_up, "font_revision", 0) or 0
    if w_rev != r_rev:
        return f"higher revision ({w_rev:g} over {r_rev:g})"
    w_ver = parse_version(_get(winner, "version_string", ""))
    r_ver = parse_version(_get(runner_up, "version_string", ""))
    if w_ver != r_ver and (w_ver is not None or r_ver is not None):
        return f"higher version string ({_get(winner, 'version_string', '')!r} over {_get(runner_up, 'version_string', '')!r})"
    if coverage_score(winner, group) != coverage_score(runner_up, group):
        return "same version; more coverage (glyphs, languages, features)"
    w_mod, r_mod = _get(winner, "head_modified", 0), _get(runner_up, "head_modified", 0)
    if w_ver is not None and w_mod != r_mod:
        return "same version; newer modification date"
    return "no way to tell which is newer; kept the earlier file name"


def same_face(a: Any, b: Any) -> bool:
    """True when two fonts could be copies of one face: same weight and slant, glyph counts within 10%.

    Unknown weight or slant never counts against. A variable font may pair with a static of another weight
    (its default instance is not the weight of the static it sits next to).
    """
    either_variable = bool(_get(a, "is_variable", False)) or bool(_get(b, "is_variable", False))
    wa, wb = _get(a, "weight_class", None), _get(b, "weight_class", None)
    if not either_variable and wa is not None and wb is not None and wa != wb:
        return False
    ia, ib = _get(a, "is_italic", None), _get(b, "is_italic", None)
    if ia is not None and ib is not None and bool(ia) != bool(ib):
        return False
    ga, gb = _get(a, "glyph_count", 0), _get(b, "glyph_count", 0)
    if ga and gb and abs(ga - gb) > GLYPH_COUNT_TOLERANCE * max(ga, gb):
        return False
    return True


def cluster_same_face(metas: Sequence[Any]) -> List[List[Any]]:
    """Split a group into clusters of fonts that could be the same face. One cluster means real duplicates."""
    clusters: List[List[Any]] = []
    for meta in sorted(metas, key=_name):
        for cluster in clusters:
            if all(same_face(meta, other) for other in cluster):
                cluster.append(meta)
                break
        else:
            clusters.append([meta])
    return clusters
