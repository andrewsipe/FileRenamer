"""One decision, shared by every tool, on whether a font is Upright or Italic.

Each tool used to decide this on its own (FontFixer from the names, FileRenamer from the filename and tables, fontnameid from
the flags), so they disagreed and stale values fed each other. This module decides once and says why.

Two passes, read-only:

1. Upright is the safe assumption. Every font is grouped with the others that share its name once the slope words
   (Italic, Oblique, Slanted, Upright, Roman, Regular) are taken out, plus the same axes. A font that has no partner stays
   Upright, unless the tables positively say Italic (a STAT ``ital`` value of 1, or the filename, flags, names and the
   measured lean all agree).
2. A group of exactly two is a pair, and one of them is the Italic. STAT decides first, then the measured lean of the
   outlines (the one that leans more, in either direction, by at least ``LEAN_MARGIN`` degrees), then the filename, names
   and flags. When the evidence conflicts, or nothing separates the two, both stay Upright and the pair is flagged for a
   person to look at.

A font with a free ``ital`` or ``slnt`` axis holds both slopes and takes no part. The lean is measured from the outlines
(``core_slant_measure``), never read from ``post.italicAngle``, which is often wrong. Nothing here changes a font.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

LEAN_MARGIN = 2.5  # degrees: one font of a pair must lean this much more than the other for the lean to decide
LONE_ITALIC_LEAN = 3.0  # a lone font keeps Italic from agreeing evidence only when it leans at least this much
STORED_ITALIC_ANGLE = 4.9  # a stored angle leaning right by this much is a design value: outlines that read flat do not overrule it
GLYPH_COUNT_TOLERANCE = 0.25  # past this gap in glyph counts a pair is only 'medium' at best: italics are often subsets

_ITALIC_WORD = re.compile(r"italic|oblique|slanted|slant", re.I)
_UPRIGHT_WORD = re.compile(r"upright|roman", re.I)
_ALT_WORD = re.compile(r"cursive|corsiva|kursiv", re.I)  # a design word in the slope slot: an alternate, italic or upright
_SLOPE_WORDS = re.compile(r"italic|oblique|slanted|slant|upright|roman|regular|cursive|corsiva|kursiv", re.I)


@dataclass(frozen=True)
class FontEvidence:
    """What one font and its filename say about slope. ``stat`` is 'italic', 'upright' or None; ``filename`` also 'alt'."""

    path: str
    key: str
    stat: Optional[str] = None
    filename: Optional[str] = None
    names_italic: bool = False
    bits_italic: bool = False
    lean: Optional[float] = None  # degrees from vertical, absolute; None when the outlines cannot be measured
    declared: float = 0.0  # post.italicAngle as stored (information only)
    glyphs: int = 0
    free_slope: bool = False  # a free ital or slnt axis: both slopes in one file
    error: Optional[str] = None


@dataclass(frozen=True)
class SlopeDecision:
    path: str
    slope: str  # upright | italic | free | unknown
    confidence: str  # high | medium | low
    role: str  # lone | pair | group | free | unreadable
    reasons: tuple = ()
    review: bool = False
    partner: Optional[str] = None
    differs: tuple = ()  # which of filename / bits / names disagree with the decision


def _slot(stem: str) -> tuple[str, str]:
    """Split a filename stem into (family part, style slot): the text after Variable, else after the last hyphen."""
    m = list(re.finditer("variable", stem, re.I))
    if m:
        i = m[-1].end()
        return stem[:i], stem[i:]
    if "-" in stem:
        head, _, tail = stem.rpartition("-")
        return head + "-", tail
    return stem, ""


def filename_slope(path: str, name: Optional[str] = None) -> Optional[str]:
    _, slot = _slot(Path(name or path).stem)
    if _ITALIC_WORD.search(slot):
        return "italic"
    if _UPRIGHT_WORD.search(slot):
        return "upright"
    if _ALT_WORD.search(slot):
        return "alt"  # neutral: the outlines say whether this alternate is an italic or an upright one
    return None


def family_key(path: str, axes: tuple = (), name: Optional[str] = None) -> str:
    """The folder, the filename with its slope words removed, and the non-slope axes. ``name`` is the filename to read when
    the file currently sits under another (a temporary name during a rename)."""
    stem = re.sub(r"~\d+$", "", Path(name or path).stem)  # a ~001 suffix from a rename clash is not part of the name
    head, slot = _slot(stem)
    squashed = re.sub(r"[^a-z0-9]", "", (head + _SLOPE_WORDS.sub("", slot)).lower())
    return f"{Path(path).parent}|{squashed}|{axes}"


def _lean_of(font: Any) -> Optional[float]:
    from FontCore.core_slant_measure import check_slant, shape_slant

    check = check_slant(font)
    if check.measured is not None and check.verdict != "unmeasurable":
        return abs(check.measured)
    shape = shape_slant(font)
    return abs(shape) if shape is not None else None


def gather_evidence(path: str, name: Optional[str] = None) -> FontEvidence:
    """Read one font. A font that cannot be read comes back with ``error`` set. ``name`` is the filename it should be judged
    by when the file sits under another name for now (a rename's temporary name)."""
    from fontTools.ttLib import TTFont

    from FontCore.core_variable_traits import _stat_vote, analyze_variable_traits

    try:
        font = TTFont(path, lazy=True, fontNumber=0, recalcTimestamp=False)
        stem = Path(name or path).stem
        stat = None
        free = False
        axes: tuple = ()
        if "fvar" in font:
            traits = analyze_variable_traits(font, stem)
            free = traits.slope == "free"
            stat = traits.slope_votes.get("stat")
            axes = tuple(sorted((a.axisTag, round(a.minValue), round(a.maxValue)) for a in font["fvar"].axes
                                if a.axisTag not in ("ital", "slnt")))
        else:
            stat = _stat_vote(font)
        name_table = font["name"]
        names = " ".join(filter(None, (name_table.getDebugName(i) for i in (2, 4, 6, 17))))
        os2 = font["OS/2"].fsSelection if "OS/2" in font else 0
        mac = font["head"].macStyle if "head" in font else 0
        return FontEvidence(
            path=path,
            key=family_key(path, axes, name),
            stat=stat,
            filename=filename_slope(path, name),
            names_italic=bool(_ITALIC_WORD.search(names)),
            bits_italic=bool(os2 & 1 or os2 & (1 << 9) or mac & 2),
            lean=None if free else _lean_of(font),
            declared=round(float(font["post"].italicAngle), 2) if "post" in font else 0.0,
            glyphs=font["maxp"].numGlyphs if "maxp" in font else 0,
            free_slope=free,
        )
    except Exception as exc:  # unreadable or damaged: say so, go on
        return FontEvidence(path=path, key=family_key(path, (), name), error=f"{type(exc).__name__}: {exc}")


def _score(e: FontEvidence) -> int:
    """Italic-ward evidence from the tables and names, without the lean. Names and flags are weak: they go stale."""
    s = 0
    s += {"italic": 3, "upright": -3}.get(e.stat or "", 0)
    s += {"italic": 2, "upright": -2}.get(e.filename or "", 0)
    s += 1 if e.names_italic else 0
    s += 1 if e.bits_italic else 0
    return s


def _claims(e: FontEvidence) -> dict:
    return {"filename": e.filename == "italic", "bits": e.bits_italic, "names": e.names_italic}


def _differs(e: FontEvidence, slope: str) -> tuple:
    want = slope == "italic"
    out = []
    for field_name, says_italic in _claims(e).items():
        if field_name == "filename" and e.filename is None:
            continue  # the filename states nothing; it has nothing to contradict
        if says_italic != want:
            out.append(field_name)
    return tuple(out)


def _decision(e: FontEvidence, slope: str, confidence: str, role: str, reasons, review=False, partner=None) -> SlopeDecision:
    return SlopeDecision(e.path, slope, confidence, role, tuple(reasons), review, partner, _differs(e, slope))


def _lone(e: FontEvidence) -> SlopeDecision:
    claims = [k for k, v in _claims(e).items() if v]
    flat = e.lean is not None and e.lean < 1.0
    if e.stat == "italic" and e.filename == "upright" and flat:
        return _decision(e, "upright", "medium", "lone",
                         ["the filename says Upright and the outlines do not lean; STAT says italic, which nothing else supports"])
    if e.stat == "italic":
        return _decision(e, "italic", "high", "lone", ["no partner, but STAT says italic"])
    if len(claims) == 3 and (e.lean or 0) >= LONE_ITALIC_LEAN:
        return _decision(e, "italic", "medium", "lone",
                         [f"no partner; the filename, flags and names say italic and the outlines lean {e.lean:g} degrees"])
    leans = (e.lean or 0) >= LONE_ITALIC_LEAN
    if leans and ((e.filename == "italic" and (e.bits_italic or e.names_italic) and e.stat != "upright")
                  or (e.filename == "alt" and (e.bits_italic or e.names_italic) and e.stat != "upright")
                  or (e.filename == "alt" and e.bits_italic and e.names_italic)):  # an alternate's flags and lean outvote STAT
        return _decision(e, "italic", "medium", "lone",
                         [f"no partner here; its filename word, flags or names and its {e.lean:g} degree lean agree on italic"])
    reasons = ["no partner: Upright is the safe default"]
    review = False
    if claims:
        reasons.append("its " + ", ".join(claims) + " say italic, which nothing else supports")
        # Acted on when the outlines lean less than an italic would (under LONE_ITALIC_LEAN degrees) and STAT does not say
        # italic. Outlines that lean more, or that cannot be measured, leave the tool's own rules in charge, flagged.
        stored_lean = e.declared <= -STORED_ITALIC_ANGLE  # italicAngle is negative for a right-leaning italic
        settled = e.lean is not None and e.lean < LONE_ITALIC_LEAN and e.stat != "italic" and not stored_lean
        review = not settled
        if stored_lean and e.lean is not None and e.lean < LONE_ITALIC_LEAN:
            reasons.append(f"but the font stores an angle of {e.declared:g} degrees and its outlines measure only {e.lean:g}: "
                           "the designer may have meant a lean the straight capitals do not show")
        elif settled and not flat:
            reasons.append(f"the outlines lean only {e.lean:g} degrees: a script or a pattern, not an italic")
        elif review:
            reasons.append(f"the outlines lean {e.lean:g} degrees: Upright with a slant, or an Italic without its Upright?"
                           if e.lean is not None else "the outlines cannot be measured, so the italic claims are not overruled")
    return _decision(e, "upright", "high" if (not claims or not review) else "medium", "lone", reasons, review)


def _word(e: FontEvidence) -> str:
    """The slope word as written in the filename (Italic, Oblique, Slanted, Cursive), lower case."""
    _, slot = _slot(Path(e.path).stem)
    m = _ITALIC_WORD.search(slot) or _ALT_WORD.search(slot)
    return m.group(0).lower() if m else ""


def _pair(a: FontEvidence, b: FontEvidence) -> tuple[SlopeDecision, SlopeDecision]:
    if (not (a.stat and b.stat and a.stat != b.stat) and a.lean is not None and b.lean is not None
            and min(a.lean, b.lean) >= LONE_ITALIC_LEAN and abs(a.lean - b.lean) < LEAN_MARGIN):
        both = f"both lean ({a.lean:g} and {b.lean:g} degrees)"
        if _word(a) and _word(b) and _word(a) != _word(b):
            why = [f"{both} and carry different slope words: two slanted cuts of one family"]
            return (_decision(a, "italic", "medium", "pair", why, False, b.path), _decision(b, "italic", "medium", "pair", why, False, a.path))
        why = [f"{both}: two slanted files share one name, not an Upright and an Italic"]
        return (_decision(a, "italic", "low", "pair", why, True, b.path), _decision(b, "italic", "low", "pair", why, True, a.path))

    explicit = _score(a) - _score(b)
    lean = (a.lean - b.lean) if (a.lean is not None and b.lean is not None) else None
    italic: Optional[FontEvidence] = None
    confidence, review, reasons = "low", False, []

    stat_split = a.stat and b.stat and a.stat != b.stat
    lean_says = None
    if lean is not None and abs(lean) >= LEAN_MARGIN:
        lean_says = a if lean > 0 else b
    explicit_says = a if explicit >= 2 else b if explicit <= -2 else None

    if stat_split:
        italic = a if a.stat == "italic" else b
        confidence = "high"
        reasons.append("STAT marks it italic and its partner upright")
        if lean_says is not None and lean_says is not italic:
            review = True
            reasons.append("but the other file leans more: check the pair")
    elif lean_says is not None:
        italic = lean_says
        other = b if italic is a else a
        reasons.append(f"leans {italic.lean:g} degrees against {other.lean:g} for its partner")
        if explicit_says is italic:
            confidence = "high"
            reasons.append("the filename, names and flags agree")
        elif explicit_says is not None:
            confidence = "low"
            review = True
            reasons.append("but the filename, names and flags say the other one: check the pair")
        else:
            confidence = "medium"
    elif explicit_says is not None:
        italic = explicit_says
        confidence = "medium"
        reasons.append("the filename, names and flags mark it italic; the lean does not separate the pair")
    else:
        review = True
        reasons.append("nothing separates the two: both stay Upright")
        reasons.append(f"leans {a.lean} and {b.lean} degrees" if lean is not None else "the outlines could not be compared")

    if italic is None:
        return (_decision(a, "upright", "low", "pair", reasons, True, b.path), _decision(b, "upright", "low", "pair", reasons, True, a.path))
    if a.glyphs and b.glyphs and abs(a.glyphs - b.glyphs) / max(a.glyphs, b.glyphs) > GLYPH_COUNT_TOLERANCE:
        reasons.append(f"the glyph counts differ ({a.glyphs} and {b.glyphs}); an italic is often a subset")
        if confidence == "high":
            confidence = "medium"
    other = b if italic is a else a
    return (_decision(italic, "italic", confidence, "pair", reasons, review, other.path),
            _decision(other, "upright", confidence, "pair", [f"partner {Path(italic.path).name} is the italic"], review, italic.path))


def _group(members: list) -> list:
    """Three or more files with one name: the Upright is the baseline, and each other file is Italic when it leans clearly
    more than that baseline (Control's Upright, Italic and Cursive; ElmaTrio's Upright, Italic and Slanted)."""
    names = ", ".join(Path(m.path).name for m in members[:4]) + (" ..." if len(members) > 4 else "")

    def flagged(why):
        return [_decision(m, "upright", "low", "group", [why], True) for m in members]

    if any(m.lean is None for m in members):
        return flagged(f"{len(members)} files share this name ({names}) and one cannot be measured")
    baseline = [m for m in members if m.filename == "upright"]
    if len(baseline) != 1:
        return flagged(f"{len(members)} files share this name ({names}) and none or several are marked Upright")
    base = baseline[0]
    out = [_decision(base, "upright", "medium", "group", [f"the Upright of {len(members)} files with one name"], False)]
    for m in members:
        if m is base:
            continue
        gap = m.lean - base.lean
        if gap >= LEAN_MARGIN:
            out.append(_decision(m, "italic", "medium", "group",
                                 [f"leans {m.lean:g} degrees against {base.lean:g} for the Upright of {len(members)} files"], False,
                                 base.path))
        elif m.filename == "italic" or m.stat == "italic":
            out.append(_decision(m, "upright", "low", "group",
                                 [f"named or marked italic but leans only {gap:g} degrees more than the Upright"], True, base.path))
        else:
            out.append(_decision(m, "upright", "medium", "group",
                                 [f"leans {m.lean:g} degrees, like the Upright: an upright alternate"], False, base.path))
    return out


def load_overrides(path: Optional[str] = None) -> list:
    """(pattern, slope) pairs from the overrides file: one ``PATTERN upright|italic`` per line, ``#`` for comments, the
    pattern matched against the filename, case-insensitively. For fonts that only a person can know (a bitmap font whose
    outlines look like nothing). The file is ``$FONTCORE_SLOPE_OVERRIDES`` or ``~/.config/fontcore/slope_overrides.txt``."""
    import os

    path = path or os.environ.get("FONTCORE_SLOPE_OVERRIDES") or os.path.expanduser("~/.config/fontcore/slope_overrides.txt")
    out: list = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.split("#", 1)[0].strip()
                parts = line.rsplit(None, 1)
                if len(parts) == 2 and parts[1].lower() in ("upright", "italic"):
                    out.append((parts[0], parts[1].lower()))
    except OSError:
        pass
    return out


def _apply_overrides(decisions: list, evidence: dict, overrides: list) -> list:
    import fnmatch

    if not overrides:
        return decisions
    done = []
    for d in decisions:
        name = Path(d.path).name.lower()
        hit = next((slope for pattern, slope in overrides if fnmatch.fnmatch(name, pattern.lower())), None)
        if hit is None or d.slope in ("free", "unknown"):
            done.append(d)
            continue
        e = evidence.get(d.path)
        done.append(SlopeDecision(d.path, hit, "high", d.role, ("set in your slope_overrides.txt",), False, d.partner,
                                  _differs(e, hit) if e else ()))
    return done


def resolve_slopes(evidence: Iterable[FontEvidence], overrides: Optional[list] = None) -> list[SlopeDecision]:
    """Decide every font. Pure: it reads only the evidence records, so it can be tested without fonts."""
    evidence = list(evidence)
    by_path = {e.path: e for e in evidence}
    out: dict[str, SlopeDecision] = {}
    groups: dict[str, list[FontEvidence]] = {}
    for e in evidence:
        if e.error:
            out[e.path] = SlopeDecision(e.path, "unknown", "low", "unreadable", (e.error,), True)
        elif e.free_slope:
            out[e.path] = SlopeDecision(e.path, "free", "high", "free", ("a free slope axis holds both slopes",))
        else:
            groups.setdefault(e.key, []).append(e)
    for members in groups.values():
        if len(members) == 1:
            d = _lone(members[0])
            out[d.path] = d
        elif len(members) == 2:
            for d in _pair(*members):
                out[d.path] = d
        else:
            for d in _group(members):
                out[d.path] = d
    return _apply_overrides(list(out.values()), by_path, load_overrides() if overrides is None else overrides)


def clear_decisions(decisions: Iterable[SlopeDecision]) -> dict:
    """path -> "italic" | "upright" for the decisions a tool may act on: not flagged, and not a free slope axis."""
    import os

    return {os.path.abspath(d.path): d.slope for d in decisions if not d.review and d.slope in ("upright", "italic")}


def decide_slopes(paths: Iterable[str]) -> list[SlopeDecision]:
    """Gather the evidence for these fonts and decide them together (one folder at a time keeps pairs together)."""
    return resolve_slopes([gather_evidence(str(p)) for p in paths])


__all__ = ["load_overrides", "clear_decisions", "decide_slopes", "FontEvidence", "SlopeDecision", "gather_evidence", "resolve_slopes", "family_key", "filename_slope"]
