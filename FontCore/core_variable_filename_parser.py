"""
Parse variable-font filenames into semantic slots for NameID policy builders.

Supports static-aligned ({Family}{Optical?}-{Width?}Variable{Slope?}) and legacy
width-in-family ({Family}{Width?}{Optical?}-Variable{Slope?}) dialects.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import List, Optional

from FontCore.core_filename_parts_parser import ParsedName, format_pascal_words, parse_filename
from FontCore.core_font_style_dictionaries import (
    ALL_OPTICAL_TERMS,
    ALL_WIDTH_TERMS,
    KNOWN_VF_SLOPES,
    ELIDABLE_VF_FILENAME_SLOPES,
    STYLE_WORDS,
)

_RE_VF_IN_STRING = re.compile(r"(?i)variable")
_RE_VF_ALIAS = re.compile(r"(?<![a-z])vf(?![a-z])|\b(?:gx|flex|var)\b", re.I)
_RE_INVERTED_FAMILY = re.compile(r"(?i)variable")

# Weight terms in subfamily before Variable indicate static-style misuse
_WEIGHT_TERMS = frozenset(
    w
    for w in STYLE_WORDS
    if w
    in {
        "Hairline",
        "Thin",
        "Extralight",
        "Ultralight",
        "Light",
        "Semilight",
        "Book",
        "Regular",
        "Normal",
        "Roman",
        "Medium",
        "Demibold",
        "Semibold",
        "Bold",
        "Extrabold",
        "Ultrabold",
        "Black",
        "Heavy",
        "Extrablack",
        "Ultrablack",
        "Fat",
        "100",
        "200",
        "300",
        "400",
        "500",
        "600",
        "700",
        "800",
        "900",
        "1000",
    }
)

_SLOPE_CANONICAL = {
    "italic": "Italic",
    "oblique": "Oblique",
    "slanted": "Slanted",
    "slant": "Slanted",
    "inclined": "Inclined",
    "upright": "Upright",
}

_ELIDABLE_AFTER_VARIABLE = frozenset({"regular", "roman", "normal", "upright"})


class VariableFilenameDialect(str, Enum):
    STATIC_ALIGNED = "static_aligned"
    LEGACY_WIDTH_FAMILY = "legacy_width_family"
    INVERTED = "inverted"


@dataclass
class VariableFilenameSlots:
    """Semantic slots extracted from a variable-font filename."""

    root_family: str
    optical: Optional[str] = None
    width: Optional[str] = None
    slope: Optional[str] = None
    bespoke: Optional[str] = None
    # Area 5: ``axes`` is the extras on top of weight and a free slope: "Width", "OpticalSize", "Custom" or a run of them
    # ("WidthOpticalSizeCustom"); None when there are none. ``weight`` is a locked weight word. A bare Variable is a free
    # slope (``slope_free``); ``slope`` is a locked slope word; ``upright`` is a locked upright. ``collection`` puts
    # Collection in the family name (all four registered axes free).
    axes: Optional[str] = None
    weight: Optional[str] = None
    upright: Optional[bool] = None
    slope_free: Optional[bool] = None
    collection: bool = False
    # The family part of the filename as written (underscores and all), so a rename changes only what the
    # convention needs; ``width_in_family`` says whether it already carries the locked width.
    family_text: Optional[str] = None
    width_in_family: bool = False
    dialect: VariableFilenameDialect = VariableFilenameDialect.STATIC_ALIGNED
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any("invalid" in w.lower() for w in self.warnings)


def _sorted_terms_by_length(terms: set[str]) -> list[str]:
    return sorted(terms, key=lambda t: (-len(t), t.lower()))


def _peel_trailing_term(raw: str, terms: set[str]) -> tuple[str, Optional[str]]:
    """Remove the longest matching trailing term from PascalCase raw string."""
    for term in _sorted_terms_by_length(terms):
        if raw.endswith(term) and len(raw) > len(term):
            return raw[: -len(term)], term
        # Case-insensitive fallback
        if raw.lower().endswith(term.lower()) and len(raw) > len(term):
            return raw[: -len(term)], raw[-len(term) :]
    return raw, None


def _canonical_slope(token: str) -> Optional[str]:
    key = token.strip().lower()
    if key in _SLOPE_CANONICAL:
        return _SLOPE_CANONICAL[key]
    if key in {s.lower() for s in KNOWN_VF_SLOPES}:
        return token.strip().title()
    for slope in ELIDABLE_VF_FILENAME_SLOPES:
        if slope.lower() == key:
            return slope
    return None


def _is_bespoke_suffix(token: str) -> bool:
    t = token.strip()
    if not t:
        return False
    if _canonical_slope(t):
        return False
    if t.lower() in _ELIDABLE_AFTER_VARIABLE:
        return False
    if t in _WEIGHT_TERMS or t.lower() in {w.lower() for w in _WEIGHT_TERMS}:
        return False
    if t in ALL_WIDTH_TERMS or t.lower() in {w.lower() for w in ALL_WIDTH_TERMS}:
        return False
    return True


def _contains_vf_token(text: str) -> bool:
    """True when text contains a variable-font marker (including glued forms)."""
    if not text:
        return False
    return bool(_RE_VF_IN_STRING.search(text) or _RE_VF_ALIAS.search(text))


_RE_AXIS_COUNT = re.compile(r"(\d+)ax(?:is|es)", re.I)  # old count words (3Axis, 4Axes): read and dropped
_AXES_WORDS = {"opticalsize": "OpticalSize", "width": "Width", "custom": "Custom"}
_AXES_ORDER = ("Width", "OpticalSize", "Custom")
_LEGACY_DROPPED = {"slope"}  # the old word for a free slope: read and dropped
_UPRIGHT_WORDS = {"upright", "roman"}
_LEGACY_ELIDED_WEIGHTS = {"regular", "normal"}


def _weight_words() -> dict[str, str]:
    return {w.lower(): w for w in _WEIGHT_TERMS if w.lower() not in _UPRIGHT_WORDS}


def _slope_words() -> dict[str, str]:
    words = {k: v for k, v in _SLOPE_CANONICAL.items() if k not in _UPRIGHT_WORDS}
    for slope in KNOWN_VF_SLOPES:
        words.setdefault(slope.lower(), slope)
    for word in ("slant", "backslant", "reclined", "reverse", "cursive", "kursiv", "rotalic", "tiltalic", "corsiva"):  # as the designer spells them
        words[word] = word.capitalize()
    return words


def _classify_after_variable(after_raw: str):
    """Split the text after ``Variable`` into (axes, weight, slope, upright, slope_free, bespoke)."""
    text = re.sub(r"[\s_-]+", "", after_raw or "")
    axes_found: set = set()
    weight = slope = bespoke = None
    upright = False
    weights, slopes = _weight_words(), _slope_words()
    words = sorted({*weights, *slopes, *_UPRIGHT_WORDS, *_AXES_WORDS, *_LEGACY_DROPPED}, key=lambda w: (-len(w), w))
    leftovers: list[str] = []
    i = 0
    while i < len(text):
        rest = text[i:]
        m = _RE_AXIS_COUNT.match(rest)
        if m:
            i += m.end()
            continue
        low = rest.lower()
        hit = next((w for w in words if low.startswith(w)), None)
        if hit is None:
            leftovers.append(rest[0])
            i += 1
            continue
        if hit in _AXES_WORDS:
            axes_found.add(_AXES_WORDS[hit])
        elif hit in _LEGACY_DROPPED:
            pass
        elif hit in _UPRIGHT_WORDS:
            upright = True
        elif hit in slopes:
            slope = slopes[hit]
        else:
            weight = weights[hit]
        i += len(hit)
    if leftovers:
        bespoke = "".join(leftovers)
    axes = "".join(a for a in _AXES_ORDER if a in axes_found) or None
    if weight and weight.lower() in _LEGACY_ELIDED_WEIGHTS and axes is None:
        weight = None  # the old pairing marker: Probe-VariableRegular was just Probe-Variable
    if slope:
        upright = False
    return axes, weight, slope, (True if upright else None), not (slope or upright), bespoke

def _split_subfamily_on_variable(subfamily_raw: str) -> tuple[str, str]:
    """Split subfamily raw string into (before Variable, after Variable)."""
    if not subfamily_raw:
        return "", ""

    s = subfamily_raw.strip()
    # Normalize aliases to Variable for splitting
    s = re.sub(r"(?i)\b(VF|GX|Flex|Var)\b", "Variable", s)

    # Ritmica-Upright-Variable → subfamily_raw may be "Upright-Variable"
    s = re.sub(r"(?i)-", "", s)

    match = re.search(r"(?i)Variable", s)
    if not match:
        return s, ""

    before = s[: match.start()]
    after = s[match.end() :]
    return before, after


def _format_peeled_term(term: Optional[str]) -> Optional[str]:
    if not term:
        return None
    return format_pascal_words(term) or None


def _parse_inverted(parsed: ParsedName) -> Optional[VariableFilenameSlots]:
    """Vendor pattern: FamilyVariable-StaticSubfamily (e.g. BrisbaneVariable-Regular)."""
    family_raw = parsed.family_raw or ""
    if not _RE_INVERTED_FAMILY.search(family_raw):
        return None
    if _contains_vf_token(parsed.subfamily_raw or ""):
        return None

    # Peel Variable from family stem
    root_raw = re.sub(r"(?i)variable", "", family_raw).strip("_-")
    root_family = format_pascal_words(root_raw) or family_raw
    warnings = [
        "inverted dialect: Variable token in family stem; manual review recommended"
    ]
    return VariableFilenameSlots(
        root_family=root_family,
        dialect=VariableFilenameDialect.INVERTED,
        warnings=warnings,
    )


def parse_variable_filename(
    input_name: str, *, strip_extension: bool = True
) -> Optional[VariableFilenameSlots]:
    """
    Parse a variable-font filename into VariableFilenameSlots.

    Returns None if the filename does not appear to be a variable font.
    """
    parsed = parse_filename(input_name, strip_extension=strip_extension)
    subfamily_raw = parsed.subfamily_raw or ""
    family_raw = parsed.family_raw or ""

    has_vf_in_subfamily = _contains_vf_token(subfamily_raw)
    has_inverted = bool(
        _RE_INVERTED_FAMILY.search(family_raw) and not has_vf_in_subfamily
    )

    if not has_vf_in_subfamily and not has_inverted:
        return None

    if has_inverted:
        return _parse_inverted(parsed)

    warnings: list[str] = []
    dialect = VariableFilenameDialect.STATIC_ALIGNED

    # Split subfamily on Variable anchor
    before_raw, after_raw = _split_subfamily_on_variable(subfamily_raw)
    subfamily_width = _format_peeled_term(before_raw) if before_raw.strip() else None

    if subfamily_width:
        # Check for weight misuse (AugureStereo-BoldVariable)
        width_words = subfamily_width.split()
        for w in width_words:
            if w in _WEIGHT_TERMS or w.lower() in {x.lower() for x in _WEIGHT_TERMS}:
                warnings.append(
                    f"invalid: weight term '{w}' before Variable in subfamily"
                )
                subfamily_width = None
                break

    # Peel width then optical from family stem (end-first: width closer to hyphen)
    stem = family_raw
    peeled_width: Optional[str] = None
    peeled_optical: Optional[str] = None

    if not subfamily_width:
        stem, peeled_width = _peel_trailing_term(stem, ALL_WIDTH_TERMS)
        if peeled_width:
            dialect = VariableFilenameDialect.LEGACY_WIDTH_FAMILY

    stem, peeled_optical = _peel_trailing_term(stem, ALL_OPTICAL_TERMS)

    root_family = format_pascal_words(stem) or parsed.family
    optical = _format_peeled_term(peeled_optical)
    width = subfamily_width or _format_peeled_term(peeled_width)

    # Classify after-Variable tokens: {Axes}{Weight}{Slope}, anything else is bespoke
    axes, weight, slope, upright, slope_free, bespoke = _classify_after_variable(after_raw)

    return VariableFilenameSlots(
        root_family=root_family,
        optical=optical,
        width=width,
        slope=slope,
        bespoke=bespoke,
        axes=axes,
        weight=weight,
        upright=upright,
        slope_free=slope_free,
        family_text=family_raw or None,
        width_in_family=peeled_width is not None,
        dialect=dialect,
        warnings=warnings,
    )


def parse_variable_filename_from_parsed(
    parsed: ParsedName,
) -> Optional[VariableFilenameSlots]:
    """Parse from an existing ParsedName (basename should already be normalized)."""
    subfamily_raw = parsed.subfamily_raw or ""
    family_raw = parsed.family_raw or ""

    has_vf_in_subfamily = _contains_vf_token(subfamily_raw)
    has_inverted = bool(
        _RE_INVERTED_FAMILY.search(family_raw) and not has_vf_in_subfamily
    )

    if not has_vf_in_subfamily and not has_inverted:
        return None

    # Reuse main logic by reconstructing a synthetic basename
    base = parsed.base or f"{family_raw}-{subfamily_raw}" if subfamily_raw else family_raw
    return parse_variable_filename(base, strip_extension=False)


def filename_has_variable_marker(stem: str) -> bool:
    """True when a filename stem already contains a variable-font marker."""
    if not stem or not str(stem).strip():
        return False
    s = str(stem).strip()
    if _contains_vf_token(s):
        return True
    # Hyphenated stems: check each segment (e.g. Ritmica-Upright-Variable)
    if "-" in s:
        left, right = s.split("-", 1)
        return _contains_vf_token(left) or _contains_vf_token(right)
    return False


def variable_slots_from_path(filepath: str) -> Optional[VariableFilenameSlots]:
    """Parse variable-font slots from a file path basename."""
    basename = os.path.basename(filepath)
    return parse_variable_filename(basename, strip_extension=True)


def _display_tokens_to_pascal(tokens: List[str]) -> str:
    return "".join(t.replace(" ", "") for t in tokens if t and t.strip())


def format_variable_filename(slots: VariableFilenameSlots) -> str:
    """
    Build a variable-font filename stem from slots: ``{Family}{Optical}{Width}-Variable{Axes}{Weight}{Slope}`` (no slope word = free).

    Examples:
        Probe-Variable
        ProbeDisplayCondensed-Variable3AxisItalic
        RobotoFlex-Variable13Axis
    """
    if slots.family_text:
        left = slots.family_text.replace(" ", "")
        if slots.width and not slots.width_in_family:
            left += slots.width.replace(" ", "")
    else:
        left_tokens: list[str] = []
        if slots.root_family:
            left_tokens.extend(slots.root_family.split())
        if slots.optical:
            left_tokens.append(slots.optical)
        if slots.width:
            left_tokens.append(slots.width)
        left = _display_tokens_to_pascal(left_tokens)
    if slots.bespoke and not left.endswith(slots.bespoke.replace(" ", "")):
        left += slots.bespoke.replace(" ", "")  # a design variant (Alt, Cut, Swash) belongs to the family
    if slots.collection and not left.endswith("Collection"):
        left += "Collection"

    right = "Variable" + (slots.axes or "") + (slots.weight or "")
    if slots.slope:
        right += slots.slope.replace(" ", "")
    elif slots.upright:
        right += "Upright"
    return f"{left}-{right}"


def slots_from_plain_filename(input_name: str, *, strip_extension: bool = True) -> VariableFilenameSlots:
    """Slots for a variable font whose filename has no variable marker (``Probe-Regular.ttf``).

    Only the family, locked optical size and locked width are read; what varies comes from the font's tables.
    """
    parsed = parse_filename(input_name, strip_extension=strip_extension)
    stem = parsed.family_raw or ""
    stem, peeled_width = _peel_trailing_term(stem, ALL_WIDTH_TERMS)
    stem, peeled_optical = _peel_trailing_term(stem, ALL_OPTICAL_TERMS)
    return VariableFilenameSlots(
        root_family=format_pascal_words(stem) or parsed.family,
        optical=_format_peeled_term(peeled_optical),
        width=_format_peeled_term(peeled_width),
        family_text=parsed.family_raw or None,
        width_in_family=peeled_width is not None,
    )


def apply_variable_traits(slots: VariableFilenameSlots, traits) -> VariableFilenameSlots:
    """Let the font's tables decide the extra axes and the slope (DECISIONS.md, Area 5); the filename keeps the design."""
    warnings = list(slots.warnings) + list(traits.warnings)
    if traits.weight_free:
        if slots.weight:
            warnings.append(f"weight word {slots.weight!r} ignored: the font has a free weight axis")
        weight = None
    else:
        weight = slots.weight  # only a weight the filename states; never derived from usWeightClass
    if traits.slope == "italic":
        slope, upright, slope_free = traits.slope_word, None, False
    elif traits.slope == "free":
        slope, upright, slope_free = None, None, True
    else:
        slope, upright, slope_free = None, True, False
    return replace(
        slots, axes=traits.axes_word, weight=weight, slope=slope, upright=upright, slope_free=slope_free,
        collection=traits.collection, bespoke=slots.bespoke, warnings=warnings,
    )


def slots_usable_for_policy(slots: Optional[VariableFilenameSlots]) -> bool:
    """True when slots are safe to drive NameID policy builders."""
    if slots is None:
        return False
    if slots.dialect == VariableFilenameDialect.INVERTED:
        return False
    return slots.is_valid


def resolve_variable_slots(filepath: str) -> Optional[VariableFilenameSlots]:
    """Variable-font slots for a font file, or None for a static font.

    The font's tables decide whether it is variable, which axes are free and the slope (DECISIONS.md, Area 5); the
    filename supplies the family and the locked optical size, width and weight. A file that cannot be opened falls
    back to its filename alone.
    """
    from FontCore.core_variable_traits import analyze_variable_traits

    basename = os.path.basename(filepath)
    traits = None
    try:
        from fontTools.ttLib import TTFont

        font = TTFont(filepath, lazy=True, fontNumber=0, recalcTimestamp=False)
        try:
            traits = analyze_variable_traits(font, filename=basename)
        finally:
            font.close()
    except Exception:
        traits = None

    slots = variable_slots_from_path(filepath)
    if traits is None:
        return slots if slots_usable_for_policy(slots) else None
    if not traits.is_variable:
        return None
    if slots is None:
        slots = slots_from_plain_filename(basename)
    if not slots_usable_for_policy(slots):
        return None
    return apply_variable_traits(slots, traits)


__all__ = [
    "resolve_variable_slots",
    "apply_variable_traits",
    "slots_from_plain_filename",
    "VariableFilenameDialect",
    "VariableFilenameSlots",
    "filename_has_variable_marker",
    "format_variable_filename",
    "parse_variable_filename",
    "parse_variable_filename_from_parsed",
    "slots_usable_for_policy",
    "variable_slots_from_path",
]
