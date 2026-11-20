"""
Parse variable-font filenames into semantic slots for NameID policy builders.

Supports static-aligned ({Family}{Optical?}-{Width?}Variable{Slope?}) and legacy
width-in-family ({Family}{Width?}{Optical?}-Variable{Slope?}) dialects.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
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
_RE_VF_ALIAS = re.compile(r"(?<![a-z])vf(?![a-z])|\b(?:gx|flex)\b", re.I)
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


def _split_subfamily_on_variable(subfamily_raw: str) -> tuple[str, str]:
    """Split subfamily raw string into (before Variable, after Variable)."""
    if not subfamily_raw:
        return "", ""

    s = subfamily_raw.strip()
    # Normalize aliases to Variable for splitting
    s = re.sub(r"(?i)\b(VF|GX|Flex)\b", "Variable", s)

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

    # Classify after-Variable tokens
    slope: Optional[str] = None
    bespoke: Optional[str] = None

    if after_raw.strip():
        after_formatted = format_pascal_words(after_raw) or after_raw
        after_words = after_formatted.split()
        if len(after_words) == 1:
            token = after_words[0]
            canon = _canonical_slope(token)
            if canon:
                slope = canon
            elif token.lower() in _ELIDABLE_AFTER_VARIABLE:
                # Regular/Normal: fully elided (not a pairing marker).
                # Upright/Roman are handled above via ELIDABLE_VF_FILENAME_SLOPES.
                slope = None
            elif _is_bespoke_suffix(token):
                bespoke = token
            else:
                bespoke = token
        else:
            # Multi-word bespoke or compound slope
            first_slope = _canonical_slope(after_words[0])
            if first_slope and len(after_words) == 1:
                slope = first_slope
            else:
                bespoke = after_formatted

    return VariableFilenameSlots(
        root_family=root_family,
        optical=optical,
        width=width,
        slope=slope,
        bespoke=bespoke,
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
    Build a static-aligned variable-font filename stem from slots.

    Examples:
        ReaderPro-CondensedVariable
        RoslindaleText-Variable
        FL_RareText-VariableItalic  (underscores preserved when present in root)
    """
    left_tokens: list[str] = []
    if slots.root_family:
        left_tokens.extend(slots.root_family.split())
    if slots.optical:
        left_tokens.append(slots.optical)
    left = _display_tokens_to_pascal(left_tokens)

    right_parts: list[str] = []
    if slots.width:
        right_parts.append(slots.width.replace(" ", ""))
    right_parts.append("Variable")
    if slots.bespoke:
        right_parts.append(slots.bespoke.replace(" ", ""))
    elif slots.slope:
        right_parts.append(slots.slope.replace(" ", ""))
    right = "".join(right_parts)
    return f"{left}-{right}"


def slots_usable_for_policy(slots: Optional[VariableFilenameSlots]) -> bool:
    """True when slots are safe to drive NameID policy builders."""
    if slots is None:
        return False
    if slots.dialect == VariableFilenameDialect.INVERTED:
        return False
    return slots.is_valid


__all__ = [
    "VariableFilenameDialect",
    "VariableFilenameSlots",
    "filename_has_variable_marker",
    "format_variable_filename",
    "parse_variable_filename",
    "parse_variable_filename_from_parsed",
    "slots_usable_for_policy",
    "variable_slots_from_path",
]
