"""
Centralized ID policy helpers for constructing NameID values.

Scope (display-layer policy only):
- ID1 (Family), ID4 (Full), ID16 (Typographic Family), ID17 (Typographic Subfamily)
- Handles Variable naming overrides
- Applies style/slope normalization for ID1/ID4 using NameSubfamilyPolicies

Variable font naming (tier model):
    Superfamily / optical / width / slope / bespoke product files share one typographic
    family (ID16 = ``{root} Variable``). Filename slots come from
    ``parse_variable_filename()`` (static-aligned and legacy width-in-family dialects).

    Per-ID rules from slots:
    - ID1: root + optical + width (no Variable, no elided slopes)
    - ID4: ID1 + ``Variable`` + bespoke + non-elided slope
    - ID16: ``{root} Variable`` (flat across all VF siblings)
    - ID17: optical + width + bespoke + non-elided slope; default ``Regular``

    Upright and Roman are filename pairing markers elided from ID1/ID4/ID17
    (like Regular), used when a family has separate upright/roman and italic VF files.

    ID17 + STAT contract (replacers wire this in a follow-up):
    1. Preserve fvar/STAT strings linked at nameIDs <= 255 via
       ``preserve_low_nameids_in_fvar_stat_*`` in core_ttx_table_io.
    2. Set ID17 from filename slots, not ``compute_stat_default_style_name_*``.

Additional shared policies consolidated:
- ID2 subfamily mapping and RIBBI flag computation
- ID3 composition and sanitizers (version/vendor/filename)
- ID5 version string formatting
- ID6 PostScript name sanitization
- Family-level Regular-equivalent detection for non-standard families

Demo and Testing:
    Run 'python CoreDemoTool.py policies --help' to see examples of NameID building,
    PostScript sanitization, and variable slot parsing.

Maintenance Note:
    When adding new policy functions to this module, update CoreDemoTool.py to showcase
    the new functionality in the 'policies' subcommand.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Optional, Dict, List, TYPE_CHECKING

# New imports for enhanced functionality
from FontCore.core_logging_config import get_logger
from FontCore.core_string_utils import normalize_empty, is_empty, join_nonempty
from FontCore.core_font_style_dictionaries import (
    ELIDABLE_VF_FILENAME_SLOPES,
    ITALIC_LIKE_SLOPE_TERMS,
)

if TYPE_CHECKING:
    from FontCore.core_variable_filename_parser import VariableFilenameSlots

logger = get_logger(__name__)

DEFAULT_REGULAR_SYNONYMS_MODE = "regular_only"

# Slopes elided from variable-font display names (ID4) and ID17
ELIDABLE_VF_DISPLAY_SLOPES: frozenset[str] = frozenset(
    {"Regular", "Roman", "Normal", "Plain", "Standard", "Upright"}
) | ELIDABLE_VF_FILENAME_SLOPES

# Canonical list of valid regular-equivalent terms
VALID_REGULAR_EQUIVALENTS = frozenset(
    [
        "Regular",
        "Roman",
        "Plain",
        "Normal",
        "Standard",
        "Book",
        "Text",
        "Medium",
        "Light",
    ]
)


class RegularEquivalentError(ValueError):
    """Raised when an invalid regular_equivalent value is provided."""

    pass


def validate_regular_equivalent(
    value: str | None, *, strict: bool = False
) -> str | None:
    """
    Validate regular_equivalent parameter.

    Args:
        value: The regular_equivalent value to validate
        strict: If True, raise exception on invalid values.
                If False, log warning and return None.

    Returns:
        Validated value or None if invalid

    Raises:
        RegularEquivalentError: If strict=True and value is invalid
    """
    normalized = normalize_empty(value)

    # None or empty is always valid
    if normalized is None:
        return None

    # Check against valid list (case-insensitive)
    if normalized.title() not in VALID_REGULAR_EQUIVALENTS:
        valid_list = ", ".join(sorted(VALID_REGULAR_EQUIVALENTS))
        error_msg = (
            f"Invalid regular_equivalent '{normalized}'. Must be one of: {valid_list}"
        )

        if strict:
            raise RegularEquivalentError(error_msg)
        else:
            logger.warning(f"{error_msg}. Ignoring this parameter.")
            return None

    # Return title-cased version for consistency
    return normalized.title()


def normalize_nfc(value: str | None) -> str | None:
    """Return Unicode NFC-normalized string; pass through None."""
    if value is None:
        return None
    try:
        return unicodedata.normalize("NFC", str(value))
    except Exception:
        return value


# Compound modifier detection for filename parsing warnings
COMPOUND_MODIFIERS = ["semi", "demi", "extra", "ultra", "super", "x"]


def detect_compound_modifier_patterns(
    family: str, style: str, slope: str = None
) -> tuple[bool, list]:
    """Detect compound modifier patterns across family, style, and slope.

    Catches PascalCase splits across ALL font attributes:

    WEIGHTS: "Extra Bold", "Semi Light", "Ultra Black"
    WIDTHS: "Semi Condensed", "Extra Condensed", "Ultra Expanded"
    SLOPES: "Ultra Italic", "Super Italic"

    Returns: (detected, list_of_instances)
    Each instance is: {"source": "family|style|slope", "modifier": "extra", "parsed_as": "Extra Bold"}
    """
    instances = []

    # Check family
    if family:
        words = family.lower().split()
        if len(words) >= 2 and words[0] in COMPOUND_MODIFIERS:
            instances.append(
                {"source": "family", "modifier": words[0], "parsed_as": family}
            )

    # Check style
    if style:
        words = style.lower().split()
        if len(words) >= 2 and words[0] in COMPOUND_MODIFIERS:
            instances.append(
                {"source": "style", "modifier": words[0], "parsed_as": style}
            )

    # Check slope
    if slope:
        words = slope.lower().split()
        if len(words) >= 2 and words[0] in COMPOUND_MODIFIERS:
            instances.append(
                {"source": "slope", "modifier": words[0], "parsed_as": slope}
            )

    return len(instances) > 0, instances


RE_REGULAR = re.compile(r"\b(Regular|Roman)\b", re.I)
RE_REVERSE_ITALIC = re.compile(r"\bReverse\s+Italic\b", re.I)
RE_REVERSE_SLANTED = re.compile(r"\bReverse\s+Slanted\b", re.I)
RE_ITALIC = re.compile(r"\bItalic\b", re.I)
RE_OBLIQUE = re.compile(r"\bOblique\b", re.I)
RE_BACKSLANTED = re.compile(r"\bBack\s*Slanted\b|\bBackslanted\b", re.I)
RE_BACKSLANT = re.compile(r"\bBackslant\b", re.I)
RE_SLANTED = re.compile(r"\bSlanted\b", re.I)
RE_SLANT = re.compile(r"\bSlant\b", re.I)
RE_INCLINED = re.compile(r"\bInclined\b", re.I)
RE_RETALIC = re.compile(r"\bRetalic\b", re.I)
RE_REVERSE = re.compile(r"\bReverse\b", re.I)
RE_CURSIVE = re.compile(r"\b(Cursive|Kursiv)\b", re.I)
RE_BOOK = re.compile(r"\bBook\b", re.I)
RE_NORMAL = re.compile(r"\bNormal\b", re.I)
RE_VARIABLE_TOKENS = re.compile(r"\b(Variable|VF|GX|Flex)\b", re.I)

# (pattern, canonical slope label) — compound / longer forms first
_STYLE_SLOPE_EXTRACTORS: tuple[tuple[re.Pattern[str], str], ...] = (
    (RE_REVERSE_ITALIC, "Reverse Italic"),
    (RE_REVERSE_SLANTED, "Reverse Slanted"),
    (RE_ITALIC, "Italic"),
    (RE_OBLIQUE, "Oblique"),
    (RE_BACKSLANTED, "Backslanted"),
    (RE_BACKSLANT, "Backslant"),
    (RE_SLANTED, "Slanted"),
    (RE_INCLINED, "Inclined"),
    (RE_RETALIC, "Retalic"),
    (RE_CURSIVE, "Cursive"),
    (RE_REVERSE, "Reverse"),
    (RE_SLANT, "Slant"),
)


def _strip_token(pattern: "re.Pattern[str]", text: str) -> tuple[str, bool]:
    new_text, count = pattern.subn("", text)
    new_text = " ".join(new_text.split())
    return new_text, count > 0


def _extract_slope_from_style(style: str) -> tuple[str, str | None]:
    """Extract slope term from style, returning (cleaned_style, slope)."""
    for pattern, label in _STYLE_SLOPE_EXTRACTORS:
        if pattern.search(style):
            style, _ = _strip_token(pattern, style)
            return style, label
    return style, None


def _apply_regular_synonym_mode(
    style: str, mode: str, drop_book: bool | None, drop_normal: bool | None
) -> str:
    """Apply Book/Normal stripping based on mode and explicit flags."""
    # Explicit flags take precedence
    if drop_book is True:
        style, _ = _strip_token(RE_BOOK, style)
    if drop_normal is True:
        style, _ = _strip_token(RE_NORMAL, style)

    # Apply mode if no explicit flags
    if drop_book is None and drop_normal is None:
        mode_lower = (mode or DEFAULT_REGULAR_SYNONYMS_MODE).lower()

        if mode_lower == "loose":
            style, _ = _strip_token(RE_BOOK, style)
            style, _ = _strip_token(RE_NORMAL, style)
        elif mode_lower == "conservative":
            candidate = style.strip().lower()
            if candidate in {"book", "normal"}:
                style, _ = _strip_token(RE_BOOK, style)
                style, _ = _strip_token(RE_NORMAL, style)

    return style


def normalize_style_and_slope_for_id1_id4(
    subfamily_style: str | None,
    slope: str | None,
    *,
    regular_synonyms_mode: str = DEFAULT_REGULAR_SYNONYMS_MODE,
    drop_book: bool | None = None,
    drop_normal: bool | None = None,
    regular_equivalent: str | None = None,
) -> tuple[str | None, str | None]:
    """Normalize a style string for ID1/ID4 policy.

    - Remove Regular/Roman from style
    - If regular_equivalent is provided, also remove that term
    - If Italic/Oblique/Slanted appear in style, move them into slope when slope is not set
    - Idempotent: can be executed repeatedly safely
    """
    # Validate regular_equivalent
    regular_equivalent = validate_regular_equivalent(regular_equivalent, strict=False)

    # Normalize inputs
    subfamily_style = normalize_empty(subfamily_style)
    slope = normalize_empty(slope)

    if is_empty(subfamily_style):
        return None, slope

    style = subfamily_style
    style, _ = _strip_token(RE_REGULAR, style)

    # Strip regular-equivalent term if provided and valid
    if regular_equivalent and regular_equivalent.lower() != "regular":
        equiv_pattern = re.compile(rf"\b{re.escape(regular_equivalent)}\b", re.I)
        style, _ = _strip_token(equiv_pattern, style)

    # Extract and remove slope terms if slope not already set
    slope_norm = slope
    if is_empty(slope_norm):
        style, extracted_slope = _extract_slope_from_style(style)
        slope_norm = extracted_slope
    else:
        # Slope already set, just remove slope terms from style
        style, _ = _extract_slope_from_style(style)

    # Apply Book/Normal handling
    style = _apply_regular_synonym_mode(
        style, regular_synonyms_mode, drop_book, drop_normal
    )

    style = " ".join(style.split())
    return normalize_empty(style), slope_norm


def normalize_subfamily_term(
    term: str, axis_type: str = "unknown", stat_elidable_terms: Optional[set] = None
) -> str:
    """
    Normalize a subfamily term by cleaning up default/suppressible values.

    Args:
        term: The subfamily term to normalize (e.g., "Normal Thin", "Regular")
        axis_type: The axis type ("width", "weight", "slope", "unknown")
        stat_elidable_terms: Optional set of terms marked as elidable in STAT

    Returns:
        Normalized term with suppressible defaults removed

    Examples:
        >>> normalize_subfamily_term("Normal Thin", "weight")
        "Thin"
        >>> normalize_subfamily_term("Normal Regular", "weight")
        "Regular"
        >>> normalize_subfamily_term("Condensed", "width")
        "Condensed"
    """
    # Define suppressible terms per axis type
    WIDTH_SUPPRESSIBLE = {"regular", "normal", "standard", "roman"}
    SLOPE_SUPPRESSIBLE = {"roman", "upright", "normal", "regular"}

    # Clean the term
    cleaned = term.strip()
    cleaned_lower = cleaned.lower()

    # For width/slope: suppress if term matches suppressible list
    if axis_type == "width":
        if any(sup_term in cleaned_lower for sup_term in WIDTH_SUPPRESSIBLE):
            return ""
    elif axis_type == "slope":
        if any(sup_term in cleaned_lower for sup_term in SLOPE_SUPPRESSIBLE):
            return ""

    # For weight: clean up "Normal" prefix but never suppress weight terms
    if axis_type == "weight":
        if cleaned_lower.startswith("normal "):
            cleaned = cleaned[7:].strip()
        elif cleaned_lower == "normal":
            cleaned = "Regular"

    # Use STAT elidable info if provided
    if stat_elidable_terms and cleaned_lower in {
        t.lower() for t in stat_elidable_terms
    }:
        return ""

    return cleaned


def normalize_fvar_name(
    fvar_name: str,
    stat_values: Optional[Dict[str, Dict[float, str]]] = None,
    coordinates: Optional[Dict[str, float]] = None,
) -> str:
    """
    Normalize an fvar instance name using STAT-informed logic.

    Args:
        fvar_name: The fvar instance name (e.g., "Normal Thin", "Condensed Regular")
        stat_values: Optional STAT value mappings {axis_tag: {value: name}}
        coordinates: Optional instance coordinates for axis type detection

    Returns:
        Normalized name with suppressible terms removed

    Examples:
        >>> normalize_fvar_name("Normal Thin")
        "Thin"
        >>> normalize_fvar_name("Condensed Regular")
        "Condensed Regular"
        >>> normalize_fvar_name("Normal Regular")
        "Regular"
    """
    if not fvar_name or fvar_name == "Unknown":
        return fvar_name

    # Split into parts
    parts = fvar_name.split()
    normalized_parts = []

    # Try to classify each part by axis type
    for part in parts:
        part_lower = part.lower()

        # Classify based on common patterns
        # Width terms
        if part_lower in {
            "condensed",
            "compressed",
            "narrow",
            "extended",
            "expanded",
            "wide",
        }:
            axis_type = "width"
        # Weight terms
        elif part_lower in {
            "thin",
            "extralight",
            "light",
            "regular",
            "medium",
            "semibold",
            "bold",
            "extrabold",
            "black",
            "heavy",
        }:
            axis_type = "weight"
        # Slope terms
        elif part_lower in {"italic", "oblique", "slanted", "slant"}:
            axis_type = "slope"
        # Suppressible defaults (could be any axis)
        elif part_lower in {"normal", "standard", "roman", "upright"}:
            # Try to infer from position/context
            # If first word and followed by weight, it's likely width
            # If last word, it's likely slope
            # If standalone or with non-weight, suppress
            idx = parts.index(part)
            if idx == 0 and len(parts) > 1:
                axis_type = "width"  # Likely "Normal Thin" case
            elif idx == len(parts) - 1:
                axis_type = "slope"
            else:
                axis_type = "unknown"
        else:
            axis_type = "unknown"

        # Normalize the part
        normalized = normalize_subfamily_term(part, axis_type)
        if normalized:
            normalized_parts.append(normalized)

    # Join and return, or "Regular" if empty
    result = " ".join(normalized_parts)
    return result if result else "Regular"


def _sanitize_asterisk_for_id1_id4(text: str | None) -> str | None:
    """Sanitize asterisk characters for ID1/ID4 by preserving asterisks.

    Asterisks are now preserved in family names. This function is kept for
    backward compatibility but no longer modifies asterisks.
    Returns None if input is None.
    """
    if text is None:
        return None
    # Preserve asterisks - return text as-is
    return text


def is_elidable_vf_slope(slope: str | None) -> bool:
    """True when a variable-font slope token should be omitted from ID4/ID17."""
    if not slope or not str(slope).strip():
        return True
    return slope.strip().lower() in {s.lower() for s in ELIDABLE_VF_DISPLAY_SLOPES}


def build_id1_from_variable_slots(slots: "VariableFilenameSlots") -> str:
    """Construct ID1 from parsed variable-font filename slots."""
    return (
        join_nonempty(slots.root_family, slots.optical, slots.width) or slots.root_family
    )


def build_id4_from_variable_slots(slots: "VariableFilenameSlots") -> str:
    """Construct ID4 from parsed variable-font filename slots."""
    base = build_id1_from_variable_slots(slots)
    parts: list[str | None] = [base, "Variable", slots.bespoke]
    if slots.slope and not is_elidable_vf_slope(slots.slope):
        parts.append(slots.slope)
    return join_nonempty(*parts)


def build_id16_from_variable_slots(slots: "VariableFilenameSlots") -> str:
    """Construct ID16 typographic family: flat ``{root} Variable`` umbrella."""
    return join_nonempty(slots.root_family, "Variable")


def build_id17_from_variable_slots(slots: "VariableFilenameSlots") -> str:
    """Construct ID17 typographic subfamily from variable-font filename slots."""
    slope = slots.slope if not is_elidable_vf_slope(slots.slope) else None
    out = join_nonempty(slots.optical, slots.width, slots.bespoke, slope)
    return out if out else "Regular"


def build_id1(
    family: str,
    modifier: str | None,
    style: str | None,
    slope: str | None,
    *,
    is_variable: bool = False,
    variable_family_override: str | None = None,
    variable_slots: "VariableFilenameSlots | None" = None,
    use_filename_normalization: bool = True,
    regular_synonyms_mode: str = DEFAULT_REGULAR_SYNONYMS_MODE,
    regular_equivalent: str | None = None,
) -> str:
    """Construct ID1 (Family) string with policy."""
    if variable_slots is not None:
        return build_id1_from_variable_slots(variable_slots)

    # Validate regular_equivalent
    regular_equivalent = validate_regular_equivalent(regular_equivalent, strict=False)

    # Variable font policy: Strip "Variable" tokens from family name
    if is_variable:
        if variable_family_override:
            # Full name from filename: strip only Variable token, keep prefix/suffix (e.g. "Font Variable Black" -> "Font Black")
            return strip_only_variable_token(variable_family_override) or variable_family_override
        base = family
        # Asterisks are preserved in family names
        return strip_variable_tokens(base) or base

    style_eff = style
    slope_eff = slope
    if use_filename_normalization:
        style_eff, slope_eff = normalize_style_and_slope_for_id1_id4(
            style_eff,
            slope_eff,
            regular_synonyms_mode=regular_synonyms_mode,
            regular_equivalent=regular_equivalent,
        )

    # ID1: omit slope if it's "Italic"
    if slope_eff and slope_eff.strip().lower() == "italic":
        slope_eff = None

    # Asterisks are preserved in ID1 (no sanitization needed)
    return join_nonempty(family, modifier, style_eff, slope_eff)


def build_id4(
    family: str,
    modifier: str | None,
    style: str | None,
    slope: str | None,
    *,
    is_variable: bool = False,
    variable_family_override: str | None = None,
    variable_slots: "VariableFilenameSlots | None" = None,
    is_italic_font: bool | None = None,
    slope_from_filename: str | None = None,
    prefix_from_filename: str | None = None,
    suffix_from_filename: str | None = None,
    use_filename_normalization: bool = True,
    regular_synonyms_mode: str = DEFAULT_REGULAR_SYNONYMS_MODE,
    regular_equivalent: str | None = None,
) -> str:
    """Construct ID4 (Full) string with policy."""
    if variable_slots is not None:
        return build_id4_from_variable_slots(variable_slots)

    # Variable font policy: preserve prefix/suffix order around "Variable" (prefix before, suffix after)
    if is_variable:
        base = variable_family_override if variable_family_override else family
        # Asterisks are preserved in family names

        # Prefer explicit prefix/suffix from filename (order preserved: Family prefix Variable suffix)
        if prefix_from_filename is not None or suffix_from_filename is not None:
            suffix = (
                None
                if is_elidable_vf_slope(suffix_from_filename)
                else suffix_from_filename
            )
            return join_nonempty(base, prefix_from_filename, "Variable", suffix)

        # Legacy: single slope_from_filename (Variable then slope)
        if slope_from_filename and not is_elidable_vf_slope(slope_from_filename):
            return join_nonempty(base, "Variable", slope_from_filename)

        # Fallback to italic detection
        suffix = "Variable Italic" if is_italic_font else "Variable"
        return join_nonempty(base, suffix)

    style_eff = style
    slope_eff = slope
    if use_filename_normalization:
        style_eff, slope_eff = normalize_style_and_slope_for_id1_id4(
            style_eff,
            slope_eff,
            regular_synonyms_mode=regular_synonyms_mode,
            regular_equivalent=regular_equivalent,
        )

    # Asterisks are preserved in ID4 (no sanitization needed)
    return join_nonempty(family, modifier, style_eff, slope_eff)


def build_id16(
    family: str,
    *,
    is_variable: bool = False,
    variable_family_override: str | None = None,
    variable_slots: "VariableFilenameSlots | None" = None,
) -> str:
    """Construct ID16 (Typographic Family). Variable mode appends "Variable"."""
    if variable_slots is not None:
        return build_id16_from_variable_slots(variable_slots)

    # Variable font policy: Append "Variable" to family
    if is_variable:
        base = variable_family_override if variable_family_override else family
        return join_nonempty(base, "Variable")

    # Static font logic
    return family


def build_id17(
    modifier: str | None,
    style: str | None,
    slope: str | None,
) -> str:
    """Construct ID17 (Typographic Subfamily).

    ID17 retains Regular/Italic tokens; do not apply filename normalization here.
    Fallback to "Regular" if empty.
    """
    out = join_nonempty(modifier, style, slope)
    return out if out else "Regular"


__all__ = [
    "build_id1",
    "build_id4",
    "build_id16",
    "build_id17",
    "normalize_style_and_slope_for_id1_id4",
    "detect_compound_modifier_patterns",
    # ID2
    "allowed_id2_subfamilies",
    "map_metrics_to_id2_subfamily",
    "compute_ribbi_flags",
    # ID3
    "build_id3",
    "format_vendor_id",
    "is_bad_vendor",
    "prepare_vendor_for_achvendid",
    "sanitize_postscript",
    "sanitize_cff_name_string",
    # ID5
    "format_version_number",
    "build_id5_version_string",
    # Family-level Regular-equivalent detection
    "REGULAR_FALLBACK_PRIORITY",
    "group_fonts_by_family_filename",
    "identify_family_regular_equivalent",
    "get_regular_equivalent_for_families",
    # Variable helpers
    "is_elidable_vf_slope",
    "build_id1_from_variable_slots",
    "build_id4_from_variable_slots",
    "build_id16_from_variable_slots",
    "build_id17_from_variable_slots",
    "ELIDABLE_VF_DISPLAY_SLOPES",
    "split_variable_subfamily",
    "strip_only_variable_token",
    "strip_variable_tokens",
    "variable_filename_fragment",
    "build_id17_variable_default",
    # CFF/CFF2 helpers
    "has_cff_table",
    "has_cff2_table",
    "coerce_usable_nametable_string",
    "sanitize_nametable_string",
    "get_name_string_unicode_fallback",
    "get_name_string_win_english",
    "sync_cff_names_binary",
]


# ---------- ID2 (Subfamily) policies ----------

allowed_id2_subfamilies = {"Regular", "Italic", "Bold", "Bold Italic"}


def map_metrics_to_id2_subfamily(*, is_bold: bool, is_italic: bool) -> str:
    """Map boolean metrics to one of the allowed ID2 subfamilies."""
    if is_bold and is_italic:
        return "Bold Italic"
    if is_bold:
        return "Bold"
    if is_italic:
        return "Italic"
    return "Regular"


def compute_ribbi_flags(subfamily: str) -> tuple[int, int]:
    """Return (fsSelection, macStyle) integers based on RIBBI subfamily."""
    sub = (subfamily or "").strip().lower()
    is_bold = "bold" in sub
    is_italic = "italic" in sub

    fs_sel = 0
    if is_italic:
        fs_sel |= 0x0001
    if is_bold:
        fs_sel |= 0x0020
    if not is_bold and not is_italic:
        fs_sel |= 0x0040

    mac = 0
    if is_bold:
        mac |= 0x01
    if is_italic:
        mac |= 0x02
    return fs_sel, mac


# ---------- ID3 (Unique identifier) policies ----------

BAD_VENDOR_PATTERNS = {
    "NONE",
    "XXXX",
    "PYRS",
    "HL",
    "HL  ",
    "PFED",
    "TN",
    "TN  ",
}


def format_vendor_id(vendor_value: Any) -> str:
    """Format a vendor value (bytes or string) to a 4-char display string."""
    if vendor_value is None:
        return "UKWN"
    if isinstance(vendor_value, bytes):
        try:
            vendor_str = vendor_value.decode("ascii", errors="ignore")
        except Exception:
            vendor_str = ""
    else:
        vendor_str = str(vendor_value)
    vendor_str = vendor_str.replace("\x00", " ")
    return vendor_str.ljust(4)[:4]


def prepare_vendor_for_achvendid(vendor_str: str) -> bytes:
    """Prepare vendor for OS/2.achVendID (4 chars, spaces padded, ASCII)."""
    return vendor_str[:4].ljust(4).encode("ascii", errors="replace")


def is_bad_vendor(vendor_str: str | None) -> bool:
    if vendor_str is None:
        return True
    normalized = vendor_str.replace("\x00", " ").upper()
    if normalized.strip() == "":
        return True
    if normalized.strip() in BAD_VENDOR_PATTERNS:
        return True
    if set(normalized) <= {" "}:
        return True
    return False


def sanitize_postscript(name: str) -> str:
    """Sanitize PostScript-like names; keep '-', '_', '.', '?', '!', '&', '*'; remove spaces; replace others with '-'."""
    name = name.replace(" ", "")
    return re.sub(r"[^A-Za-z0-9\-\._\?\!\&\*]", "-", name)


# Typographic Unicode that commonly appears in trial/promo nameIDs but cannot live in CFF strings.
_CFF_UNICODE_REPLACEMENTS = (
    ("\u2014", "-"),  # em dash
    ("\u2013", "-"),  # en dash
    ("\u2212", "-"),  # minus sign
    ("\u2010", "-"),  # hyphen
    ("\u2011", "-"),  # non-breaking hyphen
    ("\u00ad", "-"),  # soft hyphen
    ("\u2018", "'"),  # left single quotation mark
    ("\u2019", "'"),  # right single quotation mark
    ("\u201a", "'"),  # single low-9 quotation mark
    ("\u201b", "'"),  # single high-reversed-9 quotation mark
    ("\u201c", '"'),  # left double quotation mark
    ("\u201d", '"'),  # right double quotation mark
    ("\u201e", '"'),  # double low-9 quotation mark
    ("\u2032", "'"),  # prime
    ("\u2033", '"'),  # double prime
    ("\u2026", "..."),  # ellipsis
    ("\u00a0", " "),  # no-break space
)


def sanitize_cff_name_string(name: str) -> str:
    """Sanitize human-readable CFF TopDict strings (FullName, FamilyName) to latin-1."""
    if not name:
        return name
    for src, dst in _CFF_UNICODE_REPLACEMENTS:
        name = name.replace(src, dst)
    out: List[str] = []
    for ch in name:
        try:
            ch.encode("latin-1")
            out.append(ch)
        except UnicodeEncodeError:
            out.append("-")
    return "".join(out)


def _sanitize_for_cff_field(field_name: str, value: str) -> str:
    if field_name == "FontName":
        return sanitize_postscript(value)
    return sanitize_cff_name_string(value)


def build_id3(version: str, vendor: str, filename: str) -> str:
    """Compose ID3 content: version;vendor;filename (already sanitized upstream)."""
    return f"{version};{vendor};{filename}"


# ---------- ID5 (Version) policies ----------


def format_version_number(value: Any) -> str:
    """Format a version number to 'x.xxx' string (e.g., 1.0 -> '1.000')."""
    try:
        num = float(value)
        return f"{num:.3f}"
    except Exception:
        return str(value)


def build_id5_version_string(version: str | float) -> str:
    return f"Version {format_version_number(version)}"


# ---------- Family-level Regular-equivalent detection ----------

REGULAR_FALLBACK_PRIORITY = [
    "Roman",
    "Plain",
    "Normal",
    "Book",
    "Text",
    "Medium",
    "Light",
]


def group_fonts_by_family_filename(font_paths: list[str]) -> dict[str, list[str]]:
    """Group font file paths by family name extracted from filename."""
    try:
        from FontCore.core_filename_parts_parser import parse_filename
    except ImportError:
        return _group_fonts_fallback(font_paths)

    families: dict[str, list[str]] = {}
    for path in font_paths:
        parsed = parse_filename(path, strip_extension=True)
        fam = parsed.family if parsed.family else "Unknown"
        families.setdefault(fam, []).append(path)
    return families


def _group_fonts_fallback(font_paths: list[str]) -> dict[str, list[str]]:
    """Fallback grouping when parser not available."""
    import os

    families: dict[str, list[str]] = {}
    for path in font_paths:
        base = os.path.basename(path)
        base_no_ext = os.path.splitext(base)[0]
        fam = base_no_ext.split("-", 1)[0] if "-" in base_no_ext else base_no_ext
        families.setdefault(fam, []).append(path)
    return families


def _extract_weight_term_from_subfamily(subfamily: str) -> str | None:
    """Extract first recognized weight term from subfamily string."""
    if not subfamily:
        return None

    sub_lower = subfamily.lower()

    for term in REGULAR_FALLBACK_PRIORITY:
        if term.lower() in sub_lower:
            return term

    if "regular" in sub_lower:
        return "Regular"

    return None


def _check_text_standalone(subfamily: str) -> bool:
    """Check if 'Text' appears as a standalone optical size, not with weight terms."""
    if not subfamily or "text" not in subfamily.lower():
        return False

    sub_lower = subfamily.lower()

    # Exclude if combined with weight terms
    weight_disqualifiers = [
        "bold",
        "book",
        "normal",
        "medium",
        "black",
        "heavy",
        "extra",
        "semi",
        "demi",
    ]
    if any(w in sub_lower for w in weight_disqualifiers):
        return False

    # Remove slope tokens and see if we're left with just "text"
    temp = sub_lower
    temp = re.sub(r"\b(italic|oblique|slanted)\b", "", temp)
    temp = re.sub(r"[^a-z]+", "", temp)

    return temp == "text"


def _get_usweightclass_from_font(font_path: str) -> int | None:
    """Read usWeightClass from a font file without full table loading."""
    try:
        from fontTools.ttLib import TTFont

        font = TTFont(font_path, lazy=True)
        if "OS/2" in font:
            weight = font["OS/2"].usWeightClass
            font.close()
            return int(weight)
        else:
            logger.debug(f"Font '{font_path}' has no OS/2 table")
    except Exception as e:
        logger.warning(f"Failed to read usWeightClass from '{font_path}': {e}")
    return None


def _check_for_regular_in_filenames(family_paths: list[str]) -> bool:
    """Check if any font in family has 'Regular' in filename."""
    try:
        from FontCore.core_filename_parts_parser import parse_filename
    except ImportError:
        return False

    for path in family_paths:
        parsed = parse_filename(path, strip_extension=True)
        if parsed.subfamily and "regular" in parsed.subfamily.lower():
            return True
    return False


def _find_closest_to_400_weight_term(family_paths: list[str]) -> str | None:
    """Find regular-equivalent term by analyzing which term is closest to weight 400.

    Logic:
    1. For each recognized term (Roman, Plain, Normal, Book, etc.), find fonts with that term
    2. Get average weight for that term across all width variants
    3. Return term closest to 400
    4. On ties, use priority order (REGULAR_FALLBACK_PRIORITY)
    """
    try:
        from FontCore.core_filename_parts_parser import parse_filename
    except ImportError:
        return None

    # Build term -> list of weights mapping
    term_weights: Dict[str, List[int]] = {}

    for path in family_paths:
        weight = _get_usweightclass_from_font(path)
        if weight is None:
            continue

        parsed = parse_filename(path, strip_extension=True)
        if not parsed.subfamily:
            continue

        term = _extract_weight_term_from_subfamily(parsed.subfamily)
        if term:
            if term not in term_weights:
                term_weights[term] = []
            term_weights[term].append(weight)

    if not term_weights:
        return None

    # Calculate average weight for each term
    term_avg_weights: Dict[str, float] = {}
    for term, weights in term_weights.items():
        term_avg_weights[term] = sum(weights) / len(weights)

    # Find term(s) closest to 400
    closest_distance = float("inf")
    closest_terms = []

    for term, avg_weight in term_avg_weights.items():
        distance = abs(avg_weight - 400)
        if distance < closest_distance:
            closest_distance = distance
            closest_terms = [term]
        elif distance == closest_distance:
            closest_terms.append(term)

    # If single winner, return it
    if len(closest_terms) == 1:
        logger.debug(
            f"Found single closest term: {closest_terms[0]} (avg weight: {term_avg_weights[closest_terms[0]]:.1f}, distance: {closest_distance:.1f})"
        )
        return closest_terms[0]

    # Multiple terms at same distance - use priority order
    for candidate in REGULAR_FALLBACK_PRIORITY:
        if candidate in closest_terms:
            logger.debug(
                f"Tie resolved by priority: {candidate} (avg weight: {term_avg_weights[candidate]:.1f}, distance: {closest_distance:.1f})"
            )
            return candidate

    # Fallback to first term found
    result = closest_terms[0] if closest_terms else None
    if result:
        logger.debug(
            f"Fallback selection: {result} (avg weight: {term_avg_weights[result]:.1f}, distance: {closest_distance:.1f})"
        )
    return result


def _find_term_from_filenames(family_paths: list[str]) -> str | None:
    """Find fallback term from filenames by scanning in priority order."""
    try:
        from FontCore.core_filename_parts_parser import parse_filename
    except ImportError:
        return None

    term_counts: dict[str, int] = {}
    text_standalone_count = 0

    for path in family_paths:
        parsed = parse_filename(path, strip_extension=True)
        if not parsed.subfamily:
            continue

        if _check_text_standalone(parsed.subfamily):
            text_standalone_count += 1

        term = _extract_weight_term_from_subfamily(parsed.subfamily)
        if term:
            term_counts[term] = term_counts.get(term, 0) + 1

    # Return first term found in priority order
    for candidate in REGULAR_FALLBACK_PRIORITY:
        if candidate == "Text":
            if text_standalone_count > 0:
                return "Text"
        elif candidate in term_counts:
            return candidate

    return None


def identify_family_regular_equivalent(family_paths: list[str]) -> str | None:
    """Identify which weight term acts as "Regular" for this font family.

    Detection priority:
    1. Check filenames for "Regular"
    2. Check usWeightClass == 400 (must be exactly one match)
    3. Check filenames for fallback terms in priority order
    4. Special handling for "Text" (must be standalone optical size)
    """
    if not family_paths:
        return None

    logger.debug(f"Analyzing {len(family_paths)} fonts for regular equivalent")

    # Step 1: Check for "Regular" in filenames
    if _check_for_regular_in_filenames(family_paths):
        logger.debug("Found 'Regular' in filenames")
        return "Regular"

    # Step 2: Find term closest to weight 400 across all variants
    term = _find_closest_to_400_weight_term(family_paths)
    if term:
        logger.debug(f"Found closest-to-400 term: {term}")
        return term

    # Step 3: Check filenames for fallback terms
    term = _find_term_from_filenames(family_paths)
    if term:
        logger.debug(f"Found fallback term in filenames: {term}")
    else:
        logger.info(
            f"Could not determine regular equivalent for family with {len(family_paths)} fonts"
        )
    return term


def get_regular_equivalent_for_families(font_paths: list[str]) -> dict[str, str | None]:
    """Analyze font files and return per-family regular-equivalent mapping."""
    families = group_fonts_by_family_filename(font_paths)
    return {
        fam: identify_family_regular_equivalent(paths)
        for fam, paths in families.items()
    }


# ---------- Variable font helpers ----------


# Variable font detection functions are now imported from FontCore.core_variable_font_detection

# Split on Variable, including glued pairing/italic markers (VariableItalic, VariableUpright, VariableRoman)
_RE_VARIABLE_WORD = re.compile(r"\bVariable(?:Italic|Upright|Roman)?\b", re.I)


def split_variable_subfamily(subfamily: str | None) -> tuple[str, str]:
    """Split subfamily into (prefix, suffix) around Variable / VariableItalic / pairing markers.

    .. deprecated::
        Prefer ``parse_variable_filename()`` and ``build_id*_from_variable_slots()``.
        Kept for NameID replacers until they are migrated.

    Prefix is text before the token, suffix is text after. Both are normalized
    (strip, collapse spaces). Used for ID4 (prefix + suffix as third part) and
    ID17 (prefix in fallback). Glued pairing markers (VariableUpright, VariableRoman)
    and VariableItalic are consumed as the Variable token (empty suffix).

    Examples:
        "Black Variable" -> ("Black", "")
        "Variable Black" -> ("", "Black")
        "Black Variable Italic" -> ("Black", "Italic")
        "VariableItalic" -> ("", "")
        "VariableUpright" -> ("", "")
        "VariableRoman" -> ("", "")
    """
    if not subfamily or not subfamily.strip():
        return "", ""
    s = str(subfamily).strip()
    parts = _RE_VARIABLE_WORD.split(s, maxsplit=1)
    prefix = " ".join(parts[0].strip().split()) if parts[0].strip() else ""
    suffix = (
        " ".join(parts[1].strip().split())
        if len(parts) > 1 and parts[1].strip()
        else ""
    )
    return prefix, suffix


def strip_only_variable_token(text: str | None) -> str | None:
    """Remove Variable / VariableItalic / VariableUpright / VariableRoman (word boundaries).

    Does not remove VF/GX/Flex. Used for ID1 variable so "Font Variable Black" -> "Font Black".
    """
    text = normalize_empty(text)
    if is_empty(text):
        return None
    s = _RE_VARIABLE_WORD.sub(" ", str(text))
    s = " ".join(s.split()).strip()
    return normalize_empty(s)


def strip_variable_tokens(text: str | None) -> str | None:
    """Strip Variable/VF/GX/Flex tokens from text.

    Also strips glued pairing markers (VariableUpright, VariableRoman) and spaced
    forms ("Variable Upright", "Variable Roman") so family cleanup does not leave
    an orphaned upright/roman token. Longer Variable+marker forms are removed
    before bare ``Variable`` so spaced pairing markers are not split apart.
    """
    text = normalize_empty(text)
    if is_empty(text):
        return None

    s = str(text)
    # Variable + optional glued/spaced marker (before bare Variable strip)
    s = re.sub(
        r"(?i)(?<![A-Za-z0-9])Variable(?:\s+(?:Italic|Upright|Roman)|(?:Italic|Upright|Roman))?(?![A-Za-z0-9])",
        " ",
        s,
    )
    s, _ = RE_VARIABLE_TOKENS.subn(" ", s)
    s = re.sub(r"(?i)(?<![A-Za-z0-9])(VF|GX|Flex)(?![A-Za-z0-9])", " ", s)
    # Drop separators left at the edges after token removal
    s = re.sub(r"^[-_\s]+|[-_\s]+$", "", s)
    s = " ".join(s.split())

    return normalize_empty(s)


def _strip_trailing_slope_tokens(text: str) -> str:
    s = re.sub(
        r"[-_\s]*(Reverse\s+Italic|Reverse\s+Slanted|Italic|Oblique|Back\s*Slanted|"
        r"Backslanted|Backslant|Slanted|Inclined|Retalic|Cursive|Kursiv|Reverse|Slant)$",
        "",
        text,
        flags=re.I,
    ).strip()
    return s


def _collapse_hyphens(text: str) -> str:
    s = re.sub(r"-{2,}", "-", text)
    s = re.sub(r"[-_\s]+$", "", s)
    return s


def normalize_family_for_postscript(family_like: str) -> str:
    s = strip_variable_tokens(family_like) or family_like
    s = _strip_trailing_slope_tokens(s)
    s = _collapse_hyphens(s)
    return sanitize_postscript(s)


def variable_filename_fragment(family: str, is_italic: bool) -> str:
    suffix = "VariableItalic" if is_italic else "Variable"
    clean_family = normalize_family_for_postscript(family)
    return f"{clean_family}-{suffix}"


def build_id17_variable_default(
    is_italic: bool,
    slope_from_filename: str | None = None,
    prefix_from_filename: str | None = None,
    variable_slots: "VariableFilenameSlots | None" = None,
) -> str:
    """Build ID17 for variable fonts from filename-derived prefix/slope or italic detection.

    When ``variable_slots`` is provided, delegates to ``build_id17_from_variable_slots``.

    prefix_from_filename: part before "Variable" in subfamily (e.g. Black, Bold).
    slope_from_filename: part after "Variable" or slope term (e.g. Italic).
    If the only part is a slope term (Italic/Oblique/Slanted), prefix with "Regular ".
    """
    if variable_slots is not None:
        return build_id17_from_variable_slots(variable_slots)

    slope = slope_from_filename
    if is_elidable_vf_slope(slope):
        slope = None
    if slope is None and is_italic:
        slope = "Italic"
    prefix = normalize_empty(prefix_from_filename)
    slope_n = normalize_empty(slope)
    combined = join_nonempty(prefix, slope_n)
    if not combined:
        return "Regular"
    # Pure slope term -> prefix with "Regular "
    if slope_n and not prefix and slope_n.strip().lower() in {
        s.lower() for s in ITALIC_LIKE_SLOPE_TERMS
    }:
        return f"Regular {slope_n}"
    return combined


def ensure_regular_prefix_for_pure_slope(subfamily: str | None) -> str | None:
    """If subfamily is just a known slope term, prefix with 'Regular '."""
    if not subfamily:
        return subfamily
    s = (subfamily or "").strip()
    if s.lower() in {t.lower() for t in ITALIC_LIKE_SLOPE_TERMS}:
        return f"Regular {s}"
    return subfamily


# ---------- CFF/CFF2 helpers ----------


def has_cff_table(font: Any) -> bool:
    try:
        return "CFF " in font
    except Exception:
        return False


def has_cff2_table(font: Any) -> bool:
    try:
        return "CFF2" in font
    except Exception:
        return False


def get_name_string_win_english(font: Any, name_id: int) -> str | None:
    try:
        if "name" not in font:
            return None
        rec = font["name"].getName(name_id, 3, 1, 0x409)
        if rec is None:
            return None
        try:
            return rec.toUnicode()
        except Exception:
            return str(rec)
    except Exception:
        return None


def sanitize_nametable_string(value: str) -> str:
    """Strip control chars (incl. DEL) from a decoded namerecord; collapse outer whitespace."""
    if not value:
        return ""
    out: List[str] = []
    for ch in value:
        o = ord(ch)
        if o == 0x7F or (o < 0x20 and ch not in "\t\n\r"):
            continue
        out.append(ch)
    return "".join(out).strip()


def coerce_usable_nametable_string(raw: str | None) -> str | None:
    """
    Windows name slots are often populated with placeholders (``.``, ``.&`` with control chars)
    while Mac records remain valid. Return a cleaned display string only if it has real content
    (at least one Unicode letter or digit).
    """
    if raw is None:
        return None
    cleaned = sanitize_nametable_string(str(raw))
    if not cleaned:
        return None
    if not any(ch.isalnum() for ch in cleaned):
        return None
    return cleaned


def get_name_string_unicode_fallback(font: Any, name_id: int) -> str | None:
    """
    Prefer Windows UCS-2 English (US), then fall back to other namerecords.

    If the Windows string decodes but is placeholder junk (``.``, control glyphs, punctuation
    without letters), it is skipped and Macintosh / other platforms are considered—matching
    real-world subsets where Mac names were preserved while Windows IDs were rewritten.
    """
    win_raw = get_name_string_win_english(font, name_id)
    win = coerce_usable_nametable_string(win_raw)
    if win:
        return win

    tbl = getattr(font.get("name", None), "names", None)
    if not tbl:
        return None

    def sort_key(rec: Any) -> tuple:
        lid = getattr(rec, "langID", 0) or 0
        pid, eid = int(rec.platformID), int(rec.platEncID)
        try:
            lid = int(lid)
        except Exception:
            lid = 0
        if pid == 3 and eid == 1 and lid == 0x409:
            return (0, 0, 0)
        if pid == 3 and eid in (1, 10):
            return (1, lid, eid)
        if pid == 3:
            return (2, eid, lid)
        if pid == 1:
            return (3, lid, eid)
        return (9, pid, eid, lid)

    candidates: list[tuple[tuple, str]] = []
    for rec in tbl:
        if int(rec.nameID) != name_id:
            continue
        try:
            txt = rec.toUnicode()
        except Exception:
            try:
                bs = getattr(rec, "string", b"")
                if isinstance(bs, bytes):
                    txt = bs.decode("latin-1", errors="replace")
                else:
                    txt = str(bs)
            except Exception:
                continue
        txt_ok = coerce_usable_nametable_string(txt)
        if not txt_ok:
            continue
        candidates.append((sort_key(rec), txt_ok))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _update_cff_topdict_field(top, field_name: str, value: str) -> bool:
    """Update a single CFF TopDict field if different."""
    if not value:
        return False

    safe_value = _sanitize_for_cff_field(field_name, value)
    if not safe_value:
        return False

    current = getattr(top, field_name, None)
    if current == safe_value:
        return False

    try:
        setattr(top, field_name, safe_value)
        return True
    except Exception:
        return False


def _sync_cff_table(
    font: Any, ps_name: str | None, full_name: str | None, family_name: str | None
) -> bool:
    """Sync CFF table names."""
    if not has_cff_table(font):
        return False

    changed = False
    try:
        cff_table = font["CFF "]
        cff = getattr(cff_table, "cff", None)
        if cff and hasattr(cff, "topDictIndex"):
            for top in cff.topDictIndex:  # type: ignore[attr-defined]
                if ps_name:
                    changed |= _update_cff_topdict_field(top, "FontName", ps_name)
                if full_name:
                    changed |= _update_cff_topdict_field(top, "FullName", full_name)
                if family_name:
                    changed |= _update_cff_topdict_field(top, "FamilyName", family_name)
    except Exception:
        pass

    return changed


def _sync_cff2_table(font: Any, ps_name: str | None) -> bool:
    """Sync CFF2 table names (FontName only)."""
    if not has_cff2_table(font):
        return False

    changed = False
    try:
        cff2_table = font["CFF2"]
        cff2 = getattr(cff2_table, "cff", None)
        if cff2 and hasattr(cff2, "topDictIndex"):
            for top in cff2.topDictIndex:  # type: ignore[attr-defined]
                if ps_name:
                    changed |= _update_cff_topdict_field(top, "FontName", ps_name)
    except Exception:
        pass

    return changed


def sync_cff_names_binary(font: Any) -> bool:
    """Sync CFF/CFF2 TopDict names from name table (ID 4, 6, 16/1)."""
    try:
        ps_name = get_name_string_win_english(font, 6)
        full_name = get_name_string_win_english(font, 4)
        family16 = get_name_string_win_english(font, 16)
        family1 = get_name_string_win_english(font, 1)
        family_name = family16 or family1

        changed = False
        changed |= _sync_cff_table(font, ps_name, full_name, family_name)
        changed |= _sync_cff2_table(font, ps_name)

        return changed
    except Exception:
        return False
