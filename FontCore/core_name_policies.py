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
from dataclasses import dataclass
from typing import Any, Optional, Dict, List, TYPE_CHECKING

# New imports for enhanced functionality
from FontCore.core_logging_config import get_logger
from FontCore.core_string_utils import normalize_empty, is_empty, join_nonempty
from FontCore.core_font_style_dictionaries import (
    ELIDABLE_VF_FILENAME_SLOPES,
    ITALIC_LIKE_SLOPE_TERMS,
    ensure_weight_word,
)

if TYPE_CHECKING:
    from FontCore.core_variable_filename_parser import VariableFilenameSlots

logger = get_logger(__name__)

# Slopes elided from variable-font display names (ID4) and ID17
ELIDABLE_VF_DISPLAY_SLOPES: frozenset[str] = frozenset(
    {"Regular", "Roman", "Normal", "Plain", "Standard", "Upright"}
) | ELIDABLE_VF_FILENAME_SLOPES

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


RE_VARIABLE_TOKENS = re.compile(r"\b(Variable|VF|GX|Flex)\b", re.I)

# Words ID1 leaves out: the RIBBI words. Every other word is an ordinary style word and stays in ID1.
_ID1_DROPPED_WORDS = frozenset({"regular", "italic", "bold"})

# A measured italic angle at or beyond this many degrees (forward or back) means a slanted face.
ITALIC_ANGLE_FLOOR = 2.0


def id1_style(style: str | None) -> str | None:
    """Style words for ID1: drop only the exact words Regular, Italic and Bold.

    No slope vocabulary: Oblique, Cursive, Backslanted and every other word stay, so ID1 + ID2 stay
    unique (ID2 says Italic for all of them). Roman, Book, Normal, Plain, Medium, Light and Text stay too.
    """
    style = normalize_empty(style)
    if is_empty(style):
        return None
    kept = [w for w in str(style).split() if w.lower() not in _ID1_DROPPED_WORDS]
    return normalize_empty(" ".join(kept))


@dataclass(frozen=True)
class RibbiDecision:
    """ID2 and the style it implies: one of Regular, Italic, Bold, Bold Italic."""

    subfamily: str
    bold: bool
    italic: bool
    notes: tuple[str, ...] = ()


_SLOPE_LIKE = {"oblique", "slanted", "slant", "backslanted", "backslant", "inclined", "cursive", "kursiv"}

_FS_ITALIC, _FS_BOLD = 0x0001, 0x0020
_MAC_BOLD, _MAC_ITALIC = 0x0001, 0x0002


def decide_ribbi(
    style: str | None,
    *,
    fs_selection: int = 0,
    mac_style: int = 0,
    italic_angle: float | None = None,
) -> RibbiDecision:
    """Decide ID2 from the name, then the flags, then the italic angle.

    The name wins. A style that says anything (a weight, Regular, a width) is never turned Italic by
    the angle alone: a slanted upright-named design (a script, a banner) is not an italic variant.

    - Bold: the word Bold in the style. Flags decide only when the style says nothing at all.
      usWeightClass is not used (it is not always 700).
    - Italic: the word Italic in the style. When the style says nothing at all, the flags decide, then a
      measured angle with |angle| >= ITALIC_ANGLE_FLOOR (forward or back slants). A name that says
      Italic is never demoted by a flat angle, and a slanted font whose name says nothing about slope
      keeps Regular; both return a note instead.
    """
    text = normalize_empty(style) or ""
    words = {w.lower() for w in text.split()}
    says_nothing = not words

    bold = "bold" in words
    if not bold and says_nothing:
        bold = bool(fs_selection & _FS_BOLD or mac_style & _MAC_BOLD)

    notes: list[str] = []
    slanted = italic_angle is not None and abs(italic_angle) >= ITALIC_ANGLE_FLOOR
    italic = "italic" in words
    if not italic and says_nothing:
        italic = bool(fs_selection & _FS_ITALIC or mac_style & _MAC_ITALIC)
    # The angle can add Italic only when the style names a slope (Oblique, Backslant, Cursive, ...) or
    # says nothing. Any other style (a weight, Regular, a width, or a bespoke word like Bulky) is an
    # upright name, so a slanted design keeps Regular: a script is not an italic variant, and an
    # unknown word next to an Italic sibling must not collide with it (PolySansMono Bulky / Bulky Italic).
    might_name_slope = says_nothing or bool(words & _SLOPE_LIKE)
    if not italic and slanted and might_name_slope:
        italic = True
    if slanted and not italic:
        notes.append(f"the italic angle is {italic_angle:g} degrees but the name does not say Italic")
    if "italic" in words and italic_angle is not None and not slanted:
        notes.append(f"the name says Italic but the italic angle is {italic_angle:g} degrees")

    subfamily = "Bold Italic" if bold and italic else "Bold" if bold else "Italic" if italic else "Regular"
    return RibbiDecision(subfamily, bold, italic, tuple(notes))


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


def _variable_family_base(slots: "VariableFilenameSlots") -> str:
    """Root family plus optical size and width, without the word Variable."""
    return (
        join_nonempty(slots.root_family, slots.optical, slots.width) or slots.root_family
    )


def _spaced(token: str | None) -> str | None:
    """"3Axis" -> "3 Axis", "OpticalSize" -> "Optical Size" (display form of a filename token)."""
    if not token:
        return None
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", token)


def _variable_what_varies(slots: "VariableFilenameSlots") -> tuple[str | None, str | None]:
    return _spaced(slots.axes), slots.weight


def build_id16_from_variable_slots(slots: "VariableFilenameSlots") -> str:
    """ID16 typographic family: the fixed design plus Variable ("Probe Display Condensed Variable")."""
    return join_nonempty(_variable_family_base(slots), "Variable")


def build_id1_from_variable_slots(slots: "VariableFilenameSlots") -> str:
    """ID1: ID16 plus the free axes and any locked weight, without the slope (DECISIONS.md, Area 5).

    An angle word other than Italic (Oblique, Reverse, ...) stays in ID1 as naming; ID2 carries Italic.
    """
    axes, weight = _variable_what_varies(slots)
    angle = slots.slope if slots.slope and slots.slope.lower() != "italic" else None
    return join_nonempty(build_id16_from_variable_slots(slots), axes, weight, angle, slots.bespoke)


def build_id17_from_variable_slots(slots: "VariableFilenameSlots") -> str:
    """ID17: the filename's style part, spaced and without Variable, plus Upright when the slope is fixed upright."""
    axes, weight = _variable_what_varies(slots)
    slope = slots.slope or ("Upright" if slots.upright else None)
    return join_nonempty(axes, weight, slots.bespoke, slope) or "Upright"


def build_id4_from_variable_slots(slots: "VariableFilenameSlots") -> str:
    """ID4: ID16 plus ID17."""
    return join_nonempty(build_id16_from_variable_slots(slots), build_id17_from_variable_slots(slots))


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
) -> str:
    """Construct ID1 (Family) string with policy."""
    if variable_slots is not None:
        return build_id1_from_variable_slots(variable_slots)

    # Variable font policy: Strip "Variable" tokens from family name
    if is_variable:
        if variable_family_override:
            # Full name from filename: keep one "Variable" and any prefix/suffix, drop pairing markers
            # (e.g. "Font Variable Black" stays, "Font VariableItalic" -> "Font Variable")
            kept = keep_one_variable_token(variable_family_override)
            return kept if kept and "variable" in kept.lower() else join_nonempty(kept, "Variable")
        base = family
        # Asterisks are preserved in family names
        return join_nonempty(strip_variable_tokens(base) or base, "Variable")

    # ID1 keeps every style word except Regular, Italic and Bold (no slope vocabulary)
    style_eff = id1_style(join_nonempty(style, slope))

    # Asterisks are preserved in ID1 (no sanitization needed)
    return join_nonempty(family, modifier, style_eff)


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

    # ID4 names the weight: "Probe Regular", "Probe Regular Italic" (house convention, DECISIONS.md C1)
    style_eff = ensure_weight_word(join_nonempty(modifier, style, slope) or "")

    # Asterisks are preserved in ID4 (no sanitization needed)
    return join_nonempty(family, style_eff)


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

    ID17 keeps Regular and always carries a weight word: a regular italic is "Regular Italic", and
    "Wide Italic" becomes "Wide Regular Italic". No filename normalization is applied here.
    """
    return ensure_weight_word(join_nonempty(modifier, style, slope) or "")


__all__ = [
    "build_id1",
    "build_id4",
    "build_id16",
    "build_id17",
    "id1_style",
    "decide_ribbi",
    "RibbiDecision",
    "ITALIC_ANGLE_FLOOR",
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
    # Variable helpers
    "is_elidable_vf_slope",
    "build_id1_from_variable_slots",
    "build_id4_from_variable_slots",
    "build_id16_from_variable_slots",
    "build_id17_from_variable_slots",
    "ELIDABLE_VF_DISPLAY_SLOPES",
    "split_variable_subfamily",
    "strip_only_variable_token",
    "keep_one_variable_token",
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


def keep_one_variable_token(text: str | None) -> str | None:
    """Reduce Variable / VariableItalic / Variable Upright / VariableRoman to a single "Variable".

    Used for ID1 of a variable font: the word stays, the pairing marker (Italic, Upright, Roman) goes.
    """
    text = normalize_empty(text)
    if is_empty(text):
        return None
    s = re.sub(
        r"(?i)(?<![A-Za-z0-9])Variable(?:\s+(?:Italic|Upright|Roman)|(?:Italic|Upright|Roman))?(?![A-Za-z0-9])",
        "Variable",
        str(text),
    )
    return normalize_empty(" ".join(s.split()))


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
