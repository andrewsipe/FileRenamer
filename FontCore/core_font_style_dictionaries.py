#!/usr/bin/env python3
"""
Centralized font style dictionaries for filename processing.

Philosophy: Minimal dictionaries for deterministic outcomes and ambiguity resolution.
Pattern matching handles variations programmatically.
"""

import re


# ================================================================================================
# 1. COMPOUND NORMALIZATIONS (~30 entries)
# ================================================================================================
# Arbitrary style choices where no pattern can decide the correct casing

COMPOUND_NORMALIZATIONS = {
    # Weight modifiers - lowercase the weight after modifier
    "SemiBold": "Semibold",
    "DemiBold": "Demibold",
    "ExtraBold": "Extrabold",
    "UltraBold": "Ultrabold",
    "SemiLight": "Semilight",
    "ExtraLight": "Extralight",
    "UltraLight": "Ultralight",
    "ExtraBlack": "Extrablack",
    "UltraBlack": "Ultrablack",
    "ExtraThin": "Extrathin",
    "UltraThin": "Ultrathin",
    "ExtraHeavy": "Extraheavy",
    "UltraHeavy": "Ultraheavy",
    # Special cases
    "SmallCaps": "Smallcaps",
    "Small-Caps": "Smallcaps",
    "ItalicSmallCaps": "SmallcapsItalic",
    "ObliqueSmallCaps": "SmallcapsOblique",
    # Variable font cleanup
    "VariableRegular-Variable": "Variable",
    "VariableItalic-Variable": "VariableItalic",
    "VariableOblique-Variable": "VariableOblique",
    "VariableSlanted-Variable": "VariableSlanted",
    "VariableUpright-Variable": "VariableUpright",
    "VariableRoman-Variable": "VariableRoman",
    "Variable-Variable": "Variable",
    "Variable-Italic": "VariableItalic",
    "Variable-Oblique": "VariableOblique",
    "Variable-Slanted": "VariableSlanted",
    "Variable-Upright": "VariableUpright",
    "Variable-Roman": "VariableRoman",
    "VariableVariable": "Variable",
    "VariableSlanted": "Variable Slanted",
    "VariableUpright": "Variable Upright",
    "VariableRoman": "Variable Roman",
    # Common substitutions
    # "Roman": "Regular",  # REMOVED - Roman is a valid regular-equivalent
    "Round-ed": "Rounded",
    "Slant-ed": "Slanted",
    "Semi-Mono": "SemiMono",
    "Semi Mono": "SemiMono",
    "Semimono": "SemiMono",
    # Cleanup
    "--": "-",
}


# ================================================================================================
# 2. BASE TERMS FOR PATTERN GENERATION (~50 entries)
# ================================================================================================
# Complete words used for truncation detection and pattern matching

STYLE_WORDS = {
    # Weights
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
    "Super",
    "Ultra",
    # Numeric weights
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
    # Widths
    "Compressed",
    "Condensed",
    "Compact",
    "Narrow",
    "Tight",
    "Extended",
    "Expanded",
    "Wide",
    "SemiCompressed",
    "ExtraCompressed",
    "UltraCompressed",
    "SemiCondensed",
    "ExtraCondensed",
    "UltraCondensed",
    "SemiCompact",
    "ExtraCompact",
    "UltraCompact",
    "SemiNarrow",
    "ExtraNarrow",
    "UltraNarrow",
    "SemiExtended",
    "ExtraExtended",
    "UltraExtended",
    "SemiExpanded",
    "ExtraExpanded",
    "UltraExpanded",
    "SemiWide",
    "ExtraWide",
    "UltraWide",
    # Slopes
    "Italic",
    "Oblique",
    "Slanted",
    "Slant",
    "Inclined",
    "Backslanted",
    "Backslant",
    "Reverse",
    "Retalic",
    # Optical Sizes
    "Caption",
    "Display",
    "Text",
    "Poster",
    "Headline",
    "Subhead",
    "Title",
    "Titling",
    "Deck",
    "Micro",
    "Banner",
    "Fine",
    "Large",
    "Small",
    "Big",
    "Tall",
    # Other
    "Rounded",
    "Round",
    "Mono",
    "Monospace",
    "Variable",
    "Smallcaps",
    "Unicase",
    "Capitals",
    # Decorative
    "Rough",
    "Vintage",
    "Antique",
    "Shaded",
    "Shadow",
    "Line",
    "Inline",
    "DoubleLine",
    "Monoline",
    "Printed",
    "Pressed",
    "Distressed",
}


# ================================================================================================
# 3. BASE TERMS FOR PROGRAMMATIC VARIATION GENERATION
# ================================================================================================

WIDTH_BASES = {
    "Condensed",
    "Compressed",
    "Compact",
    "Narrow",
    "Tight",
    "Extended",
    "Expanded",
    "Wide",
}

OPTICAL_BASES = {
    "Caption",
    "Display",
    "Text",
    "Poster",
    "Headline",
    "Subhead",
    "Title",
    "Deck",
    "Micro",
    "Banner",
}

# Core slope vocabulary used by filename tools, NameID slope inference, and audits.
# Keep aligned with Filename_Tools/FNT_ReorderSlopes.BASE_SLOPE_TERMS (+ common aliases).
SLOPE_BASES = {
    "Italic",
    "Oblique",
    "Slanted",
    "Slant",
    "Inclined",
    "Backslanted",
    "Backslant",
    "Reverse",
    "Retalic",
    "Cursive",
    "Kursiv",
}

# Multi-word slopes that count as one unit (not "double slope").
ACCEPTABLE_COMPOUND_SLOPES: frozenset[str] = frozenset(
    {
        "Reverse Italic",
        "Reverse Slanted",
    }
)

# Slopes retained in display names (ID4) and ID17 when present in filename
KNOWN_VF_SLOPES: frozenset[str] = frozenset(SLOPE_BASES) | ACCEPTABLE_COMPOUND_SLOPES

# Lowercase tokens that mean "this name already has a slope" (do not append Italic).
# Includes spaced "Back Slanted" as seen in some filenames/parsed subfamilies.
ITALIC_LIKE_SLOPE_TERMS: frozenset[str] = frozenset(
    {s.lower() for s in SLOPE_BASES}
    | {s.lower() for s in ACCEPTABLE_COMPOUND_SLOPES}
    | {"back slanted"}
)

# Filename pairing markers: elide from ID1/ID4/ID17 (like Regular).
# Used when a family has separate upright/roman and italic VF files.
ELIDABLE_VF_FILENAME_SLOPES: frozenset[str] = frozenset({"Upright", "Roman"})

MODIFIERS = {"Semi", "Demi", "Extra", "Ultra", "Super"}

# Weight words that take a modifier (Semi + Bold, Extra + Light, ...). House style: weight compounds are
# one word (Semibold, Extralight), and width compounds are joined with their capitals (SemiCondensed).
WEIGHT_BASES = {"Hairline", "Thin", "Light", "Medium", "Bold", "Black", "Heavy"}


# ================================================================================================
# 4. HYPHEN PLACEMENT RULES (~30 entries)
# ================================================================================================

# Terms that prefer hyphen on the LEFT (Foo-Condensed)
HYPHEN_LEFT_TERMS = {
    # Widths
    "Condensed",
    "SemiCondensed",
    "ExtraCondensed",
    "UltraCondensed",
    "Compressed",
    "SemiCompressed",
    "ExtraCompressed",
    "UltraCompressed",
    "Compact",
    "SemiCompact",
    "ExtraCompact",
    "UltraCompact",
    "Narrow",
    "SemiNarrow",
    "ExtraNarrow",
    "UltraNarrow",
    "Expanded",
    "SemiExpanded",
    "ExtraExpanded",
    "UltraExpanded",
    "Extended",
    "SemiExtended",
    "ExtraExtended",
    "UltraExtended",
    "Wide",
    "SemiWide",
    "ExtraWide",
    "UltraWide",
    # Variable-font marker
    "Variable",
}

# Terms that prefer hyphen on the RIGHT (Display-)
HYPHEN_RIGHT_TERMS = {
    # Optical sizes and non-WWS terms
    "Display",
    "Text",
    "Caption",
    "Subhead",
    "Headline",
    "Title",
    "Poster",
    "Deck",
    "Micro",
    "Round",
    "Mono",
    "Sans",
    "Slab",
    "Serif",
}

# Prefixes that should NOT have hyphen after them (Semi-Bold → SemiBold)
COMPOUND_PREFIXES = {"Semi", "Extra", "Demi", "Ultra", "Super", "X"}


# ================================================================================================
# 5. PATTERN GENERATION HELPERS
# ================================================================================================


def generate_all_width_variations() -> set[str]:
    """
    Generate all width term variations programmatically.

    Returns ~80+ variations from 8 base terms:
    - Base terms (Condensed, Wide, etc.)
    - Modifier + base (SemiCondensed, ExtraWide, etc.)
    - X variations (XCondensed, XXCondensed, XXXCondensed, etc.)
    """
    variations = set(WIDTH_BASES)

    # Add modifier combinations
    for mod in MODIFIERS:
        for base in WIDTH_BASES:
            variations.add(f"{mod}{base}")

    # Add X variations (1-7 X's)
    for base in WIDTH_BASES:
        for x_count in range(1, 8):
            variations.add(f"{'X' * x_count}{base}")

    return variations


def generate_all_optical_variations() -> set[str]:
    """
    Generate all optical size term variations.

    Returns base optical terms (no modifiers typically used with optical sizes).
    """
    return set(OPTICAL_BASES)


# Words that fill the weight position of a style name. "Roman" and the regular equivalents count: they are
# how a family names its upright, so nothing needs adding. "Text" is a weight only when it stands alone;
# next to another word it is an optical size (see has_weight_word).
WEIGHT_WORDS = frozenset(
    {
        "hairline", "thin", "extralight", "ultralight", "light", "semilight", "demilight",
        "book", "regular", "normal", "plain", "standard", "roman", "medium",
        "demibold", "semibold", "bold", "extrabold", "ultrabold", "superbold",
        "black", "heavy", "extrablack", "ultrablack", "extraheavy", "ultraheavy", "fat", "super", "ultra",
    }
    | {f"{m.lower()}{w.lower()}" for m in MODIFIERS for w in WEIGHT_BASES}
    | {str(n) for n in range(100, 1100, 100)}
)


def has_weight_word(text: str) -> bool:
    """True when the style text has a word in the weight position (Bold, Semibold, Regular, Roman, 400...).

    `Text` counts only when it is the only word: with another word it is an optical size.
    """
    tokens = [t.lower() for t in (text or "").split()]
    if not tokens:
        return False
    if tokens == ["text"]:
        return True
    return any(t in WEIGHT_WORDS for t in tokens)


_WIDTH_LOWER = None


def _width_lower() -> frozenset:
    global _WIDTH_LOWER
    if _WIDTH_LOWER is None:
        _WIDTH_LOWER = frozenset(
            {w.lower() for w in WIDTH_BASES}
            | {f"{m}{w}".lower() for m in MODIFIERS for w in WIDTH_BASES}
            | {f"{'x' * n}{w}".lower() for n in range(1, 8) for w in WIDTH_BASES}
        )
    return _WIDTH_LOWER


def unknown_weight_candidates(text: str) -> list:
    """Words in a style that has no weight word and are neither a width nor Italic.

    Such a word may already be the weight under a name the tool does not know (Air, Fine...), so a
    weight is not added next to it. Empty when the style has a weight word or holds only width/Italic words.
    """
    text = " ".join((text or "").split())
    if not text or has_weight_word(text):
        return []
    return [t for t in text.split() if t.lower() != "italic" and t.lower() not in _width_lower()]


def ensure_weight_word(text: str) -> str:
    """Add Regular when the style has no weight word and nothing in it could be one.

    Added only when the style is empty or holds only width and Italic words: "Wide Italic" ->
    "Wide Regular Italic", "Condensed" -> "Condensed Regular". When the style has any other word
    (Air, Oblique, Circle...) it is returned unchanged: that word may be the weight, and guessing
    would label it Regular. Safe to run twice.
    """
    text = " ".join((text or "").split())
    if not text:
        return "Regular"
    if has_weight_word(text) or unknown_weight_candidates(text):
        return text
    tokens = text.split()
    if tokens[-1].lower() == "italic":
        tokens.insert(len(tokens) - 1, "Regular")
    else:
        tokens.append("Regular")
    return " ".join(tokens)


WEIGHT_POSITIONS = ("last", "first")


def apply_weight_override(text: str, word: str | None, position: str = "last") -> str:
    """Put a weight the user named into a style that has none.

    - word None: nothing changes.
    - word "none" (any case): nothing changes; the user is saying the style has no weight to add.
    - Only a style with no weight word is touched. A style that already names a weight keeps it.
    - position "last" (default): before a trailing Italic, otherwise at the end ("Carved" -> "Carved Light").
      position "first": at the front ("Carved" -> "Light Carved"), for bespoke words that follow the weight.
    Safe to run twice: once the word is in, the style has a weight word.
    """
    text = " ".join((text or "").split())
    word = (word or "").strip()
    if not word or word.lower() == "none" or has_weight_word(text):
        return text
    word = join_style_compounds(word)
    tokens = text.split()
    if position == "first":
        tokens.insert(0, word)
    elif tokens and tokens[-1].lower() == "italic":
        tokens.insert(len(tokens) - 1, word)
    else:
        tokens.append(word)
    return " ".join(tokens)


def join_style_compounds(text: str) -> str:
    """Join the pieces of compound style words that PascalCase splitting pulled apart.

    - modifier + weight is one word, second part lowercase: "Semi Bold" -> "Semibold",
      "Extra Light" -> "Extralight"
    - modifier + width (or X-run + width) is joined with capitals kept: "Semi Condensed" ->
      "SemiCondensed", "XX Condensed" -> "XXCondensed"
    - "Small Caps" -> "Smallcaps" (the one exception to keeping capitals)

    Everything else is left alone, so the function is safe to run twice. Apply it to a style
    (subfamily) string, not to a family name.
    """
    if not text:
        return text
    tokens = text.split()
    out = []
    i = 0
    modifiers = {m.lower() for m in MODIFIERS}
    weights = {w.lower() for w in WEIGHT_BASES}
    widths = {w.lower() for w in WIDTH_BASES}
    while i < len(tokens):
        token = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if nxt is not None:
            low, nxt_low = token.lower(), nxt.lower()
            is_modifier = low in modifiers
            is_x_run = bool(re.fullmatch(r"X{1,7}", token))
            if is_modifier and nxt_low in weights:
                out.append(token.capitalize() + nxt_low)
                i += 2
                continue
            if is_modifier and nxt_low == "mono":
                # Semi + Mono is one word wherever it appears (SemiMono), like the width compounds
                out.append(token.capitalize() + nxt.capitalize())
                i += 2
                continue
            if (is_modifier or is_x_run) and nxt_low in widths:
                first = token if is_x_run else token.capitalize()
                out.append(first + nxt.capitalize())
                i += 2
                continue
            if low == "small" and nxt_low == "caps":
                out.append("Smallcaps")
                i += 2
                continue
        out.append(token)
        i += 1
    return " ".join(out)


def join_mono_compounds(text: str) -> str:
    """Join "Semi Mono" (modifier + Mono) into "SemiMono" and leave everything else alone.

    Used on family names, where the other style compounds must not be touched.
    """
    tokens = (text or "").split()
    modifiers = {m.lower() for m in MODIFIERS}
    out, i = [], 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i].lower() in modifiers and tokens[i + 1].lower() == "mono":
            out.append(tokens[i].capitalize() + "Mono")
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out)


# Cache generated variations at module load time
ALL_WIDTH_TERMS = generate_all_width_variations()
ALL_OPTICAL_TERMS = generate_all_optical_variations()


# ================================================================================================
# MODULE INFO
# ================================================================================================

DICTIONARY_VERSION = "1.0.0"

if __name__ == "__main__":
    print(f"Font Style Dictionaries v{DICTIONARY_VERSION}")
    print(f"\nCompound normalizations: {len(COMPOUND_NORMALIZATIONS)}")
    print(f"Style words: {len(STYLE_WORDS)}")
    print(f"Width base terms: {len(WIDTH_BASES)}")
    print(f"Generated width variations: {len(ALL_WIDTH_TERMS)}")
    print(f"Optical base terms: {len(OPTICAL_BASES)}")
    print(f"Generated optical variations: {len(ALL_OPTICAL_TERMS)}")
    print(f"Hyphen left terms: {len(HYPHEN_LEFT_TERMS)}")
    print(f"Hyphen right terms: {len(HYPHEN_RIGHT_TERMS)}")
    print(f"Compound prefixes: {len(COMPOUND_PREFIXES)}")
    print(
        f"\nTotal dictionary entries: ~{len(COMPOUND_NORMALIZATIONS) + len(STYLE_WORDS) + len(WIDTH_BASES) + len(OPTICAL_BASES) + len(HYPHEN_LEFT_TERMS) + len(HYPHEN_RIGHT_TERMS)}"
    )
    print(
        f"Total pattern matches: {len(ALL_WIDTH_TERMS) + len(ALL_OPTICAL_TERMS) + len(STYLE_WORDS)}"
    )
