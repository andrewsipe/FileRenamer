#!/usr/bin/env python3
"""
Centralized font style dictionaries for filename processing.

Philosophy: Minimal dictionaries for deterministic outcomes and ambiguity resolution.
Pattern matching handles variations programmatically.
"""

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

# Terms that suggest elidable=True when used as a weight name (STAT AxisValue names)
ELIDABLE_WEIGHT_NAMES: frozenset[str] = frozenset(
    {
        "Regular",
        "Normal",
        "Roman",
        "Plain",
        "Standard",
        "Book",
        "Text",
    }
)

# Terms that suggest elidable=True when used as a width name
ELIDABLE_WIDTH_NAMES: frozenset[str] = frozenset(
    {
        "Normal",
        "Regular",
    }
)


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
    "Banner",
    "Round",
    "Rounded",
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
