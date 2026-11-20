"""
core_filename_parts_parser: Split font filename strings into Family/Subfamily and tokenize PascalCase.

Goals:
- Split input at the first hyphen into family and subfamily raw parts.
- Convert each raw part from PascalCase into human-readable words, keeping all-caps runs
  together when they are not followed by a lowercase (e.g., "KWAKGrotesk" -> "KWAK Grotesk").
- Underscores act as additional breakpoint separators alongside PascalCase boundaries
  (e.g., "ABC_EFGRegular" -> "ABC EFG Regular").
- Segments starting with lowercase preserve their original casing (e.g., "oook-Regular" -> "oook Regular").
- Digit sequences followed by lowercase-only tokens are merged without spaces, preserving lowercase
  (e.g., "35mm" stays "35mm" not "35 Mm").

This module is designed to be imported by other name-related scripts.

Examples (doctests):
>>> format_pascal_words("QueenSansExtra")
'Queen Sans Extra'
>>> split_family_subfamily("QueenSansExtra-ExtralightItalic")
('Queen Sans Extra', 'Extralight Italic')
>>> family_from_filename("KWAKGrotesk-ExtraBold")
'KWAK Grotesk'
>>> parse_filename("QueenSansExtra-ExtralightItalic").family
'Queen Sans Extra'
>>> parse_filename("QueenSansExtra-ExtralightItalic").subfamily
'Extralight Italic'

Edge cases:
>>> format_pascal_words("UIUXKit")
'UIUX Kit'
>>> format_pascal_words("ABCD")
'ABCD'
>>> split_family_subfamily("NoHyphenName")
('No Hyphen Name', '')

Underscore breakpoints:
>>> format_pascal_words("ABC_EFG")
'ABC EFG'
>>> format_pascal_words("ABC_EFGRegular")
'ABC EFG Regular'
>>> split_family_subfamily("ABC_EFG-Regular")
('ABC EFG', 'Regular')
>>> format_pascal_words("_ABC_EFG_")
'ABC EFG'

Lowercase preservation:
>>> format_pascal_words("oook")
'oook'
>>> split_family_subfamily("oook-Regular")
('oook', 'Regular')
>>> format_pascal_words("ABC_oook")
'ABC oook'
>>> format_pascal_words("oookABC")
'oook ABC'

Digit-lowercase merging:
>>> format_pascal_words("35mm")
'35mm'
>>> format_pascal_words("10px")
'10px'
>>> split_family_subfamily("OTMissouri-35mm")
('OT Missouri', '35mm')
>>> format_pascal_words("35mm50px")
'35mm 50px'

Ampersand and single uppercase letters:
>>> format_pascal_words("FitU&lc")
'Fit U & lc'
>>> split_family_subfamily("FitU&lc-ExtraWide")
('Fit U & lc', 'Extra Wide')
>>> format_pascal_words("Condensed&Wide")
'Condensed & Wide'
>>> split_family_subfamily("Cougar-Condensed&SuperWide")
('Cougar', 'Condensed & Super Wide')

CFF2 extension handling (internal format info is ignored):
>>> split_family_subfamily("Helvetica-Regular.CFF2.otf")
('Helvetica', 'Regular')
>>> split_family_subfamily("Arial-Bold.CFF2.ttf")
('Arial', 'Bold')

Exclamation point handling (trailing space when followed by content):
>>> format_pascal_words("Font!Name")
'Font! Name'
>>> format_pascal_words("Font!NameRegular")
'Font! Name Regular'
>>> split_family_subfamily("Font!Name-Bold")
('Font! Name', 'Bold')
>>> format_pascal_words("ABC_EFG!")
'ABC EFG!'
>>> format_pascal_words("Font!_Name")
'Font! Name'

Asterisk handling (no spaces around asterisk):
>>> format_pascal_words("Font*Name")
'Font*Name'
>>> format_pascal_words("Font*NameRegular")
'Font*Name Regular'
>>> split_family_subfamily("Font*Name-Bold")
('Font*Name', 'Bold')
>>> format_pascal_words("ABC_EFG*")
'ABC EFG*'

Leading-space characters:
>>> format_pascal_words("Font(Condensed")
'Font (Condensed'
>>> format_pascal_words("Font{Condensed")
'Font {Condensed'
>>> format_pascal_words("Font[Condensed")
'Font [Condensed'

Dollar sign handling (numeric attachment rules):
>>> format_pascal_words("Price$19.99")
'Price $19.99'
>>> format_pascal_words("Price19.99$")
'Price 19.99$'
>>> format_pascal_words("$100")
'$100'

Percent sign handling (numeric attachment rules):
>>> format_pascal_words("100%Success")
'100% Success'
>>> format_pascal_words("50%")
'50%'

Forbidden character replacement (tested at higher level via _normalize_basename):
>>> from FontCore.core_filename_parts_parser import split_family_subfamily
>>> split_family_subfamily("Font:Name.ttf")
('Font Name', '')
>>> # Backslash needs to be escaped in Python strings
>>> split_family_subfamily("Font" + chr(92) + "Name.ttf")  # chr(92) is backslash
('Font Name', '')

Demo and Testing:
    Run 'python CoreDemoTool.py parser --help' to see examples and test with real files.
    The demo tool shows PascalCase formatting and filename parsing capabilities.

Maintenance Note:
    When adding new parsing functions to this module, update CoreDemoTool.py to showcase
    the new functionality in the 'parser' subcommand.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Tuple


# PascalCase tokenization pattern components:
# 1. Acronym runs before CamelCase: "XMLHttp" -> "XML", "Http"
# 2. Single capital + number (not preceded by capital): "G1", "H2", "B1.2.3" -> kept together
# 3. Decimal numbers: "0.1", "1.2", "2.3" -> preserved with period
# 4. Standard capitalized words: "QueenSans" -> "Queen", "Sans"
# 5. All-caps tokens: "ABCD" -> "ABCD"
# 6. Numeric sequences: "UI2Kit" -> "UI", "2", "Kit"
# 7. Special characters: "&", "!", "*", "(", "{", "[", "]", "}", "?", ",", ";", "$", "%", "@" are preserved as separate tokens
_ACRONYM_BEFORE_CAMEL = r"[A-Z]+(?=[A-Z][a-z])"  # XML in XMLHttp
# Single capital letter followed by digits/decimal (not preceded by capital to avoid matching in "ABC123")
_SINGLE_CAP_NUMBER = (
    r"(?<![A-Z])[A-Z]\d+(?:\.\d+)*"  # G1, H2, B1.2.3 (but not "C123" in "ABC123")
)
# Multi-part decimal numbers (1.2.3, 2.3.4) - must have at least two periods
_DECIMAL_NUMBER = r"\d+(?:\.\d+)+"  # 0.1, 1.2, 1.2.3, 2.3.4.5
# Capitalized word, non-greedy to stop before capital, single-cap-number, digit, or special characters
_CAPITALIZED_WORD = r"[A-Z]?[a-z]+?(?=[A-Z]|(?<![A-Z])[A-Z]\d|\d|[&!*(){}\[\]?,;$\-%@']|$)"  # Queen, Sans, ttp (stops before capital+number, digits, or special characters including apostrophe
_ALL_CAPS_SEQUENCE = r"[A-Z]{2,}"  # KWAK, UI (but not single letters like "R")
_SINGLE_UPPERCASE = r"[A-Z]"  # Single uppercase letter (e.g., "U" in "FitU&lc")
_DIGITS = r"\d+"  # 2, 10, 123
_AMPERSAND = r"&"  # Preserved as separate token
_EXCLAMATION = r"!"  # Preserved as separate token
_ASTERISK = r"\*"  # Preserved as separate token (escaped for regex)
_LEFT_PAREN = r"\("  # Preserved as separate token (escaped for regex)
_LEFT_BRACE = r"\{"  # Preserved as separate token (escaped for regex)
_LEFT_BRACKET = r"\["  # Preserved as separate token (escaped for regex)
_RIGHT_BRACKET = r"\]"  # Preserved as separate token (escaped for regex)
_RIGHT_BRACE = r"\}"  # Preserved as separate token (escaped for regex)
_QUESTION = r"\?"  # Preserved as separate token (escaped for regex)
_COMMA = r","  # Preserved as separate token
_SEMICOLON = r";"  # Preserved as separate token
_DOLLAR = r"\$"  # Preserved as separate token (escaped for regex)
_PERCENT = r"%"  # Preserved as separate token
_AT = r"@"  # Preserved as separate token

_TOKEN_PATTERN = re.compile(
    f"{_ACRONYM_BEFORE_CAMEL}|{_SINGLE_CAP_NUMBER}|{_DECIMAL_NUMBER}|{_CAPITALIZED_WORD}|{_ALL_CAPS_SEQUENCE}|{_SINGLE_UPPERCASE}|{_DIGITS}|{_AMPERSAND}|{_EXCLAMATION}|{_ASTERISK}|{_LEFT_PAREN}|{_LEFT_BRACE}|{_LEFT_BRACKET}|{_RIGHT_BRACKET}|{_RIGHT_BRACE}|{_QUESTION}|{_COMMA}|{_SEMICOLON}|{_DOLLAR}|{_PERCENT}|{_AT}"
)

# Character class constants for spacing rules
NO_SPACE = {"*", "@", "'"}  # Apostrophe has no spacing (attaches to adjacent words)
LEADING_SPACE = {"(", "{", "["}
TRAILING_SPACE = {"!", "?", ",", ";", ")", "]", "}"}
SPECIAL_MONEY_PERCENT = {"$", "%"}


def sanitize_forbidden_characters(name: str) -> str:
    """Replace macOS-forbidden characters with undersFontCore."""
    return name.replace(":", "_").replace("\\", "_")


def _normalize_basename(input_name: str, strip_extension: bool = True) -> str:
    """Convert input to a bare filename optionally without extension.

    Strips .CFF2 from filenames like 'Helvetica-Regular.CFF2.otf' before processing,
    treating it as internal information only.
    """
    candidate = os.path.basename(input_name)
    candidate = sanitize_forbidden_characters(candidate)

    # Remove .CFF2 if present (internal format info, not naming relevant)
    # Handle both .CFF2.otf and .CFF2.ttf cases
    if ".CFF2." in candidate:
        # Replace .CFF2. with . to remove the internal format marker
        candidate = candidate.replace(".CFF2.", ".")
    elif candidate.endswith(".CFF2"):
        # Remove .CFF2 if it's the only extension
        candidate = candidate[:-5]

    return os.path.splitext(candidate)[0] if strip_extension else candidate


def tokenize_pascal_case(value: str) -> List[str]:
    """Tokenize a PascalCase or MixedCase string into component tokens.

    Uses rules that keep consecutive capitals together when not followed by a lowercase
    letter, so "KWAKGrotesk" -> ["KWAK", "Grotesk"].

    Preserves all characters not explicitly matched by patterns, including unicode
    and special characters like apostrophes.
    """
    if not value:
        return []

    tokens = []
    last_end = 0

    # Find all matches with their positions
    for match in _TOKEN_PATTERN.finditer(value):
        # Add any unmatched characters before this match
        if match.start() > last_end:
            unmatched = value[last_end : match.start()]
            tokens.append(unmatched)

        # Add the matched token
        tokens.append(match.group())
        last_end = match.end()

    # Add any remaining unmatched characters at the end
    if last_end < len(value):
        tokens.append(value[last_end:])

    return tokens


def _format_token(token: str) -> str:
    """Format a single token for display.

    All-uppercase tokens are preserved (e.g., "UI", "KWAK").
    All-lowercase tokens are preserved (e.g., "lc", "px").
    Single capital + number patterns (G1, V1, B1.2.3) preserve their original casing.
    Special characters are preserved as-is.
    Other tokens are converted to Title Case.
    """
    if not token:
        return token
    # Preserve special characters as-is
    if token in ("!", "*", "&", "(", "{", "[", "]", "}", "?", ",", ";", "$", "%", "@"):
        return token
    if token.isupper():
        return token
    if token.islower():
        return token
    # Preserve casing for single capital + number patterns (G1, H2, V1, B1.2.3, etc.)
    if re.match(r"^[A-Z]\d", token):
        return token
    return token[0].upper() + token[1:].lower()


def _is_numeric_token(token: str) -> bool:
    """Check if token is purely numeric (digits only, including decimals)."""
    if not token:
        return False
    # Check if token is all digits or matches decimal pattern
    return token.isdigit() or bool(re.match(r"^\d+(?:\.\d+)+$", token))


def _is_alphanumeric_token(token: str) -> bool:
    """Check if token starts with a letter (alphanumeric word)."""
    if not token:
        return False
    return bool(token) and token[0].isalpha()


def apply_spacing_rules(
    tokens: List[str], segment_boundaries: List[int] = None
) -> List[str]:
    """Apply contextual spacing rules to formatted tokens.

    Contextual spacing principle: Only add spaces when adjacent tokens exist
    (no spaces at start/end).

    Rules:
    - Leading-space chars ((, {, [): Add space before ONLY if previous token exists and is alphanumeric
    - Trailing-space chars (!, ?, ,, ;, ), ], }): Add space after ONLY if next token exists and is not punctuation requiring no gap
    - %: No space before if previous token exists and is numeric; space after ONLY if next token exists and starts with letter/word
    - $: No space after if next token exists and is numeric; no space before if previous token exists and is numeric;
         otherwise treat as leading-space char (space before ONLY if previous token exists and is alphanumeric)
    - * and @: No spacing on either side, UNLESS crossing a segment boundary (underscore separator)
    - Regular tokens: Space before if previous token exists and allows spacing
    - Segment boundaries: Always add space between segments separated by underscores
    """
    if not tokens:
        return []

    # Filter out empty tokens
    tokens = [t for t in tokens if t]
    if not tokens:
        return []

    # Normalize segment_boundaries to a set for fast lookup
    segment_boundary_set = set(segment_boundaries) if segment_boundaries else set()

    result = []

    for i, token in enumerate(tokens):
        prev_token = tokens[i - 1] if i > 0 else None
        next_token = tokens[i + 1] if i + 1 < len(tokens) else None
        is_segment_start = i in segment_boundary_set

        # Determine if we need space before this token
        needs_space_before = False
        if prev_token:
            # Special case: if this is the start of a new segment, always add space
            # (underscores should create word breaks)
            if is_segment_start:
                needs_space_before = True
            elif token in NO_SPACE:
                # No-space chars never get space before (unless segment boundary)
                needs_space_before = False
            elif token in TRAILING_SPACE:
                # Trailing-space chars never get space before (space goes after them)
                needs_space_before = False
            elif token == "$":
                # $: no space before if previous is numeric, otherwise space if previous is alphanumeric
                if not _is_numeric_token(prev_token) and _is_alphanumeric_token(
                    prev_token
                ):
                    needs_space_before = True
            elif token == "%":
                # %: no space before if previous is numeric, otherwise space if previous is alphanumeric
                if not _is_numeric_token(prev_token) and _is_alphanumeric_token(
                    prev_token
                ):
                    needs_space_before = True
            elif token in LEADING_SPACE:
                # Leading-space chars: space before if previous is alphanumeric
                if _is_alphanumeric_token(prev_token):
                    needs_space_before = True
            elif prev_token in NO_SPACE:
                # Previous is no-space char, no space (unless segment boundary handled above)
                needs_space_before = False
            elif prev_token in TRAILING_SPACE:
                # Previous is trailing-space char, already has space after it
                needs_space_before = False
            elif prev_token in LEADING_SPACE:
                # Previous is leading-space char, no space after it
                needs_space_before = False
            elif prev_token == "$" and _is_numeric_token(token):
                # $ before numeric, no space
                needs_space_before = False
            elif prev_token == "%":
                # % already handled its trailing space
                needs_space_before = False
            else:
                # Regular case: space before if previous token exists
                needs_space_before = True

        # Determine if we need space after this token
        needs_space_after = False
        if next_token:
            if token in NO_SPACE:
                # No-space chars never get space after
                needs_space_after = False
            elif token == "$":
                # $: no space after if next is numeric
                if not _is_numeric_token(next_token):
                    # But we don't add space after $ unless it's a trailing-space case
                    needs_space_after = False
            elif token == "%":
                # %: space after if next starts with letter/word
                if _is_alphanumeric_token(next_token):
                    needs_space_after = True
            elif token in TRAILING_SPACE:
                # Trailing-space chars: space after if next is not no-space
                if next_token not in NO_SPACE:
                    needs_space_after = True
            else:
                # Regular tokens don't add space after themselves
                needs_space_after = False

        # Build the token with spacing
        spaced_token = ""
        if needs_space_before:
            spaced_token += " "
        spaced_token += token
        if needs_space_after:
            spaced_token += " "

        result.append(spaced_token)

    return result


def format_pascal_words(value: str) -> str:
    """Convert a PascalCase/mixed string into space-separated words.

    Example: "QueenSansExtra" -> "Queen Sans Extra".
    Acronym runs are preserved: "UIUXKit" -> "UIUX Kit".
    Underscores act as additional breakpoint separators: "ABC_EFGRegular" -> "ABC EFG Regular".
    Segments starting with lowercase preserve their original casing: "oook-Regular" -> "oook Regular".
    Digit sequences followed by lowercase-only tokens are merged: "35mm" stays "35mm" not "35 Mm".
    Special characters have contextual spacing rules applied (leading-space, trailing-space, no-space, etc.).
    """
    if not value:
        return ""

    # Split by underscores and filter out empty segments
    segments = [seg for seg in value.split("_") if seg]

    # Tokenize each segment and format tokens
    formatted_tokens = []
    segment_boundaries = []  # Track indices where new segments start
    for segment_idx, segment in enumerate(segments):
        segment_tokens = tokenize_pascal_case(segment)

        # Merge digit sequences followed by lowercase-only tokens
        merged_tokens = []
        i = 0
        while i < len(segment_tokens):
            current_token = segment_tokens[i]
            # Check if current token is all digits and next token exists and is all lowercase
            if (
                current_token.isdigit()
                and i + 1 < len(segment_tokens)
                and segment_tokens[i + 1].islower()
            ):
                # Merge: digit token + lowercase token (preserve lowercase, never format)
                merged_token = current_token + segment_tokens[i + 1]
                merged_tokens.append((merged_token, True))  # True = preserve casing
                i += 2  # Skip both tokens
            else:
                merged_tokens.append((current_token, False))  # False = format normally
                i += 1

        # Format tokens based on segment start case and merge status
        segment_formatted = []
        if segment and segment[0].islower():
            # Preserve original casing for all tokens in this segment
            for token, is_merged in merged_tokens:
                segment_formatted.append(token)
        else:
            # Format tokens normally, except merged digit+lowercase tokens
            for token, is_merged in merged_tokens:
                if is_merged:
                    # Merged digit+lowercase: preserve casing
                    segment_formatted.append(token)
                else:
                    # Normal formatting (preserves all-caps, Title Cases others)
                    segment_formatted.append(_format_token(token))

        # Mark segment boundary (first token of this segment)
        if segment_formatted:
            segment_boundaries.append(len(formatted_tokens))
        formatted_tokens.extend(segment_formatted)

    # Apply spacing rules to the combined token list, with segment boundary information
    spaced_tokens = apply_spacing_rules(
        formatted_tokens, segment_boundaries=segment_boundaries
    )

    # Join with empty string since spacing is now explicit in tokens
    return "".join(spaced_tokens).strip()


def _split_at_hyphen(base: str) -> Tuple[str, str]:
    """Split filename at first hyphen into left and right parts."""
    if "-" not in base:
        return base, ""
    return base.split("-", 1)


def split_family_subfamily(
    input_name: str, *, strip_extension: bool = True
) -> Tuple[str, str]:
    """Split an input filename into formatted Family and Subfamily parts.

    - Split at the first hyphen only
    - Family is the left side, Subfamily is the right side (or empty if no hyphen)
    - Each side is converted to space-separated words respecting acronym rules

    Returns a tuple: (family, subfamily)
    """
    base = _normalize_basename(input_name, strip_extension=strip_extension)
    left, right = _split_at_hyphen(base)

    family = format_pascal_words(left)
    subfamily = format_pascal_words(right) if right else ""

    return family, subfamily


def family_from_filename(input_name: str, *, strip_extension: bool = True) -> str:
    """Extract just the formatted family part from an input filename."""
    family, _ = split_family_subfamily(input_name, strip_extension=strip_extension)
    return family


def subfamily_from_filename(input_name: str, *, strip_extension: bool = True) -> str:
    """Extract just the formatted subfamily part from an input filename.

    Note: Further subfamily analysis (weight/width/style parsing) can be built on top of
    this base segmentation in future iterations.
    """
    _, subfamily = split_family_subfamily(input_name, strip_extension=strip_extension)
    return subfamily


@dataclass(frozen=True)
class ParsedName:
    original: str
    base: str
    family_raw: str
    subfamily_raw: str
    family: str
    subfamily: str


def parse_filename(input_name: str, *, strip_extension: bool = True) -> ParsedName:
    """Parse a font filename into structured parts.

    Returns a ParsedName containing both raw and formatted components.
    """
    base = _normalize_basename(input_name, strip_extension=strip_extension)
    left, right = _split_at_hyphen(base)

    return ParsedName(
        original=input_name,
        base=base,
        family_raw=left,
        subfamily_raw=right,
        family=format_pascal_words(left),
        subfamily=format_pascal_words(right) if right else "",
    )


__all__ = [
    "tokenize_pascal_case",
    "format_pascal_words",
    "split_family_subfamily",
    "family_from_filename",
    "subfamily_from_filename",
    "ParsedName",
    "parse_filename",
]
