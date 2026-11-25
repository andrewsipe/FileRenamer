#!/usr/bin/env python3
"""
Font Extension Validation Utility

Provides functions to detect actual font format from file content and validate/fix
file extensions to match the actual format.

Features:
- Detects font format from magic bytes (TTF, OTF, WOFF, WOFF2, TTC)
- Validates file extensions match actual format
- Auto-fixes mismatched extensions with duplicate handling
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple


def detect_font_format(filepath: Path | str) -> str:
    """
    Determine the actual font type by reading the file's magic bytes.

    Args:
        filepath: Path to the font file

    Returns:
        'TTF' for TrueType fonts
        'OTF' for OpenType fonts (with CFF outlines)
        'WOFF' for Web Open Font Format
        'WOFF2' for Web Open Font Format 2
        'TTC' for TrueType Collection
        'UNKNOWN' for unrecognized formats
        'ERROR' if file cannot be read
    """
    try:
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            # Read the first 4 bytes
            header = f.read(4)

            if len(header) < 4:
                return "UNKNOWN"

            # Check for various font signatures
            # TrueType: starts with 0x00010000 or 'true' or 'typ1'
            if header == b"\x00\x01\x00\x00" or header == b"true" or header == b"typ1":
                return "TTF"

            # OpenType with CFF: starts with 'OTTO'
            elif header == b"OTTO":
                return "OTF"

            # WOFF: starts with 'wOFF'
            elif header == b"wOFF":
                return "WOFF"

            # WOFF2: starts with 'wOF2'
            elif header == b"wOF2":
                return "WOFF2"

            # TrueType Collection
            elif header == b"ttcf":
                return "TTC"

            else:
                return "UNKNOWN"

    except Exception:
        return "ERROR"


def validate_and_fix_extension(
    filepath: Path, auto_fix: bool = True
) -> Tuple[bool, Optional[Path]]:
    """
    Validate that file extension matches actual font format, and optionally fix it.

    Args:
        filepath: Path to the font file to validate
        auto_fix: If True, automatically rename file to correct extension if mismatch found

    Returns:
        Tuple of (is_valid, fixed_path_or_none):
        - is_valid: True if extension matches format, False if mismatch
        - fixed_path: Path object if file was renamed, None if no fix needed or not fixed
    """
    filepath = Path(filepath)

    if not filepath.exists():
        return False, None

    # Detect actual format
    actual_format = detect_font_format(filepath)

    # Handle error cases
    if actual_format == "ERROR":
        return False, None

    # A collection (TTC) is never renamed: ".ttc" is an extension the tools do not collect, so the file would drop
    # out of every later run. Leave it alone; describe_extension_mismatch says what it is.
    if actual_format == "TTC":
        return False, None

    # UNKNOWN format - can't validate, assume valid
    if actual_format == "UNKNOWN":
        return True, None

    # Get current extension (uppercase, without dot)
    current_ext = filepath.suffix.upper()[1:] if filepath.suffix else ""

    # Check if extension matches
    if actual_format == current_ext:
        return True, None

    # Mismatch detected
    if not auto_fix:
        return False, None

    # Auto-fix: rename to correct extension
    try:
        new_extension = actual_format.lower()
        new_path = filepath.with_suffix(f".{new_extension}")

        # Handle duplicate filename conflicts
        if new_path.exists() and new_path != filepath:
            # Use tilde format: FontName~001.ttf
            base_name = filepath.stem
            counter = 1
            while new_path.exists():
                new_name = f"{base_name}~{counter:03d}.{new_extension}"
                new_path = filepath.parent / new_name
                counter += 1
                # Safety limit to prevent infinite loop
                if counter > 9999:
                    return False, None

        # Rename the file
        filepath.rename(new_path)
        return False, new_path

    except Exception:
        # Permission error, file locked, etc.
        return False, None


__all__ = ["detect_font_format", "validate_and_fix_extension"]


def describe_extension_mismatch(filepath: Path | str) -> Optional[str]:
    """One sentence when the extension does not match the font's real format, else None. Never renames anything."""
    filepath = Path(filepath)
    actual = detect_font_format(filepath)
    if actual in ("ERROR", "UNKNOWN"):
        return None
    current = filepath.suffix.upper()[1:] if filepath.suffix else ""
    if actual == current:
        return None
    if actual == "TTC":
        return (
            f"{filepath.name} is a font collection (TTC), not a single font; the tools do not read collections, "
            "so it was left alone"
        )
    shown = filepath.suffix or "(none)"
    return f"{filepath.name} is a {actual} font but its extension is {shown}"


def is_readable(filepath: Path | str) -> Tuple[bool, Optional[str]]:
    """Open the font and read its table directory. (True, None) when it opens, else (False, reason).

    detect_font_format only reads four bytes, so a file cut off right after a valid header still says "TTF";
    this is the check that tells it is damaged.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return False, "the file does not exist"
    try:
        size = filepath.stat().st_size
    except OSError as exc:
        return False, f"cannot read the file: {exc}"
    if size < 12:
        return False, f"too small to be a font ({size} bytes)"
    try:
        from fontTools.ttLib import TTFont

        font = TTFont(str(filepath), lazy=True, recalcTimestamp=False)
        try:
            reader = getattr(font, "reader", None)
            tables = getattr(reader, "tables", None)
            if tables is not None and len(tables) == 0:
                return False, "the font has no tables"
            if "head" not in font and "CFF " not in font and "glyf" not in font:
                return False, "the font has none of the core tables (head, glyf, CFF)"
            font["head"] if "head" in font else None  # read one table so a bad directory fails here
        finally:
            font.close()
    except Exception as exc:  # fontTools raises many kinds of errors for a damaged file
        return False, f"{type(exc).__name__}: {exc}"[:160]
    return True, None
