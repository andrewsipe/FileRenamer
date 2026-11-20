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
