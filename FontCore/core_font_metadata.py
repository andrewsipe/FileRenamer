#!/usr/bin/env python3
"""
Font metadata extraction utilities.

Provides shared functions for extracting metadata from font files,
including PostScript names, version information, vendor IDs, and format detection.
"""

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

# ruff: noqa: E402
try:
    from fontTools.ttLib import TTFont, TTCollection
except ImportError:
    TTFont = None
    TTCollection = None


@dataclass
class FontMetadata:
    """Metadata extracted from a font file"""

    ps_name: str
    font_revision: float
    version_string: str
    file_size: int
    glyph_count: int
    head_created: Optional[float]
    head_modified: Optional[float]
    file_path: str
    original_filename: Optional[str] = None
    detected_format: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FontMetadata":
        """Create from dictionary."""
        return cls(**data)


def detect_font_format(font) -> str:
    """
    Detect font format from font object.

    Args:
        font: TTFont object

    Returns:
        Format string: "ttf", "otf", "woff", or "woff2"
    """
    try:
        # Check sfntVersion in head table
        head_table = font.get("head")
        if head_table:
            sfnt_version = head_table.sfntVersion

            if sfnt_version == b"wOFF":
                return "woff"
            elif sfnt_version == b"wOF2":
                return "woff2"
            elif sfnt_version == b"OTTO":
                return "otf"
            elif sfnt_version == b"\x00\x01\x00\x00" or sfnt_version == b"true":
                return "ttf"
        else:
            # Fallback: check for CFF table (OTF uses CFF, TTF uses glyf)
            if "CFF " in font:
                return "otf"
            return "ttf"
    except Exception:
        # If detection fails, default to ttf
        return "ttf"


def extract_metadata(font_path: Path) -> Optional[FontMetadata]:
    """
    Extract metadata from a font file.

    Args:
        font_path: Path to font file

    Returns:
        FontMetadata object or None if extraction fails
    """
    if TTFont is None:
        raise ImportError("fonttools library is required")

    font = None
    coll = None
    try:
        suffix = font_path.suffix.lower()
        # Use TTCollection for .ttc/.otc
        if suffix in (".ttc", ".otc"):
            if TTCollection is None:
                raise ImportError("fonttools library is required")
            coll = TTCollection(str(font_path))
            # policy: use first member
            font = coll[0]
        else:
            font = TTFont(str(font_path))

        # PostScript name (nameID 6)
        name_record = font["name"].getName(6, 3, 1, 0x409)
        ps_name = name_record.toUnicode() if name_record else ""

        # Version string (nameID 5)
        version_record = font["name"].getName(5, 3, 1, 0x409)
        version_string = version_record.toUnicode() if version_record else ""

        # head table data
        head_table = font.get("head")
        font_revision = head_table.fontRevision if head_table else 0.0
        head_created = head_table.created if head_table else None
        head_modified = head_table.modified if head_table else None

        # maxp table data
        maxp_table = font.get("maxp")
        glyph_count = maxp_table.numGlyphs if maxp_table else 0

        # Detect font format from file data
        detected_format = detect_font_format(font)

        file_size = font_path.stat().st_size

        return FontMetadata(
            ps_name=ps_name,
            font_revision=font_revision,
            version_string=version_string,
            file_size=file_size,
            glyph_count=glyph_count,
            head_created=head_created,
            head_modified=head_modified,
            file_path=str(font_path),
            original_filename=font_path.name,
            detected_format=detected_format,
        )
    except Exception:
        return None
    finally:
        try:
            if coll is not None and hasattr(coll, "close"):
                coll.close()
            elif font is not None and hasattr(font, "close"):
                font.close()
        except Exception:
            pass


def extract_metadata_with_error(
    font_path: Path,
) -> Tuple[Optional[FontMetadata], Optional[str]]:
    """
    Extract metadata from font file, capturing any exceptions.

    Args:
        font_path: Path to font file

    Returns:
        Tuple of (FontMetadata or None, error_message or None)
    """
    try:
        metadata = extract_metadata(font_path)
        if metadata:
            return metadata, None
        else:
            return None, "Failed to extract metadata (no error details available)"
    except Exception as e:
        # Capture the actual exception message, with fallback for empty messages
        error_msg = str(e) if str(e) else repr(e)
        if not error_msg:
            error_msg = f"Exception of type {type(e).__name__} occurred"
        return None, error_msg


def get_vendor_id(font_path: Path) -> str:
    """
    Extract achVendID robustly and return sanitized vendor id or codes 'UKWN'/'ERROR'.

    Args:
        font_path: Path to font file

    Returns:
        Sanitized vendor ID string, "UKWN" for unknown, or "ERROR" for read failures
    """
    if TTFont is None:
        return "ERROR"

    font = None
    coll = None
    try:
        suffix = font_path.suffix.lower()
        # Use TTCollection for .ttc/.otc
        if suffix in (".ttc", ".otc"):
            if TTCollection is None:
                return "ERROR"
            coll = TTCollection(str(font_path))
            # policy: use first member
            font = coll[0]
        else:
            font = TTFont(str(font_path))

        os2_table = font.get("OS/2") if font is not None else None

        if not os2_table:
            vendor = "UKWN"
        else:
            raw = getattr(os2_table, "achVendID", None)
            if raw is None:
                vendor = "UKWN"
            else:
                if isinstance(raw, bytes):
                    vendor = raw.decode("latin-1", "ignore")
                else:
                    vendor = str(raw)
                vendor = vendor.replace("\x00", "").strip()
                if not vendor:
                    vendor = "UKWN"

        # sanitize vendor for safe directory name: replace unsafe chars with '_'
        vendor = "".join(c for c in vendor if c.isprintable())
        vendor = re.sub(r"[^A-Za-z0-9 ._-]", "_", vendor).strip()
        vendor = re.sub(r"\s+", "_", vendor)
        if not vendor:
            vendor = "UKWN"

        return vendor

    except Exception:
        return "ERROR"  # Error reading file

    finally:
        try:
            if coll is not None and hasattr(coll, "close"):
                coll.close()
            elif font is not None and hasattr(font, "close"):
                font.close()
        except Exception:
            pass
