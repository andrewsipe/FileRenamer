#!/usr/bin/env python3
"""
Font File Renamer - PostScript name-based renaming with intelligent quality scoring

Renames font files to their PostScript names with comprehensive quality analysis:
- Two-pass renaming (temp UUID → PostScript names)
- Quality-aware priority (considers revision, language support, features, glyphs)
- Multiple fonts with same PS name get ~001, ~002, etc. suffixes
- Per-directory isolation (processes each directory independently)
- Cached metadata support (speeds up repeated runs)

Usage:
    python FontFiles_Rename.py /path/to/fonts/
    python FontFiles_Rename.py font1.otf font2.otf
    python FontFiles_Rename.py /directory/ -r
    python FontFiles_Rename.py /directory/ -n

Options:
    -r, --recursive     Process directories recursively
    -n, --dry-run       Preview changes without renaming
    -ra, --rename-all   Rename even fonts with invalid PostScript names
    -v, --verbose       Show detailed processing information
    --show-quality      Display quality scores in preview
"""

import json
import os
import re
import shutil
import uuid
import argparse
import time
from multiprocessing import Pool, cpu_count, current_process
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager

from fontTools.ttLib import TTFont

# Platform-specific file locking
try:
    import fcntl  # Unix

    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

try:
    import msvcrt  # Windows

    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False

# Add project root to path for FontCore imports
import sys
from pathlib import Path as PathLib

# ruff: noqa: E402
_project_root = PathLib(__file__).parent
while (
    not (_project_root / "FontCore").exists() and _project_root.parent != _project_root
):
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Core module imports
import FontCore.core_console_styles as cs  # noqa: E402
from FontCore.core_file_collector import collect_font_files  # noqa: E402
from FontCore.core_font_extension import (  # noqa: E402
    detect_font_format as detect_font_format_from_file,
    validate_and_fix_extension,
)
from FontCore.core_variable_font_detection import (  # noqa: E402
    is_variable_font,
    VariableFontMode,
)

console = cs.get_console()

# ============================================================================
# Constants
# ============================================================================

INDEX_FILENAME = ".font_rename_cache.json"
TRANSACTION_FILENAME = ".font_rename_transaction.json"
FONT_EXTENSIONS = {".ttf", ".otf", ".woff", ".woff2"}

# Quality scoring weights
WEIGHT_REVISION = 400  # 40% - Font revision number
WEIGHT_LANGUAGE = 2.5  # 25% - Language support breadth
WEIGHT_FEATURES = 2.0  # 20% - OpenType features
WEIGHT_GLYPHS = 100  # 10% - Meaningful glyph count increase
WEIGHT_RECENCY = 50  # 5% - Creation date recency

# Magic numbers extracted to constants
ESTIMATED_TEMP_SPACE_PER_FILE = 100 * 1024  # 100KB buffer per file for temp operations
# Maximum conflict suffixes for resolve_name_conflict (_conflict001, ...).
# 999 allows up to 999 alternate filenames per base before raising NameConflictError.
MAX_CONFLICT_COUNTER = 999
MAC_EPOCH_2020 = (
    3786825600.0  # Mac epoch timestamp for Jan 1, 2020 (baseline for recency scoring)
)
SECONDS_PER_YEAR = (
    86400 * 365
)  # Seconds in a year (approximate, for recency calculation)
RECENCY_YEARS = 5  # Years from 2020 baseline for recency normalization

# File locking retry configuration
LOCK_RETRY_MAX_ATTEMPTS = 5  # Maximum retry attempts for file lock acquisition
LOCK_RETRY_BASE_DELAY = 0.1  # Base delay in seconds for exponential backoff

# Cache validation: allow 1 second tolerance for filesystem mtime comparison
CACHE_MTIME_TOLERANCE_SEC = 1.0

# Parallel metadata extraction: use multiprocessing when uncached file count >= threshold
PARALLEL_EXTRACT_THRESHOLD = 30
PARALLEL_MAX_WORKERS_DEFAULT = max(1, (cpu_count() or 2) - 1)

# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class FontMetadata:
    """
    Enhanced metadata with quality scoring for font file renaming.

    This class stores comprehensive metadata extracted from font files, including
    PostScript names, version information, language support, OpenType features,
    and quality scores used for prioritizing fonts with the same PostScript name.

    Fields:
        ps_name: PostScript name (nameID 6) - used as primary filename
        font_revision: Font revision number from head table
        version_string: Version string (nameID 5)
        file_size: File size in bytes
        glyph_count: Number of glyphs in the font
        head_created: Creation timestamp (Mac epoch: seconds since 1904-01-01)
        head_modified: Modification timestamp (Mac epoch)
        file_path: Full path to the font file
        original_filename: Original filename before renaming
        detected_format: Detected format (ttf, otf, woff, woff2)
        language_support: Set of detected language scripts (latin, cyrillic, etc.)
        opentype_features: Set of OpenType feature tags (kern, liga, etc.)
        quality_score: Calculated quality score for prioritization (higher is better)
        typographic_family: Typographic family name (nameID 16)
        typographic_subfamily: Typographic subfamily name (nameID 17)
        is_variable: Whether this font is a variable font (has fvar table)
        file_mtime: Filesystem mtime when cached (for cache invalidation; optional)
    """

    # Original fields
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

    # Cache validation: filesystem mtime when entry was cached (None = legacy cache)
    file_mtime: Optional[float] = None

    # New quality indicators
    language_support: set = field(default_factory=set)
    opentype_features: set = field(default_factory=set)
    quality_score: Optional[float] = None

    # Typographic names (nameID 16 and 17)
    typographic_family: Optional[str] = None
    typographic_subfamily: Optional[str] = None

    # Variable font detection
    is_variable: bool = False

    def to_dict(self) -> dict:
        data = asdict(self)
        # Convert sets to lists for JSON serialization
        data["language_support"] = list(self.language_support)
        data["opentype_features"] = list(self.opentype_features)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "FontMetadata":
        # Convert lists back to sets
        if "language_support" in data:
            data["language_support"] = set(data["language_support"])
        else:
            data["language_support"] = set()
        if "opentype_features" in data:
            data["opentype_features"] = set(data["opentype_features"])
        else:
            data["opentype_features"] = set()
        # Handle backward compatibility: is_variable may not exist in old cache files
        if "is_variable" not in data:
            data["is_variable"] = False
        # file_mtime may not exist in old cache files (then we validate by size only)
        if "file_mtime" not in data:
            data["file_mtime"] = None
        return cls(**data)

    def calculate_quality_score(self, fonts_in_group: List["FontMetadata"]) -> float:
        """
        Calculate comprehensive quality score for prioritization.
        Higher score = better font to keep as primary.

        Quality scoring algorithm:
        The score is a weighted sum of multiple factors:

        1. Font Revision (40% weight): Higher revision numbers indicate newer/more
           refined versions. Multiplied by WEIGHT_REVISION (400).

        2. Language Support (25% weight): Broader language coverage is valued.
           - Pan-European (Cyrillic + Latin-Extended): 100 points
           - Cyrillic or Greek: 50 points
           - Latin-Extended: 25 points
           - Vietnamese: 20 points
           Multiplied by WEIGHT_LANGUAGE (2.5).

        3. OpenType Features (20% weight): More features indicate better typography.
           - Valuable features (kern, liga, etc.): 10 points each
           - Other features: 2 points each
           Capped at 200 points, multiplied by WEIGHT_FEATURES (2.0).

        4. Glyph Count (10% weight): Fonts with significantly more glyphs (10%+ above
           median) get bonus points. Multiplied by WEIGHT_GLYPHS (100).

        5. Creation Date Recency (5% weight): Newer fonts (created after 2020) get
           bonus based on recency. Uses Mac epoch (seconds since 1904-01-01).
           Baseline is Jan 1, 2020 (MAC_EPOCH_2020). Multiplied by WEIGHT_RECENCY (50).

        Args:
            fonts_in_group: List of all fonts in the same group for comparison

        Returns:
            Quality score (higher is better)
        """
        score = 0.0

        # 1. Font Revision (weight: 40%)
        score += (self.font_revision or 0.0) * WEIGHT_REVISION

        # 2. Language Support (weight: 25%)
        lang_score = 0
        if (
            "cyrillic" in self.language_support
            and "latin-extended" in self.language_support
        ):
            lang_score = 100  # Pan-European
        elif "cyrillic" in self.language_support or "greek" in self.language_support:
            lang_score = 50
        elif "latin-extended" in self.language_support:
            lang_score = 25
        elif "vietnamese" in self.language_support:
            lang_score = 20
        score += lang_score * WEIGHT_LANGUAGE

        # 3. OpenType Features (weight: 20%)
        valuable_features = {
            "kern",
            "liga",
            "dlig",
            "smcp",
            "c2sc",
            "onum",
            "lnum",
            "tnum",
            "frac",
            "sups",
            "subs",
        }
        feature_score = len(self.opentype_features & valuable_features) * 10
        feature_score += len(self.opentype_features - valuable_features) * 2
        score += min(feature_score, 200) * WEIGHT_FEATURES

        # 4. Meaningful Glyph Count (weight: 10%)
        if fonts_in_group and len(fonts_in_group) > 1:
            glyph_counts = [f.glyph_count for f in fonts_in_group]
            median_glyphs = sorted(glyph_counts)[len(glyph_counts) // 2]

            if median_glyphs > 0 and self.glyph_count >= median_glyphs * 1.10:
                # Reward 10%+ increases
                glyph_bonus = ((self.glyph_count / median_glyphs) - 1.0) * WEIGHT_GLYPHS
                score += min(glyph_bonus, WEIGHT_GLYPHS)

        # 5. Creation Date Recency (weight: 5%)
        # Younger creation date suggests newer revision
        # Mac epoch: seconds since 1904-01-01 00:00:00 UTC
        # 2020 baseline chosen as modern font production standard
        if self.head_created:
            if self.head_created >= MAC_EPOCH_2020:
                # Scale from 2020 to 2020+RECENCY_YEARS
                recency = min(
                    (self.head_created - MAC_EPOCH_2020)
                    / (SECONDS_PER_YEAR * RECENCY_YEARS),
                    1.0,
                )
                score += recency * WEIGHT_RECENCY

        return score

    def get_quality_breakdown(self) -> Dict[str, float]:
        """Get breakdown of quality score components for display"""
        return {
            "revision": (self.font_revision or 0.0) * WEIGHT_REVISION,
            "language": self._get_language_score() * WEIGHT_LANGUAGE,
            "features": self._get_feature_score() * WEIGHT_FEATURES,
            "total": self.quality_score or 0.0,
        }

    def _get_language_score(self) -> float:
        """Calculate language score component"""
        if (
            "cyrillic" in self.language_support
            and "latin-extended" in self.language_support
        ):
            return 100.0
        elif "cyrillic" in self.language_support or "greek" in self.language_support:
            return 50.0
        elif "latin-extended" in self.language_support:
            return 25.0
        elif "vietnamese" in self.language_support:
            return 20.0
        return 0.0

    def _get_feature_score(self) -> float:
        """Calculate feature score component"""
        valuable_features = {
            "kern",
            "liga",
            "dlig",
            "smcp",
            "c2sc",
            "onum",
            "lnum",
            "tnum",
            "frac",
            "sups",
            "subs",
        }
        feature_score = len(self.opentype_features & valuable_features) * 10
        feature_score += len(self.opentype_features - valuable_features) * 2
        return min(feature_score, 200)


@dataclass
class RenameStats:
    """Statistics for rename operations"""

    total_files: int = 0
    renamed: int = 0
    skipped: int = 0
    invalid: int = 0
    errors: List[Tuple[str, str]] = field(default_factory=list)

    def add_error(self, filename: str, reason: str):
        self.errors.append((filename, reason))
        self.skipped += 1


class NameConflictError(Exception):
    """Exception raised when name conflict resolution fails"""

    def __init__(self, message: str, reason: str = ""):
        super().__init__(message)
        self.reason = reason


# ============================================================================
# Metadata Cache
# ============================================================================


@contextmanager
def _file_lock(file_path: Path, mode: str = "r"):
    """
    Context manager for file locking with retry logic and exponential backoff.
    Supports Unix (fcntl) and Windows (msvcrt) platforms.
    """
    file_handle = None
    lock_acquired = False
    attempt = 0

    while attempt < LOCK_RETRY_MAX_ATTEMPTS and not lock_acquired:
        try:
            if mode == "r":
                file_handle = open(
                    file_path, "r", encoding="utf-8", errors="replace"
                )
            else:
                file_handle = open(
                    file_path, "w", encoding="utf-8", errors="replace"
                )

            # Try to acquire lock
            if HAS_FCNTL:
                # Unix: use fcntl for advisory locking
                fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_acquired = True
            elif HAS_MSVCRT:
                # Windows: use msvcrt for file locking
                msvcrt.locking(file_handle.fileno(), msvcrt.LK_NBLCK, 1)
                lock_acquired = True
            else:
                # No locking available - proceed without lock (better than nothing)
                lock_acquired = True

        except (IOError, OSError, BlockingIOError) as e:
            if file_handle:
                try:
                    file_handle.close()
                except Exception:
                    pass
                file_handle = None

            if attempt < LOCK_RETRY_MAX_ATTEMPTS - 1:
                # Exponential backoff
                delay = LOCK_RETRY_BASE_DELAY * (2**attempt)
                time.sleep(delay)
                attempt += 1
            else:
                # Last attempt failed
                if console:
                    cs.StatusIndicator("warning").with_explanation(
                        f"Could not acquire file lock after {LOCK_RETRY_MAX_ATTEMPTS} attempts: {e}"
                    ).emit()
                # Proceed without lock as fallback
                lock_acquired = True
                break

    try:
        yield file_handle
    finally:
        if file_handle:
            try:
                if HAS_FCNTL:
                    fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)
                elif HAS_MSVCRT:
                    msvcrt.locking(file_handle.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
            file_handle.close()


def load_cache(directory: Path) -> Dict[str, FontMetadata]:
    """
    Load metadata cache from directory with corruption recovery and file locking.
    Cached entries are validated on use (in process_single_font_metadata) by file
    size and, when present, filesystem mtime; stale entries are re-extracted.

    Cache file format:
    {
        "filename.ttf": {
            "ps_name": "FontName-Regular",
            "font_revision": 1.001,
            "version_string": "Version 1.001",
            "file_size": 123456,
            "glyph_count": 500,
            "head_created": 3786825600.0,
            "head_modified": 3786825600.0,
            "file_path": "/path/to/file.ttf",
            "original_filename": "oldname.ttf",
            "detected_format": "ttf",
            "file_mtime": 1707321600.0,
            "language_support": ["latin", "cyrillic"],
            "opentype_features": ["kern", "liga"],
            "quality_score": 1234.5,
            "typographic_family": "Font Name",
            "typographic_subfamily": "Regular"
        }
    }
    """
    cache_path = directory / INDEX_FILENAME
    if not cache_path.exists():
        return {}

    try:
        with _file_lock(cache_path, "r") as f:
            if f is None:
                # Lock acquisition failed, return empty cache
                return {}
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("Cache is not a dictionary")

        cache = {}
        for filename, meta in data.items():
            try:
                if not isinstance(meta, dict):
                    continue
                cache[filename] = FontMetadata.from_dict(meta)
            except Exception as e:
                if console:
                    cs.StatusIndicator("warning").add_file(
                        filename, filename_only=True
                    ).with_explanation(f"Invalid cache entry: {e}").emit()
                continue

        return cache
    except json.JSONDecodeError as e:
        if console:
            cs.StatusIndicator("warning").with_explanation(
                f"Cache file corrupted (invalid JSON): {e}. Recreating cache."
            ).emit()
        try:
            cache_path.unlink()
        except Exception:
            pass
        return {}
    except Exception as e:
        if console:
            cs.StatusIndicator("warning").with_explanation(
                f"Failed to load cache: {e}. Recreating cache."
            ).emit()
        try:
            cache_path.unlink()
        except Exception:
            pass
        return {}


def save_cache(directory: Path, cache: Dict[str, FontMetadata]) -> None:
    """Save metadata cache to directory with file locking"""
    cache_path = directory / INDEX_FILENAME
    try:
        data = {filename: meta.to_dict() for filename, meta in cache.items()}
        with _file_lock(cache_path, "w") as f:
            if f is None:
                # Lock acquisition failed
                if console:
                    cs.StatusIndicator("warning").with_explanation(
                        "Could not acquire lock to save cache"
                    ).emit()
                return
            json.dump(data, f, indent=2)
    except Exception as e:
        if console:
            cs.StatusIndicator("warning").with_explanation(
                f"Failed to save cache: {e}"
            ).emit()


def cleanup_cache(directory: Path) -> None:
    """Remove metadata cache from directory with error handling"""
    cache_path = directory / INDEX_FILENAME
    if cache_path.exists():
        try:
            cache_path.unlink()
        except PermissionError as e:
            if console:
                cs.StatusIndicator("warning").with_explanation(
                    f"Cannot remove cache file (permission denied): {e}"
                ).emit()
        except OSError as e:
            if console:
                cs.StatusIndicator("warning").with_explanation(
                    f"Cannot remove cache file: {e}"
                ).emit()
        except Exception as e:
            if console:
                cs.StatusIndicator("warning").with_explanation(
                    f"Unexpected error removing cache: {e}"
                ).emit()


# ============================================================================
# Font Metadata Extraction
# ============================================================================


# Languages that trigger early exit when all are detected (single-pass optimization)
PRIORITY_LANGUAGES = frozenset({"latin", "latin-extended", "cyrillic", "greek"})


def detect_language_support_from_font(font: TTFont) -> set:
    """
    Detect language support from Unicode coverage in opened font.

    Optimized single-pass iteration through codepoints for better performance.
    Early exit when all priority languages (latin, latin-extended, cyrillic, greek) are detected.
    """
    try:
        cmap = font.getBestCmap()
        if not cmap:
            return {"latin"}

        codepoints = set(cmap.keys())
        languages: set = set()

        # Common languages to detect (ordered by frequency)
        # Single pass through codepoints instead of multiple any() calls
        for cp in codepoints:
            # Latin: U+0000-U+007F (Basic Latin)
            if 0x0000 <= cp <= 0x007F:
                languages.add("latin")
            # Extended Latin (Pan-European): U+0100-U+017F
            elif 0x0100 <= cp <= 0x017F:
                languages.add("latin-extended")
            # Cyrillic: U+0400-U+04FF
            elif 0x0400 <= cp <= 0x04FF:
                languages.add("cyrillic")
            # Greek: U+0370-U+03FF
            elif 0x0370 <= cp <= 0x03FF:
                languages.add("greek")
            # Vietnamese: U+1E00-U+1EFF
            elif 0x1E00 <= cp <= 0x1EFF:
                languages.add("vietnamese")
            # Arabic: U+0600-U+06FF
            elif 0x0600 <= cp <= 0x06FF:
                languages.add("arabic")
            # Hebrew: U+0590-U+05FF
            elif 0x0590 <= cp <= 0x05FF:
                languages.add("hebrew")

            # Early exit when all priority languages are detected
            if len(languages & PRIORITY_LANGUAGES) == 4:
                break

        return languages if languages else {"latin"}
    except Exception:
        return {"latin"}


def extract_opentype_features_from_font(font: TTFont) -> set:
    """Extract OpenType feature tags from GSUB/GPOS tables in opened font"""
    features = set()

    try:
        # GSUB table (substitution features)
        if "GSUB" in font:
            gsub = font["GSUB"]
            if hasattr(gsub, "table") and hasattr(gsub.table, "FeatureList"):
                for feature in gsub.table.FeatureList.FeatureRecord:
                    features.add(feature.FeatureTag)

        # GPOS table (positioning features)
        if "GPOS" in font:
            gpos = font["GPOS"]
            if hasattr(gpos, "table") and hasattr(gpos.table, "FeatureList"):
                for feature in gpos.table.FeatureList.FeatureRecord:
                    features.add(feature.FeatureTag)
    except Exception:
        pass

    return features


def extract_metadata(
    font_path: Path, *, silent: bool = False
) -> Optional[FontMetadata]:
    """Extract enhanced metadata from a font file.
    When silent=True, no console output (for use in worker processes).
    """
    try:
        font = TTFont(str(font_path))

        # PostScript name (nameID 6)
        name_record = font["name"].getName(6, 3, 1, 0x409)
        ps_name = name_record.toUnicode() if name_record else ""

        # Version string (nameID 5)
        version_record = font["name"].getName(5, 3, 1, 0x409)
        version_string = version_record.toUnicode() if version_record else ""

        # Typographic Family (nameID 16)
        family_record = font["name"].getName(16, 3, 1, 0x409)
        typographic_family = (
            family_record.toUnicode().strip() if family_record else None
        )

        # Typographic Subfamily (nameID 17)
        subfamily_record = font["name"].getName(17, 3, 1, 0x409)
        typographic_subfamily = (
            subfamily_record.toUnicode().strip() if subfamily_record else None
        )

        # head table data
        head_table = font.get("head")
        font_revision = head_table.fontRevision if head_table else 0.0
        head_created = head_table.created if head_table else None
        head_modified = head_table.modified if head_table else None

        # maxp table data
        maxp_table = font.get("maxp")
        glyph_count = maxp_table.numGlyphs if maxp_table else 0

        # Detect font format from file magic bytes (preserves WOFF/WOFF2; TTFont
        # reader exposes inner SFNT version and would misreport container format)
        actual_format = detect_font_format_from_file(font_path)
        if actual_format in ("UNKNOWN", "ERROR"):
            detected_format = font_path.suffix.lower().lstrip(".") or "ttf"
        else:
            detected_format = actual_format.lower()

        # Language support detection
        language_support = detect_language_support_from_font(font)

        # OpenType features extraction
        opentype_features = extract_opentype_features_from_font(font)

        # Variable font detection (use LENIENT mode to catch technically valid variable fonts)
        is_variable = is_variable_font(font, mode=VariableFontMode.LENIENT)

        file_stat = font_path.stat()
        file_size = file_stat.st_size
        file_mtime = file_stat.st_mtime

        font.close()

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
            file_mtime=file_mtime,
            language_support=language_support,
            opentype_features=opentype_features,
            typographic_family=typographic_family,
            typographic_subfamily=typographic_subfamily,
            is_variable=is_variable,
        )
    except Exception as e:
        if not silent and console:
            cs.StatusIndicator("error").add_file(
                font_path.name, filename_only=True
            ).with_explanation(f"Failed to read: {e}").emit()
        return None


def _extract_metadata_worker(extract_path: Path) -> Optional[FontMetadata]:
    """Worker for parallel metadata extraction; no console output."""
    return extract_metadata(extract_path, silent=True)


def contains_problematic_pattern(ps_name: str) -> Tuple[bool, str]:
    """Check for problematic patterns in PostScript name"""
    problematic_patterns = [
        "copyright",
        "Copyright",
        "©",
        "(c)",
        "(C)",
        "fontname",
        "Fontname",
    ]
    for pattern in problematic_patterns:
        if pattern in ps_name:
            return True, pattern
    return False, ""


def _validate_font_name(name: str, context: str = "name") -> Tuple[bool, str]:
    """
    Common validation for font names safe for use as filenames (PostScript or typographic).
    Checks empty/whitespace, control chars, problematic path chars, leading/trailing space,
    forbidden first character, and problematic content patterns.

    Args:
        name: The name string to validate
        context: Label for error messages (e.g. "PostScript name", "typographic name")

    Returns:
        Tuple of (is_valid, reason); reason is empty when valid.
    """
    if not name or name.strip() == "":
        return False, "empty name"

    if name.isspace():
        return False, "contains only spaces"

    for char in name:
        code = ord(char)
        if code < 32 or code == 127:
            return False, f"contains control character (ASCII {code})"

    problematic_chars = ["?", "/", "\\", ":", "*", '"', "<", ">", "|"]
    for char in problematic_chars:
        if char in name:
            return False, f"contains '{char}'"

    if name.startswith(" ") or name.endswith(" "):
        return False, "begins or ends with a space"

    forbidden_first_chars = ["_", "-", "."]
    if name[0] in forbidden_first_chars:
        return False, f"begins with '{name[0]}'"

    has_problem, pattern = contains_problematic_pattern(name)
    if has_problem:
        return False, f"contains '{pattern}'"

    return True, ""


def is_valid_postscript_name(ps_name: str) -> Tuple[bool, str]:
    """Validate PostScript name is safe for filename."""
    return _validate_font_name(ps_name, "PostScript name")


def validate_typographic_name(name: str) -> Tuple[bool, str]:
    """
    Validate typographic name using same rules as PostScript name validation.

    Args:
        name: Typographic name to validate

    Returns:
        Tuple of (is_valid, reason) where reason is empty if valid
    """
    return _validate_font_name(name, "typographic name")


def generate_typographic_filename(
    typographic_family: Optional[str], typographic_subfamily: Optional[str]
) -> Optional[str]:
    """
    Generate filename from typographic family and subfamily.
    Validates the generated name for filename safety.

    Args:
        typographic_family: nameID 16 (Typographic Family)
        typographic_subfamily: nameID 17 (Typographic Subfamily)

    Returns:
        Normalized filename in format "Family-Style" (spaces removed),
        or None if either field is empty/None or validation fails
    """
    if not typographic_family or not typographic_subfamily:
        return None

    # Remove all spaces from both fields
    family_normalized = typographic_family.replace(" ", "")
    subfamily_normalized = typographic_subfamily.replace(" ", "")

    # Return None if normalization resulted in empty strings
    if not family_normalized or not subfamily_normalized:
        return None

    # Combine as "Family-Style"
    combined_name = f"{family_normalized}-{subfamily_normalized}"

    # Validate the combined name
    is_valid, reason = validate_typographic_name(combined_name)
    if not is_valid:
        if console:
            cs.StatusIndicator("warning").with_explanation(
                f"Typographic name validation failed: {reason}. Using PostScript name instead."
            ).emit()
        return None

    return combined_name


# ============================================================================
# Path Normalization
# ============================================================================


def normalize_path(path: Path) -> Path:
    """
    Normalize path for consistent comparison and matching.
    Uses resolve() to get absolute, canonical paths.

    Args:
        path: Path to normalize

    Returns:
        Normalized absolute path
    """
    return path.resolve()


# ============================================================================
# Quality-Based Priority Sorting
# ============================================================================


def explain_quality_comparison(
    font_a: FontMetadata,
    font_b: FontMetadata,
    fonts_in_group: List[FontMetadata],
) -> Dict[str, Tuple[float, float, float]]:
    """
    Explain why one font has a higher quality score than another.
    Returns breakdown of score differences by component.

    Args:
        font_a: First font (should be higher quality)
        font_b: Second font (should be lower quality)
        fonts_in_group: All fonts in the group for context

    Returns:
        Dictionary mapping component names to (score_a, score_b, difference) tuples
    """
    breakdown = {}

    # Revision component
    rev_a = (font_a.font_revision or 0.0) * WEIGHT_REVISION
    rev_b = (font_b.font_revision or 0.0) * WEIGHT_REVISION
    breakdown["revision"] = (rev_a, rev_b, rev_a - rev_b)

    # Language component
    lang_a = font_a._get_language_score() * WEIGHT_LANGUAGE
    lang_b = font_b._get_language_score() * WEIGHT_LANGUAGE
    breakdown["language"] = (lang_a, lang_b, lang_a - lang_b)

    # Features component
    feat_a = font_a._get_feature_score() * WEIGHT_FEATURES
    feat_b = font_b._get_feature_score() * WEIGHT_FEATURES
    breakdown["features"] = (feat_a, feat_b, feat_a - feat_b)

    # Glyph component (relative to group)
    glyph_a = 0.0
    glyph_b = 0.0
    if fonts_in_group and len(fonts_in_group) > 1:
        glyph_counts = [f.glyph_count for f in fonts_in_group]
        median_glyphs = sorted(glyph_counts)[len(glyph_counts) // 2]

        if median_glyphs > 0:
            if font_a.glyph_count >= median_glyphs * 1.10:
                glyph_a = min(
                    ((font_a.glyph_count / median_glyphs) - 1.0) * WEIGHT_GLYPHS,
                    WEIGHT_GLYPHS,
                )
            if font_b.glyph_count >= median_glyphs * 1.10:
                glyph_b = min(
                    ((font_b.glyph_count / median_glyphs) - 1.0) * WEIGHT_GLYPHS,
                    WEIGHT_GLYPHS,
                )

    breakdown["glyphs"] = (glyph_a, glyph_b, glyph_a - glyph_b)

    # Recency component
    recency_a = 0.0
    recency_b = 0.0
    if font_a.head_created and font_a.head_created >= MAC_EPOCH_2020:
        recency_a = (
            min(
                (font_a.head_created - MAC_EPOCH_2020)
                / (SECONDS_PER_YEAR * RECENCY_YEARS),
                1.0,
            )
            * WEIGHT_RECENCY
        )
    if font_b.head_created and font_b.head_created >= MAC_EPOCH_2020:
        recency_b = (
            min(
                (font_b.head_created - MAC_EPOCH_2020)
                / (SECONDS_PER_YEAR * RECENCY_YEARS),
                1.0,
            )
            * WEIGHT_RECENCY
        )

    breakdown["recency"] = (recency_a, recency_b, recency_a - recency_b)

    return breakdown


def sort_by_quality_score(metadata_list: List[FontMetadata]) -> List[FontMetadata]:
    """
    Sort fonts by comprehensive quality score.
    Considers: revision, language support, features, glyphs, creation date.
    """
    # Calculate quality scores for all fonts in the group
    for meta in metadata_list:
        meta.quality_score = meta.calculate_quality_score(metadata_list)

    # Sort by quality score (highest first)
    return sorted(metadata_list, key=lambda m: m.quality_score or 0.0, reverse=True)


# ============================================================================
# Two-Pass Renaming
# ============================================================================


def rename_to_temp(font_files: List[Path], dry_run: bool = False) -> Dict[Path, Path]:
    """
    Phase 1: Rename all files to temporary UUID names to avoid collisions
    Returns mapping of temp_path -> original_path
    """
    temp_mapping: Dict[Path, Path] = {}

    for font_path in font_files:
        temp_name = f"_tmp_{uuid.uuid4().hex[:12]}{font_path.suffix.lower()}"
        temp_path = font_path.parent / temp_name

        if dry_run:
            temp_mapping[temp_path] = font_path
        else:
            try:
                font_path.rename(temp_path)
                temp_mapping[temp_path] = font_path
            except Exception as e:
                if console:
                    cs.StatusIndicator("error").add_file(
                        font_path.name, filename_only=True
                    ).with_explanation(f"Failed temp rename: {e}").emit()

    return temp_mapping


def assign_final_names(
    ps_name_groups: Dict[str, List[FontMetadata]],
    use_typographic_names: bool = False,
) -> Dict[Path, str]:
    """
    Assign final names based on PostScript name or typographic names and quality score.
    Highest quality gets clean name, others get ~001, ~002, etc.
    Uses normalized paths for consistent mapping.

    Variable font conflict resolution:
    - Static and variable fonts with the same PostScript name are processed separately.
    - Static fonts get the base name (e.g. Family-Style.otf); duplicates get ~001, ~002.
    - Variable fonts get the same base name when the group has only variable fonts.
    - When both static and variable exist in the group, variable fonts get a "-Variable"
      suffix (e.g. Family-Style-Variable.otf) so both can coexist without collision.
    - Suffix format ~001..~999 is used for duplicate base names within each subgroup;
      subgroups larger than 999 use ~1000, ~1001, etc. (no hard cap).

    Args:
        ps_name_groups: Fonts grouped by PostScript name and format
        use_typographic_names: If True, use nameID 16/17 for filenames when available
    """
    rename_map: Dict[Path, str] = {}
    typographic_used_count = 0
    postscript_fallback_count = 0

    for group_key, metadata_list in ps_name_groups.items():
        # Use quality-based sorting
        sorted_fonts = sort_by_quality_score(metadata_list)

        # Separate static and variable fonts
        static_fonts = [f for f in sorted_fonts if not f.is_variable]
        variable_fonts = [f for f in sorted_fonts if f.is_variable]
        has_conflict = len(static_fonts) > 0 and len(variable_fonts) > 0

        # Determine base name for this group
        base_name = None
        if use_typographic_names:
            # Try to use typographic name from highest quality font
            top_font = sorted_fonts[0]
            typo_name = generate_typographic_filename(
                top_font.typographic_family, top_font.typographic_subfamily
            )
            if typo_name:
                base_name = typo_name
                typographic_used_count += 1

        # Fall back to PostScript name if typographic name not available
        if base_name is None:
            base_name = sorted_fonts[0].ps_name
            if use_typographic_names:
                postscript_fallback_count += 1

        # When use_typographic_names is True, fonts are grouped by PostScript name
        # (e.g. blank ID6 puts all in one group). Add ~### only for actual duplicates
        # of the same target base name; otherwise every font after the first got a counter.
        def _effective_base_name(meta: FontMetadata, default: str) -> str:
            if use_typographic_names:
                typo_name = generate_typographic_filename(
                    meta.typographic_family, meta.typographic_subfamily
                )
                return typo_name if typo_name else meta.ps_name
            return default

        def _assign_names(
            fonts: List[FontMetadata],
            variable_suffix: str,
        ) -> None:
            # Subgroup by effective target base name so ~### is only used for real duplicates
            subgroups: Dict[str, List[FontMetadata]] = {}
            for meta in fonts:
                font_base = _effective_base_name(meta, base_name)
                key = f"{font_base}{variable_suffix}"
                if key not in subgroups:
                    subgroups[key] = []
                subgroups[key].append(meta)
            for _key, sublist in subgroups.items():
                for idx, meta in enumerate(sublist):
                    original_path = normalize_path(Path(meta.file_path))
                    if meta.detected_format:
                        ext = f".{meta.detected_format}"
                    else:
                        ext = Path(meta.file_path).suffix.lower()
                    font_base_name = _effective_base_name(meta, base_name)
                    if idx == 0:
                        new_name = f"{font_base_name}{variable_suffix}{ext}"
                    else:
                        new_name = f"{font_base_name}{variable_suffix}~{idx:03d}{ext}"
                    rename_map[original_path] = new_name

        # Process static fonts (get clean name; ~### only when same typographic name)
        _assign_names(static_fonts, "")

        # Process variable fonts (-Variable when conflict with static; ~### only for duplicates)
        var_suffix = "-Variable" if has_conflict else ""
        _assign_names(variable_fonts, var_suffix)

    # Log typographic name usage if enabled
    if (
        use_typographic_names
        and console
        and (typographic_used_count > 0 or postscript_fallback_count > 0)
    ):
        cs.StatusIndicator("info").add_message(
            f"Typographic names used: {typographic_used_count}, "
            f"PostScript fallback: {postscript_fallback_count}"
        ).emit()

    return rename_map


def resolve_name_conflict(base_name: str, parent_dir: Path, exclude_path: Path) -> str:
    """
    Resolve naming conflicts by adding _conflict001, _conflict002, etc.

    Args:
        base_name: Base filename to resolve
        parent_dir: Directory where file will be placed
        exclude_path: Path to exclude from conflict check (the file being renamed)

    Returns:
        Resolved filename

    Raises:
        NameConflictError: If too many conflicts (>MAX_CONFLICT_COUNTER) or
                          permission issues prevent resolution
    """
    target_path = normalize_path(parent_dir / base_name)
    exclude_path_normalized = normalize_path(exclude_path) if exclude_path else None

    if not target_path.exists() or target_path == exclude_path_normalized:
        try:
            if not parent_dir.exists():
                parent_dir.mkdir(parents=True, exist_ok=True)
            if not os.access(parent_dir, os.W_OK):
                raise NameConflictError(
                    f"Cannot resolve name conflict for '{base_name}'",
                    reason="permission denied",
                )
        except (OSError, PermissionError) as e:
            raise NameConflictError(
                f"Cannot resolve name conflict for '{base_name}'",
                reason=f"directory access error: {e}",
            )
        return base_name

    stem = Path(base_name).stem
    ext = Path(base_name).suffix

    for counter in range(1, MAX_CONFLICT_COUNTER + 1):
        new_name = f"{stem}_conflict{counter:03d}{ext}"
        target_path = normalize_path(parent_dir / new_name)
        if not target_path.exists() or target_path == exclude_path_normalized:
            try:
                if not os.access(parent_dir, os.W_OK):
                    raise NameConflictError(
                        f"Cannot resolve name conflict for '{base_name}'",
                        reason="permission denied",
                    )
            except (OSError, PermissionError) as e:
                raise NameConflictError(
                    f"Cannot resolve name conflict for '{base_name}'",
                    reason=f"directory access error: {e}",
                )
            return new_name

    raise NameConflictError(
        f"Cannot resolve name conflict for '{base_name}'",
        reason=f"too many conflicts (>{MAX_CONFLICT_COUNTER})",
    )


def execute_single_rename(
    temp_path: Path, new_name: str, original_name: str, dry_run: bool, verbose: bool
) -> Tuple[bool, Optional[str]]:
    """Execute or preview a single rename"""
    # Use same StatusIndicator for both dry-run and normal mode
    # DRY prefix will be added automatically when dry_run=True
    if console and verbose:
        cs.StatusIndicator("updated", dry_run=dry_run).add_values(
            old_value=original_name, new_value=new_name
        ).emit()

    if dry_run:
        return True, None

    try:
        target_path = temp_path.parent / new_name
        temp_path.rename(target_path)
        return True, None
    except Exception as e:
        if console:
            cs.StatusIndicator("error").add_file(
                original_name, filename_only=True
            ).with_explanation(f"Failed to rename: {e}").emit()
        return False, str(e)


def execute_final_renames(
    rename_map: Dict[Path, str],
    font_metadata: Dict[Path, FontMetadata],
    dry_run: bool = False,
    verbose: bool = False,
) -> RenameStats:
    """Phase 2: Execute final renames from temp names to PostScript names"""
    stats = RenameStats()

    for temp_path, new_name in rename_map.items():
        meta = font_metadata.get(temp_path)
        original_name = meta.original_filename if meta else temp_path.name

        # Normalize paths for consistent comparison
        temp_path_normalized = normalize_path(temp_path)
        target_path = normalize_path(temp_path.parent / new_name)

        # Skip if name unchanged
        if temp_path_normalized == target_path:
            stats.skipped += 1
            continue

        # Handle naming conflicts
        try:
            resolved_name = resolve_name_conflict(new_name, temp_path.parent, temp_path)
        except NameConflictError as e:
            stats.add_error(original_name, e.reason)
            if console:
                cs.StatusIndicator("error").add_file(
                    original_name, filename_only=True
                ).with_explanation(
                    f"Name conflict resolution failed: {e.reason}"
                ).emit()
            continue

        # Execute rename
        success, error = execute_single_rename(
            temp_path, resolved_name, original_name, dry_run, verbose
        )

        if success:
            stats.renamed += 1
        else:
            stats.add_error(original_name, error or "unknown error")

    return stats


# ============================================================================
# Directory Processing
# ============================================================================


def restore_temp_file(temp_path: Path, original_path: Path, dry_run: bool) -> bool:
    """
    Restore a temp file to its original name with error handling.

    Returns:
        True if restored or dry_run, False if restore failed.
    """
    if dry_run:
        return True
    try:
        if temp_path.exists():
            temp_path.rename(original_path)
        return True
    except FileNotFoundError:
        return True
    except PermissionError as e:
        if console:
            cs.StatusIndicator("warning").add_file(
                original_path.name, filename_only=True
            ).with_explanation(
                f"Cannot restore temp file (permission denied): {e}"
            ).emit()
        return False
    except OSError as e:
        if console:
            cs.StatusIndicator("warning").add_file(
                original_path.name, filename_only=True
            ).with_explanation(f"Cannot restore temp file: {e}").emit()
        return False
    except Exception as e:
        if console:
            cs.StatusIndicator("warning").add_file(
                original_path.name, filename_only=True
            ).with_explanation(f"Unexpected error restoring temp file: {e}").emit()
        return False


def cleanup_temp_files(temp_mapping: Dict[Path, Path], dry_run: bool) -> int:
    """
    Restore all temp files to their original names.
    Used for cleanup on error or cancellation.

    Args:
        temp_mapping: Mapping of temp_path -> original_path
        dry_run: If True, no actual file operations are performed

    Returns:
        Number of restorations that failed (0 on success or dry_run).
    """
    if dry_run:
        return 0
    failures = 0
    for temp_path, original_path in temp_mapping.items():
        if not restore_temp_file(temp_path, original_path, dry_run):
            failures += 1
    return failures


def process_single_font_metadata(
    temp_path: Path,
    original_path: Path,
    cache: Dict[str, FontMetadata],
    rename_all: bool,
    dry_run: bool,
    verbose: bool,
) -> Optional[FontMetadata]:
    """
    Extract and validate metadata for a single font.
    Handles dry-run mode where temp_path may not exist.
    """
    original_name = original_path.name

    # Try cache first (validate by file size and, when available, mtime)
    metadata = None
    if original_name in cache:
        cached = cache[original_name]
        # In dry-run mode, temp_path won't exist, so check original_path
        check_path = original_path if dry_run else temp_path
        if check_path.exists():
            st = check_path.stat()
            size_ok = cached.file_size == st.st_size
            mtime_ok = (
                cached.file_mtime is None
                or abs(cached.file_mtime - st.st_mtime) < CACHE_MTIME_TOLERANCE_SEC
            )
            if size_ok and mtime_ok:
                metadata = cached
                metadata.file_path = str(temp_path if not dry_run else original_path)

    # Extract if not cached
    if metadata is None:
        # In dry-run mode, use original_path since temp files don't exist
        extract_path = original_path if dry_run else temp_path
        metadata = extract_metadata(extract_path)

    if metadata is None:
        if console:
            cs.StatusIndicator("warning").add_file(
                original_name, filename_only=True
            ).with_explanation("Skipping invalid font").emit()
        restore_temp_file(temp_path, original_path, dry_run)
        return None

    metadata.original_filename = original_name

    # Validate PostScript name
    is_valid, reason = is_valid_postscript_name(metadata.ps_name)
    if not is_valid and not rename_all:
        if console and verbose:
            cs.StatusIndicator("warning").add_file(
                original_name, filename_only=True
            ).with_explanation(f"Skipping: {reason}").emit()
        restore_temp_file(temp_path, original_path, dry_run)
        return None

    return metadata


def collect_directory_fonts(directory: Path) -> List[Path]:
    """Collect font files from directory, excluding temp files, cache, and symlinks"""
    font_files = []
    for ext in FONT_EXTENSIONS:
        font_files.extend(directory.glob(f"*{ext}"))
        font_files.extend(directory.glob(f"*{ext.upper()}"))

    seen = set()
    result = []
    for f in font_files:
        if f in seen:
            continue
        if f.is_symlink():
            continue
        if not f.name.startswith("_tmp_") and f.name != INDEX_FILENAME:
            seen.add(f)
            result.append(f)

    return result


def check_filesystem_space(directory: Path, required_bytes: int) -> bool:
    """Check if filesystem has sufficient space for operations"""
    try:
        stat = shutil.disk_usage(directory)
        free_space = stat.free
        return free_space >= (required_bytes * 2)
    except OSError:
        return True


def _prepare_directory(
    directory: Path,
    dry_run: bool,
    verbose: bool,
    specific_files: Optional[List[Path]] = None,
) -> Tuple[List[Path], Dict[Path, Path]]:
    """Prepare directory for processing: collect files, check space, rename to temp"""
    if specific_files is not None:
        # Use provided specific files, filtering to only those in this directory
        # Normalize paths for consistent comparison; reject symlinks
        normalized_dir = normalize_path(directory)
        font_files = []
        for f in specific_files:
            if normalize_path(f.parent) != normalized_dir:
                continue
            try:
                if f.is_symlink():
                    if console:
                        cs.StatusIndicator("warning").add_file(
                            f.name, filename_only=True
                        ).with_explanation("Skipping symlink").emit()
                    continue
            except OSError:
                continue
            font_files.append(f)
    else:
        # Collect all fonts in directory (backward compatible behavior)
        font_files = collect_directory_fonts(directory)
    if not font_files:
        return [], {}

    if console and verbose:
        cs.StatusIndicator("info").add_message(
            f"Processing {cs.fmt_count(len(font_files))} files in {cs.fmt_file_compact(str(directory))}"
        ).emit()

    if not dry_run:
        estimated_space = len(font_files) * ESTIMATED_TEMP_SPACE_PER_FILE
        if not check_filesystem_space(directory, estimated_space):
            if console:
                cs.StatusIndicator("warning").with_explanation(
                    "Insufficient filesystem space - operations may fail"
                ).emit()

    temp_mapping = rename_to_temp(font_files, dry_run)

    return font_files, temp_mapping


def _process_extracted_metadata(
    temp_path: Path,
    original_path: Path,
    metadata: Optional[FontMetadata],
    cache: Dict[str, FontMetadata],
    rename_all: bool,
    dry_run: bool,
    verbose: bool,
    stats: RenameStats,
    font_metadata: Dict[Path, FontMetadata],
) -> None:
    """Apply validation and store (or restore on skip). Shared by sequential and parallel paths."""
    original_name = original_path.name
    if metadata is None:
        stats.invalid += 1
        if console:
            cs.StatusIndicator("warning").add_file(
                original_name, filename_only=True
            ).with_explanation("Skipping invalid font").emit()
        restore_temp_file(temp_path, original_path, dry_run)
        return
    metadata.original_filename = original_name
    is_valid, reason = is_valid_postscript_name(metadata.ps_name)
    if not is_valid and not rename_all:
        if console and verbose:
            cs.StatusIndicator("warning").add_file(
                original_name, filename_only=True
            ).with_explanation(f"Skipping: {reason}").emit()
        restore_temp_file(temp_path, original_path, dry_run)
        return
    font_metadata[temp_path] = metadata
    cache[original_name] = metadata


def _extract_and_group_metadata(
    temp_mapping: Dict[Path, Path],
    cache: Dict[str, FontMetadata],
    rename_all: bool,
    dry_run: bool,
    verbose: bool,
    stats: RenameStats,
) -> Tuple[Dict[Path, FontMetadata], Dict[str, List[FontMetadata]]]:
    """Extract metadata from fonts and group by PostScript name and format.
    Uses parallel extraction when uncached file count >= PARALLEL_EXTRACT_THRESHOLD.
    """
    font_metadata: Dict[Path, FontMetadata] = {}

    cached_items: List[Tuple[Path, Path, FontMetadata]] = []
    need_extraction: List[Tuple[Path, Path, Path]] = []

    for temp_path, original_path in temp_mapping.items():
        original_name = original_path.name
        metadata = None
        if original_name in cache:
            cached = cache[original_name]
            check_path = original_path if dry_run else temp_path
            if check_path.exists():
                st = check_path.stat()
                size_ok = cached.file_size == st.st_size
                mtime_ok = (
                    cached.file_mtime is None
                    or abs(cached.file_mtime - st.st_mtime)
                    < CACHE_MTIME_TOLERANCE_SEC
                )
                if size_ok and mtime_ok:
                    metadata = cached
                    metadata.file_path = str(
                        temp_path if not dry_run else original_path
                    )
        if metadata is not None:
            cached_items.append((temp_path, original_path, metadata))
        else:
            extract_path = original_path if dry_run else temp_path
            need_extraction.append((temp_path, original_path, extract_path))

    for temp_path, original_path, meta in cached_items:
        _process_extracted_metadata(
            temp_path,
            original_path,
            meta,
            cache,
            rename_all,
            dry_run,
            verbose,
            stats,
            font_metadata,
        )

    if need_extraction:
        use_parallel = len(need_extraction) >= PARALLEL_EXTRACT_THRESHOLD
        extract_paths = [t[2] for t in need_extraction]

        if use_parallel:
            n_workers = min(
                PARALLEL_MAX_WORKERS_DEFAULT,
                len(need_extraction),
            )
            if console and verbose:
                cs.StatusIndicator("info").with_explanation(
                    f"Extracting metadata in parallel ({n_workers} workers, "
                    f"{len(need_extraction)} files)"
                ).emit()
            with Pool(processes=n_workers) as pool:
                results = pool.map(_extract_metadata_worker, extract_paths)
        else:
            results = [
                extract_metadata(extract_path) for extract_path in extract_paths
            ]

        for (temp_path, original_path, _), meta in zip(need_extraction, results):
            _process_extracted_metadata(
                temp_path,
                original_path,
                meta,
                cache,
                rename_all,
                dry_run,
                verbose,
                stats,
                font_metadata,
            )

    if not font_metadata:
        return {}, {}

    # Group by PostScript name AND format
    ps_name_groups: Dict[str, List[FontMetadata]] = {}
    for metadata in font_metadata.values():
        ps_name = metadata.ps_name
        if metadata.detected_format:
            format_key = metadata.detected_format
        else:
            format_key = Path(metadata.file_path).suffix.lower().lstrip(".")
        group_key = f"{ps_name}|{format_key}"
        if group_key not in ps_name_groups:
            ps_name_groups[group_key] = []
        ps_name_groups[group_key].append(metadata)

    return font_metadata, ps_name_groups


def _validate_and_prepare(
    directory: Path,
    dry_run: bool,
    verbose: bool,
    specific_files: Optional[List[Path]] = None,
) -> Tuple[List[Path], Dict[Path, Path], Dict[str, FontMetadata]]:
    """
    Validate directory and prepare for processing.
    Returns font_files, temp_mapping, and cache.
    Rejects symlinked directories and symlinked font files for safety.
    """
    try:
        if directory.is_symlink():
            raise ValueError(
                f"Symlinks not allowed: {directory}. Use the real path."
            )
    except OSError as e:
        raise ValueError(f"Cannot resolve directory {directory}: {e}") from e

    font_files, temp_mapping = _prepare_directory(
        directory, dry_run, verbose, specific_files
    )
    cache = load_cache(directory)
    return font_files, temp_mapping, cache


def _execute_rename_phase(
    rename_map: Dict[Path, str],
    font_metadata: Dict[Path, FontMetadata],
    dry_run: bool,
    verbose: bool,
) -> RenameStats:
    """
    Execute the final rename phase from temp names to final names.
    Returns rename statistics.
    """
    return execute_final_renames(rename_map, font_metadata, dry_run, verbose)


def process_directory(
    directory: Path,
    rename_all: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    use_typographic_names: bool = False,
    specific_files: Optional[List[Path]] = None,
) -> RenameStats:
    """
    Process all font files in a single directory.
    Includes temp file cleanup guarantee via try-finally.
    """
    stats = RenameStats()
    temp_mapping: Dict[Path, Path] = {}

    try:
        font_files, temp_mapping, cache = _validate_and_prepare(
            directory, dry_run, verbose, specific_files
        )
        if not font_files:
            return stats

        stats.total_files = len(font_files)

        font_metadata, ps_name_groups = _extract_and_group_metadata(
            temp_mapping, cache, rename_all, dry_run, verbose, stats
        )

        if not dry_run:
            save_cache(directory, cache)

        if not font_metadata:
            return stats

        rename_map = assign_final_names(
            ps_name_groups, use_typographic_names=use_typographic_names
        )

        rename_stats = _execute_rename_phase(
            rename_map, font_metadata, dry_run, verbose
        )
        stats.renamed = rename_stats.renamed
        stats.skipped += rename_stats.skipped
        stats.errors.extend(rename_stats.errors)

        if not dry_run:
            cleanup_cache(directory)

    except Exception as e:
        # Cleanup temp files on any error
        if temp_mapping:
            if console:
                cs.StatusIndicator("warning").with_explanation(
                    f"Error during processing, restoring temp files: {e}"
                ).emit()
            cleanup_failures = cleanup_temp_files(temp_mapping, dry_run)
            if cleanup_failures > 0 and console:
                cs.StatusIndicator("warning").with_explanation(
                    f"{cleanup_failures} temp file(s) could not be restored; "
                    "check permissions and restore manually if needed."
                ).emit()
        raise

    return stats


# ============================================================================
# Preview & Analysis
# ============================================================================


@dataclass
class RenamePreview:
    """Preview information for a single file rename"""

    original_path: Path
    original_name: str
    new_name: str
    ps_name: str
    priority: int
    quality_score: float = 0.0
    metadata: Optional[FontMetadata] = None


def _group_metadata_for_preview(
    font_metadata: Dict[Path, FontMetadata],
) -> Tuple[Dict[str, List[FontMetadata]], Dict[Path, Path]]:
    """
    Group metadata by PostScript name and format for preview.
    Returns ps_name_groups and path_to_original mapping.
    """
    ps_name_groups: Dict[str, List[FontMetadata]] = {}
    path_to_original: Dict[Path, Path] = {}

    for font_path, metadata in font_metadata.items():
        ps_name = metadata.ps_name
        if metadata.detected_format:
            format_key = metadata.detected_format
        else:
            format_key = Path(metadata.file_path).suffix.lower().lstrip(".")
        group_key = f"{ps_name}|{format_key}"
        if group_key not in ps_name_groups:
            ps_name_groups[group_key] = []
        ps_name_groups[group_key].append(metadata)
        # Map the file_path (as Path) back to original font_path key using normalized paths
        path_to_original[normalize_path(Path(metadata.file_path))] = normalize_path(
            font_path
        )

    return ps_name_groups, path_to_original


def analyze_renames(
    font_paths: List[str],
    rename_all: bool = False,
    use_typographic_names: bool = False,
) -> Dict[str, List[RenamePreview]]:
    """Analyze what renames would occur without actually performing them"""
    dirs_to_process = group_files_by_directory(font_paths)
    previews_by_dir: Dict[str, List[RenamePreview]] = {}

    for directory, files_in_dir in dirs_to_process.items():
        font_files = [Path(f) for f in files_in_dir]

        cache = load_cache(directory)

        font_metadata: Dict[Path, FontMetadata] = {}
        for font_path in font_files:
            metadata = extract_metadata(font_path)

            if metadata is None:
                continue

            is_valid, _ = is_valid_postscript_name(metadata.ps_name)
            if not is_valid and not rename_all:
                continue

            metadata.original_filename = font_path.name
            font_metadata[font_path] = metadata
            cache[font_path.name] = metadata

        if not font_metadata:
            continue

        # Group by PostScript name AND format
        ps_name_groups, path_to_original = _group_metadata_for_preview(font_metadata)

        # assign_final_names calculates quality scores via sort_by_quality_score
        # Since metadata objects are shared, scores are set on the objects in font_metadata
        rename_map = assign_final_names(
            ps_name_groups, use_typographic_names=use_typographic_names
        )

        previews = []
        for original_path, new_name in rename_map.items():
            if original_path.name == new_name:
                continue

            # Resolve paths for consistent matching using normalized paths
            resolved_path = normalize_path(original_path)
            original_font_path = path_to_original.get(resolved_path)
            if original_font_path is None:
                # Fallback: try direct lookup with normalized path
                original_font_path = original_path
            meta = font_metadata.get(original_font_path)
            if not meta:
                continue

            ps_name = meta.ps_name
            priority = 0
            if "~" in new_name:
                try:
                    match = re.search(r"~(\d{3})", new_name)
                    if match:
                        priority = int(match.group(1))
                except (ValueError, AttributeError):
                    pass

            previews.append(
                RenamePreview(
                    original_path=original_path,
                    original_name=original_path.name,
                    new_name=new_name,
                    ps_name=ps_name,
                    priority=priority,
                    quality_score=meta.quality_score or 0.0,
                    metadata=meta,
                )
            )

        if previews:
            previews_by_dir[str(directory)] = previews

    return previews_by_dir


def highlight_differences_pair(original: str, new: str) -> Tuple[str, str]:
    """Highlight differences between two strings using StatusIndicator colors"""
    if not cs.RICH_AVAILABLE or original == new:
        return original, new

    prefix_len = 0
    min_len = min(len(original), len(new))
    while prefix_len < min_len and original[prefix_len] == new[prefix_len]:
        prefix_len += 1

    suffix_len = 0
    orig_remaining = len(original) - prefix_len
    new_remaining = len(new) - prefix_len
    while (
        suffix_len < min(orig_remaining, new_remaining)
        and original[len(original) - 1 - suffix_len] == new[len(new) - 1 - suffix_len]
    ):
        suffix_len += 1

    orig_parts = []
    if prefix_len > 0:
        orig_parts.append(original[:prefix_len])
    if prefix_len < len(original) - suffix_len:
        diff_part = original[prefix_len : len(original) - suffix_len]
        orig_parts.append(f"[value.before]{diff_part}[/value.before]")
    if suffix_len > 0:
        orig_parts.append(original[-suffix_len:])
    highlighted_original = "".join(orig_parts)

    new_parts = []
    if prefix_len > 0:
        new_parts.append(new[:prefix_len])
    if prefix_len < len(new) - suffix_len:
        diff_part = new[prefix_len : len(new) - suffix_len]
        new_parts.append(f"[value.after]{diff_part}[/value.after]")
    if suffix_len > 0:
        new_parts.append(new[-suffix_len:])
    highlighted_new = "".join(new_parts)

    return highlighted_original, highlighted_new


def format_language_support(lang_set: set) -> str:
    """Format language support for display"""
    if not lang_set:
        return "Latin"

    langs = []
    if "cyrillic" in lang_set and "latin-extended" in lang_set:
        langs.append("Pan-European")
    elif "cyrillic" in lang_set:
        langs.append("Cyrillic")
    elif "latin-extended" in lang_set:
        langs.append("Latin-Ext")

    if "greek" in lang_set and "greek" not in [lang.lower() for lang in langs]:
        langs.append("Greek")
    if "vietnamese" in lang_set:
        langs.append("Vietnamese")
    if "arabic" in lang_set:
        langs.append("Arabic")
    if "hebrew" in lang_set:
        langs.append("Hebrew")

    return ", ".join(langs) if langs else "Latin"


def show_preflight_preview(
    previews_by_dir: Dict[str, List[RenamePreview]],
    show_quality: bool = False,
    explain_quality: bool = False,
) -> None:
    """
    Display a preview of what will be renamed.

    Args:
        previews_by_dir: Dictionary mapping directory paths to list of previews
        show_quality: If True, show quality scores and details
        explain_quality: If True, show quality comparison explanations
    """
    cs.emit("")
    cs.StatusIndicator("info").add_message("Rename Preview").emit()

    total_files = sum(len(previews) for previews in previews_by_dir.values())
    total_dirs = len(previews_by_dir)

    cs.emit(f"{cs.indent(1)}Total files to rename: {cs.fmt_count(total_files)}")
    cs.emit(f"{cs.indent(1)}Directories affected: {cs.fmt_count(total_dirs)}")
    cs.emit("")

    if cs.RICH_AVAILABLE and console:
        table = cs.create_table(show_header=True)
        if table:
            table.add_column("Original Name", style="lighttext", no_wrap=False)
            table.add_column("New Name", style="lighttext", no_wrap=False)

            if show_quality:
                table.add_column("Quality", style="cyan", justify="right", width=8)
                table.add_column("Rev", style="dim", justify="right", width=6)
                table.add_column("Languages", style="dim", no_wrap=False, width=15)
                table.add_column("Features", style="dim", justify="right", width=8)

            for dir_path, previews in sorted(previews_by_dir.items()):
                # Group previews by PostScript name for quality explanation
                grouped_previews: Dict[str, List[RenamePreview]] = {}
                for preview in previews:
                    if preview.ps_name not in grouped_previews:
                        grouped_previews[preview.ps_name] = []
                    grouped_previews[preview.ps_name].append(preview)

                for preview in sorted(previews, key=lambda p: p.original_name):
                    highlighted_orig, highlighted_new = highlight_differences_pair(
                        preview.original_name, preview.new_name
                    )

                    if show_quality and preview.metadata:
                        meta = preview.metadata
                        table.add_row(
                            highlighted_orig,
                            highlighted_new,
                            f"{meta.quality_score:.0f}",
                            f"{meta.font_revision:.2f}",
                            format_language_support(meta.language_support),
                            f"{len(meta.opentype_features)}",
                        )
                    else:
                        table.add_row(highlighted_orig, highlighted_new)

                # Show quality explanations if requested
                if explain_quality and show_quality:
                    for ps_name, group_previews in grouped_previews.items():
                        if len(group_previews) > 1:
                            # Sort by quality score
                            sorted_group = sorted(
                                group_previews,
                                key=lambda p: p.quality_score or 0.0,
                                reverse=True,
                            )
                            if len(sorted_group) >= 2:
                                top_font = sorted_group[0]
                                second_font = sorted_group[1]
                                if top_font.metadata and second_font.metadata:
                                    # Get all fonts in group for comparison
                                    all_metadata = [
                                        p.metadata for p in sorted_group if p.metadata
                                    ]
                                    breakdown = explain_quality_comparison(
                                        top_font.metadata,
                                        second_font.metadata,
                                        all_metadata,
                                    )

                                    cs.emit("")
                                    cs.StatusIndicator("info").add_message(
                                        f"Quality comparison for '{ps_name}'"
                                    ).emit()
                                    cs.emit(
                                        f"{cs.indent(1)}{top_font.new_name} wins over {second_font.new_name} because:"
                                    )
                                    for component, (
                                        score_a,
                                        score_b,
                                        diff,
                                    ) in breakdown.items():
                                        if (
                                            abs(diff) > 0.1
                                        ):  # Only show significant differences
                                            sign = "+" if diff > 0 else ""
                                            cs.emit(
                                                f"{cs.indent(2)}• {component.capitalize()}: {sign}{diff:.0f} points "
                                                f"({score_a:.0f} vs {score_b:.0f})"
                                            )

            console.print(table)
    else:
        for dir_path, previews in sorted(previews_by_dir.items()):
            for preview in sorted(previews, key=lambda p: p.original_name):
                cs.emit(f"{cs.indent(1)}{preview.original_name} -> {preview.new_name}")

    cs.emit("")


# ============================================================================
# Main Entry Point
# ============================================================================


def group_files_by_directory(font_paths: List[str]) -> Dict[Path, List[Path]]:
    """Group font file paths by their parent directory"""
    dirs_to_process: Dict[Path, List[Path]] = {}
    for font_path_str in font_paths:
        font_path = Path(font_path_str)
        parent_dir = font_path.parent
        if parent_dir not in dirs_to_process:
            dirs_to_process[parent_dir] = []
        dirs_to_process[parent_dir].append(font_path)
    return dirs_to_process


def recover_orphaned_temp_files(directory: Path) -> int:
    """
    Recover orphaned temp files from previous interrupted runs.
    Looks for files matching _tmp_*.{ext} pattern and attempts to restore them.

    Args:
        directory: Directory to scan for orphaned temp files

    Returns:
        Number of files recovered
    """
    recovered_count = 0

    # Look for temp files
    temp_files = []
    for ext in FONT_EXTENSIONS:
        temp_files.extend(directory.glob(f"_tmp_*{ext}"))
        temp_files.extend(directory.glob(f"_tmp_*{ext.upper()}"))

    if not temp_files:
        return 0

    # Try to load transaction log if it exists
    transaction_path = directory / TRANSACTION_FILENAME
    temp_to_original: Dict[str, str] = {}

    if transaction_path.exists():
        try:
            with open(
                transaction_path, "r", encoding="utf-8", errors="replace"
            ) as f:
                transaction_data = json.load(f)
                if (
                    isinstance(transaction_data, dict)
                    and "temp_mapping" in transaction_data
                ):
                    temp_to_original = transaction_data["temp_mapping"]
        except Exception as e:
            if console:
                cs.StatusIndicator("warning").with_explanation(
                    f"Could not read transaction log: {e}"
                ).emit()

    for temp_file in temp_files:
        original_name = temp_to_original.get(temp_file.name)
        if not original_name:
            # Try to extract from cache if available
            cache = load_cache(directory)
            for cached_name, metadata in cache.items():
                if metadata.file_path == str(temp_file):
                    original_name = cached_name
                    break

        if original_name:
            original_path = directory / original_name
            if not original_path.exists():
                try:
                    temp_file.rename(original_path)
                    recovered_count += 1
                    if console:
                        cs.StatusIndicator("updated").add_values(
                            old_value=temp_file.name, new_value=original_name
                        ).emit()
                except Exception as e:
                    if console:
                        cs.StatusIndicator("error").add_file(
                            temp_file.name, filename_only=True
                        ).with_explanation(f"Failed to recover: {e}").emit()
            else:
                # Original exists, temp is truly orphaned - delete it
                try:
                    temp_file.unlink()
                    if console:
                        cs.StatusIndicator("info").add_file(
                            temp_file.name, filename_only=True
                        ).with_explanation(
                            "Removed orphaned temp file (original exists)"
                        ).emit()
                except Exception as e:
                    if console:
                        cs.StatusIndicator("warning").add_file(
                            temp_file.name, filename_only=True
                        ).with_explanation(
                            f"Could not remove orphaned temp file: {e}"
                        ).emit()
        else:
            if console:
                cs.StatusIndicator("warning").add_file(
                    temp_file.name, filename_only=True
                ).with_explanation(
                    "Cannot determine original name - manual recovery needed"
                ).emit()

    return recovered_count


def show_directory_stats(dir_stats: RenameStats, verbose: bool) -> None:
    """Display statistics for a single directory"""
    if console and not verbose:
        cs.StatusIndicator("info").add_message(
            f"Renamed: {cs.fmt_count(dir_stats.renamed)} | Skipped: {cs.fmt_count(dir_stats.skipped)} | Invalid: {cs.fmt_count(dir_stats.invalid)}"
        ).emit()


def main():
    # On Windows (spawn), child processes re-import this module and would run main();
    # exit immediately so only the parent runs the CLI and creates Pools.
    if current_process().name != "MainProcess":
        return 0
    parser = argparse.ArgumentParser(
        description="Rename font files to PostScript names with intelligent quality scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/fonts/              # Rename all fonts in directory
  %(prog)s font1.otf font2.otf         # Rename specific files
  %(prog)s /fonts/ -r         # Process directory recursively
  %(prog)s /fonts/ -n           # Preview changes
  %(prog)s /fonts/ --show-quality  # Show quality scores in preview
        """,
    )

    parser.add_argument(
        "paths",
        nargs="*",
        help="Font files or directories to process (default: current directory)",
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true", help="Process directories recursively"
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Preview changes without renaming files",
    )
    parser.add_argument(
        "-ra",
        "--rename-all",
        action="store_true",
        help="Rename files even with invalid PostScript names",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed processing information",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip preflight preview and proceed directly",
    )
    parser.add_argument(
        "--show-quality",
        action="store_true",
        help="Display quality scores and details in preview",
    )
    parser.add_argument(
        "--use-typographic-names",
        action="store_true",
        help="Use nameID 16 (Family) and 17 (Style) for filenames instead of PostScript names",
    )
    parser.add_argument(
        "--explain-quality",
        action="store_true",
        help="Show detailed quality score comparisons explaining why one font wins over another",
    )
    parser.add_argument(
        "--recover",
        action="store_true",
        help="Recover orphaned temp files from previous interrupted runs",
    )

    args = parser.parse_args()

    if not args.paths:
        args.paths = ["."]

    # Handle recovery mode
    if args.recover:
        recovered_total = 0
        for path_str in args.paths:
            path = Path(path_str)
            if path.is_dir():
                if args.recursive:
                    for dir_path in path.rglob("*"):
                        if dir_path.is_dir():
                            recovered_total += recover_orphaned_temp_files(dir_path)
                else:
                    recovered_total += recover_orphaned_temp_files(path)
            elif path.is_file():
                recovered_total += recover_orphaned_temp_files(path.parent)

        if console:
            cs.print_panel(
                f"Recovered files: {cs.fmt_count(recovered_total)}",
                title="Recovery Complete",
                border_style="green",
            )
        return 0

    font_paths = collect_font_files(
        args.paths, recursive=args.recursive, allowed_extensions=FONT_EXTENSIONS
    )

    # Validate and fix extensions
    validated_paths = []
    for path_str in font_paths:
        path = Path(path_str)
        is_valid, fixed_path = validate_and_fix_extension(path, auto_fix=True)
        if fixed_path:
            if console:
                # INFO: Show filename with "Fixed extension" message
                cs.StatusIndicator("info").add_file(
                    str(path), filename_only=True
                ).with_explanation("Fixed extension").emit()

                # UPDATED: Show old → new with proper colors
                cs.StatusIndicator("updated").add_values(
                    old_value=path.name, new_value=fixed_path.name
                ).emit()

                # SAVED: Show new filename
                cs.StatusIndicator("saved").add_file(
                    str(fixed_path), filename_only=True
                ).emit()
            validated_paths.append(str(fixed_path))
        else:
            validated_paths.append(path_str)
    font_paths = validated_paths

    if not font_paths:
        if console:
            cs.StatusIndicator("error").with_explanation("No font files found").emit()
        return 1

    dirs_to_process = group_files_by_directory(font_paths)

    if console:
        # Use same mode string for both dry-run and normal mode
        # DRY prefix will be added automatically by StatusIndicator when dry_run=True
        mode = "RENAME"
        cs.print_panel(
            f"Mode: {mode}\n"
            f"Files: {cs.fmt_count(len(font_paths))}\n"
            f"Directories: {cs.fmt_count(len(dirs_to_process))}",
            title="Font File Renamer (Quality-Aware)",
            border_style="blue",
        )

    if not args.no_preview:
        previews_by_dir = analyze_renames(
            font_paths,
            rename_all=args.rename_all,
            use_typographic_names=args.use_typographic_names,
        )

        if previews_by_dir:
            show_preflight_preview(
                previews_by_dir,
                show_quality=args.show_quality,
                explain_quality=args.explain_quality,
            )

            if not args.dry_run:
                if not cs.prompt_confirm(
                    "Ready to rename font files to PostScript names",
                    action_prompt="Proceed with renaming?",
                    default=True,
                ):
                    if console:
                        cs.StatusIndicator("info").add_message(
                            "Operation cancelled"
                        ).emit()
                    return 0
        else:
            if console:
                cs.StatusIndicator("info").add_message(
                    "No files require renaming"
                ).emit()
            return 0

    total_stats = RenameStats()
    for idx, (directory, files_in_dir) in enumerate(sorted(dirs_to_process.items()), 1):
        if console:
            cs.StatusIndicator("info").add_message(
                f"Directory {idx}/{len(dirs_to_process)}: {cs.fmt_file_compact(str(directory))}"
            ).emit()

        # Pass specific files to process_directory to avoid processing all fonts in directory
        specific_files = files_in_dir if files_in_dir else None

        dir_stats = process_directory(
            directory,
            rename_all=args.rename_all,
            dry_run=args.dry_run,
            verbose=args.verbose,
            use_typographic_names=args.use_typographic_names,
            specific_files=specific_files,
        )

        total_stats.total_files += dir_stats.total_files
        total_stats.renamed += dir_stats.renamed
        total_stats.skipped += dir_stats.skipped
        total_stats.invalid += dir_stats.invalid
        total_stats.errors.extend(dir_stats.errors)

        show_directory_stats(dir_stats, args.verbose)

    if console:
        cs.print_panel(
            f"Total files: {cs.fmt_count(total_stats.total_files)}\n"
            f"Renamed: {cs.fmt_count(total_stats.renamed)}\n"
            f"Skipped: {cs.fmt_count(total_stats.skipped)}\n"
            f"Invalid: {cs.fmt_count(total_stats.invalid)}",
            title="Summary",
            border_style="green",
        )

    return 0


if __name__ == "__main__":
    exit(main())
