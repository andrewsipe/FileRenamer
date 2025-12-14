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
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field

from fontTools.ttLib import TTFont

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

console = cs.get_console()

# ============================================================================
# Constants
# ============================================================================

INDEX_FILENAME = ".font_rename_cache.json"
FONT_EXTENSIONS = {".ttf", ".otf", ".woff", ".woff2"}

# Quality scoring weights
WEIGHT_REVISION = 400  # 40% - Font revision number
WEIGHT_LANGUAGE = 2.5  # 25% - Language support breadth
WEIGHT_FEATURES = 2.0  # 20% - OpenType features
WEIGHT_GLYPHS = 100  # 10% - Meaningful glyph count increase
WEIGHT_RECENCY = 50  # 5% - Creation date recency

# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class FontMetadata:
    """Enhanced metadata with quality scoring"""

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

    # New quality indicators
    language_support: set = field(default_factory=set)
    opentype_features: set = field(default_factory=set)
    quality_score: Optional[float] = None

    # Typographic names (nameID 16 and 17)
    typographic_family: Optional[str] = None
    typographic_subfamily: Optional[str] = None

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
        return cls(**data)

    def calculate_quality_score(self, fonts_in_group: List["FontMetadata"]) -> float:
        """
        Calculate comprehensive quality score for prioritization.
        Higher score = better font to keep as primary.
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
        if self.head_created:
            # Mac epoch for Jan 1, 2020: 3786825600.0
            # Normalize to 0-50 range based on recency
            year_2020 = 3786825600.0
            if self.head_created >= year_2020:
                # Scale from 2020 to 2025 (5 years)
                recency = min((self.head_created - year_2020) / (86400 * 365 * 5), 1.0)
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


# ============================================================================
# Metadata Cache
# ============================================================================


def load_cache(directory: Path) -> Dict[str, FontMetadata]:
    """Load metadata cache from directory with corruption recovery"""
    cache_path = directory / INDEX_FILENAME
    if not cache_path.exists():
        return {}

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
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
                    cs.StatusIndicator("warning").add_file(filename).with_explanation(
                        f"Invalid cache entry: {e}"
                    ).emit()
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
    """Save metadata cache to directory"""
    cache_path = directory / INDEX_FILENAME
    try:
        data = {filename: meta.to_dict() for filename, meta in cache.items()}
        with open(cache_path, "w", encoding="utf-8") as f:
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


def detect_font_format(font: TTFont) -> str:
    """Detect font format from the font file data"""
    try:
        sfnt_version = font.reader.sfntVersion
        if sfnt_version == b"wOFF":
            return "woff"
        elif sfnt_version == b"wOF2":
            return "woff2"
        elif sfnt_version == b"OTTO":
            return "otf"
        elif sfnt_version == b"\x00\x01\x00\x00" or sfnt_version == b"true":
            return "ttf"
        else:
            if "CFF " in font:
                return "otf"
            return "ttf"
    except Exception:
        return "ttf"


def detect_language_support_from_font(font: TTFont) -> set:
    """Detect language support from Unicode coverage in opened font"""
    try:
        cmap = font.getBestCmap()
        if not cmap:
            return {"latin"}

        codepoints = set(cmap.keys())
        languages = set()

        # Latin: U+0000-U+007F (Basic Latin)
        if any(0x0000 <= cp <= 0x007F for cp in codepoints):
            languages.add("latin")

        # Cyrillic: U+0400-U+04FF
        if any(0x0400 <= cp <= 0x04FF for cp in codepoints):
            languages.add("cyrillic")

        # Greek: U+0370-U+03FF
        if any(0x0370 <= cp <= 0x03FF for cp in codepoints):
            languages.add("greek")

        # Extended Latin (Pan-European): U+0100-U+017F
        if any(0x0100 <= cp <= 0x017F for cp in codepoints):
            languages.add("latin-extended")

        # Vietnamese: U+1E00-U+1EFF
        if any(0x1E00 <= cp <= 0x1EFF for cp in codepoints):
            languages.add("vietnamese")

        # Arabic: U+0600-U+06FF
        if any(0x0600 <= cp <= 0x06FF for cp in codepoints):
            languages.add("arabic")

        # Hebrew: U+0590-U+05FF
        if any(0x0590 <= cp <= 0x05FF for cp in codepoints):
            languages.add("hebrew")

        return languages
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


def extract_metadata(font_path: Path) -> Optional[FontMetadata]:
    """Extract enhanced metadata from a font file"""
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

        # Detect font format from file data
        detected_format = detect_font_format(font)

        # Language support detection
        language_support = detect_language_support_from_font(font)

        # OpenType features extraction
        opentype_features = extract_opentype_features_from_font(font)

        file_size = font_path.stat().st_size

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
            language_support=language_support,
            opentype_features=opentype_features,
            typographic_family=typographic_family,
            typographic_subfamily=typographic_subfamily,
        )
    except Exception as e:
        if console:
            cs.StatusIndicator("error").add_file(font_path.name).with_explanation(
                f"Failed to read: {e}"
            ).emit()
        return None


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


def is_valid_postscript_name(ps_name: str) -> Tuple[bool, str]:
    """Validate PostScript name is safe for filename"""
    if not ps_name or ps_name.strip() == "":
        return False, "empty name"

    if ps_name.isspace():
        return False, "contains only spaces"

    for char in ps_name:
        code = ord(char)
        if code < 32 or code == 127:
            return False, f"contains control character (ASCII {code})"

    problematic_chars = ["?", "/", "\\", ":", "*", '"', "<", ">", "|"]
    for char in problematic_chars:
        if char in ps_name:
            return False, f"contains '{char}'"

    if ps_name.startswith(" ") or ps_name.endswith(" "):
        return False, "begins or ends with a space"

    forbidden_first_chars = ["_", "-", "."]
    if ps_name[0] in forbidden_first_chars:
        return False, f"begins with '{ps_name[0]}'"

    has_problem, pattern = contains_problematic_pattern(ps_name)
    if has_problem:
        return False, f"contains '{pattern}'"

    return True, ""


def generate_typographic_filename(
    typographic_family: Optional[str], typographic_subfamily: Optional[str]
) -> Optional[str]:
    """
    Generate filename from typographic family and subfamily.

    Args:
        typographic_family: nameID 16 (Typographic Family)
        typographic_subfamily: nameID 17 (Typographic Subfamily)

    Returns:
        Normalized filename in format "Family-Style" (spaces removed),
        or None if either field is empty/None
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
    return f"{family_normalized}-{subfamily_normalized}"


# ============================================================================
# Quality-Based Priority Sorting
# ============================================================================


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
                        font_path.name
                    ).with_explanation(f"Failed temp rename: {e}").emit()

    return temp_mapping


def assign_final_names(
    ps_name_groups: Dict[str, List[FontMetadata]],
    use_typographic_names: bool = False,
) -> Dict[Path, str]:
    """
    Assign final names based on PostScript name or typographic names and quality score.
    Highest quality gets clean name, others get ~001, ~002, etc.

    Args:
        ps_name_groups: Fonts grouped by PostScript name and format
        use_typographic_names: If True, use nameID 16/17 for filenames when available
    """
    rename_map: Dict[Path, str] = {}

    for group_key, metadata_list in ps_name_groups.items():
        # Use quality-based sorting
        sorted_fonts = sort_by_quality_score(metadata_list)

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

        # Fall back to PostScript name if typographic name not available
        if base_name is None:
            base_name = sorted_fonts[0].ps_name

        for idx, meta in enumerate(sorted_fonts):
            original_path = Path(meta.file_path)
            if meta.detected_format:
                ext = f".{meta.detected_format}"
            else:
                ext = original_path.suffix.lower()

            # Determine base name for this specific font
            font_base_name = base_name
            if use_typographic_names:
                typo_name = generate_typographic_filename(
                    meta.typographic_family, meta.typographic_subfamily
                )
                if typo_name:
                    # Use typographic name for this font
                    font_base_name = typo_name
                else:
                    # Fall back to this font's PostScript name
                    font_base_name = meta.ps_name

            if idx == 0:
                # Highest quality gets clean name
                new_name = f"{font_base_name}{ext}"
            else:
                # Lower quality gets counter suffix
                new_name = f"{font_base_name}~{idx:03d}{ext}"

            rename_map[original_path] = new_name

    return rename_map


def resolve_name_conflict(
    base_name: str, parent_dir: Path, exclude_path: Path
) -> Optional[str]:
    """
    Resolve naming conflicts by adding _conflict001, _conflict002, etc.
    Returns resolved name, or None if too many conflicts (>999).
    """
    target_path = parent_dir / base_name

    if not target_path.exists() or target_path == exclude_path:
        try:
            if not parent_dir.exists():
                parent_dir.mkdir(parents=True, exist_ok=True)
            if not os.access(parent_dir, os.W_OK):
                return None
        except (OSError, PermissionError):
            return None
        return base_name

    stem = Path(base_name).stem
    ext = Path(base_name).suffix

    for counter in range(1, 1000):
        new_name = f"{stem}_conflict{counter:03d}{ext}"
        target_path = parent_dir / new_name
        if not target_path.exists() or target_path == exclude_path:
            try:
                if not os.access(parent_dir, os.W_OK):
                    return None
            except (OSError, PermissionError):
                return None
            return new_name

    return None


def execute_single_rename(
    temp_path: Path, new_name: str, original_name: str, dry_run: bool, verbose: bool
) -> Tuple[bool, Optional[str]]:
    """Execute or preview a single rename"""
    if dry_run:
        if console and verbose:
            cs.StatusIndicator("info", dry_run=True).add_values(
                old_value=original_name, new_value=new_name
            ).emit()
        return True, None

    try:
        target_path = temp_path.parent / new_name
        temp_path.rename(target_path)
        if console and verbose:
            cs.StatusIndicator("updated").add_values(
                old_value=original_name, new_value=new_name
            ).emit()
        return True, None
    except Exception as e:
        if console:
            cs.StatusIndicator("error").add_file(original_name).with_explanation(
                f"Failed to rename: {e}"
            ).emit()
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

        # Skip if name unchanged
        target_path = temp_path.parent / new_name
        if temp_path == target_path:
            stats.skipped += 1
            continue

        # Handle naming conflicts
        resolved_name = resolve_name_conflict(new_name, temp_path.parent, temp_path)
        if resolved_name is None:
            stats.add_error(original_name, "too many naming conflicts")
            if console:
                cs.StatusIndicator("error").add_file(original_name).with_explanation(
                    "Too many conflicts"
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


def restore_temp_file(temp_path: Path, original_path: Path, dry_run: bool) -> None:
    """Restore a temp file to its original name with error handling"""
    if not dry_run:
        try:
            if temp_path.exists():
                temp_path.rename(original_path)
        except FileNotFoundError:
            pass
        except PermissionError as e:
            if console:
                cs.StatusIndicator("warning").add_file(
                    original_path.name
                ).with_explanation(
                    f"Cannot restore temp file (permission denied): {e}"
                ).emit()
        except OSError as e:
            if console:
                cs.StatusIndicator("warning").add_file(
                    original_path.name
                ).with_explanation(f"Cannot restore temp file: {e}").emit()
        except Exception as e:
            if console:
                cs.StatusIndicator("warning").add_file(
                    original_path.name
                ).with_explanation(f"Unexpected error restoring temp file: {e}").emit()


def process_single_font_metadata(
    temp_path: Path,
    original_path: Path,
    cache: Dict[str, FontMetadata],
    rename_all: bool,
    dry_run: bool,
    verbose: bool,
) -> Optional[FontMetadata]:
    """Extract and validate metadata for a single font"""
    original_name = original_path.name

    # Try cache first
    metadata = None
    if original_name in cache:
        cached = cache[original_name]
        if temp_path.exists() and cached.file_size == temp_path.stat().st_size:
            metadata = cached
            metadata.file_path = str(temp_path)

    # Extract if not cached
    if metadata is None:
        metadata = extract_metadata(temp_path)

    if metadata is None:
        if console:
            cs.StatusIndicator("warning").add_file(original_name).with_explanation(
                "Skipping invalid font"
            ).emit()
        restore_temp_file(temp_path, original_path, dry_run)
        return None

    metadata.original_filename = original_name

    # Validate PostScript name
    is_valid, reason = is_valid_postscript_name(metadata.ps_name)
    if not is_valid and not rename_all:
        if console and verbose:
            cs.StatusIndicator("warning").add_file(original_name).with_explanation(
                f"Skipping: {reason}"
            ).emit()
        restore_temp_file(temp_path, original_path, dry_run)
        return None

    return metadata


def collect_directory_fonts(directory: Path) -> List[Path]:
    """Collect font files from directory, excluding temp files and cache"""
    font_files = []
    for ext in FONT_EXTENSIONS:
        font_files.extend(directory.glob(f"*{ext}"))
        font_files.extend(directory.glob(f"*{ext.upper()}"))

    seen = set()
    result = []
    for f in font_files:
        if f not in seen:
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
    directory: Path, dry_run: bool, verbose: bool
) -> Tuple[List[Path], Dict[Path, Path]]:
    """Prepare directory for processing: collect files, check space, rename to temp"""
    font_files = collect_directory_fonts(directory)
    if not font_files:
        return [], {}

    if console and verbose:
        cs.StatusIndicator("info").add_message(
            f"Processing {cs.fmt_count(len(font_files))} files in {cs.fmt_file(str(directory))}"
        ).emit()

    if not dry_run:
        estimated_space = len(font_files) * 100 * 1024
        if not check_filesystem_space(directory, estimated_space):
            if console:
                cs.StatusIndicator("warning").with_explanation(
                    "Insufficient filesystem space - operations may fail"
                ).emit()

    temp_mapping = rename_to_temp(font_files, dry_run)

    return font_files, temp_mapping


def _extract_and_group_metadata(
    temp_mapping: Dict[Path, Path],
    cache: Dict[str, FontMetadata],
    rename_all: bool,
    dry_run: bool,
    verbose: bool,
    stats: RenameStats,
) -> Tuple[Dict[Path, FontMetadata], Dict[str, List[FontMetadata]]]:
    """Extract metadata from fonts and group by PostScript name and format"""
    font_metadata: Dict[Path, FontMetadata] = {}

    for temp_path, original_path in temp_mapping.items():
        metadata = process_single_font_metadata(
            temp_path, original_path, cache, rename_all, dry_run, verbose
        )

        if metadata is None:
            stats.invalid += 1
            continue

        font_metadata[temp_path] = metadata
        cache[original_path.name] = metadata

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


def process_directory(
    directory: Path,
    rename_all: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    use_typographic_names: bool = False,
) -> RenameStats:
    """Process all font files in a single directory"""
    stats = RenameStats()

    font_files, temp_mapping = _prepare_directory(directory, dry_run, verbose)
    if not font_files:
        return stats

    stats.total_files = len(font_files)

    cache = load_cache(directory)

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

    rename_stats = execute_final_renames(rename_map, font_metadata, dry_run, verbose)
    stats.renamed = rename_stats.renamed
    stats.skipped += rename_stats.skipped
    stats.errors.extend(rename_stats.errors)

    if not dry_run:
        cleanup_cache(directory)

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
        ps_name_groups: Dict[str, List[FontMetadata]] = {}
        # Create reverse mapping from file_path string to original font_path key
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
            # Map the file_path (as Path) back to original font_path key
            path_to_original[Path(metadata.file_path).resolve()] = font_path.resolve()

        # assign_final_names calculates quality scores via sort_by_quality_score
        # Since metadata objects are shared, scores are set on the objects in font_metadata
        rename_map = assign_final_names(
            ps_name_groups, use_typographic_names=use_typographic_names
        )

        previews = []
        for original_path, new_name in rename_map.items():
            if original_path.name == new_name:
                continue

            # Resolve paths for consistent matching
            resolved_path = original_path.resolve()
            original_font_path = path_to_original.get(resolved_path)
            if original_font_path is None:
                # Fallback: try direct lookup
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
    previews_by_dir: Dict[str, List[RenamePreview]], show_quality: bool = False
) -> None:
    """Display a preview of what will be renamed"""
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


def show_directory_stats(dir_stats: RenameStats, verbose: bool) -> None:
    """Display statistics for a single directory"""
    if console and not verbose:
        cs.StatusIndicator("info").add_message(
            f"Renamed: {cs.fmt_count(dir_stats.renamed)} | Skipped: {cs.fmt_count(dir_stats.skipped)} | Invalid: {cs.fmt_count(dir_stats.invalid)}"
        ).emit()


def main():
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

    args = parser.parse_args()

    if not args.paths:
        args.paths = ["."]

    font_paths = collect_font_files(
        args.paths, recursive=args.recursive, allowed_extensions=FONT_EXTENSIONS
    )

    if not font_paths:
        if console:
            cs.StatusIndicator("error").with_explanation("No font files found").emit()
        return 1

    dirs_to_process = group_files_by_directory(font_paths)

    if console:
        mode = "DRY RUN" if args.dry_run else "RENAME"
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
            show_preflight_preview(previews_by_dir, show_quality=args.show_quality)

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
                f"Directory {idx}/{len(dirs_to_process)}: {cs.fmt_file(str(directory))}"
            ).emit()

        dir_stats = process_directory(
            directory,
            rename_all=args.rename_all,
            dry_run=args.dry_run,
            verbose=args.verbose,
            use_typographic_names=args.use_typographic_names,
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
