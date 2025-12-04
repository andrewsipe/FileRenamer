#!/usr/bin/env python3
"""
Font File Renamer - PostScript name-based renaming with version priority

Renames font files to their PostScript names with intelligent version handling:
- Two-pass renaming (temp UUID → PostScript names)
- Version-aware priority naming (highest version gets clean name)
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
    -ra, --rename-all     Rename even fonts with invalid PostScript names
    -V, --verbose,        Show detailed processing information

Future enhancement note:
    Could integrate term reordering (width/weight/style) like Filename_Reorder_*.py
    to normalize PostScript name structure before renaming files.
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

# Core module imports
import FontCore.core_console_styles as cs
from FontCore.core_file_collector import collect_font_files

console = cs.get_console()

# ============================================================================
# Constants
# ============================================================================

INDEX_FILENAME = ".font_rename_cache.json"
FONT_EXTENSIONS = {".ttf", ".otf", ".woff", ".woff2"}

# ============================================================================
# Data Classes
# ============================================================================


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
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FontMetadata":
        return cls(**data)


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

        # Validate cache structure
        if not isinstance(data, dict):
            raise ValueError("Cache is not a dictionary")

        cache = {}
        for filename, meta in data.items():
            try:
                if not isinstance(meta, dict):
                    continue
                cache[filename] = FontMetadata.from_dict(meta)
            except Exception as e:
                # Skip invalid entries but continue loading others
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
        # Remove corrupted cache
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
        # Remove problematic cache
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
    """
    Detect font format from the font file data.
    Returns lowercase format string: "otf", "ttf", "woff", or "woff2"
    """
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
            # Fallback: check for CFF table (OTF uses CFF, TTF uses glyf)
            if "CFF " in font:
                return "otf"
            return "ttf"
    except Exception:
        # If detection fails, default to ttf
        return "ttf"


def extract_metadata(font_path: Path) -> Optional[FontMetadata]:
    """Extract metadata from a font file"""
    try:
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

    # Check for whitespace-only
    if ps_name.isspace():
        return False, "contains only spaces"

    # Check for control characters
    for char in ps_name:
        code = ord(char)
        if code < 32 or code == 127:
            return False, f"contains control character (ASCII {code})"

    # Check for problematic characters
    problematic_chars = ["?", "/", "\\", ":", "*", '"', "<", ">", "|"]
    for char in problematic_chars:
        if char in ps_name:
            return False, f"contains '{char}'"

    # Check for leading/trailing spaces
    if ps_name.startswith(" ") or ps_name.endswith(" "):
        return False, "begins or ends with a space"

    # Check for forbidden first characters
    forbidden_first_chars = ["_", "-", "."]
    if ps_name[0] in forbidden_first_chars:
        return False, f"begins with '{ps_name[0]}'"

    # Check for problematic patterns
    has_problem, pattern = contains_problematic_pattern(ps_name)
    if has_problem:
        return False, f"contains '{pattern}'"

    return True, ""


# ============================================================================
# Priority Sorting
# ============================================================================


def sort_by_version_priority(metadata_list: List[FontMetadata]) -> List[FontMetadata]:
    """
    Sort fonts by version priority:
    1. Highest font revision
    2. Oldest creation date (original design)
    3. Most recent modification date (latest fixes)
    """

    def sort_key(meta: FontMetadata) -> tuple:
        return (
            -meta.font_revision if meta.font_revision else float("-inf"),
            meta.head_created if meta.head_created else float("inf"),
            -meta.head_modified if meta.head_modified else float("-inf"),
        )

    return sorted(metadata_list, key=sort_key)


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
) -> Dict[Path, str]:
    """
    Assign final names based on PostScript name and version priority.
    Highest version gets clean name, others get ~001, ~002, etc.
    """
    rename_map: Dict[Path, str] = {}

    for group_key, metadata_list in ps_name_groups.items():
        sorted_fonts = sort_by_version_priority(metadata_list)
        # Get ps_name from metadata (all items in group have same ps_name)
        ps_name = sorted_fonts[0].ps_name

        for idx, meta in enumerate(sorted_fonts):
            original_path = Path(meta.file_path)
            # Use detected format if available, otherwise fallback to file extension
            if meta.detected_format:
                ext = f".{meta.detected_format}"
            else:
                ext = original_path.suffix.lower()

            if idx == 0:
                # Highest priority gets clean name
                new_name = f"{ps_name}{ext}"
            else:
                # Lower priority gets counter suffix
                new_name = f"{ps_name}~{idx:03d}{ext}"

            rename_map[original_path] = new_name

    return rename_map


def resolve_name_conflict(
    base_name: str, parent_dir: Path, exclude_path: Path
) -> Optional[str]:
    """
    Resolve naming conflicts by adding _conflict001, _conflict002, etc.
    Returns resolved name, or None if too many conflicts (>999).
    Validates that target path is writable before returning.
    """
    target_path = parent_dir / base_name

    if not target_path.exists() or target_path == exclude_path:
        # Validate parent directory is writable
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
            # Validate parent directory is writable
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
    """
    Execute or preview a single rename.
    Returns (success, error_message)
    """
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
    """
    Phase 2: Execute final renames from temp names to PostScript names
    """
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
            # Temp file already gone, nothing to restore
            pass
        except PermissionError as e:
            if console:
                cs.StatusIndicator("error").add_file(
                    original_path.name
                ).with_explanation(
                    f"Cannot restore temp file (permission denied): {e}"
                ).emit()
        except OSError as e:
            if console:
                cs.StatusIndicator("error").add_file(
                    original_path.name
                ).with_explanation(f"Cannot restore temp file: {e}").emit()
        except Exception as e:
            if console:
                cs.StatusIndicator("error").add_file(
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
    """
    Extract and validate metadata for a single font.
    Returns None if font should be skipped.
    """
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
    """Collect font files from directory, excluding temp files and cache.
    Handles case-insensitive extensions (e.g., .otf, .OTF, .Otf).
    """
    font_files = []
    for ext in FONT_EXTENSIONS:
        # Search for both lowercase and uppercase extension variants
        font_files.extend(directory.glob(f"*{ext}"))
        font_files.extend(directory.glob(f"*{ext.upper()}"))

    # Deduplicate (same file might match both patterns) and filter
    seen = set()
    result = []
    for f in font_files:
        if f not in seen:
            # Exclude temp files and cache
            if not f.name.startswith("_tmp_") and f.name != INDEX_FILENAME:
                seen.add(f)
                result.append(f)

    return result


def check_filesystem_space(directory: Path, required_bytes: int) -> bool:
    """Check if filesystem has sufficient space for operations"""
    try:
        stat = shutil.disk_usage(directory)
        free_space = stat.free
        # Require at least 2x the space needed for safety
        return free_space >= (required_bytes * 2)
    except OSError:
        # If we can't check, assume it's okay (don't block operations)
        return True


def _prepare_directory(
    directory: Path, dry_run: bool, verbose: bool
) -> Tuple[List[Path], Dict[Path, Path]]:
    """
    Prepare directory for processing: collect files, check space, rename to temp.

    Returns:
        Tuple of (font_files, temp_mapping)
    """
    font_files = collect_directory_fonts(directory)
    if not font_files:
        return [], {}

    if console and verbose:
        cs.StatusIndicator("info").add_message(
            f"Processing {cs.fmt_count(len(font_files))} files in {cs.fmt_file(str(directory))}"
        ).emit()

    # Check filesystem space before operations (estimate: assume average 100KB per font)
    if not dry_run:
        estimated_space = len(font_files) * 100 * 1024
        if not check_filesystem_space(directory, estimated_space):
            if console:
                cs.StatusIndicator("warning").with_explanation(
                    "Insufficient filesystem space - operations may fail"
                ).emit()

    # Phase 1: Rename to temp names
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
    """
    Extract metadata from fonts and group by PostScript name and format.
    OTF and TTF files with the same PostScript name are treated as separate files.

    Returns:
        Tuple of (font_metadata, ps_name_groups)
    """
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

    # Group by PostScript name AND format (OTF and TTF with same name are different files)
    ps_name_groups: Dict[str, List[FontMetadata]] = {}
    for metadata in font_metadata.values():
        ps_name = metadata.ps_name
        # Use detected format, fallback to file extension if not detected
        if metadata.detected_format:
            format_key = metadata.detected_format
        else:
            # Fallback to file extension for grouping
            format_key = Path(metadata.file_path).suffix.lower().lstrip(".")
        # Composite key: PostScript name + format
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
) -> RenameStats:
    """Process all font files in a single directory"""
    stats = RenameStats()

    # Prepare directory: collect files, check space, rename to temp
    font_files, temp_mapping = _prepare_directory(directory, dry_run, verbose)
    if not font_files:
        return stats

    stats.total_files = len(font_files)

    # Load cache
    cache = load_cache(directory)

    # Extract metadata and group by PostScript name
    font_metadata, ps_name_groups = _extract_and_group_metadata(
        temp_mapping, cache, rename_all, dry_run, verbose, stats
    )

    # Save cache
    if not dry_run:
        save_cache(directory, cache)

    if not font_metadata:
        return stats

    # Assign final names with priority
    rename_map = assign_final_names(ps_name_groups)

    # Execute final renames
    rename_stats = execute_final_renames(rename_map, font_metadata, dry_run, verbose)
    stats.renamed = rename_stats.renamed
    stats.skipped += rename_stats.skipped
    stats.errors.extend(rename_stats.errors)

    # Cleanup cache
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
    priority: int  # 0 = highest priority (clean name), 1+ = ~001, ~002, etc.


def analyze_renames(
    font_paths: List[str],
    rename_all: bool = False,
) -> Dict[str, List[RenamePreview]]:
    """
    Analyze what renames would occur without actually performing them.
    Returns dict mapping directory -> list of RenamePreview objects.
    """
    dirs_to_process = group_files_by_directory(font_paths)
    previews_by_dir: Dict[str, List[RenamePreview]] = {}

    for directory, files_in_dir in dirs_to_process.items():
        # Simulate the renaming process without actually renaming
        font_files = [Path(f) for f in files_in_dir]

        # Load cache
        cache = load_cache(directory)

        # Extract metadata for all fonts (without temp renaming)
        font_metadata: Dict[Path, FontMetadata] = {}
        for font_path in font_files:
            metadata = extract_metadata(font_path)

            if metadata is None:
                continue

            # Validate PostScript name
            is_valid, _ = is_valid_postscript_name(metadata.ps_name)
            if not is_valid and not rename_all:
                continue

            metadata.original_filename = font_path.name
            font_metadata[font_path] = metadata
            cache[font_path.name] = metadata

        if not font_metadata:
            continue

        # Group by PostScript name AND format (OTF and TTF with same name are different files)
        ps_name_groups: Dict[str, List[FontMetadata]] = {}
        for metadata in font_metadata.values():
            ps_name = metadata.ps_name
            # Use detected format, fallback to file extension if not detected
            if metadata.detected_format:
                format_key = metadata.detected_format
            else:
                # Fallback to file extension for grouping
                format_key = Path(metadata.file_path).suffix.lower().lstrip(".")
            # Composite key: PostScript name + format
            group_key = f"{ps_name}|{format_key}"
            if group_key not in ps_name_groups:
                ps_name_groups[group_key] = []
            ps_name_groups[group_key].append(metadata)

        # Assign final names with priority (reuse existing function)
        rename_map = assign_final_names(ps_name_groups)

        # Build preview list
        previews = []
        for original_path, new_name in rename_map.items():
            # Skip if name unchanged
            if original_path.name == new_name:
                continue

            # Find metadata to get PS name and priority
            meta = font_metadata.get(original_path)
            if not meta:
                continue

            ps_name = meta.ps_name
            # Determine priority (0 = clean name, 1+ = ~001, ~002, etc.)
            priority = 0
            if "~" in new_name:
                try:
                    # Extract priority from ~001 format
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
                )
            )

        if previews:
            previews_by_dir[str(directory)] = previews

    return previews_by_dir


def highlight_differences_pair(original: str, new: str) -> Tuple[str, str]:
    """
    Highlight differences between two strings using StatusIndicator color scheme.
    Returns tuple of (highlighted_original, highlighted_new).

    Uses StatusIndicator colors:
    - Original differences: value.before (turquoise2)
    - New differences: value.after (magenta2)
    """
    if not cs.RICH_AVAILABLE or original == new:
        return original, new

    # Find common prefix
    prefix_len = 0
    min_len = min(len(original), len(new))
    while prefix_len < min_len and original[prefix_len] == new[prefix_len]:
        prefix_len += 1

    # Find common suffix (after prefix)
    suffix_len = 0
    orig_remaining = len(original) - prefix_len
    new_remaining = len(new) - prefix_len
    while (
        suffix_len < min(orig_remaining, new_remaining)
        and original[len(original) - 1 - suffix_len] == new[len(new) - 1 - suffix_len]
    ):
        suffix_len += 1

    # Build original with highlighted differences
    orig_parts = []
    if prefix_len > 0:
        orig_parts.append(original[:prefix_len])
    if prefix_len < len(original) - suffix_len:
        diff_part = original[prefix_len : len(original) - suffix_len]
        orig_parts.append(f"[value.before]{diff_part}[/value.before]")
    if suffix_len > 0:
        orig_parts.append(original[-suffix_len:])
    highlighted_original = "".join(orig_parts)

    # Build new with highlighted differences
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


def show_preflight_preview(previews_by_dir: Dict[str, List[RenamePreview]]) -> None:
    """Display a preview of what will be renamed."""
    cs.emit("")
    cs.StatusIndicator("info").add_message("Rename Preview").emit()

    total_files = sum(len(previews) for previews in previews_by_dir.values())
    total_dirs = len(previews_by_dir)

    cs.emit(f"{cs.indent(1)}Total files to rename: {cs.fmt_count(total_files)}")
    cs.emit(f"{cs.indent(1)}Directories affected: {cs.fmt_count(total_dirs)}")
    cs.emit("")

    # Show changes in streamlined table with highlighted differences
    if cs.RICH_AVAILABLE and console:
        table = cs.create_table(show_header=True)
        if table:
            table.add_column("Original Name", style="lighttext", no_wrap=False)
            table.add_column("New Name", style="lighttext", no_wrap=False)

            for dir_path, previews in sorted(previews_by_dir.items()):
                for preview in sorted(previews, key=lambda p: p.original_name):
                    # Highlight differences in both original and new names
                    highlighted_orig, highlighted_new = highlight_differences_pair(
                        preview.original_name, preview.new_name
                    )
                    table.add_row(
                        highlighted_orig,
                        highlighted_new,
                    )

            console.print(table)
    else:
        # Fallback for non-Rich environments
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
        description="Rename font files to PostScript names with version priority",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/fonts/              # Rename all fonts in directory
  %(prog)s font1.otf font2.otf         # Rename specific files
  %(prog)s /fonts/ -r         # Process directory recursively
  %(prog)s /fonts/ -n           # Preview changes
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

    args = parser.parse_args()

    # Default to current directory if no paths given
    if not args.paths:
        args.paths = ["."]

    # Collect all font files
    font_paths = collect_font_files(
        args.paths, recursive=args.recursive, allowed_extensions=FONT_EXTENSIONS
    )

    if not font_paths:
        if console:
            cs.StatusIndicator("error").with_explanation("No font files found").emit()
        return 1

    # Group by directory
    dirs_to_process = group_files_by_directory(font_paths)

    # Show summary
    if console:
        mode = "DRY RUN" if args.dry_run else "RENAME"
        cs.print_panel(
            f"Mode: {mode}\n"
            f"Files: {cs.fmt_count(len(font_paths))}\n"
            f"Directories: {cs.fmt_count(len(dirs_to_process))}",
            title="Font File Renamer",
            border_style="blue",
        )

    # Analyze and show preview unless --no-preview
    if not args.no_preview:
        previews_by_dir = analyze_renames(font_paths, rename_all=args.rename_all)

        if previews_by_dir:
            show_preflight_preview(previews_by_dir)

            # Ask for confirmation unless dry-run
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

    # Process each directory
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
        )

        total_stats.total_files += dir_stats.total_files
        total_stats.renamed += dir_stats.renamed
        total_stats.skipped += dir_stats.skipped
        total_stats.invalid += dir_stats.invalid
        total_stats.errors.extend(dir_stats.errors)

        show_directory_stats(dir_stats, args.verbose)

    # Final summary
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
