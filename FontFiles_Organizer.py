#!/usr/bin/env python3
"""
Font Files Simple Sorter - Organize fonts into directory structure

Collects fonts recursively, groups by superfamily/family, renames to PostScript names,
and organizes into directory structure (A-Z by default, or vendor-based).

Usage:
    python FontFiles_SimpleSorter.py /path/to/fonts/
    python FontFiles_SimpleSorter.py /path/to/fonts/ --output-dir /path/to/output/
    python FontFiles_SimpleSorter.py /path/to/fonts/ --dry-run
    python FontFiles_SimpleSorter.py /path/to/fonts/ --verbose
    python FontFiles_SimpleSorter.py /path/to/fonts/ -vs  # Sort by vendor ID

Options:
    -n, --dry-run          Preview changes without moving files
    -o, --output-dir       Specify output directory (default: sorted_fonts/ next to source)
    -v, --verbose          Show detailed processing information
    --no-preview           Skip preflight preview
    -r, --recursive        Process directories recursively (default: non-recursive)
    -vs, --vendor-sort     Sort fonts by vendor ID (OS/2 achVendID)
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

# Add project root to path for FontCore imports (works for root and subdirectory scripts)
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
import FontCore.core_console_styles as cs
from FontCore.core_file_collector import collect_font_files_with_rich_progress
from FontCore.core_font_extension import validate_and_fix_extension
from FontCore.core_font_sorter import (
    FontSorter,
    create_font_info_from_paths,
)
from FontCore.core_font_metadata import (
    FontMetadata,
    extract_metadata_with_error,
    get_vendor_id,
)
from FontCore.core_font_utils import (
    is_valid_postscript_name,
    sanitize_folder_name,
    count_items_per_group,
    format_name_with_count,
)
from FontCore.core_name_policies import is_bad_vendor

# Import from FontFiles_Renamer
from FontFiles_Renamer import sort_by_version_priority

console = cs.get_console()

# ============================================================================
# Constants
# ============================================================================

FONT_EXTENSIONS_SET = {".ttf", ".otf", ".woff", ".woff2", ".ttx"}

# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class OrganizedFont:
    """Represents a font file with its organization metadata"""

    source_path: Path
    target_path: Optional[Path]
    ps_name: str
    family_name: str
    superfamily_name: str
    alpha_folder: str
    folder_name: str  # superfamily or family name for folder
    original_filename: str
    metadata: Optional[FontMetadata] = None
    vendor_id: Optional[str] = None


@dataclass
class OrganizationStats:
    """Statistics for organization operations"""

    total_files: int = 0
    organized: int = 0
    skipped: int = 0
    quarantined: int = 0
    errors: List[Tuple[str, str]] = field(default_factory=list)
    quarantine_errors: List[Tuple[str, str]] = field(default_factory=list)

    def add_error(self, filename: str, reason: str):
        self.errors.append((filename, reason))
        self.skipped += 1

    def add_quarantine(self, filename: str, reason: str):
        self.quarantine_errors.append((filename, reason))
        self.quarantined += 1


# ============================================================================
# Helper Functions
# ============================================================================


def get_alpha_folder(name: str, fallback_filename: str) -> str:
    """
    Determine A-Z or OTHER folder based on first character of name.
    Falls back to filename if name is empty.

    Args:
        name: Superfamily or family name from font metadata
        fallback_filename: Original filename to use if name is empty

    Returns:
        "A"-"Z" or "OTHER"
    """
    # Use name if available, otherwise use filename stem
    if name and name.strip():
        first_char = name.strip()[0].upper()
    else:
        # Fallback to filename
        stem = Path(fallback_filename).stem
        if stem:
            first_char = stem[0].upper()
        else:
            return "OTHER"

    # Check if alphabetic
    if first_char.isalpha() and ord("A") <= ord(first_char) <= ord("Z"):
        return first_char

    return "OTHER"


def atomic_move_file(source: Path, target: Path) -> None:
    """
    Move file atomically to prevent race conditions.

    Uses os.rename() for same-filesystem moves (atomic) or
    temporary file + atomic rename for cross-filesystem moves.

    Args:
        source: Source file path
        target: Target file path
    """
    try:
        # Check if same filesystem
        if os.path.samefile(source.parent, target.parent):
            # Same filesystem: use os.rename (atomic)
            os.rename(str(source), str(target))
        else:
            # Cross-filesystem: use temporary file + atomic rename
            temp_path = target.parent / f".tmp_{target.name}"
            try:
                # Copy to temporary location
                shutil.copy2(str(source), str(temp_path))
                # Atomic rename to final location
                os.rename(str(temp_path), str(target))
                # Remove source file
                os.remove(str(source))
            except Exception:
                # Clean up temp file on error
                if temp_path.exists():
                    try:
                        os.remove(str(temp_path))
                    except Exception:
                        pass
                raise
    except OSError:
        # Fallback to shutil.move for cross-filesystem if rename fails
        shutil.move(str(source), str(target))


def create_directory_structure(
    output_dir: Path,
    sort_mode: str = "ALPHABETICAL",
    verbose: bool = False,
) -> None:
    """
    Create base directory structure. All folders are created on-demand
    as files are moved to prevent empty directories.

    Args:
        output_dir: Base output directory
        sort_mode: Sort mode ("ALPHABETICAL", "VENDOR") - unused, kept for API consistency
        verbose: Show detailed information
    """
    # Create base output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # All folders (A-Z, OTHER, vendor folders, _quarantine) are created on-demand
    # in move_files_to_structure() and move_files_to_quarantine() to avoid empty directories


# ============================================================================
# Font Organization Logic
# ============================================================================


def extract_font_organization_data(
    font_paths: List[str],
    verbose: bool = False,
    sort_mode: str = "ALPHABETICAL",
) -> Tuple[List[OrganizedFont], List[Tuple[Path, str]], OrganizationStats]:
    """
    Extract metadata and determine organization structure for all fonts.

    Args:
        font_paths: List of font file paths
        verbose: Show detailed processing information
        sort_mode: Sort mode ("ALPHABETICAL", "VENDOR")

    Returns:
        Tuple of (list of OrganizedFont objects, list of (failed_path, error_message), stats)
    """
    stats = OrganizationStats()
    stats.total_files = len(font_paths)

    organized_fonts: List[OrganizedFont] = []
    failed_files: List[Tuple[Path, str]] = []  # (path, error_message)

    # Extract font metadata (family, superfamily)
    font_infos = create_font_info_from_paths(font_paths, extract_metadata=True)

    if not font_infos:
        return organized_fonts, failed_files, stats

    # Group by superfamily (fallback to family if no superfamily)
    sorter = FontSorter(font_infos)

    # Try superfamily grouping first
    try:
        superfamily_groups = sorter.group_by_superfamily()

        # Count fonts that are part of merged superfamilies
        # Always use superfamily groups if ANY merges occurred
        merged_fonts = sum(
            len(fonts)
            for fonts in superfamily_groups.values()
            if len({f.family_name for f in fonts}) > 1
        )

        if merged_fonts > 0:
            # Use superfamily grouping - some fonts benefit from it
            groups = superfamily_groups
            use_superfamily = True
        else:
            # No merges occurred, use family grouping
            groups = sorter.group_by_family()
            use_superfamily = False
    except Exception as e:
        if console:
            cs.StatusIndicator("warning").with_explanation(
                f"Superfamily grouping failed: {e}. Using family grouping."
            ).emit()
        groups = sorter.group_by_family()
        use_superfamily = False

    # Extract PostScript names and organize
    # Cache metadata to avoid duplicate extraction
    metadata_cache: Dict[Path, FontMetadata] = {}
    metadata_errors: Dict[Path, str] = {}  # Track errors for failed files
    vendor_cache: Dict[Path, str] = {}  # Cache vendor IDs
    ps_name_groups: Dict[str, List[FontMetadata]] = defaultdict(list)

    for font_info in font_infos:
        font_path = Path(font_info.path).resolve()

        # Extract vendor ID (needed for vendor sorting)
        if font_path not in vendor_cache and font_path not in metadata_errors:
            vendor_id = get_vendor_id(font_path)
            # Check for bad vendor patterns and convert to UKWN
            if is_bad_vendor(vendor_id):
                vendor_id = "UKWN"
            vendor_cache[font_path] = vendor_id
        elif font_path in metadata_errors:
            continue
        else:
            vendor_id = vendor_cache.get(font_path, "UKWN")
            # Double-check cached vendor ID for bad patterns
            if is_bad_vendor(vendor_id):
                vendor_id = "UKWN"
                vendor_cache[font_path] = vendor_id

        # Extract PostScript name (cache metadata)
        if font_path not in metadata_cache and font_path not in metadata_errors:
            metadata, error_msg = extract_metadata_with_error(font_path)
            if not metadata:
                # Store error and mark for quarantine
                # Ensure we have a meaningful error message
                if error_msg and error_msg.strip():
                    error_reason = error_msg
                else:
                    error_reason = (
                        "Failed to extract metadata (no error details available)"
                    )
                metadata_errors[font_path] = error_reason
                failed_files.append((font_path, error_reason))
                if console and verbose:
                    cs.StatusIndicator("warning").add_file(
                        font_path.name
                    ).with_explanation(f"Will quarantine: {error_reason}").emit()
                continue
            metadata_cache[font_path] = metadata
        elif font_path in metadata_errors:
            # Already marked as failed
            continue
        else:
            metadata = metadata_cache[font_path]

        ps_name = metadata.ps_name

        # Validate PostScript name
        is_valid, reason = is_valid_postscript_name(ps_name)
        if not is_valid:
            # Use sanitized filename as fallback
            ps_name = Path(font_path.name).stem
            if console and verbose:
                cs.StatusIndicator("warning").add_file(font_path.name).with_explanation(
                    f"Invalid PS name, using filename: {reason}"
                ).emit()

        ps_name_groups[ps_name].append(metadata)

    # Assign final names with version priority
    rename_map: Dict[Path, str] = {}
    for ps_name, metadata_list in ps_name_groups.items():
        sorted_fonts = sort_by_version_priority(metadata_list)
        first_path = Path(sorted_fonts[0].file_path).resolve()
        ext = first_path.suffix.lower()

        for idx, meta in enumerate(sorted_fonts):
            original_path = Path(meta.file_path).resolve()

            if idx == 0:
                new_name = f"{ps_name}{ext}"
            else:
                new_name = f"{ps_name}~{idx:03d}{ext}"

            rename_map[original_path] = new_name

    # Create OrganizedFont objects
    for font_info in font_infos:
        font_path = Path(font_info.path).resolve()

        # Skip files that failed metadata extraction (they're being quarantined)
        if font_path in metadata_errors:
            continue

        if font_path not in rename_map:
            stats.add_error(font_path.name, "No rename mapping found")
            continue

        # Determine superfamily/family name for folder
        if use_superfamily:
            # Find which superfamily group this font belongs to
            folder_name = font_info.family_name  # Will be set to superfamily below
            font_path_str = str(font_path)
            for superfamily_name, fonts in groups.items():
                if any(f.path == font_path_str for f in fonts):
                    folder_name = superfamily_name
                    break
        else:
            folder_name = font_info.family_name

        # Fallback to filename if family name is empty
        if not folder_name or folder_name == "Unknown":
            folder_name = Path(font_path.name).stem

        # Sanitize folder name
        folder_name = sanitize_folder_name(folder_name)

        # Determine alpha folder (for alphabetical mode)
        alpha_folder = get_alpha_folder(folder_name, font_path.name)

        # Get PostScript name for final filename
        ps_name = rename_map[font_path]

        # Get cached metadata and vendor ID
        metadata = metadata_cache.get(font_path)
        vendor_id = vendor_cache.get(font_path, "UKWN")

        # Create OrganizedFont
        organized_font = OrganizedFont(
            source_path=font_path,
            target_path=None,  # Will be set below
            ps_name=ps_name,
            family_name=font_info.family_name or "Unknown",
            superfamily_name=folder_name if use_superfamily else "",
            alpha_folder=alpha_folder,
            folder_name=folder_name,
            original_filename=font_path.name,
            metadata=metadata,
            vendor_id=vendor_id,
        )

        organized_fonts.append(organized_font)

    return organized_fonts, failed_files, stats


def assign_target_paths(
    organized_fonts: List[OrganizedFont],
    output_dir: Path,
    sort_mode: str = "ALPHABETICAL",
) -> None:
    """
    Assign target paths to OrganizedFont objects based on directory structure.
    Handles duplicate filenames within the same folder.
    Adds file count to folder names (e.g., "Helvetica (12)").

    Args:
        organized_fonts: List of OrganizedFont objects
        output_dir: Base output directory
        sort_mode: Sort mode ("ALPHABETICAL", "VENDOR")
    """

    def get_folder_key(font: OrganizedFont) -> Tuple[str, ...]:
        """Extract folder key for grouping based on sort mode."""
        if sort_mode == "VENDOR":
            return (font.vendor_id or "UKWN", font.folder_name)
        else:
            return (font.alpha_folder, font.folder_name)

    # First pass: count fonts per folder using the utility function
    folder_counts = count_items_per_group(organized_fonts, get_folder_key)

    # Second pass: assign target paths with file counts in folder names
    folder_files: Dict[Tuple[str, ...], List[str]] = defaultdict(list)

    for font in organized_fonts:
        # Determine top-level folder based on sort mode
        folder_key = get_folder_key(font)
        count = folder_counts[folder_key]

        if sort_mode == "VENDOR":
            # Vendor mode: vendor_id/family_name (count)
            top_folder = font.vendor_id or "UKWN"
            folder_name_with_count = format_name_with_count(font.folder_name, count)
            target_dir = output_dir / top_folder / folder_name_with_count
        else:
            # Default alphabetical mode: A-Z/family_name (count)
            alpha_folder = font.alpha_folder
            folder_name_with_count = format_name_with_count(font.folder_name, count)
            target_dir = output_dir / alpha_folder / folder_name_with_count

        # Get base filename (PostScript name)
        base_filename = font.ps_name

        # Check for duplicates in this folder
        existing_files = folder_files[folder_key]

        # Handle duplicates
        if base_filename in existing_files:
            # Find next available name with counter
            stem = Path(base_filename).stem
            ext = Path(base_filename).suffix
            counter = 1
            while True:
                new_filename = f"{stem}_dup{counter:03d}{ext}"
                if new_filename not in existing_files:
                    base_filename = new_filename
                    break
                counter += 1
                if counter > 999:
                    # Too many duplicates, skip
                    break

        folder_files[folder_key].append(base_filename)

        # Set target path
        font.target_path = target_dir / base_filename


def move_files_to_quarantine(
    failed_files: List[Tuple[Path, str]],
    output_dir: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> OrganizationStats:
    """
    Move failed files to quarantine directory with error information.

    Args:
        failed_files: List of (file_path, error_message) tuples
        output_dir: Base output directory
        dry_run: Preview mode without moving files
        verbose: Show detailed processing information

    Returns:
        OrganizationStats with quarantine results
    """
    stats = OrganizationStats()
    stats.total_files = len(failed_files)

    quarantine_dir = output_dir / "_quarantine"

    # Create quarantine directory if needed
    if not dry_run:
        quarantine_dir.mkdir(parents=True, exist_ok=True)

    for font_path, error_msg in failed_files:
        try:
            # Use original filename in quarantine
            target_path = quarantine_dir / font_path.name

            # Handle duplicate filenames in quarantine
            if target_path.exists():
                stem = font_path.stem
                ext = font_path.suffix
                counter = 1
                while target_path.exists():
                    target_path = quarantine_dir / f"{stem}_dup{counter:03d}{ext}"
                    counter += 1
                    if counter > 999:
                        stats.add_error(
                            font_path.name, "Too many duplicates in quarantine"
                        )
                        break

            if dry_run:
                if console:
                    cs.StatusIndicator("warning", dry_run=True).add_file(
                        font_path.name
                    ).with_explanation(f"Would quarantine: {error_msg}").emit()
                stats.quarantined += 1
            else:
                # Check if source exists
                if not font_path.exists():
                    stats.add_error(font_path.name, "Source file does not exist")
                    continue

                # Move file to quarantine (atomic)
                atomic_move_file(font_path, target_path)

                if console and verbose:
                    cs.StatusIndicator("warning").add_file(font_path.name).add_message(
                        f"→ _quarantine/ ({error_msg})"
                    ).emit()

                stats.add_quarantine(font_path.name, error_msg)
        except PermissionError as e:
            stats.add_error(font_path.name, f"Permission denied: {e}")
        except OSError as e:
            stats.add_error(font_path.name, f"OS error: {e}")
        except Exception as e:
            stats.add_error(font_path.name, f"Unexpected error: {e}")

    return stats


def move_files_to_structure(
    organized_fonts: List[OrganizedFont],
    output_dir: Path,
    sort_mode: str = "ALPHABETICAL",
    dry_run: bool = False,
    verbose: bool = False,
) -> OrganizationStats:
    """
    Move files to organized directory structure.

    Args:
        organized_fonts: List of OrganizedFont objects with target paths set
        output_dir: Base output directory
        sort_mode: Sort mode ("ALPHABETICAL", "VENDOR")
        dry_run: Preview mode without moving files
        verbose: Show detailed processing information

    Returns:
        OrganizationStats with results
    """
    stats = OrganizationStats()
    stats.total_files = len(organized_fonts)

    # Create directory structure
    if not dry_run:
        create_directory_structure(output_dir, sort_mode, verbose)

    # Create subdirectories and move files
    for font in organized_fonts:
        if not font.target_path:
            stats.add_error(font.original_filename, "No target path assigned")
            continue

        target_dir = font.target_path.parent

        # Create subdirectory if needed
        if not dry_run:
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                stats.add_error(font.original_filename, f"Cannot create directory: {e}")
                continue

        # Use same StatusIndicator for both dry-run and normal mode
        # DRY prefix will be added automatically when dry_run=True
        if console and verbose:
            cs.StatusIndicator("updated", dry_run=dry_run).add_file(
                font.original_filename
            ).add_message(
                f"→ {cs.fmt_file(str(font.target_path.relative_to(output_dir)))}"
            ).emit()

        if dry_run:
            stats.organized += 1
            continue

        try:
            # Check if source exists
            if not font.source_path.exists():
                stats.add_error(font.original_filename, "Source file does not exist")
                continue

            # Check if target already exists - handle duplicates with ~001, ~002 suffixes
            if font.target_path.exists():
                # Check if source still exists
                if not font.source_path.exists():
                    # Source doesn't exist - file was already moved, skip
                    stats.skipped += 1
                    continue

                # Source exists but target exists - this is a duplicate
                # Find next available name with ~001, ~002, etc. suffix
                stem = font.target_path.stem
                ext = font.target_path.suffix
                counter = 1
                duplicate_target = None

                while counter < 1000:
                    duplicate_name = f"{stem}~{counter:03d}{ext}"
                    duplicate_target = font.target_path.parent / duplicate_name
                    if not duplicate_target.exists():
                        break
                    counter += 1

                if duplicate_target and counter < 1000:
                    # Use the duplicate target path
                    font.target_path = duplicate_target
                else:
                    stats.add_error(
                        font.original_filename, "Too many duplicates, cannot move"
                    )
                    continue

            # Move file (atomic)
            atomic_move_file(font.source_path, font.target_path)
            stats.organized += 1
        except PermissionError as e:
            stats.add_error(font.original_filename, f"Permission denied: {e}")
        except OSError as e:
            stats.add_error(font.original_filename, f"OS error: {e}")
        except Exception as e:
            stats.add_error(font.original_filename, f"Unexpected error: {e}")

    return stats


def preview_organization(
    organized_fonts: List[OrganizedFont], output_dir: Path
) -> None:
    """
    Display preview of organization structure.

    Args:
        organized_fonts: List of OrganizedFont objects
        output_dir: Base output directory
    """
    if not organized_fonts:
        if console:
            cs.StatusIndicator("info").add_message("No fonts to organize").emit()
        return

    cs.emit("")
    cs.StatusIndicator("info").add_message("Organization Preview").emit()

    # Group by alpha folder and folder name
    structure: Dict[str, Dict[str, List[OrganizedFont]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for font in organized_fonts:
        if font.target_path:
            structure[font.alpha_folder][font.folder_name].append(font)

    # Display structure
    total_files = len(organized_fonts)
    cs.emit(f"{cs.indent(1)}Total files: {cs.fmt_count(total_files)}")
    cs.emit(f"{cs.indent(1)}Output directory: {cs.fmt_file(str(output_dir))}")
    cs.emit("")

    # Show structure tree
    if cs.RICH_AVAILABLE and console:
        for alpha_folder in sorted(structure.keys()):
            folders = structure[alpha_folder]
            cs.emit(f"{cs.indent(1)}{alpha_folder}/")

            for folder_name in sorted(folders.keys()):
                fonts = folders[folder_name]
                cs.emit(
                    f"{cs.indent(2)}{folder_name}/ ({cs.fmt_count(len(fonts))} files)"
                )

                if console:
                    for font in sorted(fonts, key=lambda f: f.original_filename):
                        target_rel = (
                            font.target_path.relative_to(output_dir)
                            if font.target_path
                            else "?"
                        )
                        cs.emit(
                            f"{cs.indent(3)}• {cs.fmt_file(font.original_filename)} → {cs.fmt_file(str(target_rel))}"
                        )
    else:
        # Fallback for non-Rich
        for alpha_folder in sorted(structure.keys()):
            folders = structure[alpha_folder]
            print(f"  {alpha_folder}/")

            for folder_name in sorted(folders.keys()):
                fonts = folders[folder_name]
                print(f"    {folder_name}/ ({len(fonts)} files)")

                for font in sorted(fonts, key=lambda f: f.original_filename):
                    target_rel = (
                        font.target_path.relative_to(output_dir)
                        if font.target_path
                        else "?"
                    )
                    print(f"      • {font.original_filename} → {target_rel}")

    cs.emit("")


# ============================================================================
# Main Orchestration
# ============================================================================


def organize_fonts(
    source_dir: Path,
    output_dir: Optional[Path] = None,
    dry_run: bool = False,
    verbose: bool = False,
    no_preview: bool = False,
    recursive: bool = False,
    sort_mode: str = "ALPHABETICAL",
) -> OrganizationStats:
    """
    Main orchestration function for font organization.

    Args:
        source_dir: Source directory containing fonts
        output_dir: Output directory (default: sorted_fonts/ next to source)
        dry_run: Preview mode without moving files
        verbose: Show detailed processing information
        no_preview: Skip preflight preview and confirmation
        recursive: Process directories recursively (default: False)
        sort_mode: Sort mode ("ALPHABETICAL", "VENDOR")

    Returns:
        OrganizationStats with results
    """
    # Determine output directory
    if output_dir is None:
        output_dir = source_dir.parent / "Font Library"
    else:
        output_dir = Path(output_dir)

    # Collect font files with Rich progress bar
    if console:
        cs.StatusIndicator("info").add_message(
            f"Collecting fonts from {cs.fmt_file(str(source_dir))}"
        ).emit()

    # Use the centralized Rich progress bar helper
    font_paths = collect_font_files_with_rich_progress(
        paths=[str(source_dir)],
        recursive=recursive,
        allowed_extensions=FONT_EXTENSIONS_SET,
        description="Scanning for font files...",
        console=console,
    )

    # Validate and fix extensions
    validated_paths = []
    for path_str in font_paths:
        path = Path(path_str)
        is_valid, fixed_path = validate_and_fix_extension(path, auto_fix=True)
        if fixed_path:
            if console:
                cs.StatusIndicator("info").add_file(str(fixed_path)).with_explanation(
                    f"Fixed extension: {path.name} → {fixed_path.name}"
                ).emit()
            validated_paths.append(str(fixed_path))
        else:
            validated_paths.append(path_str)
    font_paths = validated_paths

    if not font_paths:
        if console:
            cs.StatusIndicator("error").with_explanation("No font files found").emit()
        return OrganizationStats()

    if console:
        cs.StatusIndicator("info").add_message(
            f"Found {cs.fmt_count(len(font_paths))} font files"
        ).emit()

    # Extract organization data
    organized_fonts, failed_files, extract_stats = extract_font_organization_data(
        font_paths,
        verbose,
        sort_mode,
    )

    # Assign target paths
    if organized_fonts:
        assign_target_paths(organized_fonts, output_dir, sort_mode)

    # Show preview unless disabled
    if not no_preview:
        if organized_fonts:
            preview_organization(organized_fonts, output_dir)

        # Show quarantine preview if there are failed files
        if failed_files:
            cs.emit("")
            cs.StatusIndicator("warning").add_message(
                f"Files to quarantine: {cs.fmt_count(len(failed_files))}"
            ).emit()
            if console:
                for font_path, error_msg in failed_files[:10]:  # Show first 10
                    cs.emit(
                        f"{cs.indent(1)}• {cs.fmt_file(font_path.name)}: {error_msg}"
                    )
                if len(failed_files) > 10:
                    cs.emit(
                        f"{cs.indent(1)}... and {cs.fmt_count(len(failed_files) - 10)} more"
                    )
            cs.emit("")

        # Ask for confirmation unless dry-run
        if not dry_run:
            mode_description = "directory structure"
            if sort_mode == "VENDOR":
                mode_description = "vendor-based directory structure"

            if not cs.prompt_confirm(
                f"Ready to organize fonts into {mode_description}",
                action_prompt="Proceed with organization?",
                default=True,
            ):
                if console:
                    cs.StatusIndicator("info").add_message("Operation cancelled").emit()
                return OrganizationStats()

    # Move organized files
    move_stats = OrganizationStats()
    if organized_fonts:
        move_stats = move_files_to_structure(
            organized_fonts, output_dir, sort_mode, dry_run, verbose
        )

    # Move failed files to quarantine
    quarantine_stats = OrganizationStats()
    if failed_files:
        quarantine_stats = move_files_to_quarantine(
            failed_files, output_dir, dry_run, verbose
        )

    # Combine stats
    final_stats = OrganizationStats()
    final_stats.total_files = extract_stats.total_files
    final_stats.organized = move_stats.organized
    final_stats.quarantined = quarantine_stats.quarantined
    final_stats.skipped = (
        extract_stats.skipped + move_stats.skipped + quarantine_stats.skipped
    )
    final_stats.errors = (
        extract_stats.errors + move_stats.errors + quarantine_stats.errors
    )
    final_stats.quarantine_errors = quarantine_stats.quarantine_errors

    return final_stats


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Organize your font collection into a clean directory structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
QUICK START:
  %(prog)s /path/to/fonts/                    # Organize fonts alphabetically (A-Z)
  %(prog)s /path/to/fonts/ -n                 # Preview what will happen first
  %(prog)s /path/to/fonts/ -r                 # Include fonts in subfolders
  %(prog)s /path/to/fonts/ -vs                # Group by font maker (vendor)

TIPS:
  • Use -n (dry-run) first to preview what will happen
  • Use -v (verbose) to see detailed progress
  • Fonts that can't be processed go into a _quarantine folder
  • Files are renamed using their embedded PostScript (NameID 6) name
        """,
    )

    # Required argument
    parser.add_argument(
        "source_dir",
        metavar="FOLDER",
        help="Folder containing fonts to organize",
    )

    # Basic options
    basic_group = parser.add_argument_group("Basic Options")
    basic_group.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Preview changes without moving any files (safe mode)",
    )
    basic_group.add_argument(
        "-o",
        "--output-dir",
        metavar="FOLDER",
        help="Where to save organized fonts (default: creates 'Font Library' folder)",
    )
    basic_group.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Look inside subfolders for fonts (default: only top-level folder)",
    )
    basic_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed progress as files are processed",
    )
    basic_group.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip the preview and start immediately",
    )

    # Organization options
    org_group = parser.add_argument_group(
        "Organization Methods", "Choose how to organize your fonts:"
    )
    org_group.add_argument(
        "-vs",
        "--vendor-sort",
        action="store_true",
        help="Organize by font maker/vendor (e.g., Adobe, Monotype, Google)",
    )

    args = parser.parse_args()

    source_dir = Path(args.source_dir).expanduser().resolve()

    if not source_dir.exists():
        if console:
            cs.StatusIndicator("error").with_explanation(
                f"Source directory does not exist: {source_dir}"
            ).emit()
        return 1

    if not source_dir.is_dir():
        if console:
            cs.StatusIndicator("error").with_explanation(
                f"Source path is not a directory: {source_dir}"
            ).emit()
        return 1

    # Determine output directory
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()

    # Parse sort mode
    sort_mode = "ALPHABETICAL"
    if args.vendor_sort:
        sort_mode = "VENDOR"

    # Show summary
    if console:
        # Use same mode string for both dry-run and normal mode
        # DRY prefix will be added automatically by StatusIndicator when dry_run=True
        mode = "ORGANIZE"
        mode_info = f"Mode: {mode}\n"
        mode_info += f"Sort by: {sort_mode}\n"
        mode_info += f"Input Directory: {cs.fmt_file(str(source_dir))}\n"
        mode_info += f"Output Directory: {cs.fmt_file(str(output_dir or (source_dir.parent / 'Font Library')))}"
        cs.print_panel(
            mode_info,
            title="Font Files Simple Sorter",
            border_style="blue",
        )

    # Organize fonts
    stats = organize_fonts(
        source_dir,
        output_dir=output_dir,
        dry_run=args.dry_run,
        verbose=args.verbose,
        no_preview=args.no_preview,
        recursive=args.recursive,
        sort_mode=sort_mode,
    )

    # Final summary
    if console:
        summary_lines = [
            f"Total files: {cs.fmt_count(stats.total_files)}",
            f"Organized: {cs.fmt_count(stats.organized)}",
        ]
        if stats.quarantined > 0:
            summary_lines.append(f"Quarantined: {cs.fmt_count(stats.quarantined)}")
        if stats.skipped > 0:
            summary_lines.append(f"Skipped: {cs.fmt_count(stats.skipped)}")

        cs.print_panel(
            "\n".join(summary_lines),
            title="Summary",
            border_style="green",
        )

        # Show quarantine details if any
        if stats.quarantine_errors:
            cs.emit("")
            cs.StatusIndicator("warning").add_message(
                f"Quarantined files ({cs.fmt_count(len(stats.quarantine_errors))}):"
            ).emit()
            for filename, reason in stats.quarantine_errors[:20]:  # Show first 20
                cs.emit(f"{cs.indent(1)}• {cs.fmt_file(filename)}: {reason}")
            if len(stats.quarantine_errors) > 20:
                cs.emit(
                    f"{cs.indent(1)}... and {cs.fmt_count(len(stats.quarantine_errors) - 20)} more"
                )
            cs.emit("")
            cs.StatusIndicator("info").add_message(
                "Quarantined files moved to: _quarantine/ directory"
            ).emit()

        # Show other errors
        if stats.errors:
            cs.emit("")
            cs.StatusIndicator("warning").add_message("Other errors occurred:").emit()
            for filename, reason in stats.errors[:10]:  # Show first 10 errors
                cs.emit(f"{cs.indent(1)}• {cs.fmt_file(filename)}: {reason}")
            if len(stats.errors) > 10:
                cs.emit(
                    f"{cs.indent(1)}... and {cs.fmt_count(len(stats.errors) - 10)} more errors"
                )

    return 0 if stats.organized > 0 or stats.skipped == stats.total_files else 1


if __name__ == "__main__":
    exit(main())
