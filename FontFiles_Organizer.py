#!/usr/bin/env python3
"""
Font Files Organizer - Organize fonts into A-Z directory structure

Collects fonts recursively, groups by superfamily/family, renames to PostScript names,
and organizes into A-Z directory structure with superfamily/family subfolders.

Usage:
    python FontFiles_Organizer.py /path/to/fonts/
    python FontFiles_Organizer.py /path/to/fonts/ --output-dir /path/to/output/
    python FontFiles_Organizer.py /path/to/fonts/ --dry-run
    python FontFiles_Organizer.py /path/to/fonts/ --verbose

Options:
    -dry, --dry-run       Preview changes without moving files
    -o, --output-dir      Specify output directory (default: sorted_fonts/ next to source)
    -V, --verbose         Show detailed processing information
    --no-preview          Skip preflight preview
    -R, --recursive       Process directories recursively (default: non-recursive)
"""

import re
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

# Core module imports
import core.core_console_styles as cs
from core.core_file_collector import collect_font_files
from core.core_font_sorter import (
    FontSorter,
    create_font_info_from_paths,
)

# Import from FontFiles_Renamer
from FontFiles_Renamer import (
    is_valid_postscript_name,
    sort_by_version_priority,
    FontMetadata,
)

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
    target_path: Path
    ps_name: str
    family_name: str
    superfamily_name: str
    alpha_folder: str
    folder_name: str  # superfamily or family name for folder
    original_filename: str
    metadata: Optional[FontMetadata] = None


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


def sanitize_folder_name(name: str) -> str:
    """
    Make folder name filesystem-safe by removing/replacing problematic characters.

    Args:
        name: Folder name to sanitize

    Returns:
        Sanitized folder name safe for filesystem
    """
    if not name or not name.strip():
        return "Unknown"

    # Remove or replace problematic characters
    # Keep: letters, numbers, spaces, hyphens, underscores
    # Replace: / \ : * ? " < > | with underscore
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", name)

    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(" .")

    # Replace multiple spaces/underscores with single underscore
    sanitized = re.sub(r"[\s_]+", "_", sanitized)

    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")

    # Ensure not empty
    if not sanitized:
        return "Unknown"

    # Limit length (filesystem limit, typically 255)
    if len(sanitized) > 200:
        sanitized = sanitized[:200]

    return sanitized


def create_directory_structure(output_dir: Path, verbose: bool = False) -> None:
    """
    Create A-Z folders and OTHER folder in output directory.

    Args:
        output_dir: Base output directory
        verbose: Show detailed information
    """
    # Create base output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create A-Z folders
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        letter_dir = output_dir / letter
        letter_dir.mkdir(exist_ok=True)

    # Create OTHER folder
    other_dir = output_dir / "OTHER"
    other_dir.mkdir(exist_ok=True)

    # Create QUARANTINE folder
    quarantine_dir = output_dir / "QUARANTINE"
    quarantine_dir.mkdir(exist_ok=True)

    if console and verbose:
        cs.StatusIndicator("info").add_message(
            f"Created directory structure in {cs.fmt_file(str(output_dir))}"
        ).emit()


# ============================================================================
# Font Organization Logic
# ============================================================================


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
    # Try to extract metadata, but catch exceptions directly to preserve error messages
    try:
        from fontTools.ttLib import TTFont

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

        file_size = font_path.stat().st_size

        font.close()

        metadata = FontMetadata(
            ps_name=ps_name,
            font_revision=font_revision,
            version_string=version_string,
            file_size=file_size,
            glyph_count=glyph_count,
            head_created=head_created,
            head_modified=head_modified,
            file_path=str(font_path),
            original_filename=font_path.name,
        )
        return metadata, None
    except Exception as e:
        # Capture the actual exception message, with fallback for empty messages
        error_msg = str(e) if str(e) else repr(e)
        if not error_msg:
            error_msg = f"Exception of type {type(e).__name__} occurred"
        return None, error_msg


def extract_font_organization_data(
    font_paths: List[str], verbose: bool = False
) -> Tuple[List[OrganizedFont], List[Tuple[Path, str]], OrganizationStats]:
    """
    Extract metadata and determine organization structure for all fonts.

    Args:
        font_paths: List of font file paths
        verbose: Show detailed processing information

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

        # Check if superfamily grouping produced meaningful results
        # (more than just individual families)
        has_superfamilies = any(
            len({f.family_name for f in fonts}) > 1
            for fonts in superfamily_groups.values()
        )

        if has_superfamilies:
            groups = superfamily_groups
            use_superfamily = True
        else:
            # Fallback to family grouping
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
    ps_name_groups: Dict[str, List[FontMetadata]] = defaultdict(list)

    for font_info in font_infos:
        font_path = Path(font_info.path).resolve()

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

        # Determine alpha folder
        alpha_folder = get_alpha_folder(folder_name, font_path.name)

        # Get PostScript name for final filename
        ps_name = rename_map[font_path]

        # Get cached metadata
        metadata = metadata_cache.get(font_path)

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
        )

        organized_fonts.append(organized_font)

    return organized_fonts, failed_files, stats


def assign_target_paths(organized_fonts: List[OrganizedFont], output_dir: Path) -> None:
    """
    Assign target paths to OrganizedFont objects based on directory structure.
    Handles duplicate filenames within the same folder.

    Args:
        organized_fonts: List of OrganizedFont objects
        output_dir: Base output directory
    """
    # Track filenames per folder to handle duplicates
    folder_files: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for font in organized_fonts:
        alpha_folder = font.alpha_folder
        folder_name = font.folder_name

        # Create target directory path
        target_dir = output_dir / alpha_folder / folder_name

        # Get base filename (PostScript name)
        base_filename = font.ps_name

        # Check for duplicates in this folder
        folder_key = (alpha_folder, folder_name)
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

    quarantine_dir = output_dir / "QUARANTINE"

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

                # Move file to quarantine
                shutil.move(str(font_path), str(target_path))

                if console and verbose:
                    cs.StatusIndicator("warning").add_file(font_path.name).add_message(
                        f"→ QUARANTINE/ ({error_msg})"
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
    dry_run: bool = False,
    verbose: bool = False,
) -> OrganizationStats:
    """
    Move files to organized directory structure.

    Args:
        organized_fonts: List of OrganizedFont objects with target paths set
        output_dir: Base output directory
        dry_run: Preview mode without moving files
        verbose: Show detailed processing information

    Returns:
        OrganizationStats with results
    """
    stats = OrganizationStats()
    stats.total_files = len(organized_fonts)

    # Create directory structure
    if not dry_run:
        create_directory_structure(output_dir, verbose)

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

        # Move file
        if dry_run:
            if console and verbose:
                cs.StatusIndicator("info", dry_run=True).add_file(
                    font.original_filename
                ).add_message(
                    f"→ {cs.fmt_file(str(font.target_path.relative_to(output_dir)))}"
                ).emit()
            stats.organized += 1
        else:
            try:
                # Check if source exists
                if not font.source_path.exists():
                    stats.add_error(
                        font.original_filename, "Source file does not exist"
                    )
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

                # Move file
                shutil.move(str(font.source_path), str(font.target_path))

                if console and verbose:
                    cs.StatusIndicator("updated").add_file(
                        font.original_filename
                    ).add_message(
                        f"→ {cs.fmt_file(str(font.target_path.relative_to(output_dir)))}"
                    ).emit()

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

    Returns:
        OrganizationStats with results
    """
    # Determine output directory
    if output_dir is None:
        output_dir = source_dir.parent / "sorted_fonts"
    else:
        output_dir = Path(output_dir)

    # Collect font files
    if console:
        cs.StatusIndicator("info").add_message(
            f"Collecting fonts from {cs.fmt_file(str(source_dir))}"
        ).emit()

    font_paths = collect_font_files(
        [str(source_dir)], recursive=recursive, allowed_extensions=FONT_EXTENSIONS_SET
    )

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
        font_paths, verbose
    )

    # Assign target paths
    if organized_fonts:
        assign_target_paths(organized_fonts, output_dir)

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
            if not cs.prompt_confirm(
                "Ready to organize fonts into A-Z directory structure",
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
            organized_fonts, output_dir, dry_run, verbose
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
        description="Organize fonts into A-Z directory structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/fonts/                    # Organize fonts in directory
  %(prog)s /path/to/fonts/ --output-dir /out/  # Specify output directory
  %(prog)s /path/to/fonts/ -n         # Preview changes
  %(prog)s /path/to/fonts/ --verbose         # Show detailed information
        """,
    )

    parser.add_argument(
        "source_dir",
        help="Source directory containing fonts to organize",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory (default: sorted_fonts/ next to source)",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Preview changes without moving files",
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
        help="Skip preflight preview",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Process directories recursively (default: non-recursive)",
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

    # Show summary
    if console:
        mode = "DRY RUN" if args.dry_run else "ORGANIZE"
        cs.print_panel(
            f"Mode: {mode}\n"
            f"Source: {cs.fmt_file(str(source_dir))}\n"
            f"Output: {cs.fmt_file(str(output_dir or (source_dir.parent / 'sorted_fonts')))}",
            title="Font Files Organizer",
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
                "Quarantined files moved to QUARANTINE/ directory"
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
