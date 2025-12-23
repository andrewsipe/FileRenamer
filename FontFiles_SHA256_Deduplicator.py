#!/usr/bin/env python3
"""
Font SHA256 Deduplicator - Remove exact byte-for-byte duplicate font files

Uses SHA256 hashing to detect files that are 100% identical at the byte level.
These are safe to remove as they are true copies with no meaningful differences.

Features:
- SHA256 hash comparison (cryptographically sound duplicate detection)
- Per-directory isolation (only compares files within same directory)
- Cross-directory comparison: Compare two directories and remove duplicates from secondary
- Smart naming: Auto-detects priority naming (~001, ~002) and keeps clean names
- Configurable duplicate selection (keep oldest, newest, or first alphabetically)
- Safe trash/move instead of delete
- Detailed reporting of removed duplicates
- Automatic empty directory cleanup after cross-directory comparison

Usage:
    python FontFiles_SHA256_Deduplicator.py /path/to/fonts/
    python FontFiles_SHA256_Deduplicator.py font1.otf font2.otf
    python FontFiles_SHA256_Deduplicator.py /directory/ -r
    python FontFiles_SHA256_Deduplicator.py /directory/ -n
    python FontFiles_SHA256_Deduplicator.py --compare-dirs /fonts1 /fonts2 -r

Options:
    -r, --recursive      Process directories recursively
    -n, --dry-run        Preview what would be removed without moving files
    --keep-strategy      Which duplicate to keep: oldest, newest, first (default: oldest)
    --trash-dir          Where to move duplicates (default: ~/.Trash/FontDeduplicator)
    --verbose, -v        Show detailed processing information
    --compare-dirs       Compare two directories and remove duplicates from secondary directory
                         (first directory keeps files, second directory loses duplicates)

Strategy Details:
    oldest  - Keep file with oldest creation timestamp (preserves original)
    newest  - Keep file with newest modification timestamp (preserves latest edits)
    first   - Keep first file alphabetically (predictable, arbitrary)

Smart Naming:
    If duplicates include both clean names and priority-suffixed versions
    (e.g., Font-Bold.otf and Font-Bold~001.otf), the deduplicator will
    automatically prefer the clean name, avoiding the need to re-run the renamer.
"""

import hashlib
import re
import shutil
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

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

import FontCore.core_console_styles as cs
from FontCore.core_file_collector import collect_font_files

console = cs.get_console()

# ============================================================================
# Constants
# ============================================================================

FONT_EXTENSIONS = {".ttf", ".otf", ".woff", ".woff2"}
DEFAULT_TRASH_DIR = Path.home() / ".Trash" / "FontDeduplicator"

# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class DeduplicationStats:
    """Statistics for deduplication operations"""

    total_files: int = 0
    unique_files: int = 0
    duplicate_files: int = 0
    duplicate_groups: int = 0
    bytes_saved: int = 0
    files_moved: int = 0
    errors: List[Tuple[str, str]] = field(default_factory=list)
    # Cross-directory comparison statistics
    primary_files_count: int = 0
    secondary_files_count: int = 0
    empty_dirs_removed_primary: int = 0
    empty_dirs_removed_secondary: int = 0

    def add_error(self, filename: str, reason: str):
        self.errors.append((filename, reason))


@dataclass
class FileInfo:
    """Information about a file for duplicate detection"""

    path: Path
    sha256: str
    size: int
    created: float
    modified: float

    @property
    def name(self) -> str:
        return self.path.name


# ============================================================================
# Hash Computation
# ============================================================================


def compute_sha256(file_path: Path) -> str:
    """
    Compute SHA256 hash of file contents.
    Reads in chunks for memory efficiency with large files.
    """
    # Validate file exists and is readable
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    # Check file size - skip 0-byte files
    file_size = file_path.stat().st_size
    if file_size == 0:
        raise ValueError("File is empty (0 bytes)")

    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):  # 64KB chunks
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except PermissionError as e:
        if console:
            cs.StatusIndicator("error").add_file(file_path.name).with_explanation(
                f"Permission denied: {e}"
            ).emit()
        raise
    except OSError as e:
        if console:
            cs.StatusIndicator("error").add_file(file_path.name).with_explanation(
                f"File system error: {e}"
            ).emit()
        raise
    except Exception as e:
        if console:
            cs.StatusIndicator("error").add_file(file_path.name).with_explanation(
                f"Failed to hash: {e}"
            ).emit()
        raise


# ============================================================================
# File Information
# ============================================================================


def get_file_info(file_path: Path) -> FileInfo:
    """Extract file information needed for duplicate detection"""
    try:
        stat = file_path.stat()
    except OSError as e:
        raise OSError(f"Cannot access file: {e}")

    sha256 = compute_sha256(file_path)

    return FileInfo(
        path=file_path,
        sha256=sha256,
        size=stat.st_size,
        created=stat.st_ctime,
        modified=stat.st_mtime,
    )


# ============================================================================
# Duplicate Detection
# ============================================================================


def _handle_file_processing_error(
    font_path: Path,
    error: Exception,
    error_type: str,
    progress_task=None,
) -> None:
    """
    Handle file processing errors with consistent error reporting and progress updates.

    Args:
        font_path: Path to the file that failed
        error: The exception that occurred
        error_type: Type of error ("warning", "error")
        progress_task: Optional progress task to update (if None, no progress update)
    """
    if error_type == "warning":
        if console:
            cs.StatusIndicator("warning").add_file(font_path.name).with_explanation(
                str(error)
            ).emit()
    else:  # error
        if console:
            cs.StatusIndicator("error").add_file(font_path.name).with_explanation(
                str(error)
            ).emit()

    if progress_task is not None:
        progress_task.update(advance=1)


def find_duplicates(
    font_files: List[Path], verbose: bool = False
) -> Dict[str, List[FileInfo]]:
    """
    Find duplicate files by SHA256 hash.
    Returns dict mapping hash -> list of FileInfo objects with that hash.
    Only includes hashes with 2+ files.
    """
    hash_map: Dict[str, List[FileInfo]] = {}
    total_files = len(font_files)

    # Show progress for large batches
    show_progress = total_files > 50 and console and cs.RICH_AVAILABLE

    if show_progress:
        progress = cs.create_progress_bar()
        task = progress.add_task("Hashing files...", total=total_files)
        progress.start()

    try:
        for idx, font_path in enumerate(font_files, 1):
            try:
                file_info = get_file_info(font_path)

                if file_info.sha256 not in hash_map:
                    hash_map[file_info.sha256] = []
                hash_map[file_info.sha256].append(file_info)

                if verbose and console and not show_progress:
                    cs.StatusIndicator("info").add_file(font_path.name).add_message(
                        "Hashed"
                    ).emit()

                if show_progress:
                    progress.update(task, advance=1)

            except ValueError as e:
                # Skip empty files silently
                if "empty" not in str(e).lower():
                    _handle_file_processing_error(
                        font_path, e, "warning", task if show_progress else None
                    )
                elif show_progress:
                    progress.update(task, advance=1)
                continue
            except PermissionError as e:
                _handle_file_processing_error(
                    font_path,
                    PermissionError(f"Permission denied: {e}"),
                    "error",
                    task if show_progress else None,
                )
                continue
            except Exception as e:
                _handle_file_processing_error(
                    font_path,
                    Exception(f"Failed to process: {e}"),
                    "error",
                    task if show_progress else None,
                )
                continue
    finally:
        if show_progress:
            progress.stop()

    # Filter to only duplicates (2+ files with same hash)
    duplicates = {sha256: files for sha256, files in hash_map.items() if len(files) > 1}

    return duplicates


def find_cross_directory_duplicates(
    primary_files: List[Path],
    secondary_files: List[Path],
    verbose: bool = False,
) -> Dict[str, Tuple[List[FileInfo], List[FileInfo]]]:
    """
    Find duplicate files between two directories.
    Returns dict mapping hash -> (primary_files, secondary_files).
    Only includes hashes where both primary and secondary have files.
    """
    primary_hash_map: Dict[str, List[FileInfo]] = {}
    secondary_hash_map: Dict[str, List[FileInfo]] = {}
    total_files = len(primary_files) + len(secondary_files)

    # Show progress for large batches
    show_progress = total_files > 50 and console and cs.RICH_AVAILABLE

    if show_progress:
        progress = cs.create_progress_bar()
        task = progress.add_task(
            "Hashing files for cross-directory comparison...", total=total_files
        )
        progress.start()

    try:
        # Process primary directory files
        for font_path in primary_files:
            try:
                file_info = get_file_info(font_path)
                hash_key = file_info.sha256

                if hash_key not in primary_hash_map:
                    primary_hash_map[hash_key] = []
                primary_hash_map[hash_key].append(file_info)

                if show_progress:
                    progress.update(task, advance=1)

            except Exception as e:
                _handle_file_processing_error(
                    font_path, e, "warning", task if show_progress else None
                )
                if show_progress:
                    progress.update(task, advance=1)
                continue

        # Process secondary directory files
        for font_path in secondary_files:
            try:
                file_info = get_file_info(font_path)
                hash_key = file_info.sha256

                if hash_key not in secondary_hash_map:
                    secondary_hash_map[hash_key] = []
                secondary_hash_map[hash_key].append(file_info)

                if show_progress:
                    progress.update(task, advance=1)

            except Exception as e:
                _handle_file_processing_error(
                    font_path, e, "warning", task if show_progress else None
                )
                if show_progress:
                    progress.update(task, advance=1)
                continue

    finally:
        if show_progress:
            progress.stop()

    # Find hashes that exist in both directories
    cross_duplicates = {}
    for hash_key in primary_hash_map:
        if hash_key in secondary_hash_map:
            cross_duplicates[hash_key] = (
                primary_hash_map[hash_key],
                secondary_hash_map[hash_key],
            )

    return cross_duplicates


# ============================================================================
# Duplicate Resolution
# ============================================================================


def has_priority_suffix(filename: str) -> Tuple[bool, str]:
    """
    Check if filename has priority suffix pattern (~001, ~002, etc.).
    Returns (has_suffix, base_name_without_suffix)
    """
    # Match pattern like: FontName-Bold~001.otf
    match = re.match(r"^(.+?)~(\d{3})(\.[^.]+)$", filename)
    if match:
        base_name = match.group(1) + match.group(3)  # FontName-Bold.otf
        return True, base_name
    return False, filename


def select_file_to_keep(
    duplicates: List[FileInfo], strategy: str = "oldest"
) -> Tuple[FileInfo, List[FileInfo]]:
    """
    Select which duplicate to keep based on strategy.
    Smart naming: If duplicates include priority-named files (with ~001 suffix),
    prefer the clean name over suffixed versions.
    Returns (file_to_keep, files_to_remove)
    """
    # Check if we have priority-named duplicates (clean name vs ~001, ~002, etc.)
    suffixed_files = []
    clean_files = []

    for file_info in duplicates:
        has_suffix, base_name = has_priority_suffix(file_info.name)
        if has_suffix:
            suffixed_files.append((file_info, base_name))
        else:
            clean_files.append(file_info)

    # If we have both clean and suffixed versions, prefer clean name
    if clean_files and suffixed_files:
        # Check if any clean file matches the base name of suffixed files
        suffixed_base_names = {base_name for _, base_name in suffixed_files}

        # Find clean files that match suffixed base names
        matching_clean = [f for f in clean_files if f.name in suffixed_base_names]

        if matching_clean:
            # Keep the clean file, remove all suffixed versions
            files_to_remove = [f for f, _ in suffixed_files]
            # Add any other clean files that don't match
            files_to_remove.extend([f for f in clean_files if f not in matching_clean])
            # Keep the first matching clean file (or apply strategy if multiple)
            if len(matching_clean) == 1:
                return matching_clean[0], files_to_remove
            # Multiple clean files match - fall through to strategy
            duplicates = matching_clean

    # Apply standard strategy
    if strategy == "oldest":
        # Keep file with oldest creation time
        sorted_files = sorted(duplicates, key=lambda f: f.created)
    elif strategy == "newest":
        # Keep file with newest modification time
        sorted_files = sorted(duplicates, key=lambda f: -f.modified)
    elif strategy == "first":
        # Keep first alphabetically
        sorted_files = sorted(duplicates, key=lambda f: f.name)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return sorted_files[0], sorted_files[1:]


# ============================================================================
# File Operations
# ============================================================================


def _retry_file_operation(
    operation: callable,
    file_path: Path,
    max_retries: int = 3,
    retry_delay: float = 0.1,
    error_message_prefix: str = "Operation failed",
) -> Tuple[bool, Optional[str]]:
    """
    Execute a file operation with retry logic for transient errors.

    Args:
        operation: Callable that performs the file operation (no args)
        file_path: Path to the file being operated on (for error messages)
        max_retries: Maximum number of retry attempts
        retry_delay: Delay in seconds between retries
        error_message_prefix: Prefix for error messages

    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    for attempt in range(max_retries):
        try:
            operation()
            return True, None
        except PermissionError as e:
            if attempt == max_retries - 1:
                return (
                    False,
                    f"{error_message_prefix} - Permission denied after {max_retries} attempts: {e}",
                )
            time.sleep(retry_delay)
        except OSError as e:
            if attempt == max_retries - 1:
                return (
                    False,
                    f"{error_message_prefix} - File system error after {max_retries} attempts: {e}",
                )
            time.sleep(retry_delay)

    return False, f"{error_message_prefix} - Max retries exceeded"


def move_to_trash(file_path: Path, trash_dir: Path, dry_run: bool = False) -> bool:
    """
    Move a file to trash directory (with conflict handling).
    Returns True if successful, False otherwise.
    """
    if dry_run:
        return True

    try:
        # Check if source file still exists
        if not file_path.exists():
            if console:
                cs.StatusIndicator("warning").add_file(file_path.name).with_explanation(
                    "File no longer exists"
                ).emit()
            return False

        # Check permissions on source file
        if not file_path.is_file():
            if console:
                cs.StatusIndicator("error").add_file(file_path.name).with_explanation(
                    "Not a regular file"
                ).emit()
            return False

        # Create trash directory with proper permissions
        try:
            trash_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            if console:
                cs.StatusIndicator("error").add_file(file_path.name).with_explanation(
                    f"Cannot create trash directory: {e}"
                ).emit()
            return False

        # Handle naming conflicts in trash
        target = trash_dir / file_path.name
        counter = 1
        while target.exists():
            stem = file_path.stem
            suffix = file_path.suffix
            target = trash_dir / f"{stem}_{counter}{suffix}"
            counter += 1
            if counter > 9999:  # Safety limit
                if console:
                    cs.StatusIndicator("error").add_file(
                        file_path.name
                    ).with_explanation("Too many conflicts in trash directory").emit()
                return False

        # Attempt move with retry for transient errors
        success, error_msg = _retry_file_operation(
            lambda: shutil.move(str(file_path), str(target)),
            file_path,
            max_retries=3,
            retry_delay=0.1,
            error_message_prefix="Failed to move file",
        )

        if success:
            return True

        if console:
            cs.StatusIndicator("error").add_file(file_path.name).with_explanation(
                error_msg or "Failed to move file"
            ).emit()
        return False

    except Exception as e:
        if console:
            cs.StatusIndicator("error").add_file(file_path.name).with_explanation(
                f"Failed to move: {e}"
            ).emit()
        return False

    return False


def format_bytes(size: int) -> str:
    """Format byte size as human-readable string"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def remove_empty_directories(root_dir: Path, verbose: bool = False) -> int:
    """
    Recursively remove empty directories starting from deepest level.
    Returns count of directories removed.
    """
    import os

    removed_count = 0

    # Walk directory tree bottom-up (deepest first)
    try:
        # Collect all directories first
        dirs_to_check = []
        for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
            dirs_to_check.append(Path(dirpath))

        # Process in reverse order (deepest first)
        for dir_path in dirs_to_check:
            try:
                # Check if directory is empty
                if dir_path.exists() and dir_path.is_dir():
                    # Check if directory has any contents
                    try:
                        next(dir_path.iterdir(), None)
                        # Directory has contents, skip
                        continue
                    except StopIteration:
                        # Directory is empty, remove it
                        try:
                            dir_path.rmdir()
                            removed_count += 1
                            if console and verbose:
                                cs.StatusIndicator("info").add_file(
                                    str(dir_path.relative_to(root_dir))
                                ).with_explanation("Removed empty directory").emit()
                        except OSError as e:
                            # Directory might have been removed already or permission error
                            if verbose:
                                cs.StatusIndicator("warning").add_file(
                                    str(dir_path.relative_to(root_dir))
                                ).with_explanation(
                                    f"Could not remove directory: {e}"
                                ).emit()
            except Exception as e:
                if verbose:
                    cs.StatusIndicator("warning").add_file(
                        str(dir_path.relative_to(root_dir))
                    ).with_explanation(f"Error checking directory: {e}").emit()
    except Exception as e:
        if console:
            cs.StatusIndicator("warning").with_explanation(
                f"Error during empty directory cleanup: {e}"
            ).emit()

    return removed_count


# ============================================================================
# Directory Processing
# ============================================================================


def collect_directory_fonts(directory: Path) -> List[Path]:
    """Collect font files from directory"""
    font_files = []
    for ext in FONT_EXTENSIONS:
        font_files.extend(directory.glob(f"*{ext}"))
    return font_files


def process_duplicate_file(
    file_info: FileInfo, trash_dir: Path, dry_run: bool, verbose: bool
) -> Tuple[bool, int]:
    """
    Process a single duplicate file (move to trash or preview).
    Returns (success, bytes_saved)
    """
    # Use same StatusIndicator for both dry-run and normal mode
    # DRY prefix will be added automatically when dry_run=True
    if console:
        if verbose:
            cs.StatusIndicator("deleted", dry_run=dry_run).add_file(
                file_info.name
            ).add_message("Removed duplicate" if not dry_run else "Would remove").emit()
        else:
            cs.StatusIndicator("deleted", dry_run=dry_run).add_file(
                file_info.name
            ).with_explanation("Removed duplicate" if not dry_run else "Would remove").emit()

    if dry_run:
        return True, file_info.size

    if move_to_trash(file_info.path, trash_dir, dry_run):
        return True, file_info.size

    return False, 0


def process_duplicate_group(
    duplicate_files: List[FileInfo],
    strategy: str,
    trash_dir: Path,
    dry_run: bool,
    verbose: bool,
) -> Tuple[int, int, List[Tuple[str, str]]]:
    """
    Process a single duplicate group.
    Returns (files_moved, bytes_saved, errors)
    """
    file_to_keep, files_to_remove = select_file_to_keep(duplicate_files, strategy)

    if console and verbose:
        cs.StatusIndicator("info").add_message(
            f"Duplicate group ({cs.fmt_count(len(duplicate_files))} files, {format_bytes(file_to_keep.size)} each):"
        ).emit()
        cs.StatusIndicator("info").add_file(file_to_keep.name).add_message(
            "KEEP"
        ).emit()

    files_moved = 0
    bytes_saved = 0
    errors = []

    for file_info in files_to_remove:
        success, saved = process_duplicate_file(file_info, trash_dir, dry_run, verbose)
        if success:
            files_moved += 1
            bytes_saved += saved
        else:
            errors.append((file_info.name, "Failed to move to trash"))

    return files_moved, bytes_saved, errors


def process_cross_directory_duplicates(
    cross_duplicates: Dict[str, Tuple[List[FileInfo], List[FileInfo]]],
    trash_dir: Path,
    dry_run: bool,
    verbose: bool,
) -> Tuple[int, int, List[Tuple[str, str]]]:
    """
    Process cross-directory duplicates by removing files from secondary directory.
    Returns (files_removed, bytes_saved, errors)
    """
    files_removed = 0
    bytes_saved = 0
    errors = []

    for hash_key, (primary_files, secondary_files) in cross_duplicates.items():
        if console and verbose:
            cs.StatusIndicator("info").add_message(
                f"Cross-directory duplicate group ({cs.fmt_count(len(primary_files))} primary, {cs.fmt_count(len(secondary_files))} secondary):"
            ).emit()
            for pf in primary_files:
                cs.StatusIndicator("info").add_file(pf.name).add_message(
                    "KEEP (primary)"
                ).emit()

        # Remove all secondary files
        for file_info in secondary_files:
            success, saved = process_duplicate_file(
                file_info, trash_dir, dry_run, verbose
            )
            if success:
                files_removed += 1
                bytes_saved += saved
            else:
                errors.append((file_info.name, "Failed to move to trash"))

    return files_removed, bytes_saved, errors


def process_directory(
    directory: Path,
    strategy: str = "oldest",
    trash_dir: Path = DEFAULT_TRASH_DIR,
    dry_run: bool = False,
    verbose: bool = False,
) -> DeduplicationStats:
    """Process all font files in a single directory"""
    stats = DeduplicationStats()

    # Collect font files
    font_files = collect_directory_fonts(directory)
    if not font_files:
        return stats

    stats.total_files = len(font_files)

    if console and verbose:
        cs.StatusIndicator("info").add_message(
            f"Analyzing {cs.fmt_count(len(font_files))} files in {cs.fmt_file(str(directory))}"
        ).emit()

    # Find duplicates
    duplicates = find_duplicates(font_files, verbose)

    if not duplicates:
        if console and verbose:
            cs.StatusIndicator("info").add_message("No duplicates found").emit()
        stats.unique_files = stats.total_files
        return stats

    stats.duplicate_groups = len(duplicates)

    # Process each duplicate group
    for sha256, duplicate_files in duplicates.items():
        files_moved, bytes_saved, errors = process_duplicate_group(
            duplicate_files, strategy, trash_dir, dry_run, verbose
        )

        stats.files_moved += files_moved
        stats.bytes_saved += bytes_saved
        for filename, reason in errors:
            stats.add_error(filename, reason)

    stats.duplicate_files = sum(len(files) - 1 for files in duplicates.values())
    stats.unique_files = stats.total_files - stats.duplicate_files

    return stats


# ============================================================================
# Main Entry Point
# ============================================================================


def collect_font_files_with_progress_bar(
    paths: List[str],
    recursive: bool,
    description: str = "Scanning directories...",
) -> List[str]:
    """
    Collect font files with a progress bar showing directory scanning progress.
    Uses same styling as Variation Analyzer - simple description, no dynamic updates.
    """
    from FontCore.core_file_collector import iter_font_files

    files_found = []
    show_progress = console and cs.RICH_AVAILABLE

    if show_progress:
        progress = cs.create_progress_bar()
        progress.add_task(description, total=None)  # Indeterminate progress
        progress.start()

        try:
            for file_path in iter_font_files(
                paths=paths,
                recursive=recursive,
                allowed_extensions=FONT_EXTENSIONS,
            ):
                files_found.append(file_path)
        finally:
            progress.stop()
    else:
        # Fallback without progress bar
        files_found = collect_font_files(
            paths, recursive=recursive, allowed_extensions=FONT_EXTENSIONS
        )

    return files_found


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


def show_directory_stats(dir_stats: DeduplicationStats, verbose: bool) -> None:
    """Display statistics for a single directory"""
    if console and not verbose:
        cs.StatusIndicator("info").add_message(
            f"Files: {cs.fmt_count(dir_stats.total_files)} | Duplicates: {cs.fmt_count(dir_stats.duplicate_files)} | Removed: {cs.fmt_count(dir_stats.files_moved)}"
        ).emit()


def show_final_errors(total_stats: DeduplicationStats, verbose: bool) -> None:
    """Display error summary"""
    if total_stats.errors and console:
        cs.StatusIndicator("warning").add_message(
            f"{cs.fmt_count(len(total_stats.errors))} errors occurred"
        ).emit()
        if verbose:
            for filename, reason in total_stats.errors:
                cs.StatusIndicator("warning").add_file(filename).with_explanation(
                    reason
                ).emit()


def main():
    parser = argparse.ArgumentParser(
        description="Remove exact byte-for-byte duplicate font files using SHA256",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/fonts/                    # Deduplicate fonts in directory
  %(prog)s font1.otf font2.otf               # Check specific files
  %(prog)s /fonts/ -r               # Process recursively
  %(prog)s /fonts/ -n                 # Preview what would be removed
  %(prog)s /fonts/ --keep-strategy newest    # Keep newest duplicate
  %(prog)s --compare-dirs /fonts1 /fonts2    # Compare two directories
  %(prog)s --compare-dirs /fonts1 /fonts2 -r  # Compare recursively

Keep Strategies:
  oldest  - Keep file with oldest creation time (default)
  newest  - Keep file with newest modification time
  first   - Keep first file alphabetically
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
        help="Preview what would be removed without moving files",
    )
    parser.add_argument(
        "-s",
        "--keep-strategy",
        choices=["oldest", "newest", "first"],
        default="oldest",
        help="Strategy for selecting which duplicate to keep (default: oldest)",
    )
    parser.add_argument(
        "-t",
        "--trash-dir",
        type=Path,
        default=DEFAULT_TRASH_DIR,
        help=f"Directory to move duplicates to (default: {DEFAULT_TRASH_DIR})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed processing information",
    )
    parser.add_argument(
        "--compare-dirs",
        nargs=2,
        metavar=("PRIMARY", "SECONDARY"),
        help="Compare two directories and remove duplicates from secondary directory",
    )

    args = parser.parse_args()

    # Validate compare-dirs argument
    if args.compare_dirs:
        primary_dir = Path(args.compare_dirs[0])
        secondary_dir = Path(args.compare_dirs[1])

        if not primary_dir.exists():
            if console:
                cs.StatusIndicator("error").with_explanation(
                    f"Primary directory does not exist: {primary_dir}"
                ).emit()
            return 1

        if not secondary_dir.exists():
            if console:
                cs.StatusIndicator("error").with_explanation(
                    f"Secondary directory does not exist: {secondary_dir}"
                ).emit()
            return 1

        if not primary_dir.is_dir():
            if console:
                cs.StatusIndicator("error").with_explanation(
                    f"Primary path is not a directory: {primary_dir}"
                ).emit()
            return 1

        if not secondary_dir.is_dir():
            if console:
                cs.StatusIndicator("error").with_explanation(
                    f"Secondary path is not a directory: {secondary_dir}"
                ).emit()
            return 1

        # Cannot use compare-dirs with regular paths
        if args.paths:
            if console:
                cs.StatusIndicator("error").with_explanation(
                    "Cannot use --compare-dirs with regular paths argument"
                ).emit()
            return 1

        # Handle cross-directory comparison mode
        if console:
            cs.StatusIndicator("info").add_message(
                f"Scanning primary directory: {primary_dir.name}"
            ).emit()
        primary_files = collect_font_files_with_progress_bar(
            [str(primary_dir)],
            recursive=args.recursive,
            description="Scanning directories...",
        )

        if console:
            cs.StatusIndicator("info").add_message(
                f"Scanning secondary directory: {secondary_dir.name}"
            ).emit()
        secondary_files = collect_font_files_with_progress_bar(
            [str(secondary_dir)],
            recursive=args.recursive,
            description="Scanning directories...",
        )

        if not primary_files and not secondary_files:
            if console:
                cs.StatusIndicator("error").with_explanation(
                    "No font files found in either directory"
                ).emit()
            return 1

        # Show summary
        if console:
            # Use same mode string for both dry-run and normal mode
            # DRY prefix will be added automatically by StatusIndicator when dry_run=True
            mode = "CROSS-DIRECTORY COMPARE"
            summary_lines = [
                f"Mode: {mode}",
                f"Primary directory: {primary_dir}",
                f"Secondary directory: {secondary_dir}",
                f"Primary files: {cs.fmt_count(len(primary_files))}",
                f"Secondary files: {cs.fmt_count(len(secondary_files))}",
                f"Recursive: {'Yes' if args.recursive else 'No'}",
                f"Trash: {args.trash_dir}",
            ]
            cs.print_panel(
                "\n".join(summary_lines),
                title="SHA256 Font Deduplicator",
                border_style="blue",
            )

        # Find cross-directory duplicates
        cross_duplicates = find_cross_directory_duplicates(
            [Path(f) for f in primary_files],
            [Path(f) for f in secondary_files],
            verbose=args.verbose,
        )

        if not cross_duplicates:
            if console:
                cs.StatusIndicator("info").add_message(
                    "No duplicates found between directories"
                ).emit()
            return 0

        # Process cross-directory duplicates
        files_removed, bytes_saved, errors = process_cross_directory_duplicates(
            cross_duplicates,
            args.trash_dir,
            args.dry_run,
            args.verbose,
        )

        # Clean up empty directories (if not dry-run)
        primary_dirs_removed = 0
        secondary_dirs_removed = 0
        if not args.dry_run:
            if console:
                cs.StatusIndicator("info").add_message(
                    "Cleaning up empty directories..."
                ).emit()
            secondary_dirs_removed = remove_empty_directories(
                secondary_dir, args.verbose
            )
            primary_dirs_removed = remove_empty_directories(primary_dir, args.verbose)

        # Show summary
        if console:
            summary_lines = [
                f"Primary files: {cs.fmt_count(len(primary_files))}",
                f"Secondary files: {cs.fmt_count(len(secondary_files))}",
                f"Duplicates found: {cs.fmt_count(len(cross_duplicates))} groups",
                f"Files removed from secondary: {cs.fmt_count(files_removed)}",
                f"Space saved: {format_bytes(bytes_saved)}",
            ]
            if primary_dirs_removed > 0 or secondary_dirs_removed > 0:
                summary_lines.append(
                    f"Empty directories removed: Primary: {primary_dirs_removed}, Secondary: {secondary_dirs_removed}"
                )
            cs.print_panel(
                "\n".join(summary_lines),
                title="Summary",
                border_style="green",
            )

            if errors:
                show_final_errors(DeduplicationStats(errors=errors), args.verbose)

        return 0 if not errors else 1

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
        # Use same mode string for both dry-run and normal mode
        # DRY prefix will be added automatically by StatusIndicator when dry_run=True
        mode = "DEDUPLICATE"
        cs.print_panel(
            f"Mode: {mode}\n"
            f"Strategy: Keep {args.keep_strategy} duplicate\n"
            f"Files: {cs.fmt_count(len(font_paths))}\n"
            f"Directories: {cs.fmt_count(len(dirs_to_process))}\n"
            f"Trash: {args.trash_dir}",
            title="SHA256 Font Deduplicator",
            border_style="blue",
        )

    # Process each directory
    total_stats = DeduplicationStats()
    for idx, (directory, files_in_dir) in enumerate(sorted(dirs_to_process.items()), 1):
        if console:
            cs.StatusIndicator("info").add_message(
                f"Directory {idx}/{len(dirs_to_process)}: {cs.fmt_file(str(directory))}"
            ).emit()

        dir_stats = process_directory(
            directory,
            strategy=args.keep_strategy,
            trash_dir=args.trash_dir,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )

        total_stats.total_files += dir_stats.total_files
        total_stats.unique_files += dir_stats.unique_files
        total_stats.duplicate_files += dir_stats.duplicate_files
        total_stats.duplicate_groups += dir_stats.duplicate_groups
        total_stats.bytes_saved += dir_stats.bytes_saved
        total_stats.files_moved += dir_stats.files_moved
        total_stats.errors.extend(dir_stats.errors)

        show_directory_stats(dir_stats, args.verbose)

    # Final summary
    if console:
        cs.print_panel(
            f"Total files scanned: {cs.fmt_count(total_stats.total_files)}\n"
            f"Unique files: {cs.fmt_count(total_stats.unique_files)}\n"
            f"Duplicate files: {cs.fmt_count(total_stats.duplicate_files)} "
            f"({total_stats.duplicate_groups} groups)\n"
            f"Files removed: {cs.fmt_count(total_stats.files_moved)}\n"
            f"Space saved: {format_bytes(total_stats.bytes_saved)}",
            title="Summary",
            border_style="green",
        )

        show_final_errors(total_stats, args.verbose)

    return 0


if __name__ == "__main__":
    exit(main())
