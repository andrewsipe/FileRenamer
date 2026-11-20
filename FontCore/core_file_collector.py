"""
core_file_collector: shared helpers to collect font files from files/dirs.

Features:
- Supports TTF, OTF, WOFF, WOFF2, TTX by default
- Optional recursive directory scanning
- Case-insensitive extension matching
- De-duplicates and returns sorted list of absolute paths

Demo and Testing:
    Run 'python CoreDemoTool.py collector --help' to see examples and test with real paths.
    The demo tool shows supported extensions and file collection capabilities.

Maintenance Note:
    When adding new file collection features to this module, update CoreDemoTool.py to showcase
    the new functionality in the 'collector' subcommand.
"""

from __future__ import annotations

import glob
import os
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Set, TypedDict


SUPPORTED_EXTENSIONS: Set[str] = {".ttf", ".otf", ".woff", ".woff2", ".ttx"}


class ProgressDict(TypedDict, total=False):
    """Type definition for progress callback dictionary."""

    dirs_scanned: int
    files_scanned: int
    matches_found: int
    current_dir: str
    error: str  # optional


def _normalize_paths(paths: Iterable[str | Path]) -> List[Path]:
    """Convert input paths to Path objects, expanding user paths."""
    norm: List[Path] = []
    for p in paths:
        try:
            norm.append(Path(p).expanduser())
        except Exception:
            continue
    return norm


def _matches_extension(path: Path, allowed_extensions: Set[str]) -> bool:
    """Check if path has an allowed extension (case-insensitive)."""
    try:
        ext = path.suffix.lower()
    except Exception:
        return False

    if not ext:
        return False

    allowed_lower = {e.lower() for e in allowed_extensions}
    return ext in allowed_lower


def _glob_patterns_for_extension(
    base_path: Path, ext: str, recursive: bool, include_uppercase: bool
) -> List[str]:
    """Generate glob patterns for a single extension."""
    ext_low = ext.lower()
    patterns = []

    prefix = "**/" if recursive else ""
    patterns.append(str(base_path / f"{prefix}*{ext_low}"))

    if include_uppercase:
        patterns.append(str(base_path / f"{prefix}*{ext_low.upper()}"))

    return patterns


def _collect_from_directory(
    path_obj: Path, allowed: Set[str], recursive: bool, include_uppercase: bool
) -> Set[str]:
    """Collect all matching files from a directory."""
    results: Set[str] = set()

    for ext in allowed:
        patterns = _glob_patterns_for_extension(
            path_obj, ext, recursive, include_uppercase
        )
        for pattern in patterns:
            results.update(glob.glob(pattern, recursive=recursive))

    return results


def _safe_absolute_path(path: str, allowed: Set[str]) -> Optional[str]:
    """Convert to absolute path if it matches allowed extensions."""
    try:
        po = Path(path)
        return str(po) if _matches_extension(po, allowed) else None
    except Exception:
        return None


class ProgressDict(TypedDict, total=False):
    """Type definition for progress callback dictionary."""

    dirs_scanned: int
    files_scanned: int
    matches_found: int
    current_dir: str
    error: str  # optional


def iter_font_files(
    paths: Iterable[str | Path],
    recursive: bool = True,
    *,
    allowed_extensions: Optional[Set[str]] = None,
    include_uppercase: bool = True,
    on_progress: Optional[Callable[[ProgressDict], None]] = None,
) -> Iterator[str]:
    """Iterate over font file paths as they are discovered, with optional progress callbacks.

    - paths: files and/or directories
    - recursive: recurse into directories (default True for progress use case)
    - allowed_extensions: override supported extensions; compared case-insensitively
    - include_uppercase: when globbing, also search uppercase extension variants
    - on_progress: optional callback with progress dict: {'dirs_scanned', 'files_scanned', 'matches_found', 'current_dir'}

    Yields:
        Absolute file paths as they are discovered
    """
    allowed = allowed_extensions or SUPPORTED_EXTENSIONS
    path_objs = _normalize_paths(paths)

    # Progress tracking
    dirs_scanned = 0
    files_scanned = 0
    matches_found = 0
    last_progress_time = 0
    current_dir = ""

    def _should_report_progress() -> bool:
        """Throttle progress updates to every 200 files or 100ms"""
        nonlocal last_progress_time
        now = time.time()
        if files_scanned % 200 == 0 or (now - last_progress_time) > 0.1:
            last_progress_time = now
            return True
        return False

    def _report_progress():
        """Report current progress if callback provided"""
        if on_progress and _should_report_progress():
            on_progress(
                {
                    "dirs_scanned": dirs_scanned,
                    "files_scanned": files_scanned,
                    "matches_found": matches_found,
                    "current_dir": current_dir,
                }
            )

    for path_obj in path_objs:
        try:
            if path_obj.is_file():
                files_scanned += 1
                if _matches_extension(path_obj, allowed):
                    matches_found += 1
                    yield str(path_obj.resolve())
                _report_progress()
            elif path_obj.is_dir():
                if recursive:
                    # Use os.walk for better performance than glob
                    try:
                        for root, dirs, files in os.walk(path_obj):
                            current_dir = root
                            dirs_scanned += 1

                            for filename in files:
                                files_scanned += 1
                                file_path = Path(root) / filename

                                if _matches_extension(file_path, allowed):
                                    matches_found += 1
                                    yield str(file_path.resolve())

                                _report_progress()
                    except (PermissionError, OSError) as e:
                        # Report error but continue with other directories
                        if on_progress:
                            on_progress(
                                {
                                    "dirs_scanned": dirs_scanned,
                                    "files_scanned": files_scanned,
                                    "matches_found": matches_found,
                                    "current_dir": str(path_obj),
                                    "error": str(e),
                                }
                            )
                        continue  # Skip this directory, continue with others
                else:
                    # Non-recursive: check files in directory only
                    current_dir = str(path_obj)
                    dirs_scanned += 1
                    try:
                        for filename in os.listdir(path_obj):
                            files_scanned += 1
                            file_path = path_obj / filename

                            if file_path.is_file() and _matches_extension(
                                file_path, allowed
                            ):
                                matches_found += 1
                                yield str(file_path.resolve())

                            _report_progress()
                    except (PermissionError, OSError):
                        continue
        except Exception:
            continue

    # Final progress report
    if on_progress:
        on_progress(
            {
                "dirs_scanned": dirs_scanned,
                "files_scanned": files_scanned,
                "matches_found": matches_found,
                "current_dir": "Complete",
            }
        )


def collect_font_files_with_progress(
    paths: Iterable[str | Path],
    recursive: bool = True,
    *,
    allowed_extensions: Optional[Set[str]] = None,
    include_uppercase: bool = True,
    on_progress: Optional[Callable[[ProgressDict], None]] = None,
) -> List[str]:
    """Collect font file paths with progress callbacks (wrapper around iter_font_files).

    - paths: files and/or directories
    - recursive: recurse into directories
    - allowed_extensions: override supported extensions; compared case-insensitively
    - include_uppercase: when globbing, also search uppercase extension variants
    - on_progress: optional callback with progress dict: {'dirs_scanned', 'files_scanned', 'matches_found', 'current_dir'}

    Returns:
        Sorted list of absolute file paths
    """
    results = list(
        iter_font_files(
            paths=paths,
            recursive=recursive,
            allowed_extensions=allowed_extensions,
            include_uppercase=include_uppercase,
            on_progress=on_progress,
        )
    )
    return sorted(set(results))


def collect_font_files(
    paths: Iterable[str | Path],
    recursive: bool = False,
    *,
    allowed_extensions: Optional[Set[str]] = None,
    include_uppercase: bool = True,
) -> List[str]:
    """Collect font file paths from a list of files and/or directories.

    - paths: files and/or directories
    - recursive: recurse into directories
    - allowed_extensions: override supported extensions; compared case-insensitively
    - include_uppercase: when globbing, also search uppercase extension variants
    """
    allowed = allowed_extensions or SUPPORTED_EXTENSIONS
    path_objs = _normalize_paths(paths)
    results: Set[str] = set()

    for path_obj in path_objs:
        try:
            if path_obj.is_file():
                if _matches_extension(path_obj, allowed):
                    results.add(str(path_obj))
            elif path_obj.is_dir():
                results.update(
                    _collect_from_directory(
                        path_obj, allowed, recursive, include_uppercase
                    )
                )
        except Exception:
            continue

    # Filter and convert to absolute paths
    filtered = [
        abs_path
        for p in results
        if (abs_path := _safe_absolute_path(p, allowed)) is not None
    ]

    return sorted(set(filtered))


def collect_font_files_with_rich_progress(
    paths: Iterable[str | Path],
    recursive: bool = True,
    *,
    allowed_extensions: Optional[Set[str]] = None,
    include_uppercase: bool = True,
    description: str = "Scanning for font files...",
    console: Optional[object] = None,
) -> List[str]:
    """Collect font files with a Rich progress bar (optional enhancement).

    This is a convenience wrapper that uses Rich progress bars if available,
    falling back to simple collection if Rich is not available or if console
    is not provided.

    Args:
        paths: Files and/or directories to scan
        recursive: Recurse into directories
        allowed_extensions: Override supported extensions
        include_uppercase: Include uppercase extension variants
        description: Progress bar description text
        console: Rich Console instance (optional, will try to get one if None)

    Returns:
        Sorted list of absolute file paths
    """
    # Try to import Rich and console_styles (optional dependencies)
    try:
        import FontCore.core_console_styles as cs
        from FontCore.core_console_styles import create_progress_bar

        rich_available = cs.RICH_AVAILABLE
        if console is None:
            console = cs.get_console()
    except ImportError:
        rich_available = False
        console = None

    # Use Rich progress bar if available
    if rich_available and console:
        try:
            progress = create_progress_bar(console)
            task_id = progress.add_task(
                f"[progress.description]{description}",
                total=None,  # Indeterminate progress
            )

            font_paths = []

            def progress_callback(progress_dict: ProgressDict) -> None:
                """Update progress bar during font file collection."""
                files_scanned = progress_dict.get("files_scanned", 0)
                matches_found = progress_dict.get("matches_found", 0)
                current_dir = progress_dict.get("current_dir", "")

                # Update progress description
                if current_dir and current_dir != "Complete":
                    dir_name = Path(current_dir).name
                    desc = f"{description} {matches_found} fonts found ({files_scanned} files scanned) - {dir_name}"
                else:
                    desc = f"{description} {matches_found} fonts found ({files_scanned} files scanned)"

                progress.update(task_id, description=desc)

            with progress:
                for font_path in iter_font_files(
                    paths=paths,
                    recursive=recursive,
                    allowed_extensions=allowed_extensions,
                    include_uppercase=include_uppercase,
                    on_progress=progress_callback,
                ):
                    font_paths.append(font_path)

            return sorted(set(font_paths))
        except Exception:
            # Fallback to simple collection if progress bar fails
            pass

    # Fallback: simple collection without progress bar
    return collect_font_files_with_progress(
        paths=paths,
        recursive=recursive,
        allowed_extensions=allowed_extensions,
        include_uppercase=include_uppercase,
    )


__all__ = [
    "SUPPORTED_EXTENSIONS",
    "collect_font_files",
    "iter_font_files",
    "collect_font_files_with_progress",
    "collect_font_files_with_rich_progress",
]
