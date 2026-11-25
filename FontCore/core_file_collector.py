"""
core_file_collector: shared helpers to collect font files from files/dirs.

One walker sits behind every function here, so they all follow the same rules (DECISIONS.md, area 3):

- Not recursive unless asked (`recursive=True`).
- Binary formats only by default (TTF, OTF, WOFF, WOFF2); `.ttx` when a tool passes `allowed_extensions=SUPPORTED_EXTENSIONS`.
- macOS junk (`._*` files, `__MACOSX` folders) and hidden files and folders are skipped unless `include_hidden=True`.
  A file or folder named directly is always honoured. Folders are never skipped by name (a tool that creates output
  folders decides for itself what to do with them).
- Each real file once: a symlink and its target, or a folder given twice, give one path (resolved).
- `on_missing(path)` is called for an input that does not exist; `on_unsupported(path)` for a recognised font file the
  tools cannot read (`.ttc`, `.dfont`, `.eot`), so a tool can say what it left out instead of staying silent.
- Case-insensitive extension matching; returns sorted absolute paths.

Demo and Testing:
    Run 'python CoreDemoTool.py collector --help' to see examples and test with real paths.

Maintenance Note:
    When adding new file collection features to this module, update CoreDemoTool.py to showcase
    the new functionality in the 'collector' subcommand.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Set, TypedDict


BINARY_EXTENSIONS: Set[str] = {".ttf", ".otf", ".woff", ".woff2"}
# Everything the tools can read: the binary formats plus TTX (XML). TTX is opt-in.
SUPPORTED_EXTENSIONS: Set[str] = BINARY_EXTENSIONS | {".ttx"}
# Font files the tools do not read; counted through `on_unsupported`, never returned.
RECOGNISED_UNSUPPORTED_EXTENSIONS: Set[str] = {".ttc", ".dfont", ".eot"}
_JUNK_DIRS = {"__macosx"}


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


class ProgressDict(TypedDict, total=False):
    """Type definition for progress callback dictionary."""

    dirs_scanned: int
    files_scanned: int
    matches_found: int
    current_dir: str
    error: str  # optional


def _is_hidden_name(name: str) -> bool:
    return name.startswith(".")


def iter_font_files(
    paths: Iterable[str | Path],
    recursive: bool = False,
    *,
    allowed_extensions: Optional[Set[str]] = None,
    include_uppercase: bool = True,  # kept for older callers; matching is always case-insensitive
    include_hidden: bool = False,
    on_progress: Optional[Callable[[ProgressDict], None]] = None,
    on_missing: Optional[Callable[[Path], None]] = None,
    on_unsupported: Optional[Callable[[Path], None]] = None,
) -> Iterator[str]:
    """Yield font file paths as they are found (absolute, resolved, each real file once).

    - paths: files and/or directories
    - recursive: go into subfolders (default False)
    - allowed_extensions: override the default set (BINARY_EXTENSIONS); compared case-insensitively
    - include_hidden: also walk hidden folders and return hidden files and macOS junk (`._*`, `__MACOSX`)
    - on_progress: callback with {'dirs_scanned', 'files_scanned', 'matches_found', 'current_dir'}
    - on_missing / on_unsupported: see the module docstring
    """
    allowed = allowed_extensions or BINARY_EXTENSIONS
    unsupported = {e.lower() for e in RECOGNISED_UNSUPPORTED_EXTENSIONS} - {e.lower() for e in allowed}
    seen: Set[str] = set()
    dirs_scanned = files_scanned = matches_found = 0
    last_progress_time = 0.0
    current_dir = ""

    def _report(force: bool = False) -> None:
        nonlocal last_progress_time
        if not on_progress:
            return
        now = time.time()
        if force or files_scanned % 200 == 0 or (now - last_progress_time) > 0.1:
            last_progress_time = now
            on_progress(
                {
                    "dirs_scanned": dirs_scanned,
                    "files_scanned": files_scanned,
                    "matches_found": matches_found,
                    "current_dir": current_dir,
                }
            )

    def _consider(file_path: Path):
        """Return the path to yield, or None."""
        nonlocal files_scanned, matches_found
        files_scanned += 1
        if _matches_extension(file_path, allowed):
            try:
                real = str(file_path.resolve())
            except OSError:
                return None
            if real in seen:
                return None
            seen.add(real)
            matches_found += 1
            return real
        if on_unsupported and file_path.suffix.lower() in unsupported:
            on_unsupported(file_path)
        return None

    for path_obj in _normalize_paths(paths):
        try:
            if path_obj.is_file():
                found = _consider(path_obj)  # a file named directly is honoured, hidden or not
                if found:
                    yield found
                _report()
            elif path_obj.is_dir():
                if recursive:
                    for root, dirs, files in os.walk(path_obj):
                        current_dir = root
                        dirs_scanned += 1
                        if not include_hidden:
                            dirs[:] = [d for d in dirs if not _is_hidden_name(d) and d.lower() not in _JUNK_DIRS]
                        for filename in sorted(files):
                            if not include_hidden and _is_hidden_name(filename):
                                continue
                            found = _consider(Path(root) / filename)
                            if found:
                                yield found
                            _report()
                else:
                    current_dir = str(path_obj)
                    dirs_scanned += 1
                    for filename in sorted(os.listdir(path_obj)):
                        if not include_hidden and _is_hidden_name(filename):
                            continue
                        file_path = path_obj / filename
                        if file_path.is_file():
                            found = _consider(file_path)
                            if found:
                                yield found
                        _report()
            elif on_missing is not None:
                on_missing(path_obj)
        except (PermissionError, OSError):
            continue

    if on_progress:
        current_dir = "Complete"
        _report(force=True)


def collect_font_files_with_progress(
    paths: Iterable[str | Path],
    recursive: bool = False,
    *,
    allowed_extensions: Optional[Set[str]] = None,
    include_uppercase: bool = True,
    include_hidden: bool = False,
    on_progress: Optional[Callable[[ProgressDict], None]] = None,
    on_missing: Optional[Callable[[Path], None]] = None,
    on_unsupported: Optional[Callable[[Path], None]] = None,
) -> List[str]:
    """Collect font file paths with progress callbacks. Sorted absolute paths; same rules as iter_font_files."""
    return sorted(
        set(
            iter_font_files(
                paths=paths,
                recursive=recursive,
                allowed_extensions=allowed_extensions,
                include_uppercase=include_uppercase,
                include_hidden=include_hidden,
                on_progress=on_progress,
                on_missing=on_missing,
                on_unsupported=on_unsupported,
            )
        )
    )


def collect_font_files(
    paths: Iterable[str | Path],
    recursive: bool = False,
    *,
    allowed_extensions: Optional[Set[str]] = None,
    include_uppercase: bool = True,
    include_hidden: bool = False,
    on_missing: Optional[Callable[[Path], None]] = None,
    on_unsupported: Optional[Callable[[Path], None]] = None,
) -> List[str]:
    """Collect font file paths from files and/or directories. Sorted absolute paths; same rules as iter_font_files."""
    return collect_font_files_with_progress(
        paths,
        recursive,
        allowed_extensions=allowed_extensions,
        include_uppercase=include_uppercase,
        include_hidden=include_hidden,
        on_missing=on_missing,
        on_unsupported=on_unsupported,
    )


def collect_font_files_with_rich_progress(
    paths: Iterable[str | Path],
    recursive: bool = False,
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
    "BINARY_EXTENSIONS",
    "RECOGNISED_UNSUPPORTED_EXTENSIONS",
    "SUPPORTED_EXTENSIONS",
    "collect_font_files",
    "iter_font_files",
    "collect_font_files_with_progress",
    "collect_font_files_with_rich_progress",
]
