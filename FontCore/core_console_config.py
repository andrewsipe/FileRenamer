#!/usr/bin/env python3
"""
Console configuration and theme definitions.

Separated from core_console_styles for easier maintenance and modification.
Contains all configuration settings, Rich theme, and status label definitions.
"""

from __future__ import annotations

import importlib.util as importlib_util
from typing import Optional

from FontCore.core_logging_config import get_logger

logger = get_logger(__name__)

# ============================================================================
# CONFIGURATION & AVAILABILITY
# ============================================================================
# Core configuration settings for console behavior and styling
CONSOLE_CONFIG = {
    "label_width": 11,  # width for labels (keeps alignment)
    "indent_size": 12,  # base indent spaces
    "pre_label_width": 5,  # width for pre-labels (smaller than regular labels)
    "pre_label_indent": 7,  # dedicated indent for non-status indenting
    "use_rich": True,  # set False to force non-rich fallback
    "use_questionary": False,  # reserved; questionary is not a FontCore dependency
    "theme_mode": "dark",  # "dark" or "light"
}

RICH_AVAILABLE: bool = (
    importlib_util.find_spec("rich") is not None
    if CONSOLE_CONFIG["use_rich"]
    else False
)

# ============================================================================
# RICH IMPORTS
# ============================================================================
# Conditional Rich imports with type aliases.
# Rich's __init__.py calls os.getcwd(); on macOS that can raise PermissionError
# when cwd is in a restricted location (e.g. Downloads). Catch and fall back.
if RICH_AVAILABLE:
    try:
        from rich.console import Console as _Console
        from rich.theme import Theme
        from rich.panel import Panel
        from rich.table import Table as _Table
        from rich import box
        from rich.align import Align
        from rich.progress import (
            Progress as _Progress,
            SpinnerColumn,
            TextColumn,
            BarColumn,
            TaskProgressColumn,
            TimeElapsedColumn,
        )
    except (PermissionError, OSError) as e:
        logger.warning(
            "Rich import failed (%s); using plain text output. "
            "Run from a permitted directory (e.g. cd ~) to enable Rich.",
            e,
        )
        RICH_AVAILABLE = False
        _Console = None
        Theme = None
        Panel = None
        _Table = None
        box = None
        Align = None
        _Progress = None
        SpinnerColumn = None
        TextColumn = None
        BarColumn = None
        TaskProgressColumn = None
        TimeElapsedColumn = None
if not RICH_AVAILABLE:
    # Type stubs for when Rich is not available
    _Console = None
    Theme = None
    Panel = None
    _Table = None
    box = None
    Align = None
    _Progress = None
    SpinnerColumn = None
    TextColumn = None
    BarColumn = None
    TaskProgressColumn = None
    TimeElapsedColumn = None

# ============================================================================
# THEME COLORS
# ============================================================================
# Base color definitions for Rich theme
# These are the actual color values that Rich will use

THEME_COLORS = {
    # Base text colors
    "darktext": "#282A39",
    "lighttext": "grey100",
    # Status background colors (referenced by STATUS_DEFINITIONS)
    "info": "dodger_blue1",
    "updated": "magenta2",
    "unchanged": "turquoise2",
    "error": "red1",
    "warning": "gold1",
    "saved": "#29A329",
    "created": "cornflower_blue",
    "removed": "medium_violet_red",
    "input": "cornsilk1",
    "parsing": "grey37",
    "success": "green_yellow",
    "preview": "gold3",
    "skipped": "orange1",
    "duplicate": "yellow3",
    "cache": "purple4",
    "discovered": "magenta",
    "mapping": "cyan3",
    "minimal": "cyan2",
    "moderate": "cyan3",
    "major": "turquoise4",
    "pushed": "spring_green2",  # Git push operations
    "committed": "chartreuse2",  # Git commit operations
    "ok": "turquoise2",  # Status checks that pass (similar to unchanged but for verification)
    # Additional theme colors (not status-specific)
    "info.bright": "deep_sky_blue1",
    "header": "deep_sky_blue1",
    # Content styling
    "value.before": "turquoise2",
    "value.after": "magenta2",
    "value.unchanged": "dim turquoise2",
    "file.name": "#29A329",
    "file.path": "grey37",
    "count": "bold turquoise2",
    "field": "honeydew2",
    "field.number": "bold honeydew2",
    # Progress bar styling
    "bar.complete": "magenta3",
    "bar.finished": "magenta2",
    "bar.pulse": "medium_violet_red",
    "progress.description": "dodger_blue1",
    "progress.percentage": "bold turquoise2",
    "progress.elapsed": "dodger_blue3",
    "progress.remaining": "dodger_blue3",
    # Rich's internal fallback color names
    "repr.number": "bold turquoise2",
    "repr.str": "grey100",
    "repr.bool_true": "italic spring_green3",
    "repr.bool_false": "italic deep_pink2",
    "repr.path": "grey37",
    "repr.filename": "green",
    "repr.file": "none",
    "repr.call": "none",
    "repr.tag_name": "hot_pink",
}

# Inline Rich markup for logger strings (pair with status labels: ERROR=red1, WARNING=gold1).
RICH_INLINE_ERROR = "orange_red1"
RICH_INLINE_WARNING = "light_goldenrod2"

# ============================================================================
# RICH THEME BUILDING
# ============================================================================
# Build Rich Theme object from color definitions
# Note: CUSTOM_THEME must remain a Rich Theme object for Rich compatibility

if RICH_AVAILABLE:
    try:
        CUSTOM_THEME = Theme(THEME_COLORS)
    except Exception as e:
        logger.error(f"Failed to initialize custom theme: {e}")
        # Fallback to default theme
        CUSTOM_THEME = Theme({})

    # Module-level console singleton
    _console_singleton: Optional[_Console] = None
else:
    CUSTOM_THEME = None
    _console_singleton = None

# ============================================================================
# STATUS DEFINITIONS
# ============================================================================
# Unified configuration for all status types
# Single source of truth combining label config and StatusIndicator behavior
# Note: Uses theme KEYS (like "info", "lighttext") not actual color values
# The _build_status_label function will look up actual colors from CUSTOM_THEME
#
# Field Descriptions:
#   - label_text: Text displayed in the status label (e.g., " INFO", " UPDATED")
#   - foreground: Theme key for text color (e.g., "lighttext", "darktext")
#   - background: Theme key for background color (e.g., "info", "error", "updated")
#   - template: Format string for StatusIndicator message structure
#               "{context}" = file/field name, "{details}" = explanation/error message
#   - value_style: How values are styled when displayed
#                  "plain" = no special styling
#                  "before" = styled as old value (for change displays)
#                  "after" = styled as new value (for change displays)
#                  "unchanged" = styled as unchanged value
#   - show_change: Whether this status type typically shows old→new value changes
#   - use_case: Description of when/where this status type is used in scripts

STATUS_DEFINITIONS = {
    "info": {
        "label_text": " INFO",
        "foreground": "lighttext",  # Theme key, not actual color
        "background": "info",  # Theme key, not actual color
        "template": "{context}{details}",
        "value_style": "plain",
        "show_change": False,
        "use_case": "General informational messages, processing status updates, operational messages. Used for non-critical notifications like 'Processing files', 'Found X font families', 'Running NameID replacer scripts'.",
    },
    "updated": {
        "label_text": " UPDATED",
        "foreground": "darktext",
        "background": "updated",
        "template": "{context}",
        "value_style": "after",
        "show_change": True,
        "use_case": "When a value, field, or file has been modified. Typically shows old→new value changes. Used for NameID updates, field modifications, file changes. Most common status for actual modifications.",
    },
    "unchanged": {
        "label_text": " NO CHANGE",
        "foreground": "darktext",
        "background": "unchanged",
        "template": "{context}",
        "value_style": "unchanged",
        "show_change": False,
        "use_case": "When a value, field, or file remains the same (no modification needed). Used to indicate files/records that were checked but didn't require changes. Often used with add_values(value='CurrentValue') to show what the current value is.",
    },
    "deleted": {
        "label_text": " REMOVED",
        "foreground": "darktext",
        "background": "removed",
        "template": "{context}",
        "value_style": "plain",
        "show_change": False,
        "use_case": "When a record, field, or nameID entry has been deleted/removed. Used in NameID_Deleter and similar deletion operations. Exported as REMOVED_LABEL for backward compatibility.",
    },
    "parsing": {
        "label_text": " PARSING",
        "foreground": "lighttext",
        "background": "parsing",
        "template": "{context}",
        "value_style": "plain",
        "show_change": False,
        "use_case": "File processing status, shown when actively parsing/processing a font file. Typically used with add_file() or add_message('Processing: filename'). Used in opentype tools, NameID scripts, and file processors to show progress.",
    },
    "saved": {
        "label_text": " SAVED TO",
        "foreground": "lighttext",
        "background": "saved",
        "template": "{context}",
        "value_style": "plain",
        "show_change": False,
        "use_case": "When a file has been successfully saved to disk. Used after write operations to confirm file persistence. Suppressed in dry-run mode. Often used with add_file() to show which file was saved.",
    },
    "success": {
        "label_text": " SUCCESS",
        "foreground": "darktext",
        "background": "success",
        "template": "{context}{details}",
        "value_style": "plain",
        "show_change": False,
        "use_case": "Successful completion of operations, validation success, processing complete. Used for final success messages, often with with_summary_block() to show statistics. Also used for individual successful operations like 'Export successful'.",
    },
    "warning": {
        "label_text": " WARNING",
        "foreground": "darktext",
        "background": "warning",
        "template": "{context}{details}",
        "value_style": "plain",
        "show_change": False,
        "use_case": "Non-critical warnings, configuration issues, missing records, potential problems. Used extensively in FontFiles_RenamerEnhanced for naming conflicts, validation warnings. Often used with with_explanation() to provide warning details. Not dimmed in dry-run mode.",
    },
    "error": {
        "label_text": " ERROR",
        "foreground": "lighttext",
        "background": "error",
        "template": "{context}: {details}",
        "value_style": "plain",
        "show_change": False,
        "use_case": "Error conditions, failures, exceptions, permission denied, file read/write failures. Used throughout all scripts for error handling. Often used with add_file() and with_explanation() to show which file failed and why. Not dimmed in dry-run mode.",
    },
    "skipped": {
        "label_text": " SKIPPED",
        "foreground": "darktext",
        "background": "skipped",
        "template": "{context}",
        "value_style": "plain",
        "show_change": False,
        "use_case": "Files or operations that were skipped during processing. Used with with_explanation() to provide skip reason (e.g., 'Invalid filename format', 'Need exactly one match', 'Already processed'). Used in UPM_Rescaler, FNT_ReorderFind-n-Replace, FontFiles_Cleaner.",
    },
    "duplicate": {
        "label_text": " DUPLICATE",
        "foreground": "darktext",
        "background": "duplicate",
        "template": "{context}",
        "value_style": "plain",
        "show_change": False,
        "use_case": "Duplicate entries detected during processing. Used when duplicate records, names, or entries are found. Less commonly used but available for duplicate detection scenarios.",
    },
    "cache": {
        "label_text": " CACHE",
        "foreground": "lighttext",
        "background": "cache",
        "template": "{context}",
        "value_style": "plain",
        "show_change": False,
        "use_case": "Cache-related operations, cache hits/misses, cache updates. Available for caching scenarios but less commonly used in current scripts.",
    },
    "discovered": {
        "label_text": " FOUND",
        "foreground": "darktext",
        "background": "discovered",
        "template": "{context}",
        "value_style": "plain",
        "show_change": False,
        "use_case": "Files discovered/found during scanning or extraction. Used in core_logging_config for font file discovery, AdobeFontsExtractor for extracted fonts. Typically used with add_file() and add_message() to show format and source.",
    },
    "mapping": {
        "label_text": " NAMING",
        "foreground": "darktext",
        "background": "mapping",
        "template": "{context}",
        "value_style": "plain",
        "show_change": False,
        "use_case": "Explains how the output filename was chosen (URL stem, CSS font-family, site handler, etc.). Shown at verbose verbosity; uses 'label: detail', not arrows, to avoid looking like a rename.",
    },
    "minimal": {
        "label_text": " MINIMAL",
        "foreground": "darktext",
        "background": "minimal",
        "template": "{context}",
        "value_style": "plain",
        "show_change": False,
        "use_case": "Severity level indicator for minimal changes. Part of the severity scale (minimal → moderate → major). Available for categorizing change severity levels.",
    },
    "moderate": {
        "label_text": " MODERATE",
        "foreground": "darktext",
        "background": "moderate",
        "template": "{context}",
        "value_style": "plain",
        "show_change": False,
        "use_case": "Severity level indicator for moderate changes. Part of the severity scale (minimal → moderate → major). Available for categorizing change severity levels.",
    },
    "major": {
        "label_text": " MAJOR",
        "foreground": "lighttext",
        "background": "major",
        "template": "{context}",
        "value_style": "plain",
        "show_change": False,
        "use_case": "Severity level indicator for major changes. Part of the severity scale (minimal → moderate → major). Available for categorizing change severity levels.",
    },
    "preview": {
        "label_text": " PREVIEW",
        "foreground": "darktext",
        "background": "preview",
        "template": "{context}",
        "value_style": "plain",
        "show_change": False,
        "use_case": "Dry-run mode messages, preview of what would happen without making changes. Used extensively in dry-run scenarios to show 'Would perform:', 'DRY RUN MODE: No changes will be made', and preview operations. Used in opentype_wrapper, UPM_Rescaler, FontFixer, core_nameid_replacer_base.",
    },
    "created": {
        "label_text": " CREATED",
        "foreground": "lighttext",
        "background": "created",
        "template": "{context}",
        "value_style": "after",
        "show_change": False,
        "use_case": "When a new record, field, or nameID entry has been created. Used in NameID operations when creating new nameID records. Typically used with add_field() and add_values(value='NewValue') to show what was created.",
    },
    "input": {
        "label_text": " INPUT",
        "foreground": "darktext",
        "background": "input",
        "template": "{context}",
        "value_style": "plain",
        "show_change": False,
        "use_case": "User input prompts, interactive input requests. Available for user interaction scenarios but less commonly used in current scripts.",
    },
    "pushed": {
        "label_text": " PUSHED",
        "foreground": "darktext",
        "background": "pushed",
        "template": "{context}",
        "value_style": "plain",
        "show_change": False,
        "use_case": "Git push operations completed successfully. Used when code has been pushed to remote repository. More specific than 'success' - indicates the push operation itself completed.",
    },
    "committed": {
        "label_text": " COMMITTED",
        "foreground": "darktext",
        "background": "committed",
        "template": "{context}",
        "value_style": "plain",
        "show_change": False,
        "use_case": "Git commit operations completed successfully. Used when changes have been committed to git. More specific than 'success' - indicates the commit operation itself completed.",
    },
    "ok": {
        "label_text": " OK",
        "foreground": "darktext",
        "background": "ok",
        "template": "{context}",
        "value_style": "plain",
        "show_change": False,
        "use_case": "Status checks that pass, verification successful, things are in correct state. Used for 'FontCore is up to date', 'No changes needed', 'Already correct' type messages. Less emphatic than 'success' - just indicates things are as expected.",
    },
}

# ============================================================================
# PRE-LABEL DEFINITIONS
# ============================================================================
# Pre-labels appear before status labels (e.g., [DRY] [UPDATED])
# Smaller width (4 spaces) than regular status labels (11 spaces)
# Used for mode indicators like dry-run, preview, test, etc.

PRE_LABEL_DEFINITIONS = {
    "dry": {
        "label_text": " DRY",  # No leading space, 4 chars total
        "foreground": "darktext",
        "background": "warning",  # Uses warning yellow (gold1)
        "use_case": "Dry-run mode indicator. Appears before all status labels when dry_run=True to indicate preview mode.",
    },
}

# ============================================================================
# STATUS LABEL BUILDER
# ============================================================================
# Workaround function: Rich doesn't allow custom theme names in components
# This function looks up theme keys to get actual colors, then builds Rich markup
# This preserves the connection between status names and their colors


def _build_status_label(
    text: str, foreground_theme_key: str, background_theme_key: str = "lighttext"
) -> str:
    """
    Build a formatted status label using theme colors.

    This is a workaround for Rich's limitation: we can't use custom theme names
    directly in Rich components. Instead, we:
    1. Take theme keys (like "info", "lighttext")
    2. Look up actual colors from CUSTOM_THEME.styles.get()
    3. Build Rich markup with actual color values

    Args:
        text: The label text to display
        foreground_theme_key: Theme key for foreground color (e.g., "lighttext")
        background_theme_key: Theme key for background color (e.g., "info")
    """
    width = CONSOLE_CONFIG.get("label_width", 11)
    if RICH_AVAILABLE and CUSTOM_THEME:
        # Look up actual colors from theme (the "cheat" that makes this work)
        foreground_color = CUSTOM_THEME.styles.get(
            foreground_theme_key, "yellow1"
        )  # yellow1 is fallback foreground
        background_color = CUSTOM_THEME.styles.get(
            background_theme_key, "red3"
        )  # red3 is fallback background
        return f"[bold {foreground_color} on {background_color}]{text:<{width}}[/bold {foreground_color} on {background_color}]"
    return f"{text:<{width}}"


def _build_pre_label(
    text: str, foreground_theme_key: str, background_theme_key: str = "lighttext"
) -> str:
    """
    Build a formatted pre-label using theme colors.

    Similar to _build_status_label() but uses pre_label_width (4 spaces)
    instead of label_width (11 spaces). Pre-labels are smaller and appear
    before status labels (e.g., [DRY] [UPDATED]).

    This uses the same workaround as status labels: we can't use custom theme
    names directly in Rich components, so we look up actual colors from CUSTOM_THEME.

    Args:
        text: The pre-label text to display (e.g., "DRY")
        foreground_theme_key: Theme key for foreground color (e.g., "darktext")
        background_theme_key: Theme key for background color (e.g., "warning")
    """
    width = CONSOLE_CONFIG.get("pre_label_width", 5)
    if RICH_AVAILABLE and CUSTOM_THEME:
        # Look up actual colors from theme (the "cheat" that makes this work)
        foreground_color = CUSTOM_THEME.styles.get(
            foreground_theme_key, "yellow1"
        )  # yellow1 is fallback foreground
        background_color = CUSTOM_THEME.styles.get(
            background_theme_key, "red3"
        )  # red3 is fallback background
        return f"[bold {foreground_color} on {background_color}]{text:<{width}}[/bold {foreground_color} on {background_color}]"
    return f"{text:<{width}}"


# ============================================================================
# VALIDATION
# ============================================================================
# Validate that all theme keys referenced in STATUS_DEFINITIONS exist in THEME_COLORS


def _validate_status_definitions() -> None:
    """Validate that all theme color references in STATUS_DEFINITIONS exist."""
    for status, defn in STATUS_DEFINITIONS.items():
        for key_name in ["foreground", "background"]:
            theme_key = defn[key_name]
            if theme_key not in THEME_COLORS:
                logger.warning(
                    f"Status '{status}' references unknown theme color '{theme_key}' "
                    f"for {key_name}. Available colors: {list(THEME_COLORS.keys())[:10]}..."
                )


def _validate_pre_label_definitions() -> None:
    """Validate that all theme color references in PRE_LABEL_DEFINITIONS exist."""
    for pre_label, defn in PRE_LABEL_DEFINITIONS.items():
        for key_name in ["foreground", "background"]:
            theme_key = defn[key_name]
            if theme_key not in THEME_COLORS:
                logger.warning(
                    f"Pre-label '{pre_label}' references unknown theme color '{theme_key}' "
                    f"for {key_name}. Available colors: {list(THEME_COLORS.keys())[:10]}..."
                )


# Run validation if Rich is available
if RICH_AVAILABLE:
    _validate_status_definitions()
    _validate_pre_label_definitions()


# ============================================================================
# STATUS LABEL CONSTANTS
# ============================================================================
# Build labels from STATUS_DEFINITIONS using theme key lookup
# These are exported as individual constants for backward compatibility
# Special case: "deleted" status maps to "REMOVED_LABEL" constant name

_LABEL_NAME_MAP = {
    "deleted": "REMOVED",  # Internal name "deleted" → export name "REMOVED_LABEL"
}

# Generate all label constants programmatically
for status_key, defn in STATUS_DEFINITIONS.items():
    # Determine the constant name (e.g., "info" → "INFO_LABEL", "deleted" → "REMOVED_LABEL")
    label_name = _LABEL_NAME_MAP.get(status_key, status_key.upper())
    constant_name = f"{label_name}_LABEL"

    # Build and assign the label to module globals
    globals()[constant_name] = _build_status_label(
        defn["label_text"],
        defn["foreground"],
        defn["background"],
    )

# ============================================================================
# PRE-LABEL CONSTANTS
# ============================================================================
# Build pre-labels from PRE_LABEL_DEFINITIONS using theme key lookup
# Pre-labels are smaller (4 spaces) and appear before status labels

# Build DRY_LABEL constant
DRY_LABEL: str = _build_pre_label(
    PRE_LABEL_DEFINITIONS["dry"]["label_text"],
    PRE_LABEL_DEFINITIONS["dry"]["foreground"],
    PRE_LABEL_DEFINITIONS["dry"]["background"],
)

# Build PRE_LABEL_THEMES dictionary (for extensibility)
PRE_LABEL_THEMES = {
    pre_label: _build_pre_label(
        defn["label_text"],
        defn["foreground"],
        defn["background"],
    )
    for pre_label, defn in PRE_LABEL_DEFINITIONS.items()
}

# Indentation constants
INDENT: str = " " * CONSOLE_CONFIG.get("indent_size", 12)
PRE_LABEL_INDENT: str = " " * CONSOLE_CONFIG.get("pre_label_indent", 6)

# ============================================================================
# STATUS INDICATOR THEMES
# ============================================================================
# Build STATUS_THEMES from STATUS_DEFINITIONS + built labels
# This maintains the structure StatusIndicator expects


def _build_status_theme(defn: dict) -> dict:
    """
    Extract StatusIndicator-relevant fields from a status definition.

    Args:
        defn: Status definition dictionary from STATUS_DEFINITIONS

    Returns:
        Dictionary with label, template, value_style, and show_change
    """
    return {
        "label": _build_status_label(
            defn["label_text"],
            defn["foreground"],
            defn["background"],
        ),
        "template": defn["template"],
        "value_style": defn["value_style"],
        "show_change": defn["show_change"],
    }


STATUS_THEMES = {
    status: _build_status_theme(defn) for status, defn in STATUS_DEFINITIONS.items()
}
