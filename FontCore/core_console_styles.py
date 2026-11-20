#!/usr/bin/env python3
"""
Shared console styling utilities for consistent output across scripts.

Primary API: StatusIndicator class for unified console output formatting.
Secondary APIs: Core formatting primitives and high-level helpers.

Usage:
from FontCore.core_console_styles import (
    RICH_AVAILABLE, UPDATED_LABEL, UNCHANGED_LABEL, ERROR_LABEL, WARNING_LABEL, INFO_LABEL,
    SAVED_LABEL, CREATED_LABEL, INPUT_LABEL, PARSING_LABEL, SUCCESS_LABEL,
    SKIPPED_LABEL, DUPLICATE_LABEL, CACHE_LABEL, DISCOVERED_LABEL, MAPPING_LABEL,
    INDENT, indent,
    fmt_change, fmt_field, fmt_file, fmt_file_compact, fmt_value, fmt_count,
    fmt_smart_underline, fmt_header,
    fmt_preflight_checklist, fmt_processing_summary, fmt_operation_description,
    emit, get_console, create_table, print_panel, print_session_header, status_message,
    create_progress_bar,
    prompt_input, prompt_confirm, prompt_text, prompt_select, QuitRequested,
    StatusIndicator,
)

These helpers auto-detect Rich. When Rich is available, they output styled markup;
otherwise, they fall back to plain text with no markup.

Demo and Testing:
    Run 'python CoreDemoTool.py console' to see a comprehensive showcase of all
    available labels, formatting helpers, and styling capabilities.
"""

from __future__ import annotations

from typing import Optional
from pathlib import Path
import re

# Enhanced functionality import
from FontCore.core_logging_config import get_logger

# Import configuration, theme, and labels from config module
import FontCore.core_console_config as _config
from FontCore.core_console_config import (
    CONSOLE_CONFIG,
    RICH_AVAILABLE,
    CUSTOM_THEME,
    _Console,
    Panel,
    _Table,
    box,
    Align,
    _Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    # Labels are re-exported for backward compatibility (scripts import from core_console_styles)
    INFO_LABEL,  # noqa: F401
    UPDATED_LABEL,  # noqa: F401
    UNCHANGED_LABEL,  # noqa: F401
    ERROR_LABEL,  # noqa: F401
    WARNING_LABEL,  # noqa: F401
    SAVED_LABEL,  # noqa: F401
    CREATED_LABEL,  # noqa: F401
    REMOVED_LABEL,  # noqa: F401
    INPUT_LABEL,  # noqa: F401
    PREVIEW_LABEL,  # noqa: F401
    PARSING_LABEL,  # noqa: F401
    SUCCESS_LABEL,  # noqa: F401
    SKIPPED_LABEL,  # noqa: F401
    DUPLICATE_LABEL,  # noqa: F401
    CACHE_LABEL,  # noqa: F401
    DISCOVERED_LABEL,  # noqa: F401
    MAPPING_LABEL,  # noqa: F401
    MINIMAL_LABEL,  # noqa: F401
    MODERATE_LABEL,  # noqa: F401
    MAJOR_LABEL,  # noqa: F401
    INDENT,
    STATUS_THEMES,
    DRY_LABEL,
    PRE_LABEL_INDENT,
)

logger = get_logger(__name__)

# ============================================================================
# CONSOLE INFRASTRUCTURE
# ============================================================================
# Core console utilities used throughout the module


def get_console() -> Optional["_Console"]:
    """
    Get the Rich console instance if available, otherwise None.
    """
    if RICH_AVAILABLE:
        if _config._console_singleton is None:
            try:
                _config._console_singleton = _Console(theme=CUSTOM_THEME)
            except Exception as e:
                logger.warning(f"Failed to initialize Rich console: {e}")
                return None
        return _config._console_singleton
    else:
        logger.debug("Rich not available, using plain text output")
    return None


def emit(
    message: str, highlight=None, console: Optional["_Console"] = None, end: str = "\n"
) -> None:
    """
    Emit a message via Rich console if available, otherwise print().
    """
    if RICH_AVAILABLE:
        (console or get_console()).print(
            message, end=end, overflow="fold", no_wrap=False
        )
    else:
        clean_message = re.sub(r"\[/?[^\]]+\]", "", message)
        print(clean_message, end=end)


def emit_spacer(lines: int = 1, console: Optional["_Console"] = None) -> None:
    """
    Emit one or more blank lines for visual separation.

    Args:
        lines: Number of blank lines to emit (default: 1)
        console: Optional console instance
    """
    for _ in range(lines):
        emit("", console=console)


def emit_section_break(console: Optional["_Console"] = None) -> None:
    """
    Emit a visual section break (2 blank lines).

    Args:
        console: Optional console instance
    """
    emit_spacer(2, console=console)


def emit_section_separator(
    char: str = "─", length: int = 60, console: Optional["_Console"] = None
) -> None:
    """
    Emit a visual separator line.

    Args:
        char: Character to use for separator (default: "─")
        length: Length of separator line (default: 60)
        console: Optional console instance
    """
    separator = char * length
    emit(separator, console=console)


def indent(level: int = 1, additional: int = 0) -> str:
    """
    Generate indentation for hierarchical output with natural wrapping support.
    """
    if level <= 0:
        return ""

    base = CONSOLE_CONFIG.get("indent_size", 12)  # default 12 spaces
    level_spacing = (level - 1) * 2  # Each level adds 2 spaces
    total_spaces = base + level_spacing + additional

    return " " * total_spaces


def get_pre_label_indent() -> str:
    """
    Get the dedicated indent for non-status indenting scenarios.

    Uses pre_label_indent (6 spaces) from CONSOLE_CONFIG, which is separate
    from the regular indent_size (12 spaces) used for status messages.

    Returns:
        Pre-label indent string (6 spaces by default)
    """
    return PRE_LABEL_INDENT


# Unicode symbol constants
ARROW = "→"
BULLET = "•"
EM_DASH = "—"


def bullet(text: str, level: int = 1) -> str:
    """
    Render a simple bullet line at the given indentation level.
    """
    return f"{indent(level)}{BULLET} {text}"


def bulleted_kv(key: str, value: str | int, level: int = 1) -> str:
    """
    Render an indented bullet with a key: value pair.
    """
    return f"{indent(level)}{BULLET} {key}: {fmt_count(value) if isinstance(value, int) else value}"


def fmt_bullet_list(items: list[str], indent_level: int = 1) -> str:
    """
    Format a list of items as indented bullets.

    Args:
        items: List of strings to format as bullets
        indent_level: Indentation level (1=12 spaces, 2=14 spaces, etc.)

    Returns:
        Formatted string with each item on a new line with bullet

    Example:
        >>> fmt_bullet_list(["Item 1", "Item 2", "Item 3"])
        "            • Item 1\n            • Item 2\n            • Item 3"
    """
    return "\n".join(f"{indent(indent_level)}{BULLET} {item}" for item in items)


# ============================================================================
# CORE FORMATTING PRIMITIVES
# ============================================================================
# Basic formatting functions used by StatusIndicator and other components


def fmt_change(old_value: str, new_value: str) -> str:
    """
    Format a change as old → new with visual emphasis on what changed.

    Example:
        >>> fmt_change("Arial-Bold", "Helvetica-Bold")
        "Arial-Bold → Helvetica-Bold"  # (with color: before=turquoise, after=magenta)
    """
    if RICH_AVAILABLE:
        return f"[value.before]{old_value}[/value.before] {ARROW} [value.after]{new_value}[/value.after]"
    return f"{old_value} -> {new_value}"


def fmt_field(field_name: str, value: str | int) -> str:
    """
    Format a field as name: value with automatic number styling.

    Example:
        >>> fmt_field("nameID", 1)
        "nameID: 1"  # (with field styling when Rich available)
    """
    if isinstance(value, int) and RICH_AVAILABLE:
        return f"[field]{field_name}[/field]: [field.number]{value}[/field.number]"
    return f"[field]{field_name}[/field]: {value}"


def fmt_value(value: str | int, style: str = "plain") -> str:
    """
    Format a value with different styling options.

    Args:
        value: The value to format
        style: "plain", "before", "after", "unchanged"

    Example:
        >>> fmt_value("MyFont-Bold")
        "MyFont-Bold"  # (plain text)
        >>> fmt_value("MyFont-Bold", "before")
        "MyFont-Bold"  # (with turquoise styling)
        >>> fmt_value("MyFont-Bold", "after")
        "MyFont-Bold"  # (with magenta styling)
        >>> fmt_value("MyFont-Bold", "unchanged")
        "MyFont-Bold"  # (with dim turquoise styling)
    """
    if not RICH_AVAILABLE:
        return str(value)

    if style == "before":
        return f"[value.before]{value}[/value.before]"
    elif style == "after":
        return f"[value.after]{value}[/value.after]"
    elif style == "unchanged":
        return f"[value.unchanged]{value}[/value.unchanged]"
    else:  # plain
        return str(value)


def fmt_count(value: int | str) -> str:
    """
    Format a count or aggregate number with emphasis.

    Accepts both integers and strings (e.g., "90%" for percentages).

    Example:
        >>> fmt_count(42)
        "42"  # (with bold turquoise styling when Rich available)
        >>> fmt_count("90%")
        "90%"  # (with bold turquoise styling when Rich available)
    """
    return f"[count]{value}[/count]" if RICH_AVAILABLE else str(value)


def _escape_markup(text: str) -> str:
    """Keep brackets in a filename from being read as a Rich style tag."""
    if not RICH_AVAILABLE:
        return text
    from rich.markup import escape

    return escape(text)


def fmt_file(path: str, filename_only: bool = True) -> str:
    """
    Format a file path with consistent styling.

    Example:
        >>> fmt_file("/path/to/MyFont-Bold.otf")
        "MyFont-Bold.otf"  # (with green filename styling when Rich available)
    """
    if not RICH_AVAILABLE:
        return Path(path).name if filename_only else path

    if filename_only:
        return f"[file.name]{_escape_markup(Path(path).name)}[/file.name]"

    path_obj = Path(path)
    parent = str(path_obj.parent) + "/" if path_obj.parent != Path(".") else ""
    return (
        f"[file.path]{_escape_markup(parent)}[/file.path]"
        f"[file.name]{_escape_markup(path_obj.name)}[/file.name]"
    )


def fmt_file_compact(path: str) -> str:
    """
    Format a file path with dimmed directory and emphasized filename for easier scanning.

    Example:
        >>> fmt_file_compact("/path/to/MyFont-Bold.otf")
        "/path/to/MyFont-Bold.otf"  # (with dimmed path + green filename)
    """
    return fmt_file(path, filename_only=False)


def fmt_kv_pair(key: str, value: str | int) -> str:
    """
    Format a single key-value pair consistently.

    Args:
        key: The key name
        value: The value (string or integer)

    Returns:
        Formatted key-value pair string

    Example:
        >>> fmt_kv_pair("nameID", 1)
        "nameID: 1"
        >>> fmt_kv_pair("Version", "1.000")
        "Version: 1.000"
    """
    return fmt_field(key, value)


def fmt_smart_underline(text: str) -> str:
    """
    Apply intelligent underlining that skips lowercase descenders for better typography.

    Example:
        >>> fmt_smart_underline("Typography is groovy")
        "Typography is groovy"  # (with smart underlines, skips descenders)
    """
    if not RICH_AVAILABLE:
        return text

    lowercase_descenders = set("gjpqyQ")

    result = []

    for char in text:
        if char in lowercase_descenders:
            result.append(char)
        else:
            result.append(f"[underline]{char}[/underline]")

    return "".join(result)


def fmt_header(text: str, console: Optional["_Console"] = None) -> None:
    """
    Create a centered header with panel styling.

    Example:
        >>> fmt_header("PROCESSING SUMMARY")
        # Creates a centered panel with the text
    """
    if RICH_AVAILABLE:
        console = console or get_console()
        centered_text = Align.center(text)
        panel = Panel(
            centered_text,
            box=box.HORIZONTALS,
            border_style="dodger_blue1",
            style="bold grey100",
            padding=0,
            expand=True,
        )

        console.print(panel)
    else:
        print(f"=== {text} ===")


# ============================================================================
# MAIN API - STATUS INDICATOR CLASS
# ============================================================================
# Primary interface for all console output formatting


class StatusIndicator:
    """
    Universal status indicator for consistent message formatting.

    Builds messages in layers:
    - Level 1: Base label (UPDATED, ERROR, etc.)
    - Level 2: Context (file, field, etc.)
    - Level 3: Values/changes based on status type
    - Level 4: Additional details with indentation

    Features:
    - Chainable builder pattern for composing complex messages
    - Automatic indentation for hierarchical output
    - Preview/dry-run mode conversion
    - Processing summary formatting
    - Support for 18 status types (updated, created, error, warning, etc.)
    - Style overrides for all builder methods (e.g., style="bold hot_pink")

    Usage:
        # Simple message
        StatusIndicator("info").add_message("Processing files").emit()

        # File operation with details
        StatusIndicator("updated")
            .add_field("nameID", 1)
            .add_file("font.otf")
            .add_values(old_value="Old", new_value="New")
            .add_item("Additional context")
            .emit()

        # File with custom styling
        StatusIndicator("saved")
            .add_file("font.otf", style="reverse")
            .emit()

        # Style overrides for emphasis
        StatusIndicator("unchanged")
            .add_field("nameID", 1, style="bold turquoise2")
            .add_file("font.otf")
            .add_values(value="CurrentValue", style="bold")
            .emit()

        # Preview mode
        StatusIndicator("updated", dry_run=True)
            .add_file("font.otf")
            .emit()

        # Processing summary
        StatusIndicator("success")
            .add_message("Processing Completed!")
            .with_summary_block(updated=10, unchanged=2, errors=0)
            .emit()

        # Dry-Run Mode:
            Pass dry_run=True to enable preview mode:
            - Adds DRY prefix label before status label (warning yellow color)
            - Dims operational labels (updated, created, deleted, parsing)
            - Suppresses 'saved' messages entirely
            - Info/warning/error labels remain normal (not dimmed, but still show DRY prefix)

            Usage:
                StatusIndicator("updated", dry_run=True)
                    .add_file("font.otf")
                    .emit()
                    # Output: [DRY] [ UPDATED   ] font.otf

        # Style Parameter:
            All builder methods accept an optional style parameter:
            - Use Rich style names: "bold", "hot_pink", "bold turquoise2"
            - Respects Rich limitations (can't combine custom theme names with modifiers)
            - Falls back gracefully when Rich is unavailable

            Examples:
                .add_message("Important!", style="bold hot_pink")
                .add_field("nameID", 1, style="bold")
                .add_values(value="NewValue", style="bold")
                .with_explanation("Error details", style="dim")
                .add_item("Note", style="bold red")
    """

    # --- STAGE 1: INITIALIZATION ---
    def __init__(self, status: str, dry_run: bool = False):
        if status not in STATUS_THEMES:
            available = ", ".join(sorted(STATUS_THEMES.keys()))
            raise ValueError(f"Unknown status: '{status}'. Available: {available}")

        self.status = status
        self.theme = STATUS_THEMES[status]
        self.dry_run = dry_run

        # Core message components
        self.context_parts = []
        self.explanation = None

        # Value display (for changes or single values)
        self.old_value = None
        self.new_value = None
        self.value = None
        self.value_style_override = None

        # Optional step log (for detailed operation tracking)
        self.step_log = None

    # --- STAGE 2: CORE CONTEXT BUILDERS (Builds Left-to-Right) ---
    def _apply_style(self, content: str, style: str = None) -> str:
        """Wrap content in Rich markup if style provided and Rich available."""
        if style and RICH_AVAILABLE:
            return f"[{style}]{content}[/{style}]"
        return content

    def add_message(self, message: str, style: str = None):
        """Add a simple message to the main context line.

        Args:
            message: The message text to add
            style: Optional Rich style to apply (e.g., "bold", "hot_pink", "bold turquoise2")
        """
        self.context_parts.append(self._apply_style(message, style))
        return self

    def add_file(self, filepath: str, filename_only: bool = True, style: str = None):
        """
        Add file context to the main message line.

        Args:
            filepath: Path to the file
            filename_only: If True, show only filename; if False, show full path
            style: Optional Rich style to apply to the file path (e.g., "reverse")
                   If None and status is "saved", automatically applies "reverse" style
        """
        # Automatically apply reverse style for "saved" status if no explicit style provided
        if style is None and self.status == "saved":
            style = "reverse #29A329"

        if style:
            # Apply custom style to the file path
            if filename_only:
                filename = _escape_markup(Path(filepath).name)
                if RICH_AVAILABLE:
                    self.context_parts.append(f"[{style}]{filename}[/{style}]")
                else:
                    self.context_parts.append(Path(filepath).name)
            else:
                path_obj = Path(filepath)
                parent = (
                    str(path_obj.parent) + "/" if path_obj.parent != Path(".") else ""
                )
                if RICH_AVAILABLE:
                    full_path = (
                        f"[file.path]{_escape_markup(parent)}[/file.path]"
                        f"[file.name]{_escape_markup(path_obj.name)}[/file.name]"
                    )
                    self.context_parts.append(f"[{style}]{full_path}[/{style}]")
                else:
                    self.context_parts.append(filepath)
        else:
            self.context_parts.append(fmt_file(filepath, filename_only))
        return self

    def add_field(self, field_name: str, value: int, style: str = None):
        """Add a structured field (e.g., 'nameID: 1') to the main message.

        Args:
            field_name: The field name (e.g., "nameID")
            value: The field value (typically an integer)
            style: Optional Rich style to apply (e.g., "bold", "hot_pink", "bold turquoise2")
        """
        field_text = fmt_field(field_name, value)
        self.context_parts.append(self._apply_style(field_text, style))
        return self

    # --- STAGE 3: VALUE SPECIFIERS ---
    def add_values(
        self,
        old_value: str = None,
        new_value: str = None,
        value: str = None,
        style: str = None,
    ):
        """Add value context (e.g., a change from old to new).

        Args:
            old_value: The original value (for change displays)
            new_value: The new value (for change displays)
            value: A single value (for unchanged/created displays)
            style: Optional Rich style to apply to the value(s) (e.g., "bold", "hot_pink")
        """
        if self.theme["show_change"] and old_value and new_value:
            self.old_value = old_value
            self.new_value = new_value
        elif value:
            self.value = value
        if style:
            self.value_style_override = style
        return self

    # --- STAGE 4: DETAIL APPENDERS (Builds Top-to-Bottom) ---
    def with_explanation(self, message: str | list[str], style: str = None):
        """
        Add a primary trailing message or reason.

        Often used with 'error' or 'info' statuses to provide context.
        Accepts both strings and lists, auto-formatting lists as bullet points.

        Args:
            message: The explanation text (string) or list of items (list[str])
            style: Optional Rich style to apply (e.g., "dim", "bold red")
        """
        if isinstance(message, list):
            # Format list as bullet points
            formatted = fmt_bullet_list(message, indent_level=1)
            self.explanation = self._apply_style(formatted, style)
        else:
            self.explanation = self._apply_style(message, style)
        return self

    def add_item(self, text: str, indent_level: int = 1, style: str = None):
        """
        Add an indented, subordinate line of information.

        Useful for itemizing changes or adding notes beneath the main message.

        Args:
            text: The item text to add
            indent_level: Indentation level (1=12 spaces, 2=14 spaces, etc.)
            style: Optional Rich style to apply (e.g., "bold red", "dim")
        """
        styled_text = self._apply_style(text, style)
        self.context_parts.append(f"\n{indent(indent_level)}{styled_text}")
        return self

    def add_bullet_list(
        self, items: list[str], indent_level: int = 1, style: str = None
    ):
        """
        Add multiple items as a bullet list to StatusIndicator.

        Args:
            items: List of strings to add as bullets
            indent_level: Indentation level (1=12 spaces, 2=14 spaces, etc.)
            style: Optional Rich style to apply to all items (e.g., "dim", "bold")
        """
        formatted = fmt_bullet_list(items, indent_level)
        if style:
            formatted = self._apply_style(formatted, style)
        self.context_parts.append(f"\n{formatted}")
        return self

    def add_kv_pairs(
        self, pairs: dict[str, str | int], indent_level: int = 1, style: str = None
    ):
        """
        Add key-value pairs as formatted items.

        Args:
            pairs: Dictionary of key-value pairs to add
            indent_level: Indentation level (1=12 spaces, 2=14 spaces, etc.)
            style: Optional Rich style to apply to all pairs (e.g., "dim", "bold")
        """
        for key, value in pairs.items():
            kv_text = fmt_kv_pair(key, value)
            if style:
                kv_text = self._apply_style(kv_text, style)
            self.context_parts.append(f"\n{indent(indent_level)}{kv_text}")
        return self

    def add_step_log(self, step_log: list):
        """
        Add a step-by-step log of operations applied.

        Args:
            step_log: List of (operation_name, name_before, name_after) tuples
        """
        self.step_log = step_log
        return self

    def with_summary_block(
        self,
        updated: int = 0,
        unchanged: int = 0,
        errors: int = 0,
        additional_info: list = None,
    ):
        """
        Append a final, formatted block of statistics.
        """
        # Add blank line before summary for better visual separation
        indent_str = PRE_LABEL_INDENT if self.dry_run else INDENT
        summary_parts = [
            fmt_field("updated", updated),
            fmt_field("unchanged", unchanged),
            fmt_field("errors", errors),
        ]
        summary = " | ".join(summary_parts)
        self.context_parts.append(f"\n{indent_str}{summary}")

        if additional_info:
            for info in additional_info:
                self.context_parts.append(f"\n{indent_str}{info}")
        return self

    def add_indent(self, level: int = 1, additional: int = 0):
        """Add indentation for hierarchical output."""
        self.context_parts.append(f"\n{indent(level, additional)}")
        return self

    # --- STAGE 5: BUILD & EMIT ---
    def _format_label(self) -> str | None:
        """Build the status label with dry-run dimming and DRY prefix if applicable."""
        # Suppress 'saved' messages entirely in dry-run mode
        if self.dry_run and self.status == "saved":
            return None

        label = self.theme["label"]

        # Dim operational labels in dry-run mode (not info/warning/error/parsing)
        if self.dry_run and self.status not in ["info", "warning", "error", "parsing"]:
            label = f"[dim]{label}[/dim]"

        # Add DRY prefix when in dry-run mode
        if self.dry_run:
            return f"{DRY_LABEL} {label}"

        return label

    def _format_context_and_details(self) -> str:
        """Build the context and details portion of the message."""
        context = " ".join(self.context_parts)
        details = self.explanation or ""

        # Handle spacing between context and details based on template
        template = self.theme["template"]
        if "{context}{details}" in template:
            # Template has no space between - add one if both exist
            if context and details and not details.startswith(" "):
                details = f" {details}"

        return template.format(context=context, details=details)

    def _get_indent(self) -> str:
        """Get the appropriate indent based on dry-run mode."""
        if self.dry_run:
            # When dry_run=True, align with content after DRY + status labels
            # Calculate: pre_label_width + space + label_width + space
            pre_label_width = CONSOLE_CONFIG.get("pre_label_width", 4)
            label_width = CONSOLE_CONFIG.get("label_width", 11)
            total_width = pre_label_width + 1 + label_width + 1
            return " " * total_width
        else:
            # When dry_run=False, use regular INDENT (aligns with content after status label)
            return INDENT

    def _format_values(self) -> str:
        """Build the value display (changes or single values)."""
        if not (self.old_value or self.new_value or self.value):
            return ""

        indent_str = self._get_indent()

        # Handle old→new changes
        if self.theme["show_change"] and self.old_value and self.new_value:
            change_text = fmt_change(self.old_value, self.new_value)
            if self.value_style_override:
                change_text = self._apply_style(change_text, self.value_style_override)
            # No extra space - indent_str already positions us at the right column
            return f"\n{indent_str}{change_text}"

        # Handle single values
        if self.value:
            # Use override style if provided, otherwise use theme's value_style
            style = self.value_style_override or self.theme.get("value_style", "plain")

            # If we have a style override, just apply it directly
            if self.value_style_override:
                value_text = self._apply_style(
                    str(self.value), self.value_style_override
                )
            else:
                # Use fmt_value which applies theme styling
                value_text = fmt_value(self.value, style)

            # No extra space - indent_str already positions us at the right column
            return f"\n{indent_str}{value_text}"

        return ""

    def _format_step_log(self) -> str:
        """Build the step log display."""
        if not self.step_log:
            return ""

        indent_str = self._get_indent()
        result = []
        for op_name, before, after in self.step_log:
            # Only show if there's an actual change
            if before != after:
                before_stem = Path(before).stem
                after_stem = Path(after).stem
                change_display = fmt_change(before_stem, after_stem)
                result.append(f"\n{indent_str}  {BULLET} {op_name}: {change_display}")

        return "".join(result)

    def build(self) -> str:
        """Build the final formatted status message."""
        # Get label (returns None if suppressed in dry-run)
        label = self._format_label()
        if label is None:
            return ""

        # Build message components
        message = self._format_context_and_details()
        message += self._format_values()
        message += self._format_step_log()

        return f"{label} {message}"

    def emit(self, console=None):
        """Build and emit the message in one call."""
        _emit = emit
        _emit(self.build(), console=console)


# ============================================================================
# HIGH-LEVEL HELPERS
# ============================================================================
# Convenience functions that use StatusIndicator internally for common patterns


def fmt_preflight_checklist(script_name: str, operations: list, console=None) -> None:
    """
    Display a standardized pre-flight checklist showing what the script will do.

    Args:
        script_name: Name of the script (e.g., "NameID 1 Replacer")
        operations: List of operation descriptions
        console: Optional console instance

    Example:
        >>> fmt_preflight_checklist("NameID 1 Replacer", ["Replace nameID 1", "Update family names"])
        # Displays formatted checklist with script name and operations
    """
    if console is None:
        console = get_console()

    emit("")
    StatusIndicator("info").add_message("Pre-flight checklist:").emit(console)
    emit(f"  Script: {script_name}", console=console)
    emit("  Operations to perform:", console=console)
    for i, operation in enumerate(operations, 1):
        emit(f"    {i}. {operation}", console=console)


def fmt_processing_summary(
    dry_run: bool = False,
    updated: int = 0,
    unchanged: int = 0,
    errors: int = 0,
    console=None,
    additional_info: list = None,
) -> None:
    """
    Display a standardized processing summary.

    Args:
        dry_run: Whether this was a dry run
        updated: Number of files that were updated
        unchanged: Number of files that were unchanged
        errors: Number of files that had errors
        console: Optional console instance
        additional_info: Optional list of additional info lines to display

    Example:
        >>> fmt_processing_summary(dry_run=False, updated=35, unchanged=3, errors=2)
        # Displays "Processing Completed! updated: 35 | unchanged: 3 | errors: 2"
    """
    if console is None:
        console = get_console()

    emit("")
    label = "Preview" if dry_run else "Processing Completed!"
    StatusIndicator("success", dry_run=dry_run).add_message(label).with_summary_block(
        updated=updated,
        unchanged=unchanged,
        errors=errors,
        additional_info=additional_info,
    ).emit(console)


def fmt_operation_description(operation_type: str, details: str) -> str:
    """
    Format a standardized operation description for pre-flight checklist.

    Args:
        operation_type: Type of operation (e.g., "Replace", "Delete", "Create")
        details: Specific details about what will be done

    Returns:
        Formatted operation description

    Example:
        >>> fmt_operation_description("Replace", "nameID 1 (Font Family)")
        "Replace nameID 1 (Font Family)"
    """
    return f"{operation_type} {details}"


def fmt_deletion_operation(
    name_ids: list = None,
    mac_records: bool = False,
    fontsquirrel: bool = False,
    windows_english_only: bool = False,
) -> str:
    """
    Format a deletion operation description.

    Args:
        name_ids: List of name IDs to delete
        mac_records: Whether to delete Macintosh records
        fontsquirrel: Whether to delete FontSquirrel records
        windows_english_only: Whether to keep only Windows/English/Latin records

    Returns:
        Formatted deletion operation description

    Example:
        >>> fmt_deletion_operation(name_ids=[1, 2])
        "Delete nameID 1, 2"
    """
    parts = []
    if name_ids:
        parts.append(f"nameID {', '.join(map(str, sorted(name_ids)))}")
    if mac_records:
        parts.append("Macintosh records (platformID=1)")
    if fontsquirrel:
        parts.append("FontSquirrel records (nameIDs: 200,201,202,203,55555)")
    if windows_english_only:
        parts.append(
            "non-Windows/English/Latin records (keep only platformID=3, platEncID=1, langID=0x409)"
        )

    return f"Delete {', '.join(parts)}"


def fmt_replacement_operation(
    name_id: int, description: str, source: str = None
) -> str:
    """
    Format a replacement operation description.

    Args:
        name_id: The name ID being replaced
        description: Description of the name ID
        source: Source of the replacement value (e.g., "filename parser", "user input")

    Returns:
        Formatted replacement operation description

    Example:
        >>> fmt_replacement_operation(1, "Font Family", "filename parser")
        "Replace nameID 1 (Font Family) using filename parser"
    """
    base = f"Replace nameID {name_id} ({description})"
    if source:
        base += f" using {source}"
    return base


# ============================================================================
# USER INTERACTION
# ============================================================================
# Interactive prompt functions for user input


class QuitRequested(Exception):
    """Exception raised when user requests to quit during batch operations."""

    pass


def prompt_input(message: str, console: Optional["_Console"] = None) -> str:
    """
    Render an INPUT-labeled prompt and return user input using a two-line layout.
    """
    emit("")  # Leading newline for spacing
    result = ""
    try:
        if RICH_AVAILABLE:
            console_instance = console or get_console()
            # 1. Print the informational message
            console_instance.print(f"{INPUT_LABEL} {message}")
            # 2. Print the input prompt on the next line
            console_instance.print(f"{INPUT_LABEL} ", end="")
            result = input()
        else:
            print(f"{INPUT_LABEL} {message}")
            print(f"{INPUT_LABEL} ", end="")
            result = input()

    except (EOFError, KeyboardInterrupt):
        # Return an empty string on interruption
        result = ""
    finally:
        # Ensures a trailing newline
        emit("")

    return result


def prompt_confirm(
    message: str,
    action_prompt: str = "Proceed?",
    default: bool = False,
    allow_quit: bool = False,
) -> bool:
    """
    Yes/no/quit confirmation with a consistent two-line layout.
    Accepts a main message and a separate action prompt for the second line.
    Supports 'q' or 'quit' to raise QuitRequested exception when allow_quit=True.
    """
    emit("")  # Leading newline for spacing

    if allow_quit:
        default_str = "Y/n/q" if default else "y/N/q"
    else:
        default_str = "Y/n" if default else "y/N"
    response_val = default

    try:
        if RICH_AVAILABLE:
            console_instance = get_console()
            # 1. Print the informational message on its own line
            console_instance.print(f"{INPUT_LABEL} {message}")
            # 2. Print the action prompt on the next line
            console_instance.print(
                f"{INPUT_LABEL} {action_prompt} [input][bold]({default_str}): [/bold][/input]",
                end="",
            )
        else:
            print(f"{INPUT_LABEL} {message}")
            print(
                f"{INPUT_LABEL} {action_prompt} [input][bold]({default_str}): [/bold][/input]",
                end="",
            )

        response = input().strip().lower()

        if response in ["y", "yes"]:
            response_val = True
        elif response in ["n", "no", "s", "skip"]:
            response_val = False
        elif response in ["q", "quit", "exit"]:
            if allow_quit:
                raise QuitRequested("User requested to quit batch operation")
            else:
                # In standalone mode, treat 'q' as 'no'
                response_val = False

    except (EOFError, KeyboardInterrupt):
        response_val = False
    finally:
        emit("")

    return response_val


def prompt_text(message: str, default: str = "") -> str:
    """
    Text input using vanilla Python with Rich styling.
    """
    emit("")
    emit(f"{INPUT_LABEL} {message}", console=get_console())

    try:
        if default:
            result = prompt_input(f"[{default}]").strip()
            return result if result else default
        else:
            result = prompt_input(":").strip()
            return result
    except (EOFError, KeyboardInterrupt):
        return default
    finally:
        emit("")


def prompt_select(message: str, choices: list, default=None) -> str:
    """
    Simple numbered selection using vanilla Python with Rich styling.
    """
    emit("")
    emit(f"{INPUT_LABEL} {message}", console=get_console())

    for i, choice in enumerate(choices, 1):
        emit(f"  {i}. {choice}", console=get_console())

    while True:
        try:
            selection = prompt_input("Enter number").strip()
            idx = int(selection) - 1
            if 0 <= idx < len(choices):
                emit("")
                return choices[idx]
            emit(
                f"{WARNING_LABEL} Please select 1-{len(choices)}", console=get_console()
            )
        except (ValueError, EOFError, KeyboardInterrupt):
            emit("")
            return default or choices[0]


# ============================================================================
# STRUCTURED OUTPUT HELPERS
# ============================================================================
# Rich components for complex output layouts


def print_panel(
    message: str,
    title: Optional[str] = None,
    border_style: str = "dodger_blue1",
    console: Optional["_Console"] = None,
) -> None:
    """
    Print a message in a Rich panel (box). Falls back to simple print if Rich unavailable.
    """
    if RICH_AVAILABLE:
        panel = Panel(message, title=title, border_style=border_style)
        (console or get_console()).print(panel)
    else:
        if title:
            print(f"=== {title} ===")
        print(message)
        if title:
            print("=" * (len(title) + 8))


def status_message(
    label: str, details: str = "", console: Optional["_Console"] = None
) -> None:
    """
    Print a status message with a label and optional details.
    """
    if details:
        emit(f"{label} {details}", console=console)
    else:
        emit(label, console=console)


def print_session_header(url: str, console: Optional["_Console"] = None) -> None:
    """
    Print a prominent session header when starting a new site/page.
    """
    if not RICH_AVAILABLE:
        print(f"\n{'=' * 80}")
        print(f"  SESSION: {url}")
        print(f"{'=' * 80}\n")
        return

    console_instance = console or get_console()

    console_instance.print()
    console_instance.print(f"[info.bright]{'═' * 80}[/info.bright]")
    console_instance.print(
        f"[bold][info]  SESSION:[/info][bold] [lighttext][bold]{url}[/lighttext][bold]"
    )
    console_instance.print(f"[info.bright]{'═' * 80}[/info.bright]")
    console_instance.print()


def create_table(
    title: Optional[str] = None,
    show_header: bool = True,
    console: Optional["_Console"] = None,
    row_styles: Optional[list] = None,
) -> Optional["_Table"]:
    """
    Create a Rich Table with consistent styling.
    """
    if not RICH_AVAILABLE:
        return None

    from rich.table import Table

    return Table(
        title=title,
        title_justify="center",
        title_style="bold deep_sky_blue1",
        show_header=show_header,
        header_style="bold dodger_blue1",
        border_style="dim",
        highlight="True",
        row_styles=row_styles,
    )


def create_progress_bar(console: Optional[_Console] = None) -> _Progress:
    """Create a standardized progress bar for consistent styling across scripts."""
    if not RICH_AVAILABLE:
        logger.error("Rich is required for progress bars but not available")
        raise ImportError(
            "Rich is required for progress bars. Install with: pip install rich"
        )

    if console is None:
        console = get_console()
        if console is None:
            logger.error("Failed to get console instance for progress bar")
            raise RuntimeError("Console instance not available")

    try:
        return _Progress(
            SpinnerColumn("dots2", style="dodger_blue1"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        )
    except Exception as e:
        logger.error(f"Failed to create progress bar: {e}")
        raise
