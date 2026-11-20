#!/usr/bin/env python3
"""
Standardized string handling utilities for font processing.

Philosophy:
- None means "no value was provided"
- Empty string means "value was provided but empty"
- Whitespace-only strings are treated as empty
- All functions preserve None vs empty distinction when meaningful

Usage:
    from FontCore.core_string_utils import normalize_empty, is_empty, safe_strip

    # Normalize user input
    family = normalize_empty(user_input)  # "" -> None

    # Check for meaningful content
    if not is_empty(family):
        process(family)

    # Safe string operations
    cleaned = safe_strip(raw_value)  # Never crashes
"""

from typing import Optional, TypeVar, Callable

T = TypeVar("T")


def is_empty(value: Optional[str]) -> bool:
    """
    Check if string is None, empty, or whitespace-only.

    Args:
        value: String to check

    Returns:
        True if value has no meaningful content

    Examples:
        >>> is_empty(None)
        True
        >>> is_empty("")
        True
        >>> is_empty("   ")
        True
        >>> is_empty("content")
        False
        >>> is_empty("  content  ")
        False
    """
    return not value or not str(value).strip()


def normalize_empty(value: Optional[str]) -> Optional[str]:
    """
    Convert empty/whitespace strings to None, strip meaningful content.

    Use this for optional text fields where empty = absent.

    Args:
        value: String to normalize

    Returns:
        None if empty/whitespace, stripped string otherwise

    Examples:
        >>> normalize_empty(None)
        None
        >>> normalize_empty("")
        None
        >>> normalize_empty("   ")
        None
        >>> normalize_empty("  content  ")
        'content'
    """
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped if stripped else None


def normalize_empty_to_default(value: Optional[str], default: str = "") -> str:
    """
    Normalize string with a fallback default value.

    Use this when you need a guaranteed non-None return.

    Args:
        value: String to normalize
        default: Default value if empty (default: "")

    Returns:
        Stripped string or default if empty

    Examples:
        >>> normalize_empty_to_default(None, "Unknown")
        'Unknown'
        >>> normalize_empty_to_default("", "Unknown")
        'Unknown'
        >>> normalize_empty_to_default("  content  ", "Unknown")
        'content'
    """
    normalized = normalize_empty(value)
    return normalized if normalized is not None else default


def safe_strip(value: Optional[str]) -> str:
    """
    Safely strip whitespace, never returns None.

    Use this when you need a string for concatenation/formatting.

    Args:
        value: String to strip

    Returns:
        Stripped string or empty string if None

    Examples:
        >>> safe_strip(None)
        ''
        >>> safe_strip("  content  ")
        'content'
    """
    return str(value).strip() if value is not None else ""


def coalesce(*values: Optional[str]) -> Optional[str]:
    """
    Return first non-empty value.

    Use this for fallback chains (e.g., ID16 -> ID1 -> "Unknown").

    Args:
        *values: Values to check in order

    Returns:
        First non-empty value or None if all empty

    Examples:
        >>> coalesce(None, "", "first")
        'first'
        >>> coalesce("", "   ", "second", "third")
        'second'
        >>> coalesce(None, "", "  ")
        None
    """
    for value in values:
        if not is_empty(value):
            return normalize_empty(value)
    return None


def ensure_value(value: Optional[str], fallback: str) -> str:
    """
    Ensure a non-empty value, using fallback if needed.

    Similar to coalesce() but always returns a string.

    Args:
        value: Primary value
        fallback: Fallback value (should be non-empty)

    Returns:
        value if not empty, otherwise fallback

    Examples:
        >>> ensure_value(None, "Unknown")
        'Unknown'
        >>> ensure_value("Valid", "Unknown")
        'Valid'
    """
    return normalize_empty_to_default(value, fallback)


def join_nonempty(*parts: Optional[str], separator: str = " ") -> str:
    """
    Join only non-empty parts with separator.

    Use this for building composite names.

    Args:
        *parts: String parts to join
        separator: Separator string

    Returns:
        Joined string, or empty if all parts empty

    Examples:
        >>> join_nonempty("Font", None, "Bold")
        'Font Bold'
        >>> join_nonempty("Font", "", "  ", "Bold")
        'Font Bold'
        >>> join_nonempty(None, "", separator="-")
        ''
    """
    cleaned = [normalize_empty(p) for p in parts]
    return separator.join(p for p in cleaned if p is not None)


def apply_if_present(
    value: Optional[str], func: Callable[[str], T], default: Optional[T] = None
) -> Optional[T]:
    """
    Apply function to value only if non-empty.

    Use this to avoid repetitive is_empty() checks.

    Args:
        value: String value to process
        func: Function to apply
        default: Return value if string is empty

    Returns:
        func(value) if value is non-empty, otherwise default

    Examples:
        >>> apply_if_present("  TEXT  ", str.lower)
        'text'
        >>> apply_if_present(None, str.lower, default="unknown")
        'unknown'
        >>> apply_if_present("", lambda x: x.split(), default=[])
        []
    """
    if is_empty(value):
        return default
    return func(normalize_empty(value))  # type: ignore
