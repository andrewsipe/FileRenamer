#!/usr/bin/env python3
"""
Font utility functions for validation and sanitization.

Provides shared functions for validating PostScript names and sanitizing
folder/filename strings for filesystem safety.
"""

import re
from collections import defaultdict
from typing import Callable, Dict, List, Tuple, TypeVar

T = TypeVar("T")
K = TypeVar("K")


def is_valid_postscript_name(ps_name: str) -> Tuple[bool, str]:
    """
    Validate PostScript name is safe for filename.

    Args:
        ps_name: PostScript name to validate

    Returns:
        Tuple of (is_valid, reason_string)
    """
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
    if ps_name and ps_name[0] in forbidden_first_chars:
        return False, f"begins with '{ps_name[0]}'"

    return True, ""


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


def count_items_per_group(
    items: List[T],
    key_func: Callable[[T], K],
) -> Dict[K, int]:
    """
    Count items per group based on a key function.

    Useful for counting fonts per family, files per folder, etc.

    Args:
        items: List of items to count
        key_func: Function that extracts a key from each item

    Returns:
        Dictionary mapping keys to counts

    Example:
        >>> fonts = [Font(family="Arial"), Font(family="Arial"), Font(family="Helvetica")]
        >>> counts = count_items_per_group(fonts, lambda f: f.family)
        >>> counts
        {'Arial': 2, 'Helvetica': 1}
    """
    counts: Dict[K, int] = defaultdict(int)
    for item in items:
        key = key_func(item)
        counts[key] += 1

    return dict(counts)


def format_name_with_count(name: str, count: int) -> str:
    """
    Format a name with a count suffix (e.g., "Helvetica (12)").

    Args:
        name: Base name
        count: Count to append

    Returns:
        Formatted name with count

    Example:
        >>> format_name_with_count("Helvetica", 12)
        'Helvetica (12)'
    """
    return f"{name} ({count})"
