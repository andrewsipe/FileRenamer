"""Shared conflict-priority helper for organizers.

Extracted from FontFiles_Renamer.py so the base renamer can be archived
without changing organizer behavior. Uses revision + head dates, not the
quality-score model in FontFiles_RenamerEnhanced.py.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, TypeVar


class _VersionPriorityMeta(Protocol):
    font_revision: Optional[float]
    head_created: Optional[float]
    head_modified: Optional[float]


T = TypeVar("T", bound=_VersionPriorityMeta)


def sort_by_version_priority(metadata_list: List[T]) -> List[T]:
    """
    Sort fonts by version priority:
    1. Highest font revision
    2. Oldest creation date (original design)
    3. Most recent modification date (latest fixes)
    """

    def sort_key(meta: _VersionPriorityMeta) -> tuple:
        return (
            -meta.font_revision if meta.font_revision else float("-inf"),
            meta.head_created if meta.head_created else float("inf"),
            -meta.head_modified if meta.head_modified else float("-inf"),
        )

    return sorted(metadata_list, key=sort_key)
