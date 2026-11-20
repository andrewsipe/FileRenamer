#!/usr/bin/env python3
"""
Core Font Sorting Utilities

Provides intelligent font grouping and sorting capabilities for font management scripts.
Supports standard family grouping, superfamily clustering, and forced group merging.

Usage:
    from FontCore.core_font_sorter import FontSorter, FontInfo

    # Create font info objects
    font_infos = [FontInfo(path, family_name) for path, family_name in font_data]

    # Sort by family
    sorter = FontSorter(font_infos)
    families = sorter.group_by_family()

    # Sort by superfamily with options
    # Note: ignore_terms is now passed to FontSorter constructor
    sorter = FontSorter(font_infos, ignore_terms={'29LT', 'Adobe'})
    superfamilies = sorter.group_by_superfamily(
        exclude_families=['Script', 'Display']
    )

    # Force specific families to merge
    forced_groups = [['Rough Love', 'Love Script'], ['Family A', 'Family B']]
    families = sorter.group_by_family(forced_groups=forced_groups)

Demo and Testing:
    Run 'python CoreDemoTool.py font --help' to see all available options and examples.
    The demo tool supports real file processing and all font sorter functionality.

Maintenance Note:
    When adding new functions to this module, update CoreDemoTool.py to showcase
    the new functionality in the 'font' subcommand.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    from FontCore.core_file_collector import collect_font_files
except ImportError:
    collect_font_files = None


@dataclass
class FontInfo:
    """Basic font information for sorting and grouping."""

    path: str
    family_name: str
    vendor: Optional[str] = None
    vendor_id: Optional[str] = None
    designer: Optional[str] = None
    style: Optional[str] = None

    def __post_init__(self):
        """Normalize family name after initialization."""
        if isinstance(self.family_name, str):
            self.family_name = unicodedata.normalize("NFC", self.family_name)
        else:
            self.family_name = "Unknown"


class FontSorter:
    """Intelligent font sorting and grouping utilities."""

    def __init__(
        self, font_infos: List[FontInfo], ignore_terms: Optional[Set[str]] = None
    ):
        """Initialize with a list of FontInfo objects.

        Args:
            font_infos: List of FontInfo objects to sort/group
            ignore_terms: Optional set of terms to ignore when normalizing family names.
                         If provided, these terms are removed from family names before
                         grouping operations. Applies to both family and superfamily grouping.
        """
        self.font_infos = font_infos
        self.ignore_terms = set(ignore_terms or [])
        # Normalize all family names upfront
        # Use path as key since FontInfo is not hashable
        self._normalized_names: Dict[str, str] = {
            fi.path: self._normalize_family_name(fi.family_name) for fi in font_infos
        }

    def group_by_family(
        self, forced_groups: Optional[List[List[str]]] = None
    ) -> Dict[str, List[FontInfo]]:
        """Group fonts by normalized family name (standard grouping).

        Uses normalized family names (with ignored terms removed) for grouping.
        Original family names are preserved in FontInfo objects for display.
        """
        families: Dict[str, List[FontInfo]] = {}
        for font_info in self.font_infos:
            normalized = self._normalized_names[font_info.path]
            families.setdefault(normalized, []).append(font_info)

        if forced_groups:
            # Normalize forced group names before applying
            normalized_forced_groups = [
                [self._normalize_family_name(name) for name in group]
                for group in forced_groups
            ]
            families = self.apply_forced_groups(families, normalized_forced_groups)

        return families

    def get_normalized_name(self, font_info: FontInfo) -> str:
        """Get normalized family name for a FontInfo object.

        Args:
            font_info: FontInfo object to get normalized name for

        Returns:
            Normalized family name (with ignored terms removed)
        """
        return self._normalized_names.get(font_info.path, font_info.family_name)

    def get_normalization_summary(self) -> Dict[str, str]:
        """Get mapping of original → normalized names for display.

        Only includes families where normalization actually changed the name.

        Returns:
            Dictionary mapping original family names to normalized names
        """
        summary = {}
        seen_originals = set()
        # Build path to FontInfo mapping for lookup
        path_to_fontinfo = {fi.path: fi for fi in self.font_infos}
        for path, normalized in self._normalized_names.items():
            fi = path_to_fontinfo.get(path)
            if (
                fi
                and fi.family_name != normalized
                and fi.family_name not in seen_originals
            ):
                summary[fi.family_name] = normalized
                seen_originals.add(fi.family_name)
        return summary

    def group_by_vendor(self) -> Dict[str, List[FontInfo]]:
        """Group fonts by vendor name."""
        return self._group_by_attribute("vendor", "Unknown")

    def group_by_designer(self) -> Dict[str, List[FontInfo]]:
        """Group fonts by designer name."""
        return self._group_by_attribute("designer", "Unknown")

    def group_by_vendor_id(self) -> Dict[str, List[FontInfo]]:
        """Group fonts by vendor ID (OS/2.achVendID)."""
        return self._group_by_attribute("vendor_id", "Unknown")

    def _group_by_attribute(
        self, attr_name: str, default: str
    ) -> Dict[str, List[FontInfo]]:
        """Generic grouping by any FontInfo attribute."""
        groups: Dict[str, List[FontInfo]] = {}
        for font_info in self.font_infos:
            key = getattr(font_info, attr_name, None) or default
            groups.setdefault(key, []).append(font_info)
        return groups

    def _build_forced_mapping(
        self, forced_groups: List[List[str]], existing_families: Set[str]
    ) -> Dict[str, str]:
        """Build mapping from family names to their forced group names."""
        family_to_forced: Dict[str, str] = {}

        for forced_group in forced_groups:
            available = [f for f in forced_group if f in existing_families]

            if len(available) >= 2:
                group_name = available[0]
                for family_name in available:
                    family_to_forced[family_name] = group_name

        return family_to_forced

    def apply_forced_groups(
        self, groups: Dict[str, List[FontInfo]], forced_groups: List[List[str]]
    ) -> Dict[str, List[FontInfo]]:
        """Apply forced groupings to existing groups."""
        if not forced_groups:
            return groups

        family_to_forced = self._build_forced_mapping(forced_groups, set(groups.keys()))
        new_groups: Dict[str, List[FontInfo]] = {}

        for group_name, fonts in groups.items():
            target_group = family_to_forced.get(group_name, group_name)
            new_groups.setdefault(target_group, []).extend(fonts)

        return new_groups

    def get_forced_groups_info(
        self, forced_groups: List[List[str]], group_type: str = "family"
    ) -> List[Dict[str, Any]]:
        """Get information about forced group merges."""
        return [
            {
                "group_name": forced_group[0],
                "merged_families": forced_group[1:],
                "group_type": group_type,
            }
            for forced_group in (forced_groups or [])
            if len(forced_group) >= 2
        ]

    def _is_weak_prefix(self, prefix: str, all_family_names: List[str]) -> bool:
        """Detect if a prefix is too generic/common to form a meaningful superfamily."""
        if len(prefix) < 2:
            return True

        families_with_prefix = [
            name
            for name in all_family_names
            if name.startswith(prefix + " ") or name == prefix
        ]

        if len(families_with_prefix) < 4:
            return False

        # Extract second tokens from families with this prefix
        second_tokens = {
            tokens[1]
            for name in families_with_prefix
            if len(tokens := name.split()) >= 2 and tokens[0] == prefix
        }

        return len(second_tokens) >= 4

    def _normalize_family_name(self, name: str) -> str:
        """Normalize family name by removing ignored terms.

        Args:
            name: Original family name

        Returns:
            Normalized family name with ignored terms removed, or original name if no normalization needed
        """
        if not self.ignore_terms:
            return name
        tokens = self._filter_tokens(name, self.ignore_terms)
        return " ".join(tokens) if tokens else name

    def _filter_tokens(self, name: str, ignore_terms: Set[str]) -> List[str]:
        """Filter out ignored terms from name tokens."""
        return [t for t in name.split() if t not in ignore_terms]

    def _get_common_prefix_tokens(
        self, tokens_a: List[str], tokens_b: List[str]
    ) -> List[str]:
        """Get common leading tokens between two token lists."""
        common = []
        for t_a, t_b in zip(tokens_a, tokens_b):
            if t_a == t_b:
                common.append(t_a)
            else:
                break
        return common

    def _is_substantial_single_token(
        self, token: str, tokens_a: List[str], tokens_b: List[str]
    ) -> bool:
        """Check if a single common token is substantial enough to group on."""
        if len(token) < 3:
            return False

        # For 3-char tokens, apply stricter rules
        if len(token) == 3:
            is_complete_name = token == " ".join(tokens_a) or token == " ".join(
                tokens_b
            )
            both_multiword = len(tokens_a) >= 2 and len(tokens_b) >= 2
            return is_complete_name or both_multiword

        # Token is complete name of either font
        if token == " ".join(tokens_a) or token == " ".join(tokens_b):
            return True

        # Both are multi-word with substantial shared first token
        if len(tokens_a) >= 2 and len(tokens_b) >= 2:
            return True

        # Single-word names that match
        return len(tokens_a) == 1 and len(tokens_b) == 1

    def _shares_prefix(
        self,
        name_a: str,
        name_b: str,
        ignore_terms: Optional[Set[str]] = None,
        all_family_names: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Check if two names share a meaningful prefix.

        Note: When used with normalized names (from constructor), ignore_terms should be None
        since names are already normalized. This parameter is kept for backward compatibility.

        Args:
            all_family_names: Full list of family names in context. When provided, single-token
                              prefixes are rejected if they are shared by 4+ families with 4+
                              distinct second tokens (i.e. a foundry/vendor prefix, not a
                              meaningful superfamily root).
        """
        # If ignore_terms provided, filter tokens (legacy behavior)
        # Otherwise, assume names are already normalized
        if ignore_terms is not None:
            tokens_a = self._filter_tokens(name_a, ignore_terms)
            tokens_b = self._filter_tokens(name_b, ignore_terms)
        else:
            tokens_a = name_a.split()
            tokens_b = name_b.split()

        if not tokens_a or not tokens_b:
            return None

        common = self._get_common_prefix_tokens(tokens_a, tokens_b)

        if not common:
            return None

        common_prefix = " ".join(common)

        # 2+ common tokens is always substantial
        if len(common) >= 2:
            return common_prefix

        # Single token: reject if it's a weak/overly-broad prefix (e.g. a foundry name
        # shared by every family in the collection)
        token = common[0]
        if all_family_names and self._is_weak_prefix(token, all_family_names):
            return None

        return (
            common_prefix
            if self._is_substantial_single_token(token, tokens_a, tokens_b)
            else None
        )

    def _partition_families(
        self, family_names: List[str], exclude_patterns: List[str]
    ) -> tuple[List[str], List[str]]:
        """Partition families into excluded and groupable lists."""
        excluded = []
        groupable = []

        for name in family_names:
            name_lower = name.lower()
            if any(pattern in name_lower for pattern in exclude_patterns):
                excluded.append(name)
            else:
                groupable.append(name)

        return excluded, groupable

    def _merge_superfamily_assignments(
        self, superfamily_map: Dict[str, str], superfam_a: str, superfam_b: str
    ) -> str:
        """Merge two superfamily assignments, choosing the shorter prefix."""
        target = (
            superfam_a
            if len(superfam_a.split()) <= len(superfam_b.split())
            else superfam_b
        )

        for fname, sfam in superfamily_map.items():
            if sfam in (superfam_a, superfam_b):
                superfamily_map[fname] = target

        return target

    def _assign_superfamily(
        self,
        superfamily_map: Dict[str, str],
        name_a: str,
        name_b: str,
        common_prefix: str,
    ) -> None:
        """Assign two families to the same superfamily."""
        superfam_a = superfamily_map.get(name_a)
        superfam_b = superfamily_map.get(name_b)

        if superfam_a and superfam_b:
            self._merge_superfamily_assignments(superfamily_map, superfam_a, superfam_b)
        elif superfam_a:
            superfamily_map[name_b] = superfam_a
        elif superfam_b:
            superfamily_map[name_a] = superfam_b
        else:
            superfamily_map[name_a] = common_prefix
            superfamily_map[name_b] = common_prefix

    def _build_superfamily_map(
        self,
        groupable_families: List[str],
        excluded_families: List[str],
    ) -> Dict[str, str]:
        """Build mapping from normalized family names to superfamily names.

        Args:
            groupable_families: List of normalized family names to group
            excluded_families: List of normalized family names to exclude from grouping

        Returns:
            Dictionary mapping normalized family names to superfamily names
        """
        superfamily_map: Dict[str, str] = {}

        # Find shared prefixes and cluster families
        # Names are already normalized, so no need to filter tokens
        all_family_names = groupable_families + excluded_families
        for i, name_a in enumerate(groupable_families):
            for name_b in groupable_families[i + 1 :]:
                common_prefix = self._shares_prefix(
                    name_a, name_b, ignore_terms=None, all_family_names=all_family_names
                )

                if common_prefix:
                    self._assign_superfamily(
                        superfamily_map, name_a, name_b, common_prefix
                    )

        # Assign unassigned groupable families to themselves
        for name in groupable_families:
            if name not in superfamily_map:
                superfamily_map[name] = name

        # Excluded families always map to themselves
        for name in excluded_families:
            superfamily_map[name] = name

        return superfamily_map

    def group_by_superfamily(
        self,
        exclude_families: Optional[List[str]] = None,
        forced_groups: Optional[List[List[str]]] = None,
    ) -> Dict[str, List[FontInfo]]:
        """Group fonts by superfamily using common prefix clustering.

        Uses normalized family names (with ignored terms removed) for prefix comparison.
        Original family names are preserved in FontInfo objects for display.

        Args:
            exclude_families: Optional list of family name patterns to exclude from superfamily grouping.
                            Patterns are matched against normalized names.
            forced_groups: Optional list of forced group merges. Family names in forced groups
                          are normalized before matching.
        """
        exclude_patterns = [pattern.lower() for pattern in (exclude_families or [])]

        # Group by normalized family name first
        normalized_names_to_fonts: Dict[str, List[FontInfo]] = {}
        for font_info in self.font_infos:
            normalized = self._normalized_names[font_info.path]
            normalized_names_to_fonts.setdefault(normalized, []).append(font_info)

        # Partition families (using normalized names)
        unique_normalized_names = list(normalized_names_to_fonts.keys())
        excluded_families, groupable_families = self._partition_families(
            unique_normalized_names, exclude_patterns
        )

        # Build superfamily mapping (names are already normalized)
        superfamily_map = self._build_superfamily_map(
            groupable_families, excluded_families
        )

        # Group fonts by superfamily (using normalized names)
        families: Dict[str, List[FontInfo]] = {}
        for font_info in self.font_infos:
            normalized = self._normalized_names[font_info.path]
            superfamily_name = superfamily_map[normalized]
            families.setdefault(superfamily_name, []).append(font_info)

        # Apply forced groupings if provided (normalize forced group names)
        if forced_groups:
            normalized_forced_groups = [
                [self._normalize_family_name(name) for name in group]
                for group in forced_groups
            ]
            families = self.apply_forced_groups(families, normalized_forced_groups)

        return families

    def get_grouping_summary(
        self, groups: Dict[str, List[FontInfo]], group_type: str = "family"
    ) -> Dict[str, Any]:
        """Get summary information about font groupings."""
        total_fonts = len(self.font_infos)
        num_groups = len(groups)
        group_sizes = [len(fonts) for fonts in groups.values()]

        return {
            "group_type": group_type,
            "total_fonts": total_fonts,
            "num_groups": num_groups,
            "avg_group_size": sum(group_sizes) / num_groups if num_groups > 0 else 0,
            "largest_group": max(group_sizes, default=0),
            "smallest_group": min(group_sizes, default=0),
            "group_sizes": group_sizes,
        }

    def get_superfamily_summary(
        self, groups: Dict[str, List[FontInfo]]
    ) -> Dict[str, Any]:
        """Get superfamily grouping summary data."""
        superfamily_members: Dict[str, Set[str]] = {}

        for group_name, fonts in groups.items():
            family_names = {f.family_name for f in fonts}
            if len(family_names) > 1:
                superfamily_members[group_name] = family_names

        return {
            "num_groups": len(groups),
            "merges_occurred": len(superfamily_members) > 0,
            "superfamily_members": {k: list(v) for k, v in superfamily_members.items()},
        }

    def get_hierarchical_groups(
        self, groups: Dict[str, List[FontInfo]], group_type: str = "family"
    ) -> Dict[str, Any]:
        """Get hierarchical group structure data."""
        result = {}

        if group_type == "superfamily":
            result = self._get_superfamily_hierarchy(groups)
        else:
            result = self._get_family_hierarchy(groups)

        return result

    def _get_superfamily_hierarchy(
        self, groups: Dict[str, List[FontInfo]]
    ) -> Dict[str, Any]:
        """Build hierarchical superfamily → family → fonts structure."""
        result = {}

        for superfamily_name, fonts in sorted(groups.items()):
            family_groups: Dict[str, List[FontInfo]] = {}
            for font in fonts:
                family_groups.setdefault(font.family_name, []).append(font)

            result[superfamily_name] = {
                "total_fonts": len(fonts),
                "families": {
                    family_name: [
                        font.path for font in sorted(family_fonts, key=lambda f: f.path)
                    ]
                    for family_name, family_fonts in sorted(family_groups.items())
                },
            }

        return result

    def _get_family_hierarchy(
        self, groups: Dict[str, List[FontInfo]]
    ) -> Dict[str, Any]:
        """Build flat family → fonts structure."""
        return {
            group_name: {
                "total_fonts": len(fonts),
                "fonts": [font.path for font in sorted(fonts, key=lambda f: f.path)],
            }
            for group_name, fonts in sorted(groups.items())
        }


# Font metadata extraction helpers


def _extract_name_record(name_table, record_id: int) -> Optional[str]:
    """Extract a name record with fallback to different platforms."""
    rec = name_table.getName(record_id, 3, 1, 0x409) or name_table.getName(
        record_id, 1, 0, 0
    )
    return str(rec.toUnicode()) if rec else None


def _extract_family_name(name_table) -> str:
    """Extract family name with priority: ID 16 (Typographic) then ID 1 (Font Family)."""
    family_name = _extract_name_record(name_table, 16)
    if not family_name:
        family_name = _extract_name_record(name_table, 1)
    return family_name or "Unknown"


def _extract_font_metadata(
    path: str,
) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Extract family name, vendor, vendor_id, and designer from font file."""
    family_name = "Unknown"
    vendor = None
    vendor_id = None
    designer = None

    try:
        from fontTools.ttLib import TTFont

        font = TTFont(path)

        if "name" in font:
            name_tbl = font["name"]
            family_name = _extract_family_name(name_tbl)
            vendor = _extract_name_record(name_tbl, 8)
            designer = _extract_name_record(name_tbl, 9)

        if "OS/2" in font:
            vendor_id_raw = getattr(font["OS/2"], "achVendID", None)
            if vendor_id_raw and isinstance(vendor_id_raw, str):
                vendor_id = vendor_id_raw.strip() or None

        font.close()

    except Exception:
        family_name = Path(path).stem

    return family_name, vendor, vendor_id, designer


def create_font_info_from_paths(
    filepaths: List[str], extract_metadata: bool = True
) -> List[FontInfo]:
    """Create FontInfo objects from file paths."""
    if not extract_metadata:
        return [FontInfo(path=path, family_name=Path(path).stem) for path in filepaths]

    font_infos = []
    for path in filepaths:
        family_name, vendor, vendor_id, designer = _extract_font_metadata(path)
        font_infos.append(
            FontInfo(
                path=path,
                family_name=family_name,
                vendor=vendor,
                vendor_id=vendor_id,
                designer=designer,
            )
        )

    return font_infos


# Convenience functions for common use cases


def sort_fonts_by_family(
    filepaths: List[str],
    forced_groups: Optional[List[List[str]]] = None,
    extract_metadata: bool = True,
) -> Dict[str, List[FontInfo]]:
    """Sort fonts by family name."""
    font_infos = create_font_info_from_paths(filepaths, extract_metadata)
    sorter = FontSorter(font_infos)
    return sorter.group_by_family(forced_groups=forced_groups)


def sort_fonts_by_superfamily(
    filepaths: List[str],
    ignore_terms: Optional[List[str]] = None,
    exclude_families: Optional[List[str]] = None,
    forced_groups: Optional[List[List[str]]] = None,
    extract_metadata: bool = True,
) -> Dict[str, List[FontInfo]]:
    """Sort fonts by superfamily using common prefix clustering.

    Args:
        filepaths: List of font file paths
        ignore_terms: Optional list of terms to ignore when normalizing family names
        exclude_families: Optional list of family name patterns to exclude
        forced_groups: Optional list of forced group merges
        extract_metadata: Whether to extract font metadata

    Returns:
        Dictionary mapping superfamily names to lists of FontInfo objects
    """
    font_infos = create_font_info_from_paths(filepaths, extract_metadata)
    ignore_terms_set = set(ignore_terms or [])
    sorter = FontSorter(font_infos, ignore_terms=ignore_terms_set)
    return sorter.group_by_superfamily(
        exclude_families=exclude_families, forced_groups=forced_groups
    )