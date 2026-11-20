#!/usr/bin/env python3
"""
Variable font detection with configurable strictness levels.

Variable fonts (per OpenType spec) should have:
- Required: fvar table (defines variation axes)
- Recommended: STAT table (style attributes)
- Optional: avar table (axis variations)
- Optional: MVAR table (metric variations)

This module provides flexible detection based on use case:
- Strict mode: Enforce spec compliance (fvar + STAT)
- Lenient mode: Accept technical validity (fvar only)
- Pedantic mode: Validate additional properties

Usage:
    from FontCore.core_variable_font_detection import (
        is_variable_font,
        VariableFontMode,
        analyze_variable_font
    )

    # Simple check (uses default strict mode)
    if is_variable_font(font):
        print("Variable font detected")

    # Lenient check (for legacy fonts)
    if is_variable_font(legacy_font, mode=VariableFontMode.LENIENT):
        print("Has fvar table")

    # Detailed analysis
    analysis = analyze_variable_font(font)
    if analysis.is_variable:
        print(f"Variable font with {analysis.axis_count} axes")
        if not analysis.has_stat:
            print("WARNING: Missing STAT table")
"""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import Any, List
from FontCore.core_logging_config import get_logger

logger = get_logger(__name__)


class VariableFontMode(Enum):
    """
    Variable font detection modes.

    STRICT: Requires fvar + STAT (spec-compliant)
    LENIENT: Requires only fvar (technically valid)
    PEDANTIC: Strict + validates axis defaults and STAT structure
    """

    STRICT = "strict"
    LENIENT = "lenient"
    PEDANTIC = "pedantic"


@dataclass
class VariableFontAnalysis:
    """
    Detailed analysis of variable font properties.

    Provides comprehensive information about a font's variable
    characteristics for both detection and diagnostics.
    """

    is_variable: bool
    has_fvar: bool
    has_stat: bool
    has_design_axis_record: bool = False
    has_avar: bool = False
    has_mvar: bool = False
    axis_count: int = 0
    axes: List[str] = None  # List of axis tags
    instance_count: int = 0
    issues: List[str] = None  # List of detected issues

    def __post_init__(self):
        """Set defaults and extract stack trace."""
        if self.axes is None:
            self.axes = []
        if self.issues is None:
            self.issues = []

    @property
    def is_spec_compliant(self) -> bool:
        """Check if font meets OpenType variable font spec."""
        return self.has_fvar and self.has_stat

    @property
    def is_technically_valid(self) -> bool:
        """Check if font is technically a variable font (has fvar)."""
        return self.has_fvar

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "is_variable": self.is_variable,
            "is_spec_compliant": self.is_spec_compliant,
            "is_technically_valid": self.is_technically_valid,
            "has_fvar": self.has_fvar,
            "has_stat": self.has_stat,
            "has_design_axis_record": self.has_design_axis_record,
            "has_avar": self.has_avar,
            "has_mvar": self.has_mvar,
            "axis_count": self.axis_count,
            "axes": self.axes,
            "instance_count": self.instance_count,
            "issues": self.issues,
        }


def _has_design_axis_record(font: Any, has_stat: bool) -> bool:
    """True when STAT has a non-empty DesignAxisRecord (required for AxisValues)."""
    if not has_stat:
        return False
    try:
        stat_table = font["STAT"].table
        return (
            hasattr(stat_table, "DesignAxisRecord")
            and stat_table.DesignAxisRecord is not None
            and len(stat_table.DesignAxisRecord.Axis) > 0
        )
    except Exception as e:
        logger.debug(f"Error checking STAT DesignAxisRecord: {e}")
        return False


def _has_design_axis_record_ttx(root: Any, has_stat: bool) -> bool:
    """TTX equivalent of _has_design_axis_record."""
    if not has_stat:
        return False
    try:
        stat = root.find(".//STAT")
        if stat is None:
            return False
        design_axis = stat.find(".//DesignAxisRecord")
        if design_axis is None:
            return False
        return len(design_axis.findall(".//Axis")) > 0
    except Exception as e:
        logger.debug(f"Error checking TTX STAT DesignAxisRecord: {e}")
        return False


def _check_table_presence(font: Any, table_tag: str) -> bool:
    """Safely check if font has a table."""
    try:
        return table_tag in font
    except Exception as e:
        logger.debug(f"Error checking for table '{table_tag}': {e}")
        return False


def _extract_axis_info(font: Any) -> tuple[int, List[str], int]:
    """
    Extract axis information from fvar table.

    Returns:
        Tuple of (axis_count, axis_tags, instance_count)
    """
    axis_count = 0
    axis_tags = []
    instance_count = 0

    try:
        if "fvar" in font:
            fvar = font["fvar"]

            # Get axes
            if hasattr(fvar, "axes"):
                axis_count = len(fvar.axes)
                axis_tags = [getattr(axis, "axisTag", "") for axis in fvar.axes]

            # Get instances
            if hasattr(fvar, "instances"):
                instance_count = len(fvar.instances)

    except Exception as e:
        logger.debug(f"Error extracting axis info: {e}")

    return axis_count, axis_tags, instance_count


def _validate_stat_structure(font: Any) -> List[str]:
    """
    Validate STAT table structure (for pedantic mode).

    Returns:
        List of issues found (empty if valid)
    """
    issues = []

    try:
        if "STAT" not in font:
            issues.append("Missing STAT table")
            return issues

        stat = font["STAT"].table

        # Check for DesignAxisRecord
        if not hasattr(stat, "DesignAxisRecord"):
            issues.append("STAT missing DesignAxisRecord")
        elif not hasattr(stat.DesignAxisRecord, "Axis"):
            issues.append("STAT DesignAxisRecord has no Axis records")
        elif not stat.DesignAxisRecord.Axis:
            issues.append("STAT DesignAxisRecord.Axis is empty")

        # Check for AxisValueArray
        if not hasattr(stat, "AxisValueArray"):
            issues.append("STAT missing AxisValueArray")
        elif not hasattr(stat.AxisValueArray, "AxisValue"):
            issues.append("STAT AxisValueArray has no AxisValue records")
        elif not stat.AxisValueArray.AxisValue:
            issues.append("STAT AxisValueArray.AxisValue is empty")

        # Check for ElidedFallbackNameID
        if not hasattr(stat, "ElidedFallbackNameID"):
            issues.append("STAT missing ElidedFallbackNameID (recommended)")

    except Exception as e:
        issues.append(f"Error validating STAT: {e}")

    return issues


def _validate_axis_defaults(font: Any) -> List[str]:
    """
    Validate that fvar axis defaults match STAT values (for pedantic mode).

    Returns:
        List of issues found (empty if valid)
    """
    issues = []

    try:
        if "fvar" not in font or "STAT" not in font:
            return issues

        # Get axis defaults from fvar
        fvar_defaults = {}
        for axis in font["fvar"].axes:
            tag = getattr(axis, "axisTag", None)
            default = getattr(axis, "defaultValue", None)
            if tag and default is not None:
                fvar_defaults[tag] = float(default)

        # Check if STAT has corresponding AxisValue records for defaults
        stat = font["STAT"].table
        if not hasattr(stat, "AxisValueArray") or not stat.AxisValueArray:
            issues.append("STAT has no AxisValue records to match fvar defaults")
            return issues

        # This is a simplified check - full validation would be more complex
        axis_tags_with_values = set()
        for av in stat.AxisValueArray.AxisValue:
            axis_index = getattr(av, "AxisIndex", None)
            if axis_index is not None and hasattr(stat, "DesignAxisRecord"):
                try:
                    axis = stat.DesignAxisRecord.Axis[axis_index]
                    axis_tags_with_values.add(getattr(axis, "AxisTag", ""))
                except (IndexError, AttributeError):
                    pass

        # Check for axes without STAT values
        for tag in fvar_defaults:
            if tag not in axis_tags_with_values:
                issues.append(f"Axis '{tag}' has no AxisValue records in STAT")

    except Exception as e:
        issues.append(f"Error validating axis defaults: {e}")

    return issues


def analyze_variable_font(
    font: Any, mode: VariableFontMode = VariableFontMode.STRICT
) -> VariableFontAnalysis:
    """
    Perform detailed analysis of variable font properties.

    Args:
        font: Font object (TTFont instance)
        mode: Detection mode (determines is_variable result)

    Returns:
        VariableFontAnalysis with detailed information

    Examples:
        >>> analysis = analyze_variable_font(font)
        >>> if analysis.is_variable:
        ...     print(f"{analysis.axis_count} axes: {', '.join(analysis.axes)}")
        >>> if analysis.issues:
        ...     for issue in analysis.issues:
        ...         print(f"WARNING: {issue}")
    """
    # Check table presence
    has_fvar = _check_table_presence(font, "fvar")
    has_stat = _check_table_presence(font, "STAT")
    has_design_axis_record = _has_design_axis_record(font, has_stat)
    has_avar = _check_table_presence(font, "avar")
    has_mvar = _check_table_presence(font, "MVAR")

    # Extract axis information
    axis_count, axes, instance_count = _extract_axis_info(font)

    # Determine if font is variable based on mode
    if mode == VariableFontMode.STRICT:
        is_variable = has_fvar and has_stat
    elif mode == VariableFontMode.LENIENT:
        is_variable = has_fvar
    elif mode == VariableFontMode.PEDANTIC:
        is_variable = has_fvar and has_stat
    else:
        is_variable = False

    # Collect issues
    issues = []

    # Basic validation
    if has_fvar and not has_stat:
        issues.append("Missing STAT table (recommended by OpenType spec)")

    if has_fvar and axis_count == 0:
        issues.append("fvar table exists but has no axes")

    # Pedantic mode: additional validation
    if mode == VariableFontMode.PEDANTIC and is_variable:
        stat_issues = _validate_stat_structure(font)
        issues.extend(stat_issues)

        axis_issues = _validate_axis_defaults(font)
        issues.extend(axis_issues)

    return VariableFontAnalysis(
        is_variable=is_variable,
        has_fvar=has_fvar,
        has_stat=has_stat,
        has_design_axis_record=has_design_axis_record,
        has_avar=has_avar,
        has_mvar=has_mvar,
        axis_count=axis_count,
        axes=axes,
        instance_count=instance_count,
        issues=issues,
    )


def is_variable_font(
    font: Any,
    mode: VariableFontMode = VariableFontMode.STRICT,
) -> bool:
    """
    Check if font is a variable font.

    Args:
        font: Font object (TTFont instance or TTX root element)
        mode: Detection strictness mode

    Returns:
        True if font is variable according to mode

    Examples:
        >>> # Strict mode (default): requires fvar + STAT
        >>> is_variable_font(font)
        True

        >>> # Lenient mode: requires only fvar
        >>> is_variable_font(legacy_font, mode=VariableFontMode.LENIENT)
        True

        >>> # Pedantic mode: strict + validates structure
        >>> is_variable_font(font, mode=VariableFontMode.PEDANTIC)
        False  # if STAT structure has issues
    """
    analysis = analyze_variable_font(font, mode)

    # Log issues in pedantic mode
    if mode == VariableFontMode.PEDANTIC and analysis.issues:
        for issue in analysis.issues:
            logger.warning(f"Variable font validation: {issue}")

    return analysis.is_variable


# TTX-specific functions
def analyze_variable_font_ttx(
    root: Any, mode: VariableFontMode = VariableFontMode.STRICT
) -> VariableFontAnalysis:
    """
    Analyze variable font properties from TTX XML root.

    Args:
        root: TTX XML root element
        mode: Detection mode

    Returns:
        VariableFontAnalysis instance

    Examples:
        >>> tree, root, _ = load_ttx("font.ttx")
        >>> analysis = analyze_variable_font_ttx(root)
        >>> print(f"Variable: {analysis.is_variable}")
    """
    try:
        # Check table presence
        has_fvar = root.find(".//fvar") is not None
        has_stat = root.find(".//STAT") is not None
        has_design_axis_record = _has_design_axis_record_ttx(root, has_stat)
        has_avar = root.find(".//avar") is not None
        has_mvar = root.find(".//MVAR") is not None

        # Extract axis information
        axis_count = 0
        axes = []
        instance_count = 0

        if has_fvar:
            fvar = root.find(".//fvar")
            axis_elements = fvar.findall(".//Axis")
            axis_count = len(axis_elements)

            for axis_el in axis_elements:
                tag_el = axis_el.find("AxisTag")
                if tag_el is not None:
                    tag = tag_el.get("value") or (tag_el.text or "").strip()
                    if tag:
                        axes.append(tag)

            instance_elements = fvar.findall(".//NamedInstance")
            instance_count = len(instance_elements)

        # Determine if variable based on mode
        if mode == VariableFontMode.STRICT:
            is_variable = has_fvar and has_stat
        elif mode == VariableFontMode.LENIENT:
            is_variable = has_fvar
        elif mode == VariableFontMode.PEDANTIC:
            is_variable = has_fvar and has_stat
        else:
            is_variable = False

        # Collect issues
        issues = []

        if has_fvar and not has_stat:
            issues.append("Missing STAT table (recommended by OpenType spec)")

        if has_fvar and axis_count == 0:
            issues.append("fvar table exists but has no axes")

        # Pedantic validation for TTX
        if mode == VariableFontMode.PEDANTIC and is_variable:
            stat = root.find(".//STAT")
            if stat is not None:
                if stat.find(".//DesignAxisRecord") is None:
                    issues.append("STAT missing DesignAxisRecord")
                if stat.find(".//AxisValueArray") is None:
                    issues.append("STAT missing AxisValueArray")
                if stat.find(".//ElidedFallbackNameID") is None:
                    issues.append("STAT missing ElidedFallbackNameID (recommended)")

        return VariableFontAnalysis(
            is_variable=is_variable,
            has_fvar=has_fvar,
            has_stat=has_stat,
            has_design_axis_record=has_design_axis_record,
            has_avar=has_avar,
            has_mvar=has_mvar,
            axis_count=axis_count,
            axes=axes,
            instance_count=instance_count,
            issues=issues,
        )

    except Exception as e:
        logger.error(f"Error analyzing TTX variable font: {e}")
        return VariableFontAnalysis(
            is_variable=False,
            has_fvar=False,
            has_stat=False,
            has_design_axis_record=False,
            issues=[f"Analysis failed: {e}"],
        )


def is_variable_font_ttx(
    root: Any,
    mode: VariableFontMode = VariableFontMode.STRICT,
) -> bool:
    """
    Check if TTX font is variable.

    Args:
        root: TTX XML root element
        mode: Detection strictness mode

    Returns:
        True if font is variable according to mode

    Examples:
        >>> tree, root, _ = load_ttx("font.ttx")
        >>> is_variable_font_ttx(root)
        True
    """
    analysis = analyze_variable_font_ttx(root, mode)

    # Log issues in pedantic mode
    if mode == VariableFontMode.PEDANTIC and analysis.issues:
        for issue in analysis.issues:
            logger.warning(f"Variable font validation: {issue}")

    return analysis.is_variable


# Convenience functions for backward compatibility
def is_variable_font_binary(font: Any, strict: bool = True) -> bool:
    """
    Check if binary font is variable (backward compatible).

    Args:
        font: TTFont instance
        strict: If True, require both fvar and STAT. If False, only fvar.

    Returns:
        True if font is variable

    Examples:
        >>> # Strict mode (default)
        >>> is_variable_font_binary(font)
        True

        >>> # Lenient mode
        >>> is_variable_font_binary(font, strict=False)
        True
    """
    mode = VariableFontMode.STRICT if strict else VariableFontMode.LENIENT
    return is_variable_font(font, mode)


__all__ = [
    "VariableFontMode",
    "VariableFontAnalysis",
    "is_variable_font",
    "analyze_variable_font",
    "is_variable_font_ttx",
    "analyze_variable_font_ttx",
    "is_variable_font_binary",  # Backward compatibility
]
