#!/usr/bin/env python3
"""
Font Variation Analyzer - Compare fonts with the same PostScript name

Analyzes fonts that share a PostScript name but have different content.
Shows detailed comparisons to help identify meaningful differences.

Purpose:
    When you have multiple files with the same PS name that aren't byte-for-byte
    identical (SHA256 different), this tool shows you WHERE they differ so you
    can decide which version to keep.

Data Points Analyzed:
    - Font type (TTF, OTF, WOFF, WOFF2)
    - Static vs Variable font
    - Version number (fontRevision)
    - Creation date
    - Glyph count
    - Table inventory (count and list)
    - OpenType feature count
    - Kern table presence
    - Metrics summary
    - File size

Usage:
    python FontFiles_Variation_Analyzer.py /path/to/fonts/
    python FontFiles_Variation_Analyzer.py font1.otf font2.otf
    python FontFiles_Variation_Analyzer.py /directory/ --recursive
    python FontFiles_Variation_Analyzer.py /directory/ --verbose

Options:
    -r, --recursive      Process directories recursively
    --verbose, -V        Show detailed table lists and extra information
    --single, -s         Include single files (no variations to compare)
    --min-diff N         Only show groups with at least N differences (default: 1)
"""

import argparse
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from fontTools.ttLib import TTFont

import core.core_console_styles as cs
from core.core_file_collector import collect_font_files

console = cs.get_console()

# Rich is required for tables
try:
    from rich.table import Table
    from rich.text import Text

    cs.RICH_AVAILABLE = True
except ImportError:
    cs.RICH_AVAILABLE = False

# ============================================================================
# Constants
# ============================================================================

FONT_EXTENSIONS = {".ttf", ".otf", ".woff", ".woff2"}
DEFAULT_TRASH_DIR = Path.home() / ".Trash" / "FontDeduplicator"

# Tables that define core font content (excluding metadata)
CORE_CONTENT_TABLES = [
    # Glyph outlines
    "glyf",  # TrueType outlines
    "CFF ",  # PostScript Type 1 outlines
    "CFF2",  # PostScript Type 2 outlines
    # OpenType features
    "GSUB",  # Glyph substitution
    "GPOS",  # Glyph positioning
    "GDEF",  # Glyph definitions
    "BASE",  # Baseline data
    # Character mapping
    "cmap",  # Character to glyph mapping
    # Color/SVG
    "svg ",  # SVG glyphs
    "COLR",  # Color layers
    "CPAL",  # Color palettes
    "sbix",  # Apple bitmap
    "CBDT",  # Color bitmap data
    "CBLC",  # Color bitmap location
    # Additional layout
    "JSTF",  # Justification
    "MATH",  # Math layout
]

# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class FontAnalysis:
    """Detailed analysis of a single font file"""

    file_path: Path
    ps_name: str

    # Basic info
    font_type: str  # TTF, OTF, WOFF, WOFF2
    is_variable: bool

    # Version info
    font_revision: float
    version_string: str
    created_date: Optional[float]
    modified_date: Optional[float]

    # Structure
    glyph_count: int
    table_count: int
    table_list: List[str]

    # Features
    has_gsub: bool
    has_gpos: bool
    gsub_feature_count: int
    gpos_feature_count: int
    has_kern: bool

    # Metrics
    units_per_em: int
    ascent: int
    descent: int
    line_gap: int

    # File info
    file_size: int

    # Content hash (optional)
    content_hash: Optional[str] = None

    # Error tracking
    extraction_errors: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.file_path.name

    @property
    def total_features(self) -> int:
        return self.gsub_feature_count + self.gpos_feature_count

    @property
    def created_datetime(self) -> Optional[str]:
        if self.created_date:
            try:
                dt = datetime.fromtimestamp(self.created_date)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                return None
        return None


@dataclass
class VariationGroup:
    """A group of fonts with the same PostScript name"""

    ps_name: str
    fonts: List[FontAnalysis]

    @property
    def has_variations(self) -> bool:
        return len(self.fonts) > 1

    def _count_differences_between_fonts(
        self, f1: FontAnalysis, f2: FontAnalysis
    ) -> int:
        """Count differences between two font analyses"""
        diffs = 0
        if f1.font_type != f2.font_type:
            diffs += 1
        if f1.is_variable != f2.is_variable:
            diffs += 1
        if f1.font_revision != f2.font_revision:
            diffs += 1
        if f1.glyph_count != f2.glyph_count:
            diffs += 1
        if f1.table_count != f2.table_count:
            diffs += 1
        if f1.total_features != f2.total_features:
            diffs += 1
        if f1.has_kern != f2.has_kern:
            diffs += 1
        if f1.file_size != f2.file_size:
            diffs += 1
        return diffs

    @property
    def difference_count(self) -> int:
        """Count how many attributes differ across the group"""
        if not self.has_variations:
            return 0

        differences = 0
        first = self.fonts[0]

        for font in self.fonts[1:]:
            differences += self._count_differences_between_fonts(first, font)

        return differences


@dataclass
class AnalysisStats:
    """Statistics for the analysis run"""

    total_files: int = 0
    valid_fonts: int = 0
    invalid_fonts: int = 0
    unique_ps_names: int = 0
    variation_groups: int = 0
    single_fonts: int = 0


# ============================================================================
# Font Analysis
# ============================================================================


def detect_font_type(font: TTFont) -> str:
    """Detect font type from sfntVersion"""
    try:
        sfnt_version = font.reader.sfntVersion
        if sfnt_version == b"wOFF":
            return "WOFF"
        elif sfnt_version == b"wOF2":
            return "WOFF2"
        elif sfnt_version == b"\x00\x01\x00\x00" or sfnt_version == b"true":
            return "TTF"
        elif sfnt_version == b"OTTO":
            return "OTF"
        else:
            return f"Unknown ({sfnt_version})"
    except Exception:
        return "Unknown"


def is_variable_font(font: TTFont) -> bool:
    """Check if font is a variable font"""
    return "fvar" in font


def get_feature_count(font: TTFont, table_tag: str) -> int:
    """Get feature count from GSUB or GPOS table"""
    try:
        if table_tag in font:
            table = font[table_tag]
            if hasattr(table, "table") and hasattr(table.table, "FeatureList"):
                return table.table.FeatureList.FeatureCount
    except Exception:
        pass
    return 0


def extract_version_info(
    font: TTFont, errors: List[str]
) -> Tuple[float, str, Optional[float], Optional[float]]:
    """Extract version information from font"""
    font_revision = 0.0
    version_string = ""
    created_date = None
    modified_date = None

    try:
        if "head" in font:
            head = font["head"]
            font_revision = head.fontRevision if hasattr(head, "fontRevision") else 0.0
            created_date = head.created if hasattr(head, "created") else None
            modified_date = head.modified if hasattr(head, "modified") else None
    except Exception as e:
        errors.append(f"Failed to read head table: {e}")

    try:
        if "name" in font:
            ver_record = font["name"].getName(5, 3, 1, 0x409)
            version_string = ver_record.toUnicode() if ver_record else ""
    except Exception as e:
        errors.append(f"Failed to read version string: {e}")

    return font_revision, version_string, created_date, modified_date


def extract_structure_info(
    font: TTFont, errors: List[str]
) -> Tuple[int, int, List[str]]:
    """Extract structure information from font"""
    glyph_count = 0
    try:
        if "maxp" in font:
            glyph_count = font["maxp"].numGlyphs
    except Exception as e:
        errors.append(f"Failed to read glyph count: {e}")

    table_list = sorted(font.keys())
    table_count = len(table_list)

    return glyph_count, table_count, table_list


def extract_feature_info(font: TTFont) -> Tuple[bool, bool, int, int, bool]:
    """Extract OpenType feature information from font"""
    has_gsub = "GSUB" in font
    has_gpos = "GPOS" in font
    gsub_feature_count = get_feature_count(font, "GSUB")
    gpos_feature_count = get_feature_count(font, "GPOS")
    has_kern = "kern" in font

    return has_gsub, has_gpos, gsub_feature_count, gpos_feature_count, has_kern


def extract_metrics_info(font: TTFont, errors: List[str]) -> Tuple[int, int, int, int]:
    """Extract metrics information from font"""
    units_per_em = 0
    ascent = 0
    descent = 0
    line_gap = 0

    try:
        if "head" in font:
            units_per_em = font["head"].unitsPerEm
    except Exception as e:
        errors.append(f"Failed to read units per em: {e}")

    try:
        if "hhea" in font:
            hhea = font["hhea"]
            ascent = hhea.ascent if hasattr(hhea, "ascent") else 0
            descent = hhea.descent if hasattr(hhea, "descent") else 0
            line_gap = hhea.lineGap if hasattr(hhea, "lineGap") else 0
    except Exception as e:
        errors.append(f"Failed to read metrics: {e}")

    return units_per_em, ascent, descent, line_gap


def compute_content_hash(file_path: Path) -> Optional[str]:
    """
    Compute SHA256 hash of only core content tables (excluding metadata).
    Returns None on failure.
    """
    import hashlib

    try:
        font = TTFont(str(file_path), lazy=True)
        content_hash = hashlib.sha256()

        # Hash only core content tables in sorted order
        for table_tag in sorted(font.keys()):
            if table_tag in CORE_CONTENT_TABLES:
                try:
                    if (
                        hasattr(font, "reader")
                        and font.reader
                        and table_tag in font.reader
                    ):
                        table_data = font.reader[table_tag]
                        content_hash.update(table_data)
                    elif hasattr(font, "getTableData"):
                        table_data = font.getTableData(table_tag)
                        if table_data:
                            content_hash.update(table_data)
                except Exception:
                    continue

        font.close()
        return content_hash.hexdigest()
    except Exception:
        return None


def analyze_font(font_path: Path) -> Optional[FontAnalysis]:
    """Extract detailed analysis from a font file"""
    errors = []

    try:
        font = TTFont(str(font_path))
    except Exception as e:
        if console:
            cs.StatusIndicator("error").add_file(font_path.name).with_explanation(
                f"Failed to open: {e}"
            ).emit()
        return None

    try:
        # Basic info
        font_type = detect_font_type(font)
        is_var = is_variable_font(font)

        # PostScript name
        ps_name = ""
        try:
            name_record = font["name"].getName(6, 3, 1, 0x409)
            ps_name = name_record.toUnicode() if name_record else font_path.stem
        except Exception as e:
            ps_name = font_path.stem
            errors.append(f"Failed to read PS name: {e}")

        # Extract various info sections
        font_revision, version_string, created_date, modified_date = (
            extract_version_info(font, errors)
        )
        glyph_count, table_count, table_list = extract_structure_info(font, errors)
        has_gsub, has_gpos, gsub_feature_count, gpos_feature_count, has_kern = (
            extract_feature_info(font)
        )
        units_per_em, ascent, descent, line_gap = extract_metrics_info(font, errors)

        # File info
        file_size = font_path.stat().st_size

        font.close()

        # Compute content hash if requested (check global flag)
        content_hash = None
        if (
            hasattr(analyze_font, "_compute_content_hash")
            and analyze_font._compute_content_hash
        ):
            content_hash = compute_content_hash(font_path)

        return FontAnalysis(
            file_path=font_path,
            ps_name=ps_name,
            font_type=font_type,
            is_variable=is_var,
            font_revision=font_revision,
            version_string=version_string,
            created_date=created_date,
            modified_date=modified_date,
            glyph_count=glyph_count,
            table_count=table_count,
            table_list=table_list,
            has_gsub=has_gsub,
            has_gpos=has_gpos,
            gsub_feature_count=gsub_feature_count,
            gpos_feature_count=gpos_feature_count,
            has_kern=has_kern,
            units_per_em=units_per_em,
            ascent=ascent,
            descent=descent,
            line_gap=line_gap,
            file_size=file_size,
            content_hash=content_hash,
            extraction_errors=errors,
        )

    except Exception as e:
        if console:
            cs.StatusIndicator("error").add_file(font_path.name).with_explanation(
                f"Failed to analyze: {e}"
            ).emit()
        try:
            font.close()
        except Exception:
            pass
        return None


# ============================================================================
# Comparison & Reporting
# ============================================================================


def format_file_size(size: int) -> str:
    """Format byte size as human-readable string"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def get_unique_values(fonts: List[FontAnalysis], attr: str) -> Set:
    """Get set of unique values for an attribute across fonts"""
    return {getattr(font, attr) for font in fonts}


def has_difference(fonts: List[FontAnalysis], attr: str) -> bool:
    """Check if fonts differ on this attribute"""
    return len(get_unique_values(fonts, attr)) > 1


def only_file_size_differs(group: VariationGroup) -> bool:
    """
    Check if only file size differs between fonts in the group.
    Returns True if file size is the only difference.
    """
    if not group.has_variations or len(group.fonts) < 2:
        return False

    first = group.fonts[0]

    # Check all attributes except file_size
    checks = [
        ("font_type", lambda f: f.font_type),
        ("is_variable", lambda f: f.is_variable),
        ("font_revision", lambda f: f.font_revision),
        ("created_date", lambda f: f.created_date),
        ("glyph_count", lambda f: f.glyph_count),
        ("table_count", lambda f: f.table_count),
        ("total_features", lambda f: f.total_features),
        ("gsub_feature_count", lambda f: f.gsub_feature_count),
        ("gpos_feature_count", lambda f: f.gpos_feature_count),
        ("has_kern", lambda f: f.has_kern),
        ("units_per_em", lambda f: f.units_per_em),
        ("ascent", lambda f: f.ascent),
        ("descent", lambda f: f.descent),
        ("line_gap", lambda f: f.line_gap),
    ]

    # All other attributes must be identical
    for attr_name, attr_getter in checks:
        first_val = attr_getter(first)
        for font in group.fonts[1:]:
            if attr_getter(font) != first_val:
                return False

    # File size must differ
    file_sizes = {f.file_size for f in group.fonts}
    return len(file_sizes) > 1


def count_differences_in_group(
    group: VariationGroup, verbose: bool = False
) -> Tuple[int, int]:
    """
    Count how many attributes differ vs are identical in a group.
    Returns: (differences_count, identical_count)
    """
    if not group.has_variations:
        return 0, 0

    # List of all attributes we check (matching what's in add_table_rows functions)
    attributes_to_check = [
        ("Type", lambda f: f.font_type),
        ("Variable", lambda f: "Yes" if f.is_variable else "No"),
        ("Revision", lambda f: f"{f.font_revision:.3f}"),
        ("Created", lambda f: f.created_datetime or "Unknown"),
        ("Glyphs", lambda f: str(f.glyph_count)),
        ("Tables", lambda f: str(f.table_count)),
        ("Total Features", lambda f: str(f.total_features)),
        ("GSUB Features", lambda f: str(f.gsub_feature_count) if f.has_gsub else "—"),
        ("GPOS Features", lambda f: str(f.gpos_feature_count) if f.has_gpos else "—"),
        ("Kern Table", lambda f: "Yes" if f.has_kern else "No"),
        ("Units/EM", lambda f: str(f.units_per_em)),
        ("Ascent", lambda f: str(f.ascent)),
        ("Descent", lambda f: str(f.descent)),
        ("Line Gap", lambda f: str(f.line_gap)),
        ("File Size", lambda f: format_file_size(f.file_size)),
    ]

    # Add verbose-only attributes
    if verbose:
        # Table list is verbose-only, but we don't count it as a standard attribute
        pass

    differences = 0
    identical = 0

    for attr_name, attr_getter in attributes_to_check:
        values = [attr_getter(f) for f in group.fonts]
        if len(set(values)) > 1:
            differences += 1
        else:
            identical += 1

    return differences, identical


def add_table_rows_basic(table: Table, group: VariationGroup, add_row_func):
    """Add basic info rows to comparison table"""
    add_row_func("Type", [f.font_type for f in group.fonts])
    add_row_func("Variable", ["Yes" if f.is_variable else "No" for f in group.fonts])
    add_row_func("Revision", [f"{f.font_revision:.3f}" for f in group.fonts])
    add_row_func("Created", [f.created_datetime or "Unknown" for f in group.fonts])


def add_table_rows_structure(
    table: Table, group: VariationGroup, add_row_func, verbose: bool
):
    """Add structure info rows to comparison table"""
    add_row_func("Glyphs", [str(f.glyph_count) for f in group.fonts])
    add_row_func("Tables", [str(f.table_count) for f in group.fonts])

    if verbose:
        table_lists = [
            ", ".join(f.table_list[:10]) + ("..." if len(f.table_list) > 10 else "")
            for f in group.fonts
        ]
        add_row_func("Table List", table_lists, highlight_diff=False)


def add_table_rows_features(table: Table, group: VariationGroup, add_row_func):
    """Add feature info rows to comparison table"""
    add_row_func("Total Features", [str(f.total_features) for f in group.fonts])
    add_row_func(
        "GSUB Features",
        [str(f.gsub_feature_count) if f.has_gsub else "—" for f in group.fonts],
    )
    add_row_func(
        "GPOS Features",
        [str(f.gpos_feature_count) if f.has_gpos else "—" for f in group.fonts],
    )
    add_row_func("Kern Table", ["Yes" if f.has_kern else "No" for f in group.fonts])


def add_table_rows_metrics(table: Table, group: VariationGroup, add_row_func):
    """Add metrics info rows to comparison table"""
    add_row_func("Units/EM", [str(f.units_per_em) for f in group.fonts])
    add_row_func("Ascent", [str(f.ascent) for f in group.fonts])
    add_row_func("Descent", [str(f.descent) for f in group.fonts])
    add_row_func("Line Gap", [str(f.line_gap) for f in group.fonts])
    add_row_func("File Size", [format_file_size(f.file_size) for f in group.fonts])


def create_comparison_table(
    group: VariationGroup, verbose: bool = False
) -> Optional[Table]:
    """Create a Rich table comparing fonts in the group - shows only differences"""
    if not cs.RICH_AVAILABLE or not console:
        return None

    table = Table(
        title=f"[bold cyan]{group.ps_name}[/bold cyan] ({len(group.fonts)} variants)",
        show_header=True,
        header_style="bold magenta",
        border_style="blue",
    )

    # Add columns
    table.add_column("Attribute", style="yellow", no_wrap=True)
    for font in group.fonts:
        # Truncate long filenames
        name = font.name
        if len(name) > 30:
            name = name[:27] + "..."
        table.add_column(name, style="cyan")

    # Track statistics
    rows_added = 0
    rows_skipped = 0

    # Helper to add row only when values differ
    def add_row(label: str, values: List[str], highlight_diff: bool = True):
        nonlocal rows_added, rows_skipped
        has_differences = len(set(values)) > 1

        if has_differences:
            # Only add rows with differences
            if highlight_diff:
                # Highlight differences
                styled_values = [Text(v, style="bold green") for v in values]
                table.add_row(label, *styled_values)
            else:
                table.add_row(label, *values)
            rows_added += 1
        else:
            rows_skipped += 1

    # Add all rows organized by category (only differences will be added)
    add_table_rows_basic(table, group, add_row)
    add_table_rows_structure(table, group, add_row, verbose)
    add_table_rows_features(table, group, add_row)
    add_table_rows_metrics(table, group, add_row)

    # If no differences found, return None (caller will show message)
    if rows_added == 0:
        return None

    return table


def show_variation_summary(group: VariationGroup):
    """Show summary of variations detected"""
    type_values = get_unique_values(group.fonts, "font_type")
    revision_values = get_unique_values(group.fonts, "font_revision")
    glyph_values = get_unique_values(group.fonts, "glyph_count")

    if len(type_values) > 1 or len(revision_values) > 1 or len(glyph_values) > 1:
        if console:
            indicator = cs.StatusIndicator("info").add_message("Variations detected")
            if len(type_values) > 1:
                indicator.add_item(f"Types: {', '.join(sorted(type_values))}")
            if len(revision_values) > 1:
                revisions = sorted(revision_values)
                indicator.add_item(
                    f"Revisions: {revisions[0]:.3f} to {revisions[-1]:.3f}"
                )
            if len(glyph_values) > 1:
                indicator.add_item(
                    f"Glyph counts: {min(glyph_values)} to {max(glyph_values)}"
                )
            indicator.emit()


def get_font_differences(font: FontAnalysis, reference: FontAnalysis) -> List[str]:
    """Get comprehensive list of differences between font and reference"""
    diffs = []
    if font.font_type != reference.font_type:
        diffs.append(f"type:{font.font_type}")
    if font.is_variable != reference.is_variable:
        diffs.append(f"variable:{'Yes' if font.is_variable else 'No'}")
    if font.font_revision != reference.font_revision:
        diffs.append(f"rev:{font.font_revision:.3f}")
    if font.created_date != reference.created_date:
        created_str = font.created_datetime or "Unknown"
        diffs.append(f"created:{created_str}")
    if font.glyph_count != reference.glyph_count:
        diffs.append(f"glyphs:{font.glyph_count}")
    if font.table_count != reference.table_count:
        diffs.append(f"tables:{font.table_count}")
    if font.total_features != reference.total_features:
        diffs.append(f"features:{font.total_features}")
    if font.gsub_feature_count != reference.gsub_feature_count:
        diffs.append(f"gsub:{font.gsub_feature_count}")
    if font.gpos_feature_count != reference.gpos_feature_count:
        diffs.append(f"gpos:{font.gpos_feature_count}")
    if font.has_kern != reference.has_kern:
        diffs.append(f"kern:{'Yes' if font.has_kern else 'No'}")
    if font.units_per_em != reference.units_per_em:
        diffs.append(f"units/em:{font.units_per_em}")
    if font.ascent != reference.ascent:
        diffs.append(f"ascent:{font.ascent}")
    if font.descent != reference.descent:
        diffs.append(f"descent:{font.descent}")
    if font.line_gap != reference.line_gap:
        diffs.append(f"linegap:{font.line_gap}")
    if font.file_size != reference.file_size:
        diffs.append(f"size:{format_file_size(font.file_size)}")
    return diffs


def detect_likely_different_fonts(group: VariationGroup) -> bool:
    """
    Detect if a group likely contains different fonts (not just versions).

    Indicators:
    - Large glyph count variance (>20% difference)
    - Many fonts with different base names in filename
    """
    glyph_counts = [f.glyph_count for f in group.fonts]
    min_glyphs = min(glyph_counts)
    max_glyphs = max(glyph_counts)

    # If glyph count variance is >20%, likely different fonts
    if min_glyphs > 0:
        variance = (max_glyphs - min_glyphs) / min_glyphs
        if variance > 0.20:  # 20% variance
            return True

    return False


def extract_base_name(filename: str) -> str:
    """
    Extract base name from filename for grouping.
    Examples:
      FaxnhamDisplay-Black.woff2 -> FaxnhamDisplay-Black
      FaxnhamText-Bold~001.woff2 -> FaxnhamText-Bold
    """
    # Remove extension
    name = Path(filename).stem

    # Remove ~001 style suffixes
    name = re.sub(r"~\d{3}$", "", name)

    # Remove " copy" and similar
    name = re.sub(r"\s*copy\s*\d*$", "", name, flags=re.IGNORECASE)

    return name


def subgroup_by_similarity(group: VariationGroup) -> Dict[str, List[FontAnalysis]]:
    """
    Sub-group fonts by base name similarity when they're likely different fonts.
    """
    subgroups: Dict[str, List[FontAnalysis]] = {}

    for font in group.fonts:
        base_name = extract_base_name(font.name)
        if base_name not in subgroups:
            subgroups[base_name] = []
        subgroups[base_name].append(font)

    return subgroups


def show_subgroup_single_font(font: FontAnalysis):
    """Display a single font in a subgroup"""
    if console:
        cs.StatusIndicator("info").add_file(font.name).add_item(
            f"Glyphs: {font.glyph_count} | Type: {font.font_type} | Size: {format_file_size(font.file_size)}"
        ).emit()


def show_subgroup_multiple_fonts(base_name: str, fonts: List[FontAnalysis]):
    """Display multiple fonts in a subgroup"""
    if console:
        indicator = cs.StatusIndicator("info").add_message(
            f"{base_name} ({len(fonts)} variants)"
        )
        for idx, font in enumerate(fonts, 1):
            if idx == 1:
                indicator.add_item(
                    f"#{idx}: {cs.fmt_file(font.name)} (reference)", style="dim"
                )
            else:
                diffs = get_font_differences(font, fonts[0])
                if diffs:
                    diff_str = " | ".join(diffs)
                    indicator.add_item(f"#{idx}: {cs.fmt_file(font.name)} → {diff_str}")
                else:
                    indicator.add_item(
                        f"#{idx}: {cs.fmt_file(font.name)} (identical)", style="dim"
                    )
        indicator.emit()


def show_subgrouped_comparison(group: VariationGroup):
    """Show comparison for likely different fonts grouped together"""
    if console:
        cs.StatusIndicator("warning").add_message(
            "Large variation detected - likely different fonts with same PS name!"
        ).add_item("Grouping by filename similarity...", style="dim").emit()

    subgroups = subgroup_by_similarity(group)

    for base_name, fonts in sorted(subgroups.items()):
        if len(fonts) == 1:
            show_subgroup_single_font(fonts[0])
        else:
            show_subgroup_multiple_fonts(base_name, fonts)


def show_normal_vertical_list(group: VariationGroup):
    """Show normal vertical list for actual variations - shows only differences"""
    if console:
        cs.StatusIndicator("info").add_message(
            "Too many variants for horizontal table - showing vertical list",
            style="dim",
        ).emit()

    # Check if all fonts are identical
    first = group.fonts[0]
    all_identical = True
    for font in group.fonts[1:]:
        if get_font_differences(font, first):
            all_identical = False
            break

    if all_identical:
        if console:
            cs.StatusIndicator("unchanged").add_message(
                f"{group.ps_name}: All attributes identical - files are duplicates"
            ).emit()
        return

    show_variation_summary(group)

    first = group.fonts[0]
    for idx, font in enumerate(group.fonts, 1):
        if console:
            if idx == 1:
                cs.StatusIndicator("info").add_file(font.name).add_message(
                    f"#{idx} (reference)", style="dim"
                ).emit()
            else:
                diffs = get_font_differences(font, first)
                if diffs:
                    diff_str = " | ".join(diffs)
                    cs.StatusIndicator("updated").add_file(font.name).add_message(
                        f"#{idx} → {diff_str}"
                    ).emit()
                else:
                    cs.StatusIndicator("unchanged").add_file(font.name).add_message(
                        f"#{idx} (same as reference)", style="dim"
                    ).emit()


def show_vertical_comparison(group: VariationGroup):
    """Vertical comparison for groups with many variants (readable for large groups)"""
    if console:
        cs.StatusIndicator("info").add_message(
            f"{group.ps_name} ({cs.fmt_count(len(group.fonts))} variants)"
        ).emit()

    # Check if this is likely multiple different fonts with same/broken PS name
    if detect_likely_different_fonts(group):
        show_subgrouped_comparison(group)
    else:
        show_normal_vertical_list(group)


def detect_content_identical_group(group: VariationGroup) -> bool:
    """
    Check if all fonts in group have identical content hashes.
    Returns True if content_hash exists and all match.
    """
    if not group.has_variations:
        return False

    # Check if all have content_hash
    if not all(f.content_hash for f in group.fonts):
        return False

    # Check if all match
    first_hash = group.fonts[0].content_hash
    return all(f.content_hash == first_hash for f in group.fonts)


def show_text_comparison(group: VariationGroup, verbose: bool = False):
    """Fallback text-based comparison when Rich not available - shows only differences"""
    if console:
        cs.StatusIndicator("info").add_message(
            f"{group.ps_name} ({len(group.fonts)} variants)"
        ).emit()

    # Check if all fonts are identical
    if len(group.fonts) > 1:
        first = group.fonts[0]
        all_identical = True
        for font in group.fonts[1:]:
            if get_font_differences(font, first):
                all_identical = False
                break

        if all_identical:
            if console:
                cs.StatusIndicator("unchanged").add_message(
                    "All attributes identical - files are duplicates"
                ).emit()
            return

    # Show differences only
    first = group.fonts[0]
    for idx, font in enumerate(group.fonts, 1):
        if console:
            if idx == 1:
                cs.StatusIndicator("info").add_file(font.name).add_message(
                    f"#{idx} (reference)", style="dim"
                ).emit()
            else:
                diffs = get_font_differences(font, first)
                if diffs:
                    diff_str = " | ".join(diffs)
                    cs.StatusIndicator("updated").add_file(font.name).add_message(
                        f"#{idx} → {diff_str}"
                    ).emit()
                else:
                    cs.StatusIndicator("unchanged").add_file(font.name).add_message(
                        f"#{idx} (same as reference)", style="dim"
                    ).emit()

        if verbose and font.extraction_errors:
            if console:
                cs.StatusIndicator("warning").add_file(font.name).add_item(
                    f"Errors: {', '.join(font.extraction_errors)}"
                ).emit()


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


# ============================================================================
# Deduplication: Grouping and Selection Strategies
# ============================================================================


@dataclass
class DeduplicationStats:
    """Statistics for deduplication operations"""

    files_moved: int = 0
    bytes_saved: int = 0
    errors: List[Tuple[str, str]] = field(default_factory=list)

    def add_error(self, filename: str, reason: str):
        self.errors.append((filename, reason))


def group_variations_by_difference_type(
    variation_groups: List[VariationGroup],
) -> Tuple[List[VariationGroup], List[VariationGroup]]:
    """
    Group variation groups by difference type.
    Returns: (file_size_only_groups, multiple_differences_groups)
    """
    file_size_only = []
    multiple_differences = []

    for group in variation_groups:
        if only_file_size_differs(group):
            file_size_only.append(group)
        else:
            multiple_differences.append(group)

    return file_size_only, multiple_differences


def select_file_to_keep(
    fonts: List[FontAnalysis], strategy: str
) -> Tuple[FontAnalysis, List[FontAnalysis]]:
    """
    Select which file to keep based on strategy.
    Returns: (file_to_keep, files_to_remove)
    """
    if len(fonts) < 2:
        return fonts[0], []

    if strategy == "oldest_creation":
        # Keep file with oldest creation date
        sorted_fonts = sorted(
            fonts,
            key=lambda f: f.created_date if f.created_date else float("inf"),
        )
    elif strategy == "newest_revision":
        # Keep file with highest font revision
        sorted_fonts = sorted(fonts, key=lambda f: f.font_revision, reverse=True)
    elif strategy == "most_glyphs":
        # Keep file with most glyphs
        sorted_fonts = sorted(fonts, key=lambda f: f.glyph_count, reverse=True)
    elif strategy == "more_tables":
        # Keep file with more tables
        sorted_fonts = sorted(fonts, key=lambda f: f.table_count, reverse=True)
    elif strategy == "most_features":
        # Keep file with most features
        sorted_fonts = sorted(fonts, key=lambda f: f.total_features, reverse=True)
    elif strategy == "largest_size":
        # Keep file with largest file size
        sorted_fonts = sorted(fonts, key=lambda f: f.file_size, reverse=True)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return sorted_fonts[0], sorted_fonts[1:]


def select_smallest_files(
    fonts: List[FontAnalysis],
) -> Tuple[List[FontAnalysis], List[FontAnalysis]]:
    """
    For file size only differences, select smallest files to remove.
    Returns: (files_to_keep, files_to_remove)
    """
    if len(fonts) < 2:
        return fonts, []

    # Find minimum file size
    min_size = min(f.file_size for f in fonts)

    # Keep files that are NOT the smallest
    files_to_keep = [f for f in fonts if f.file_size > min_size]
    files_to_remove = [f for f in fonts if f.file_size == min_size]

    # If all files are same size (shouldn't happen, but handle it)
    if not files_to_keep:
        return fonts[:1], fonts[1:]

    return files_to_keep, files_to_remove


def show_interactive_menu_multiple_differences() -> Optional[str]:
    """
    Show interactive menu for multiple differences groups.
    Returns selected strategy or None if quit.
    """
    if not console:
        return None

    print("\n" + "=" * 70)
    print("Multiple differences detected. Choose selection strategy:")
    print("=" * 70)
    print("1. Keep oldest creation date")
    print("2. Keep newest revision")
    print("3. Keep most glyphs")
    print("4. Keep more tables")
    print("5. Keep most features")
    print("6. Keep largest file size")
    print("7. Skip (don't remove anything)")
    print("8. Quit (exit deduplication process)")
    print("=" * 70)

    while True:
        try:
            choice = input("\nEnter choice (1-8): ").strip()
            if choice == "1":
                return "oldest_creation"
            elif choice == "2":
                return "newest_revision"
            elif choice == "3":
                return "most_glyphs"
            elif choice == "4":
                return "more_tables"
            elif choice == "5":
                return "most_features"
            elif choice == "6":
                return "largest_size"
            elif choice == "7":
                return "skip"
            elif choice == "8":
                return None  # Quit
            else:
                print("Invalid choice. Please enter 1-8.")
        except (EOFError, KeyboardInterrupt):
            return None  # Quit on Ctrl+C or EOF


def prompt_file_size_only() -> Optional[bool]:
    """
    Prompt user for file size only differences.
    Returns True to remove smallest, False to skip, None to quit.
    """
    if not console:
        return False

    while True:
        try:
            response = input("\nRemove smallest file size? (y/n/q): ").strip().lower()
            if response == "y":
                return True
            elif response == "n":
                return False
            elif response == "q":
                return None  # Quit
            else:
                print("Invalid response. Please enter y, n, or q.")
        except (EOFError, KeyboardInterrupt):
            return None  # Quit on Ctrl+C or EOF


def show_preview(
    groups: List[VariationGroup],
    files_to_remove: List[FontAnalysis],
    strategy_name: str,
) -> None:
    """Show preview of files that will be removed"""
    if not console:
        return

    total_bytes = sum(f.file_size for f in files_to_remove)
    cs.StatusIndicator("info").add_message(
        f"Preview: {len(files_to_remove)} file(s) will be removed ({format_file_size(total_bytes)})"
    ).add_item(f"Strategy: {strategy_name}", style="dim").emit()

    for group in groups:
        group_files_to_remove = [f for f in files_to_remove if f in group.fonts]
        if group_files_to_remove:
            if console:
                cs.StatusIndicator("warning").add_message(
                    f"{group.ps_name}: Removing {len(group_files_to_remove)} file(s)"
                ).emit()
            for font in group_files_to_remove:
                if console:
                    cs.StatusIndicator("info").add_file(font.name).add_item(
                        f"Size: {format_file_size(font.file_size)}", style="dim"
                    ).emit()


def confirm_action() -> Optional[bool]:
    """
    Confirm before taking action.
    Returns True to proceed, False to skip, None to quit.
    """
    if not console:
        return True

    while True:
        try:
            response = input("\nProceed with removal? (y/n/q): ").strip().lower()
            if response == "y":
                return True
            elif response == "n":
                return False
            elif response == "q":
                return None  # Quit
            else:
                print("Invalid response. Please enter y, n, or q.")
        except (EOFError, KeyboardInterrupt):
            return None  # Quit on Ctrl+C or EOF


def process_file_size_only_groups(
    groups: List[VariationGroup],
    trash_dir: Path,
    dry_run: bool,
    dedup_stats: DeduplicationStats,
) -> Optional[bool]:
    """
    Process file size only groups.
    Returns True if successful, False if skipped, None if quit.
    """
    if not groups:
        return True

    if console:
        cs.StatusIndicator("info").add_message(
            f"Processing {len(groups)} group(s) with file size differences only"
        ).emit()

    # Collect all files to remove
    all_files_to_remove = []
    for group in groups:
        _, files_to_remove = select_smallest_files(group.fonts)
        all_files_to_remove.extend(files_to_remove)

    if not all_files_to_remove:
        if console:
            cs.StatusIndicator("info").add_message("No files to remove").emit()
        return True

    # Show preview
    show_preview(groups, all_files_to_remove, "Remove smallest file size")

    # Prompt user
    response = prompt_file_size_only()
    if response is None:
        return None  # Quit
    if not response:
        if console:
            cs.StatusIndicator("info").add_message("Skipped").emit()
        return False

    # Confirm
    confirm = confirm_action()
    if confirm is None:
        return None  # Quit
    if not confirm:
        if console:
            cs.StatusIndicator("info").add_message("Cancelled").emit()
        return False

    # Process removal
    for font in all_files_to_remove:
        if move_to_trash(font.file_path, trash_dir, dry_run):
            dedup_stats.files_moved += 1
            dedup_stats.bytes_saved += font.file_size
            if console:
                if dry_run:
                    cs.StatusIndicator("info", dry_run=True).add_file(
                        font.name
                    ).with_explanation("Would remove").emit()
                else:
                    cs.StatusIndicator("deleted").add_file(font.name).with_explanation(
                        "Removed"
                    ).emit()
        else:
            dedup_stats.add_error(font.name, "Failed to move to trash")

    return True


def process_multiple_differences_groups(
    groups: List[VariationGroup],
    trash_dir: Path,
    dry_run: bool,
    dedup_stats: DeduplicationStats,
) -> Optional[bool]:
    """
    Process multiple differences groups.
    Returns True if successful, False if skipped, None if quit.
    """
    if not groups:
        return True

    if console:
        cs.StatusIndicator("info").add_message(
            f"Processing {len(groups)} group(s) with multiple differences"
        ).emit()

    # Show menu
    strategy = show_interactive_menu_multiple_differences()
    if strategy is None:
        return None  # Quit
    if strategy == "skip":
        if console:
            cs.StatusIndicator("info").add_message("Skipped").emit()
        return False

    # Collect all files to remove based on strategy
    all_files_to_remove = []
    for group in groups:
        _, files_to_remove = select_file_to_keep(group.fonts, strategy)
        all_files_to_remove.extend(files_to_remove)

    if not all_files_to_remove:
        if console:
            cs.StatusIndicator("info").add_message("No files to remove").emit()
        return True

    # Show preview
    strategy_names = {
        "oldest_creation": "Keep oldest creation date",
        "newest_revision": "Keep newest revision",
        "most_glyphs": "Keep most glyphs",
        "more_tables": "Keep more tables",
        "most_features": "Keep most features",
        "largest_size": "Keep largest file size",
    }
    show_preview(groups, all_files_to_remove, strategy_names.get(strategy, strategy))

    # Confirm
    confirm = confirm_action()
    if confirm is None:
        return None  # Quit
    if not confirm:
        if console:
            cs.StatusIndicator("info").add_message("Cancelled").emit()
        return False

    # Process removal
    for font in all_files_to_remove:
        if move_to_trash(font.file_path, trash_dir, dry_run):
            dedup_stats.files_moved += 1
            dedup_stats.bytes_saved += font.file_size
            if console:
                if dry_run:
                    cs.StatusIndicator("info", dry_run=True).add_file(
                        font.name
                    ).with_explanation("Would remove").emit()
                else:
                    cs.StatusIndicator("deleted").add_file(font.name).with_explanation(
                        "Removed"
                    ).emit()
        else:
            dedup_stats.add_error(font.name, "Failed to move to trash")

    return True


# ============================================================================
# Main Processing
# ============================================================================


def analyze_all_fonts(
    font_paths: List[str], stats: AnalysisStats
) -> List[FontAnalysis]:
    """Analyze all font files and return valid analyses"""
    analyses: List[FontAnalysis] = []
    total_files = len(font_paths)

    # Show progress for large batches
    show_progress = total_files > 50 and console and cs.RICH_AVAILABLE

    if show_progress:
        progress = cs.create_progress_bar()
        task = progress.add_task("Analyzing fonts...", total=total_files)
        progress.start()

    try:
        for font_path_str in font_paths:
            font_path = Path(font_path_str)
            analysis = analyze_font(font_path)

            if analysis:
                analyses.append(analysis)
                stats.valid_fonts += 1
            else:
                stats.invalid_fonts += 1

            if show_progress:
                progress.update(task, advance=1)
    finally:
        if show_progress:
            progress.stop()

    return analyses


def group_fonts_by_ps_name(
    analyses: List[FontAnalysis],
) -> Dict[str, List[FontAnalysis]]:
    """Group font analyses by PostScript name"""
    groups: Dict[str, List[FontAnalysis]] = {}
    for analysis in analyses:
        ps_name = analysis.ps_name
        if ps_name not in groups:
            groups[ps_name] = []
        groups[ps_name].append(analysis)
    return groups


def create_variation_groups(
    groups: Dict[str, List[FontAnalysis]],
    stats: AnalysisStats,
    include_singles: bool,
    min_differences: int,
) -> List[VariationGroup]:
    """Create variation groups and filter by criteria"""
    variation_groups: List[VariationGroup] = []

    for ps_name, font_list in sorted(groups.items()):
        group = VariationGroup(ps_name=ps_name, fonts=font_list)

        if group.has_variations:
            stats.variation_groups += 1
            # Filter by minimum differences
            if group.difference_count >= min_differences:
                variation_groups.append(group)
        else:
            stats.single_fonts += 1
            if include_singles:
                variation_groups.append(group)

    return variation_groups


def display_variation_groups(variation_groups: List[VariationGroup], verbose: bool):
    """Display comparison tables for all variation groups - shows only differences"""
    for group in variation_groups:
        # Count differences for summary
        diff_count, identical_count = count_differences_in_group(group, verbose)
        only_size_differs = only_file_size_differs(group)

        # Check if content-identical
        content_identical = detect_content_identical_group(group)

        if content_identical:
            if console:
                cs.StatusIndicator("info").add_message(
                    f"CONTENT IDENTICAL: {group.ps_name} ({len(group.fonts)} files)"
                ).add_item(
                    "All files have identical glyphs/features/layout", style="green"
                ).add_item(
                    "Differences are metadata only (safe duplicates)", style="dim"
                ).emit()

        # For groups with many variants (>4), use vertical format
        if len(group.fonts) > 4:
            # Show summary for vertical mode
            if diff_count > 0 and console:
                indicator = cs.StatusIndicator("info").add_message(
                    f"{diff_count} difference(s) found, {identical_count} attribute(s) identical"
                )
                if only_size_differs:
                    indicator.add_item(
                        "Note: Only file size differs - likely safe duplicates (check SHA256 to confirm)",
                        style="dim",
                    )
                indicator.emit()
            show_vertical_comparison(group)
        elif cs.RICH_AVAILABLE and console:
            table = create_comparison_table(group, verbose)
            if table:
                # Show summary before table
                if diff_count > 0:
                    if console:
                        indicator = cs.StatusIndicator("info").add_message(
                            f"{diff_count} difference(s) found, {identical_count} attribute(s) identical"
                        )
                        if only_size_differs:
                            indicator.add_item(
                                "Note: Only file size differs - likely safe duplicates (check SHA256 to confirm)",
                                style="dim",
                            )
                        indicator.emit()
                cs.emit(table)
            else:
                # Table is None means all attributes identical
                if console:
                    cs.StatusIndicator("unchanged").add_message(
                        f"{group.ps_name}: All attributes identical - files are duplicates"
                    ).add_item(
                        f"Checked {diff_count + identical_count} attributes across {len(group.fonts)} files",
                        style="dim",
                    ).emit()
        else:
            show_text_comparison(group, verbose)


def process_fonts(
    font_paths: List[str],
    verbose: bool = False,
    include_singles: bool = False,
    min_differences: int = 1,
    content_compare: bool = False,
) -> Tuple[AnalysisStats, List[VariationGroup]]:
    """Analyze fonts and group by PostScript name"""
    stats = AnalysisStats()
    stats.total_files = len(font_paths)

    # Set flag for analyze_font to compute content hashes
    if content_compare:
        analyze_font._compute_content_hash = True

    if console:
        cs.StatusIndicator("info").add_message(
            f"Analyzing {cs.fmt_count(len(font_paths))} font files..."
        ).emit()

    # Analyze all fonts
    analyses = analyze_all_fonts(font_paths, stats)

    if not analyses:
        if console:
            cs.StatusIndicator("warning").with_explanation(
                "No valid fonts to analyze"
            ).emit()
        return stats, []

    # Group by PostScript name
    groups = group_fonts_by_ps_name(analyses)
    stats.unique_ps_names = len(groups)

    # Create variation groups with filtering
    variation_groups = create_variation_groups(
        groups, stats, include_singles, min_differences
    )

    # Display results
    if console:
        cs.StatusIndicator("info").add_message(
            f"Found {cs.fmt_count(stats.variation_groups)} groups with variations"
        ).emit()
        if stats.single_fonts > 0 and not include_singles:
            cs.StatusIndicator("info").add_item(
                f"Hiding {cs.fmt_count(stats.single_fonts)} single fonts, use --single to show",
                style="dim",
            ).emit()

    # Show comparisons
    display_variation_groups(variation_groups, verbose)

    return stats, variation_groups


# ============================================================================
# Main Entry Point
# ============================================================================


def setup_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description="Analyze and compare font variations with the same PostScript name",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/fonts/              # Analyze fonts in directory
  %(prog)s font1.otf font2.otf         # Analyze specific files
  %(prog)s /fonts/ -r         # Process recursively
  %(prog)s /fonts/ --verbose           # Show detailed info
  %(prog)s /fonts/ --single            # Include single fonts (no variations)
  %(prog)s /fonts/ --min-diff 3        # Only show groups with 3+ differences
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
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed information (table lists, errors)",
    )
    parser.add_argument(
        "-s",
        "--single",
        action="store_true",
        help="Include single fonts with no variations",
    )
    parser.add_argument(
        "-m",
        "--min-diff",
        type=int,
        default=1,
        help="Minimum number of differences to show group (default: 1)",
    )
    parser.add_argument(
        "-dd",
        "--deduplicate",
        action="store_true",
        help="Enable interactive deduplication after showing comparisons",
    )
    parser.add_argument(
        "--trash-dir",
        type=Path,
        default=DEFAULT_TRASH_DIR,
        help=f"Directory to move duplicates to (default: {DEFAULT_TRASH_DIR})",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Preview what would be removed without moving files",
    )
    parser.add_argument(
        "--content-compare",
        action="store_true",
        help="Compute content hashes to identify fonts with identical glyphs but different metadata",
    )

    return parser


def show_startup_info(args, font_paths):
    """Display startup information panel"""
    if console:
        info_lines = [
            f"Files: {cs.fmt_count(len(font_paths))}",
            f"Recursive: {'Yes' if args.recursive else 'No'}",
            f"Show singles: {'Yes' if args.single else 'No'}",
            f"Min differences: {args.min_diff}",
        ]
        if args.deduplicate:
            info_lines.append(
                f"Deduplication: {'Enabled (dry-run)' if args.dry_run else 'Enabled'}"
            )
        cs.print_panel(
            "\n".join(info_lines),
            title="Font Variation Analyzer",
            border_style="blue",
        )


def show_final_summary(
    stats: AnalysisStats, dedup_stats: Optional[DeduplicationStats] = None
):
    """Display final summary panel"""
    if console:
        lines = [
            f"Total files: {cs.fmt_count(stats.total_files)}",
            f"Valid fonts: {cs.fmt_count(stats.valid_fonts)}",
            f"Invalid fonts: {cs.fmt_count(stats.invalid_fonts)}",
            f"Unique PS names: {cs.fmt_count(stats.unique_ps_names)}",
            f"Variation groups: {cs.fmt_count(stats.variation_groups)}",
            f"Single fonts: {cs.fmt_count(stats.single_fonts)}",
        ]
        if dedup_stats:
            lines.append("")
            lines.append("Deduplication:")
            lines.append(f"  Files moved: {cs.fmt_count(dedup_stats.files_moved)}")
            lines.append(f"  Bytes saved: {format_file_size(dedup_stats.bytes_saved)}")
            if dedup_stats.errors:
                lines.append(f"  Errors: {len(dedup_stats.errors)}")
        cs.print_panel(
            "\n".join(lines),
            title="Analysis Summary",
            border_style="green",
        )


def main():
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Check for Rich
    if not cs.RICH_AVAILABLE:
        cs.StatusIndicator("warning").with_explanation(
            "Rich library not available. Install with: pip install rich"
        ).emit()
        cs.StatusIndicator("info").add_message("Using fallback text output...").emit()

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

    # Show summary
    show_startup_info(args, font_paths)

    # Process fonts
    stats, variation_groups = process_fonts(
        font_paths,
        verbose=args.verbose,
        include_singles=args.single,
        min_differences=args.min_diff,
        content_compare=args.content_compare,
    )

    # Deduplication if enabled
    dedup_stats = None
    if args.deduplicate and variation_groups:
        dedup_stats = DeduplicationStats()

        # Group by difference type
        file_size_only_groups, multiple_differences_groups = (
            group_variations_by_difference_type(variation_groups)
        )

        if console:
            cs.StatusIndicator("info").add_message(
                "Starting deduplication process..."
            ).emit()

        # Process file size only groups first
        should_continue = True
        if file_size_only_groups:
            result = process_file_size_only_groups(
                file_size_only_groups,
                args.trash_dir,
                args.dry_run,
                dedup_stats,
            )
            if result is None:  # Quit
                should_continue = False
                if console:
                    cs.StatusIndicator("info").add_message(
                        "Deduplication cancelled by user"
                    ).emit()
            elif result:
                if console:
                    cs.StatusIndicator("success").add_message(
                        f"Processed {len(file_size_only_groups)} file size only group(s)"
                    ).emit()

        # Process multiple differences groups (only if didn't quit)
        if multiple_differences_groups and should_continue:
            result = process_multiple_differences_groups(
                multiple_differences_groups,
                args.trash_dir,
                args.dry_run,
                dedup_stats,
            )
            if result is None:  # Quit
                if console:
                    cs.StatusIndicator("info").add_message(
                        "Deduplication cancelled by user"
                    ).emit()
            elif result:
                if console:
                    cs.StatusIndicator("success").add_message(
                        f"Processed {len(multiple_differences_groups)} multiple differences group(s)"
                    ).emit()

    # Final summary
    show_final_summary(stats, dedup_stats)

    return 0


if __name__ == "__main__":
    exit(main())
