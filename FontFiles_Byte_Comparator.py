#!/usr/bin/env python3
"""
Font Byte Comparator - Diagnose byte-level differences between font files

When SHA256 says files differ but Variation Analyzer says they're identical,
this tool shows you EXACTLY what bytes differ and whether it matters.

Usage:
    # Direct comparison
    python FontFiles_Byte_Comparator.py font1.otf font2.otf
    python FontFiles_Byte_Comparator.py font1.otf font2.otf font3.otf

    # Directory scan
    python FontFiles_Byte_Comparator.py /directory/ --scan
    python FontFiles_Byte_Comparator.py /directory/ --scan --group-by-ps-name
    python FontFiles_Byte_Comparator.py /directory/ --scan --recursive

Options:
    --scan               Scan directory for near-duplicates
    --group-by-ps-name   Group files by PostScript name
    -r, --recursive      Process directories recursively
    --verbose, -v        Show detailed byte-level information
    --show-cosmetic      Include cosmetic differences in output
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from fontTools.ttLib import TTFont

# Add project root to path for FontCore imports (works for root and subdirectory scripts)
import sys
from pathlib import Path as PathLib

# ruff: noqa: E402
_project_root = PathLib(__file__).parent
while (
    not (_project_root / "FontCore").exists() and _project_root.parent != _project_root
):
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import FontCore.core_console_styles as cs
from FontCore.core_file_collector import collect_font_files
from FontCore.core_font_extension import validate_and_fix_extension

console = cs.get_console()

FONT_EXTENSIONS = {".ttf", ".otf", ".woff", ".woff2"}
DEFAULT_TRASH_DIR = Path.home() / ".Trash" / "FontDeduplicator"

# ============================================================================
# Data Classes
# ============================================================================


class DifferenceCategory(Enum):
    """Categories of differences between fonts"""

    IDENTICAL = "identical"
    COSMETIC = "cosmetic"  # timestamps, checksums only
    METADATA = "metadata"  # name table, version strings
    STRUCTURAL = "structural"  # glyph count, table inventory
    CONTENT = "content"  # actual glyphs, features, metrics


@dataclass
class TableDifference:
    """Difference in a specific table"""

    table_tag: str
    differs: bool
    field_diffs: Dict[str, Tuple] = field(default_factory=dict)
    byte_diff_count: int = 0


@dataclass
class DifferenceReport:
    """Complete comparison report for a set of files"""

    files: List[Path]
    ps_name: str
    category: DifferenceCategory
    tables_only_in_first: List[str] = field(default_factory=list)
    tables_only_in_others: List[str] = field(default_factory=list)
    table_differences: List[TableDifference] = field(default_factory=list)
    recommendation: str = ""

    @property
    def file_names(self) -> List[str]:
        return [f.name for f in self.files]

    def get_file_to_remove(self) -> Optional[Path]:
        """Return the file that should be removed based on recommendation"""
        if "RECOMMEND: Keep" not in self.recommendation:
            return None

        # Extract which file to keep from recommendation
        if self.files[0].name in self.recommendation:
            return self.files[1]  # Keep first, remove second
        elif self.files[1].name in self.recommendation:
            return self.files[0]  # Keep second, remove first

        return None


# ============================================================================
# Core Comparison Functions
# ============================================================================


def compare_head_table(font1: TTFont, font2: TTFont) -> Dict[str, Tuple]:
    """Compare head table fields, return dict of {field: (val1, val2)}"""
    diffs = {}
    head1 = font1["head"]
    head2 = font2["head"]

    fields_to_check = [
        "fontRevision",
        "checkSumAdjustment",
        "magicNumber",
        "flags",
        "unitsPerEm",
        "created",
        "modified",
        "xMin",
        "yMin",
        "xMax",
        "yMax",
        "macStyle",
        "lowestRecPPEM",
        "fontDirectionHint",
        "indexToLocFormat",
        "glyphDataFormat",
    ]

    for field_name in fields_to_check:
        if hasattr(head1, field_name) and hasattr(head2, field_name):
            val1 = getattr(head1, field_name)
            val2 = getattr(head2, field_name)
            if val1 != val2:
                diffs[field_name] = (val1, val2)

    return diffs


def compare_name_table(font1: TTFont, font2: TTFont) -> Dict[int, Tuple]:
    """Compare name table records, return dict of {nameID: (str1, str2)}"""
    diffs = {}

    ids1 = {record.nameID for record in font1["name"].names}
    ids2 = {record.nameID for record in font2["name"].names}

    for name_id in sorted(ids1 | ids2):
        record1 = font1["name"].getName(name_id, 3, 1, 0x409)
        record2 = font2["name"].getName(name_id, 3, 1, 0x409)

        str1 = record1.toUnicode() if record1 else None
        str2 = record2.toUnicode() if record2 else None

        if str1 != str2:
            diffs[name_id] = (str1, str2)

    return diffs


def compare_os2_table(font1: TTFont, font2: TTFont) -> Dict[str, Tuple]:
    """Compare OS/2 table fields"""
    diffs = {}
    os2_1 = font1["OS/2"]
    os2_2 = font2["OS/2"]

    fields_to_check = [
        "version",
        "xAvgCharWidth",
        "usWeightClass",
        "usWidthClass",
        "fsType",
        "sTypoAscender",
        "sTypoDescender",
        "sTypoLineGap",
        "usWinAscent",
        "usWinDescent",
        "achVendID",
        "panose",
    ]

    for field_name in fields_to_check:
        if hasattr(os2_1, field_name) and hasattr(os2_2, field_name):
            val1 = getattr(os2_1, field_name)
            val2 = getattr(os2_2, field_name)
            if val1 != val2:
                diffs[field_name] = (val1, val2)

    return diffs


def compare_table_data(font1: TTFont, font2: TTFont, table_tag: str) -> TableDifference:
    """Compare a specific table between two fonts"""
    diff = TableDifference(table_tag=table_tag, differs=False)

    try:
        data1 = font1.getTableData(table_tag)
        data2 = font2.getTableData(table_tag)

        if data1 != data2:
            diff.differs = True
            diff.byte_diff_count = sum(1 for b1, b2 in zip(data1, data2) if b1 != b2)

            # Get field-level differences for known tables
            if table_tag == "head":
                diff.field_diffs = compare_head_table(font1, font2)
            elif table_tag == "name":
                diff.field_diffs = compare_name_table(font1, font2)
            elif table_tag == "OS/2":
                diff.field_diffs = compare_os2_table(font1, font2)
            elif table_tag in ("GPOS", "GSUB"):
                # Compare feature counts for GPOS/GSUB
                diff.field_diffs = compare_feature_table(font1, font2, table_tag)

    except Exception as e:
        diff.field_diffs["error"] = (str(e), "")

    return diff


def compare_feature_table(
    font1: TTFont, font2: TTFont, table_tag: str
) -> Dict[str, Tuple]:
    """Compare GPOS or GSUB feature tables"""
    diffs = {}

    try:
        table1 = font1[table_tag]
        table2 = font2[table_tag]

        # Compare feature counts
        if hasattr(table1, "table") and hasattr(table1.table, "FeatureList"):
            count1 = table1.table.FeatureList.FeatureCount
            count2 = table2.table.FeatureList.FeatureCount
            if count1 != count2:
                diffs["FeatureCount"] = (count1, count2)

        # Compare lookup counts
        if hasattr(table1, "table") and hasattr(table1.table, "LookupList"):
            lookup1 = (
                len(table1.table.LookupList.Lookup) if table1.table.LookupList else 0
            )
            lookup2 = (
                len(table2.table.LookupList.Lookup) if table2.table.LookupList else 0
            )
            if lookup1 != lookup2:
                diffs["LookupCount"] = (lookup1, lookup2)

    except Exception:
        pass

    return diffs


def diagnose_byte_differences(file1: Path, file2: Path) -> DifferenceReport:
    """Main diagnostic function comparing two font files"""
    try:
        font1 = TTFont(str(file1), lazy=False)
        font2 = TTFont(str(file2), lazy=False)

        # Get PostScript name
        ps_name = ""
        try:
            name_rec = font1["name"].getName(6, 3, 1, 0x409)
            ps_name = name_rec.toUnicode() if name_rec else file1.stem
        except Exception:
            ps_name = file1.stem

        report = DifferenceReport(
            files=[file1, file2], ps_name=ps_name, category=DifferenceCategory.IDENTICAL
        )

        # Check table inventory
        tables1 = set(font1.keys())
        tables2 = set(font2.keys())

        report.tables_only_in_first = list(tables1 - tables2)
        report.tables_only_in_others = list(tables2 - tables1)

        # Compare common tables
        for table_tag in sorted(tables1 & tables2):
            diff = compare_table_data(font1, font2, table_tag)
            if diff.differs:
                report.table_differences.append(diff)

        # Categorize differences
        report.category = categorize_differences(report)
        report.recommendation = generate_recommendation(report)

        font1.close()
        font2.close()

        return report

    except Exception as e:
        return DifferenceReport(
            files=[file1, file2],
            ps_name="ERROR",
            category=DifferenceCategory.CONTENT,
            recommendation=f"Error comparing files: {e}",
        )


def categorize_differences(report: DifferenceReport) -> DifferenceCategory:
    """Categorize the type of differences found"""

    # No differences
    if (
        not report.tables_only_in_first
        and not report.tables_only_in_others
        and not report.table_differences
    ):
        return DifferenceCategory.IDENTICAL

    # Different table inventory = structural
    if report.tables_only_in_first or report.tables_only_in_others:
        return DifferenceCategory.STRUCTURAL

    # Check what tables differ
    differing_tables = {d.table_tag for d in report.table_differences}

    # Only cosmetic tables differ
    cosmetic_tables = {"head", "DSIG"}
    if differing_tables.issubset(cosmetic_tables):
        # Check if only timestamps/checksums differ
        for diff in report.table_differences:
            if diff.table_tag == "head":
                cosmetic_fields = {"created", "modified", "checkSumAdjustment"}
                if set(diff.field_diffs.keys()).issubset(cosmetic_fields):
                    continue
                else:
                    return DifferenceCategory.METADATA
        return DifferenceCategory.COSMETIC

    # Check if GPOS/GSUB differences are just head differences
    # (head often changes when fonts are reprocessed but content stays same)
    if (
        differing_tables == {"GPOS", "GSUB", "head"}
        or differing_tables == {"GPOS", "head"}
        or differing_tables == {"GSUB", "head"}
    ):
        # Check if head only has cosmetic differences
        head_diff = next(
            (d for d in report.table_differences if d.table_tag == "head"), None
        )
        if head_diff:
            cosmetic_fields = {"created", "modified", "checkSumAdjustment"}
            if set(head_diff.field_diffs.keys()).issubset(cosmetic_fields):
                # Head is cosmetic, check if GPOS/GSUB have field-level diffs
                feature_diffs_found = False
                for diff in report.table_differences:
                    if diff.table_tag in ("GPOS", "GSUB") and diff.field_diffs:
                        feature_diffs_found = True
                        break

                if not feature_diffs_found:
                    # No field-level diffs in features, might be byte-level only
                    return DifferenceCategory.METADATA

    # Content tables differ
    content_tables = {"glyf", "CFF ", "CFF2", "GSUB", "GPOS", "cmap", "hmtx", "vmtx"}
    if differing_tables & content_tables:
        return DifferenceCategory.CONTENT

    # Otherwise it's metadata
    return DifferenceCategory.METADATA


def generate_recommendation(report: DifferenceReport) -> str:
    """Generate recommendation based on difference category and specific differences"""
    if report.category == DifferenceCategory.IDENTICAL:
        return "Files are byte-for-byte identical"
    elif report.category == DifferenceCategory.COSMETIC:
        # Prefer file with cleaner name (without ~NNN suffix)
        file1_has_tilde = "~" in report.files[0].stem
        file2_has_tilde = "~" in report.files[1].stem

        if file1_has_tilde and not file2_has_tilde:
            return f"RECOMMEND: Keep {report.files[1].name} (cosmetic-only differences)"
        elif file2_has_tilde and not file1_has_tilde:
            return f"RECOMMEND: Keep {report.files[0].name} (cosmetic-only differences)"
        else:
            # Both have or don't have tildes - default to first file
            return f"RECOMMEND: Keep {report.files[0].name} (cosmetic-only differences)"
    elif report.category == DifferenceCategory.METADATA:
        # Check for GPOS/GSUB feature differences
        for diff in report.table_differences:
            if (
                diff.table_tag in ("GPOS", "GSUB")
                and "FeatureCount" in diff.field_diffs
            ):
                count1, count2 = diff.field_diffs["FeatureCount"]
                if count1 > count2:
                    return f"RECOMMEND: Keep {report.files[0].name} (has more {diff.table_tag} features: {count1} vs {count2})"
                elif count2 > count1:
                    return f"RECOMMEND: Keep {report.files[1].name} (has more {diff.table_tag} features: {count2} vs {count1})"

        # No feature differences - prefer file with cleaner name (without ~NNN suffix)
        file1_has_tilde = "~" in report.files[0].stem
        file2_has_tilde = "~" in report.files[1].stem

        if file1_has_tilde and not file2_has_tilde:
            return f"RECOMMEND: Keep {report.files[1].name} (metadata-only differences, cleaner filename)"
        elif file2_has_tilde and not file1_has_tilde:
            return f"RECOMMEND: Keep {report.files[0].name} (metadata-only differences, cleaner filename)"
        else:
            # Both have or don't have tildes - default to first file
            return f"RECOMMEND: Keep {report.files[0].name} (metadata-only differences)"
    elif report.category == DifferenceCategory.STRUCTURAL:
        return "Different structure - investigate before removing"
    else:  # CONTENT
        # Provide more specific guidance for content differences
        differing_tables = {d.table_tag for d in report.table_differences}

        # Check if it's just GPOS/GSUB/head
        if differing_tables.issubset({"GPOS", "GSUB", "head"}):
            # Check for feature count differences
            for diff in report.table_differences:
                if (
                    diff.table_tag in ("GPOS", "GSUB")
                    and "FeatureCount" in diff.field_diffs
                ):
                    count1, count2 = diff.field_diffs["FeatureCount"]
                    if count1 > count2:
                        return f"RECOMMEND: Keep {report.files[0].name} (has more {diff.table_tag} features: {count1} vs {count2})"
                    elif count2 > count1:
                        return f"RECOMMEND: Keep {report.files[1].name} (has more {diff.table_tag} features: {count2} vs {count1})"

            return "OpenType features differ but counts match. Use --verbose to investigate. May be safe duplicates."

        return "Different content - use --verbose to see what differs"


# ============================================================================
# Display Functions
# ============================================================================


def show_comparison_report(
    report: DifferenceReport, verbose: bool = False, show_cosmetic: bool = False
):
    """Display comparison report using StatusIndicator - compact format"""

    if console:
        # Determine status based on recommendation
        if "RECOMMEND: Keep" in report.recommendation:
            # Extract which file to keep and display compactly
            if report.files[0].name in report.recommendation:
                status = "success"
                keep_file = report.files[0].name
                remove_file = report.files[1].name
            else:
                status = "success"
                keep_file = report.files[1].name
                remove_file = report.files[0].name

            cs.StatusIndicator(status).add_message(f"{report.ps_name}").add_item(
                f"✓ Keep: {keep_file}", style="green"
            ).add_item(f"✗ Remove: {remove_file}", style="dim").add_item(
                f"Reason: {report.recommendation.split('(')[1].rstrip(')')}",
                style="dim",
            ).emit()
        elif (
            "safe duplicates" in report.recommendation.lower()
            or "metadata" in report.recommendation.lower()
        ):
            # Metadata differences - likely safe
            cs.StatusIndicator("info").add_message(f"{report.ps_name}").add_item(
                f"Files: {report.files[0].name} / {report.files[1].name}", style="dim"
            ).add_item(f"Status: {report.recommendation}", style="cyan").emit()
        else:
            # Other cases - show normally
            category_styles = {
                DifferenceCategory.IDENTICAL: ("success", "IDENTICAL"),
                DifferenceCategory.COSMETIC: ("info", "COSMETIC"),
                DifferenceCategory.METADATA: ("info", "METADATA"),
                DifferenceCategory.STRUCTURAL: ("warning", "STRUCTURAL"),
                DifferenceCategory.CONTENT: ("warning", "CONTENT"),
            }

            style, label = category_styles.get(
                report.category, ("info", str(report.category))
            )

            cs.StatusIndicator(style).add_message(f"{report.ps_name}").add_item(
                f"{report.files[0].name} / {report.files[1].name}", style="dim"
            ).add_item(f"Category: {label} | {report.recommendation}").emit()

        # Show detailed differences only in verbose mode
        if verbose and report.table_differences:
            cs.StatusIndicator("info").add_item(
                f"Tables: {', '.join([d.table_tag for d in report.table_differences])}",
                style="dim",
            ).emit()

            for diff in report.table_differences:
                if diff.field_diffs:
                    indicator = cs.StatusIndicator("info").add_message(
                        f"  {diff.table_tag}:"
                    )
                    for field, (val1, val2) in diff.field_diffs.items():
                        if field in ("created", "modified"):
                            try:
                                v1_str = (
                                    datetime.fromtimestamp(val1).strftime(
                                        "%Y-%m-%d %H:%M:%S"
                                    )
                                    if val1
                                    else "None"
                                )
                                v2_str = (
                                    datetime.fromtimestamp(val2).strftime(
                                        "%Y-%m-%d %H:%M:%S"
                                    )
                                    if val2
                                    else "None"
                                )
                                indicator.add_item(f"    {field}: {v1_str} vs {v2_str}")
                            except Exception:
                                indicator.add_item(f"    {field}: {val1} vs {val2}")
                        else:
                            indicator.add_item(f"    {field}: {val1} vs {val2}")
                    indicator.emit()


# ============================================================================
# Directory Scanning Functions
# ============================================================================


def scan_directory_for_near_duplicates(
    directory: Path, recursive: bool = False, group_by_ps_name: bool = False
) -> List[DifferenceReport]:
    """Scan directory and find files with same PS name"""

    # Collect fonts
    font_paths = collect_font_files(
        [str(directory)], recursive=recursive, allowed_extensions=FONT_EXTENSIONS
    )

    # Validate and fix extensions
    validated_paths = []
    for path_str in font_paths:
        path = Path(path_str)
        is_valid, fixed_path = validate_and_fix_extension(path, auto_fix=True)
        if fixed_path:
            if console:
                cs.StatusIndicator("info").add_file(str(fixed_path)).with_explanation(
                    f"Fixed extension: {path.name} → {fixed_path.name}"
                ).emit()
            validated_paths.append(str(fixed_path))
        else:
            validated_paths.append(path_str)
    font_paths = validated_paths

    if not font_paths:
        if console:
            cs.StatusIndicator("error").with_explanation("No font files found").emit()
        return []

    if console:
        cs.StatusIndicator("info").add_message(
            f"Scanning {cs.fmt_count(len(font_paths))} font files..."
        ).emit()

    # Group by PostScript name
    ps_name_groups = {}
    for font_path_str in font_paths:
        font_path = Path(font_path_str)
        try:
            font = TTFont(str(font_path), lazy=True)
            name_rec = font["name"].getName(6, 3, 1, 0x409)
            ps_name = name_rec.toUnicode() if name_rec else font_path.stem
            font.close()

            if ps_name not in ps_name_groups:
                ps_name_groups[ps_name] = []
            ps_name_groups[ps_name].append(font_path)
        except Exception:
            continue

    # Find groups with multiple files
    near_duplicates = {
        name: files for name, files in ps_name_groups.items() if len(files) > 1
    }

    if not near_duplicates:
        if console:
            cs.StatusIndicator("info").add_message(
                "No potential near-duplicates found"
            ).emit()
        return []

    if console:
        cs.StatusIndicator("info").add_message(
            f"Found {cs.fmt_count(len(near_duplicates))} groups with multiple files"
        ).emit()

    # Compare each group
    reports = []
    for ps_name, files in sorted(near_duplicates.items()):
        if len(files) == 2:
            report = diagnose_byte_differences(files[0], files[1])
            reports.append(report)
        else:
            # Compare first file with each other
            for other_file in files[1:]:
                report = diagnose_byte_differences(files[0], other_file)
                reports.append(report)

    return reports


# ============================================================================
# Main Entry Point
# ============================================================================


def move_to_trash(file_path: Path, trash_dir: Path) -> bool:
    """Move a file to trash directory"""
    import shutil

    try:
        trash_dir.mkdir(parents=True, exist_ok=True)

        # Handle naming conflicts
        target = trash_dir / file_path.name
        counter = 1
        while target.exists():
            stem = file_path.stem
            suffix = file_path.suffix
            target = trash_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        shutil.move(str(file_path), str(target))
        return True
    except Exception as e:
        if console:
            cs.StatusIndicator("error").add_file(file_path.name).with_explanation(
                f"Failed to move: {e}"
            ).emit()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose byte-level differences between font files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "paths",
        nargs="+",
        help="Either 2+ font files to compare, or a directory to scan for near-duplicates",
    )
    parser.add_argument(
        "--group-by-ps-name",
        action="store_true",
        help="Group results by PostScript name (directory scan mode)",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Process directories recursively (directory scan mode)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed field-level differences",
    )
    parser.add_argument(
        "--show-cosmetic",
        action="store_true",
        help="Show cosmetic differences (timestamps, checksums)",
    )
    parser.add_argument(
        "-dd",
        "--deduplicate",
        action="store_true",
        help="Remove recommended duplicates after confirmation",
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

    args = parser.parse_args()

    # Show header
    if console:
        cs.print_panel(
            "Font Byte Comparator\nDiagnosing byte-level differences",
            title="Byte Comparator",
            border_style="blue",
        )

    reports = []

    # Convert to Path objects
    paths = [Path(p) for p in args.paths]

    # Auto-detect mode based on first path
    if len(paths) == 1 and paths[0].is_dir():
        # Directory scan mode
        directory = paths[0]
        reports = scan_directory_for_near_duplicates(
            directory, recursive=args.recursive, group_by_ps_name=args.group_by_ps_name
        )
    elif len(paths) >= 2:
        # Direct comparison mode - validate all are files
        validated_paths = []
        for f in paths:
            if not f.exists():
                if console:
                    cs.StatusIndicator("error").with_explanation(
                        f"File not found: {f}"
                    ).emit()
                return 1
            if not f.is_file():
                if console:
                    cs.StatusIndicator("error").with_explanation(
                        f"Not a file: {f} (mix of files and directories not supported)"
                    ).emit()
                return 1

        # Validate and fix extensions
        validated_paths = []
        for f in paths:
            is_valid, fixed_path = validate_and_fix_extension(f, auto_fix=True)
            if fixed_path:
                if console:
                    cs.StatusIndicator("info").add_file(
                        str(fixed_path)
                    ).with_explanation(
                        f"Fixed extension: {f.name} → {fixed_path.name}"
                    ).emit()
                validated_paths.append(fixed_path)
            else:
                validated_paths.append(f)
        paths = validated_paths

        # Compare first file with each other
        for other_file in paths[1:]:
            report = diagnose_byte_differences(paths[0], other_file)
            reports.append(report)
    else:
        if console:
            cs.StatusIndicator("error").with_explanation(
                "Provide either a directory to scan, or 2+ files to compare"
            ).emit()
        return 1

    # Display reports
    for report in reports:
        show_comparison_report(
            report, verbose=args.verbose, show_cosmetic=args.show_cosmetic
        )

    # Summary
    if console and reports:
        category_counts = {}
        recommendations_count = 0
        for report in reports:
            cat = report.category
            category_counts[cat] = category_counts.get(cat, 0) + 1
            if report.get_file_to_remove():
                recommendations_count += 1

        summary_lines = [f"Total comparisons: {len(reports)}"]
        for cat, count in sorted(category_counts.items(), key=lambda x: x[0].value):
            summary_lines.append(f"{cat.value}: {count}")
        if recommendations_count > 0:
            summary_lines.append(
                f"\nFiles with removal recommendations: {recommendations_count}"
            )

        cs.print_panel("\n".join(summary_lines), title="Summary", border_style="green")

    # Deduplication if requested
    if args.deduplicate and reports:
        files_to_remove = []
        for report in reports:
            file_to_remove = report.get_file_to_remove()
            if file_to_remove:
                files_to_remove.append((file_to_remove, report))

        if not files_to_remove:
            if console:
                cs.StatusIndicator("info").add_message(
                    "No files recommended for removal"
                ).emit()
            return 0

        # Show what will be removed
        if console:
            # Use same panel text for both dry-run and normal mode
            # DRY prefix will be added automatically by StatusIndicator when dry_run=True
            cs.print_panel(
                "Deduplication Preview",
                title="Deduplication",
                border_style="yellow",
            )

            cs.StatusIndicator("warning").add_message(
                f"The following {len(files_to_remove)} files will be removed:"
            ).emit()

            for file_path, report in files_to_remove:
                cs.StatusIndicator("warning").add_file(file_path.name).add_item(
                    f"Group: {report.ps_name}", style="dim"
                ).emit()

        # Confirm unless dry-run
        if not args.dry_run:
            try:
                response = input("\nProceed with removal? (y/n): ").strip().lower()
                if response != "y":
                    if console:
                        cs.StatusIndicator("info").add_message("Cancelled").emit()
                    return 0
            except (EOFError, KeyboardInterrupt):
                if console:
                    cs.StatusIndicator("info").add_message("\nCancelled").emit()
                return 0

        # Remove files
        removed_count = 0
        failed_count = 0

        for file_path, report in files_to_remove:
            # Use same StatusIndicator for both dry-run and normal mode
            # DRY prefix will be added automatically when dry_run=True
            if console:
                cs.StatusIndicator("deleted", dry_run=args.dry_run).add_file(
                    file_path.name
                ).with_explanation(
                    "Removed" if not args.dry_run else "Would remove"
                ).emit()

            if args.dry_run:
                removed_count += 1
            else:
                if move_to_trash(file_path, args.trash_dir):
                    removed_count += 1
                else:
                    failed_count += 1

        # Final summary
        if console:
            summary_lines = [
                f"Files {'would be ' if args.dry_run else ''}removed: {removed_count}"
            ]
            if failed_count > 0:
                summary_lines.append(f"Failed: {failed_count}")
            if not args.dry_run and removed_count > 0:
                summary_lines.append(f"Moved to: {args.trash_dir}")

            cs.print_panel(
                "\n".join(summary_lines),
                title="Deduplication Complete",
                border_style="green",
            )

    return 0


if __name__ == "__main__":
    exit(main())
