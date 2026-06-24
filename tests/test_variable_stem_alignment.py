"""Tests for variable-font static-aligned stem normalization in the renamer."""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from FileRenamer.FontFiles_RenamerEnhanced import (  # noqa: E402
    FontMetadata,
    _variable_aligned_stem,
)
from FontCore.core_variable_filename_parser import (  # noqa: E402
    filename_has_variable_marker,
    format_variable_filename,
    parse_variable_filename,
)


def _vf_meta(basename: str) -> FontMetadata:
    return FontMetadata(
        ps_name="ReaderPro-Bold",
        font_revision=1.0,
        version_string="Version 1.0",
        file_size=1024,
        glyph_count=500,
        head_created=None,
        head_modified=None,
        file_path=f"/tmp/{basename}",
        original_filename=basename,
        is_variable=True,
    )


def test_variable_aligned_stem_legacy_width():
    meta = _vf_meta("ReaderProCondensed-Variable.ttf")
    assert _variable_aligned_stem(meta, "ReaderProCondensed") == "ReaderPro-CondensedVariable"


def test_variable_aligned_stem_matches_format_helper():
    basename = "ReaderProCondensed-Variable.ttf"
    slots = parse_variable_filename(basename)
    assert slots is not None
    meta = _vf_meta(basename)
    assert _variable_aligned_stem(meta, "ignored") == format_variable_filename(slots)


def test_conflict_suffix_skipped_when_stem_has_variable_marker():
    stem = "ReaderPro-CondensedVariable"
    assert filename_has_variable_marker(stem)
