# Changelog

## [0.2.2] - 2026-11-21

### Changed
- Vendored FontCore refreshed to v0.4.0 (14 modules). The renamer reads existing name records and uses FontCore only to clean PostScript names and read variable-font filenames. For 5,377 filenames and name strings from a 1,539-font library and a 802-font intake sample, `sanitize_postscript`, `filename_has_variable_marker`, `parse_variable_filename` and `format_variable_filename` return exactly what they did in 0.2.1, so no rename changes.

## [0.2.1] - 2026-11-21

### Changed
- Vendored FontCore refreshed to v0.3.1 (14 modules). Wording and counts in the console messages change ("1 file" / "3 files", `Finished`).
  The renamer reads existing name records and uses FontCore only to clean PostScript names; on a 150-font sample per stem mode (`ps`, `typo`, `full`, `legacy`, `uid`, `best`) the dry runs are identical to 0.2.0, and `sanitize_postscript` gives the same result for 7,695 names from a 1,539-font library.

## [0.2.0] - 2026-11-20

First installable release.

### Changed
- Packaged as `fontrenamer` with three commands: `fontrenamer`, `fontorganizer`, `fontdeduper`.
  They replace `FontFiles_RenamerEnhanced.py`, `FontFiles_OrganizerEnhanced.py` and
  `FontFiles_SHA256_Deduplicator.py`. Flags are unchanged.
- FontCore is vendored in `FontCore/` (14 modules, FontCore v0.2.0) instead of a gitignored symlink.
  The module list is declared in `[tool.fontcore-vendor]` in `pyproject.toml`.

### Removed
- `FontFiles_Byte_Comparator.py` and `FontFiles_Variation_Analyzer.py`. They did not work well as
  CLI programs. They remain in git history before this release.
