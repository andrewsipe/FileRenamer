# Changelog

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
