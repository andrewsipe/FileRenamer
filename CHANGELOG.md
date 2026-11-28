# Changelog

## [0.11.0] - 2026-11-27

### Changed
- A variable font's Upright or Italic word now comes from one decision made for the whole folder, so a font and its partner get different names instead of one name plus `~001`. The fonts of a folder are grouped by name (slope words taken out) and compared: STAT first, then which one's outlines lean more, then the filename, names and flags. A font alone in its family stays Upright unless its outlines are shown to be upright (stale italic flags then no longer name it Italic). FontCore's `core_slope_resolver` makes the decision; FontFixer uses the same one.
- A font the resolver cannot be sure of (unmeasurable outlines, conflicting evidence, three or more files with one name, two slanted files sharing a name, a free slope axis) keeps the slope its own tables give it, as before. `tools/resolve_slopes.py` in FontCore lists them with the reason.

## [0.10.1] - 2026-11-26

### Changed
- The preview header and the confirmation prompt say where the new names come from: variable fonts from the filename and the font tables, not from PostScript names. With `--variable-only` the header line is `Naming:` (the `--stem` option does not apply) and says static fonts are left alone.
- The preview counts variable fonts ("2 (1 to rename, 1 already named correctly)") and static fonts left alone.
- The Method column is shown only when the rows differ; one method for every row is said once above the table. The variable label is now `Variable (tables)`, so it no longer wraps.

## [0.10.0] - 2026-11-26

### Added
- `--variable-only` renames the variable fonts of a mixed folder and leaves the static ones as they are, for collections whose static filenames are already set by other means.

## [0.9.0] - 2026-11-26

FontCore 0.11.6 vendored (see `FontCore/VENDOR.md`).

### Fixed
- A variable font no longer needs a valid PostScript name, a name table or a unique PostScript name to be renamed. Fonts collected from many places often have none of these, and they were skipped or, when an upright and an italic file shared one PostScript name, left alone. A variable font's house name comes from its filename and its tables, whatever the `-N` stem option says, and fonts are grouped by the name they will get, so only true duplicates compete (one keeps the clean name, the rest become `~001`, `~002`). The preview says "Variable (filename + font tables)" in the Method column.
- A variable font whose every axis is locked is renamed as a static font.

## [0.8.3] - 2026-11-26

FontCore 0.11.5 vendored (see `FontCore/VENDOR.md`).

### Fixed
- A real rename of a variable font left it under a temporary `_tmp_` name: the family was read from the temporary name FileRenamer gives every file in its first pass. The family now comes from the original filename and the axes from the font. Previews were not affected. Tests now do a real rename, not only a preview.

## [0.8.2] - 2026-11-26

FontCore 0.11.2 vendored: a locked weight word is written only when the filename states it. The name table and `usWeightClass` are no longer read for a weight, and there is no `Weight` placeholder (`Danfo-Variable` becomes `Danfo-CustomVariableUpright`). A locked `Regular` the filename states is still written.

## [0.8.1] - 2026-11-26

FontCore 0.11.1 vendored: a locked `Regular` is written as a weight word (`AugureStereoRegular-WidthVariable` is not `AugureStereo-WidthVariable`); `Normal` and `Roman` are not read as weights, and ID2 saying Regular is not evidence of one.

## [0.8.0] - 2026-11-26

FontCore 0.11.0 vendored (see `FontCore/VENDOR.md`).

### Changed
- Variable fonts are renamed to `Family{Optical}{Width}{Variant}{Weight}-{FreeExtras}Variable{Slope}` (for example `AugureStereoLight-WidthVariableUpright`): what is fixed goes in the family, the free extras sit left of Variable and a locked slope right of it. A font whose every axis is locked is renamed as a static font.

## [0.7.2] - 2026-11-26

FontCore 0.10.2 vendored: a locked `Regular` weight word is kept (`AugureStereo-VariableRegular` is no longer mistaken for a free weight), and is dropped only when the font has a free weight axis.

## [0.7.1] - 2026-11-26

FontCore 0.10.1 vendored: a design-variant word after Variable (Alt, Cut, Swash) moves into the family part (`Probe-VariableAlt` becomes `ProbeAlt-Variable`); Rotalic, Tiltalic and Corsiva are read as slope words.

## [0.7.0] - 2026-11-26

FontCore 0.10.0 vendored (see `FontCore/VENDOR.md`).

### Changed
- Variable fonts are renamed to `{Family}{Optical}{Width}-Variable{Axes}{Slope}` under the refined rule: no axis counts and no `Slope` word, `Width`/`OpticalSize`/`Custom` for the extras, `Upright` or the designer's slope word when the slope is locked, and a font with all four registered axes free becomes `{Family}Collection-Variable`.

## [0.6.2] - 2026-11-25

FontCore 0.9.2 vendored: a variable filename keeps its family part as written, so `BT_Avance-Variable` stays `BT_Avance-...` instead of becoming `BTAvance-...`.

## [0.6.1] - 2026-11-25

FontCore 0.9.1 vendored: a variable font with no weight axis and a Regular weight class no longer gets a `Regular` word in its name.

## [0.6.0] - 2026-11-25

FontCore 0.9.0 vendored (see `FontCore/VENDOR.md`).

### Changed
- Variable fonts are renamed to `{Family}{Optical}{Width}-Variable{Axes}{Weight}{Slope}` (for example `ProbeDisplayCondensed-Variable3AxisItalic`). The font's tables decide the axes and the slope; a locked width now sits in the family part, before the hyphen.

## [0.5.0] - 2026-11-25

FontCore 0.8.2 vendored (see `FontCore/VENDOR.md`).

### Changed
- `fontrenamer`, `fontorganizer` and `fontdeduper` end with one sentence ("Renamed 3, unchanged 1, 1 problem.") under the Summary panel; the chip turns to a warning when anything went wrong.
- The renamer exits 1 when a font could not be read or renamed (it always exited 0). The organizer counts quarantined and failed files as problems, and the deduper exits 1 on errors in every mode.

## [0.4.1] - 2026-11-25

FontCore 0.7.0 vendored (see `FontCore/VENDOR.md`).

### Changed
- File collection follows one set of rules: macOS junk (`._*`, `__MACOSX`) and hidden files and folders are skipped, each real file is listed once (a symlink and its target are one file), and a path that does not exist gets a warning. Binary formats are the default; this tool asks for TTX where it handles it. `fontdeduper` no longer renames a file whose extension does not match its format: it says so and carries on (only `fontrenamer` renames). A font collection (TTC) named `.ttf` is no longer renamed to `.ttc` (which no tool reads): it is left alone with a warning.

## [0.4.0] - 2026-11-22

FontCore 0.6.0 vendored (see `FontCore/VENDOR.md`).

**Behavior change:** `fontorganizer` now leaves fonts that share a name but differ in weight, slant or size under their own filenames.

### Changed
- `fontorganizer` now has the same-face check the renamer got in 0.3.0: fonts that share a PostScript name but differ in weight, slant or size are not ranked, and each keeps its own filename (with a warning). FontCore refreshed so its font metadata records weight, slant and variable-ness.
- Help in plain words for `fontrenamer`, `fontorganizer` and `fontdeduper` (standard flag wording, no old script names), and each ends with a Heads up line: the renamer renames in place with no backup, the organizer moves and renames into the output folder, the deduper moves duplicates to the trash folder (recoverable). Preview first with `-n`.

## [0.3.0] - 2026-11-22

**Behavior change:** which font keeps the clean name when several want it can differ from 0.2.x, and fonts that share a name but are not copies of one font are left alone (see Changed).

### Changed
- Install and update instructions use `uv tool install` / `uv tool upgrade` (pipx is no longer mentioned).
- One conflict rule for `fontrenamer` and `fontorganizer` (new `fontrenamer/conflict.py`, replacing `version_priority.py`). When several fonts want the same name, the best copy keeps it and the rest get `~001`, `~002`, ranked by: higher head revision, higher version string (nameID 5), more glyphs/languages/features, then the newer modification date only when the version is read and equal (a converted or re-saved file can have a newer date and an older version), then the earlier file name. The creation date is no longer used. Before, the renamer used a quality score and the organizers a revision/date rule, and they chose a different winner in about one group in five.
- Fonts that share a PostScript name but differ in weight, slant or size (for example many fonts whose name is a copyright line) are no longer chained into `Name`, `Name~001`, `Name~002` ... even with `-ra`. None of them is renamed to the shared name; each keeps its filename and a warning says so. The organizers rank with the same rule but do not have this check yet (their metadata has no weight or slant).
- The renamer's cache file is now `.font_rename_cache_v2.json` (it records weight and slant). An old `.font_rename_cache.json` is ignored and can be deleted.

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
