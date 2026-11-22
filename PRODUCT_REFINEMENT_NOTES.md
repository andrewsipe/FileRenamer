# Product refinement notes — FileRenamer

Captured during the 2026-08-21 declutter pass. Use this when turning FileRenamer into a sharper product release. **Not** user-facing docs.

## What was archived (declutter)

| Archived path | Was | Why archived |
|---------------|-----|--------------|
| `_misc/_archive/FileRenamer/FontFiles_Renamer.py` | Base PS-name renamer | Superseded by `FontFiles_RenamerEnhanced.py` (superset of CLI flags; active tests target Enhanced only) |
| `_misc/_archive/FileRenamer/FontFiles_Organizer.py` | Simpler organizer | Superseded by `FontFiles_OrganizerEnhanced.py` (adds `-ft` / `-vt` / `-nt` / `-cs`) |

Also removed local junk: `.metrics_checkpoint.json` (personal font-library paths; not part of the tool).

## Active tree (after declutter)

| File | Role |
|------|------|
| `FontFiles_RenamerEnhanced.py` | Canonical renamer |
| `FontFiles_OrganizerEnhanced.py` | Canonical organizer |
| `version_priority.py` | Shared helper extracted so organizers do not import the archived base renamer |
| `FontFiles_SHA256_Deduplicator.py` | Hash dedupe |
| `FontFiles_Byte_Comparator.py` | Byte-level compare |
| `FontFiles_Variation_Analyzer.py` | VF variation analysis |
| `REVIEW_OPTIONS_PRO_CON.md` | Deferred Enhanced renamer refactors |

## Behavior to revisit on the product pass

### Conflict / priority model (important)

- **Archived base renamer** resolved same-stem conflicts with **version priority**: highest `fontRevision` → oldest `head` created → newest `head` modified (`sort_by_version_priority`).
- **Enhanced renamer** uses **quality scoring** (`sort_by_quality_score` / `calculate_quality_score`) — revision is only one ingredient among glyphs, languages, features, etc.
- **Organizers (still)** use **version priority** via `version_priority.py` (same logic as the archived base). Declutter deliberately did **not** switch organizers to quality scoring.

**Product decision later:** one conflict model for rename + organize, or document two intentional models.

### Base vs Enhanced — CLI gaps

Flag scan at archive time:

- Renamer base had **no** flags that Enhanced lacks.
- Organizer base had **no** flags that Enhanced lacks.
- Enhanced-only renamer flags worth productizing under cleaner names: `-N/--stem`, `-ff`, `--show-quality`, `--explain-quality`, `--recover`, `--no-progress`, `--use-typographic-names`.
- Enhanced-only organizer flags: `-ft`, `-vt`, `-nt`, `-cs`.

No unique base CLI behavior was found beyond the **version-priority** algorithm (now preserved for organizers only).

### Naming / packaging (deferred)

- Drop `Enhanced` from public names; console scripts (`font-rename`, `font-organize`, …).
- README still partially describes pre-Enhanced entry points — refresh when packaging.
- Decide whether Byte Comparator + Variation Analyzer stay in this product or split out.
- `REVIEW_OPTIONS_PRO_CON.md` remains a backlog for Enhanced renamer internals.

## Do not lose

If you delete the `_misc` backup later, keep at least:

1. This file.
2. `version_priority.py` (or fold it into FontCore with tests).
3. Awareness that archived base renamer ≠ Enhanced quality model.

## Packaging pass (2026-11-20)

- Package `fontrenamer/` (renamer, organizer, deduper, version_priority); commands `fontrenamer`, `fontorganizer`, `fontdeduper`.
- Byte_Comparator and Variation_Analyzer retired (git history keeps them).
- Organizer/deduper may later split into their own projects; they are independent modules.
- Still open: one conflict model for rename + organize (quality score vs version priority).

## Conflict model (2026-11-22)

The renamer ranked same-name fonts by quality score and the organizers by version priority; they chose a different winner in 33 of 173 groups (19%) on the intake collection (7,805 fonts), and with `-ra` the renamer chained unrelated fonts that share a bad PostScript name (`Copyright-Weekend-Type`, `~001`, ...) into one "conflict". `tools/conflict_study.py FOLDER` reproduces the numbers (read-only).

**Decided and built (`fontrenamer/conflict.py`, shared by the renamer and the organizers; `version_priority.py` is gone):**

- A same-face check comes first (same weight and slant, glyph counts within 10%; a variable font may pair with a static of another weight; unknown is not evidence). A group that fails it is left alone with a warning.
- Ranking: higher revision, higher version string (nameID 5), more coverage (glyphs, languages, features), then the newer modification date **only when the version is read and equal** (a re-saved or converted file can carry a newer date and an older version), then the file name. Creation date is not used. Our tools keep `head.modified`, so their own changes are not a "newer date"; that is intended.
- On the intake collection after the change: 30 groups (923 fonts, mostly the shared-copyright-name families) are not duplicates and are left alone; the other 143 groups (286 fonts) agree with the old quality-score winner in 118 and with the old version-priority winner in 119. The date decides 106 (equal versions), a higher version string 16, a higher revision 7, coverage 3; 11 cannot be told apart and keep the earlier file name.
- The same-face check needs weight and slant, which only the renamer's metadata has (the organizers' FontCore metadata has glyph count but not weight or slant), so the organizers share the ranking but not yet the check. Their PostScript-name fallback (invalid names use the filename) already avoids the shared-garbage-name chain.
- The renamer's cache file is now `.font_rename_cache_v2.json` (it records weight and slant); an old `.font_rename_cache.json` is left in place and ignored.
- Suffixes unchanged: `~NNN` for losers, `_conflictNNN` for a collision with an unrelated file, `_dupNNN` in quarantine.
