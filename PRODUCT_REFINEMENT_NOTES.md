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
