# Font File Renamer tools

Font file renaming, organizing, and deduplication.

## Scripts (current)

| Script | Role |
|--------|------|
| `FontFiles_RenamerEnhanced.py` | Name-table / PS-stem renaming with quality-aware conflict priority |
| `FontFiles_OrganizerEnhanced.py` | Family/vendor organize (+ filename / vendor / name-table filters) |
| `FontFiles_SHA256_Deduplicator.py` | SHA256 duplicate detection |
| `FontFiles_Byte_Comparator.py` | Byte-level file compare |
| `FontFiles_Variation_Analyzer.py` | Variable-font variation analysis |
| `version_priority.py` | Shared helper used by the organizer |

Older base `FontFiles_Renamer.py` / `FontFiles_Organizer.py` live under `_misc/_archive/FileRenamer/`. See `PRODUCT_REFINEMENT_NOTES.md` for product-pass context.

## Quick start

```bash
cd FileRenamer
python FontFiles_RenamerEnhanced.py /path/to/fonts/ -r --dry-run
python FontFiles_RenamerEnhanced.py /path/to/fonts/ -r

python FontFiles_OrganizerEnhanced.py /path/to/fonts/ --dry-run
python FontFiles_SHA256_Deduplicator.py /path/to/fonts -R --dry-run
```

## Related

- [Filename_Tools](https://github.com/andrewsipe/Filename_Tools) — clean/normalize filenames before renaming
- [FontNameID](https://github.com/andrewsipe/FontNameID) — update PostScript / name-table metadata
