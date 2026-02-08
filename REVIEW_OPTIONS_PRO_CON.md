# Code Review Options: Pro/Con Breakdown

Remaining improvements from the Feb 2026 review that were not implemented in the first round. Use this to decide which are worth pursuing.

---

## 1. Quality score normalization (0–100 or 0–1000)

**What it is:** Scale the internal quality score (currently an unbounded weighted sum, often in the hundreds or low thousands) to a fixed range (e.g. 0–100) before use or display.

| Pro | Con |
|-----|-----|
| Easier to interpret: “85/100” is clearer than “847.3”. | Current logic is “higher = better”; relative order is what matters for sorting, not the scale. |
| Consistent scale across runs and font sets. | Requires choosing and documenting a scaling method (linear vs percentile, baseline). |
| Could support a “minimum quality” filter later (e.g. only rename if score ≥ 70). | Slight extra work in `calculate_quality_score()` and any UI that shows scores. |

**Verdict:** Worth it if you add `--show-quality`/preview UIs or future filtering; otherwise low priority.

---

## 2. Unit tests (quality scoring, conflict resolution, cache, Unicode, symlinks)

**What it is:** Add pytest (or similar) tests for: quality score edge cases, name conflict resolution with many conflicts, cache invalidation, Unicode filenames, symlink handling.

| Pro | Con |
|-----|-----|
| Prevents regressions when changing scoring, cache, or validation. | Setup time: test data (sample fonts, cache files), possibly pytest in the repo. |
| Documents expected behavior (e.g. “empty metadata → score 0”). | Some tests need real font files or mocks for TTFont. |
| Enables safer refactors (e.g. parallel path, config dataclass). | Maintenance: tests must be updated when behavior intentionally changes. |

**Verdict:** High value if you plan ongoing changes or refactors; start with a few tests for quality scoring and validation, then expand.

---

## 3. Configuration dataclass (`RenameConfig`)

**What it is:** Replace the many boolean/optional parameters of `process_directory()` (and similar) with a single `RenameConfig` dataclass (e.g. `recursive`, `dry_run`, `rename_all`, `verbose`, `use_typographic_names`).

| Pro | Con |
|-----|-----|
| Cleaner function signatures; easier to add options without more parameters. | Refactor touch: every caller of `process_directory` and internal helpers must build/pass config. |
| Single place to document and validate options. | Slightly more boilerplate at the CLI (build config from argparse). |
| Easier to pass config into workers or future APIs. | No functional change; purely structural. |

**Verdict:** Nice to have when you add more options or expose an API; not urgent for current CLI-only usage.

---

## 4. Structured return type (`RenameResult`)

**What it is:** Have `process_directory()` (and possibly the main entry) return a dataclass (e.g. `RenameResult(stats, renamed_files, errors)`) instead of only printing and returning `RenameStats`.

| Pro | Con |
|-----|-----|
| Enables scripting and programmatic use (e.g. “how many failed?”, “list renamed paths”). | All call sites must be updated to use the new return type. |
| Easier to test: assert on result instead of console output. | Current design is “print and return stats”; full list of renames/errors may be large in memory. |
| Aligns with “return values, don’t only print” best practice. | If you only need stats, the current return is already sufficient. |

**Verdict:** Do it if you want to call the renamer from other scripts or tests and need structured output; otherwise optional.

---

## 5. Logging module instead of (or in addition to) console output

**What it is:** Use `logging` for warnings/errors/info and keep Rich for user-facing progress; optionally allow log level / log file configuration.

| Pro | Con |
|-----|-----|
| Logs can be written to a file for debugging or support. | Two output paths to maintain (Rich vs logging). |
| Standard levels (DEBUG/INFO/WARNING/ERROR) and filtering. | Risk of duplicate or noisy output if both Rich and logging emit the same events. |
| Better for automation (e.g. parse log files). | Current Rich usage is already clear for interactive use. |

**Verdict:** Add logging if you need file logs or integration with log aggregation; otherwise Rich-only is fine.

---

## 6. Document cache file format for troubleshooting

**What it is:** Add a short section in the script docstring or a separate doc (e.g. in `FileRenamer/`) describing the `.font_rename_cache.json` structure, when it’s used, and how to clear/corrupt it for troubleshooting.

| Pro | Con |
|-----|-----|
| Helps users and future-you when cache behaves oddly. | One-time doc effort; keep in sync if format changes. |
| Complements the existing cache-format comment in `load_cache()`. | Minimal. |

**Verdict:** Low effort, high clarity; worth doing when you next touch cache logic or docs.

---

## Summary

| Option | Effort | Value | Suggested priority |
|--------|--------|--------|--------------------|
| Quality score normalization | Low | Medium (if you show/filter by score) | When adding quality UI/filter |
| Unit tests | Medium | High (for ongoing work) | Start small, then expand |
| RenameConfig dataclass | Medium | Low–medium | When adding more options/API |
| RenameResult return type | Low–medium | Medium (for scripting/tests) | When calling from other code |
| Logging module | Medium | Low–medium | When you need file logs |
| Cache format doc | Low | Medium | Next doc/cache pass |
