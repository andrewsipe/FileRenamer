# File Renamer

Font file renaming, organizing, and deduplication tools.

## Overview

Tools for renaming font files based on their PostScript names, organizing fonts, detecting duplicates, and analyzing font variations.

## Scripts

### `FontFiles_Renamer.py`
**PostScript name-based renaming** with intelligent version handling.

Renames font files to their PostScript names with:
- Two-pass renaming (temp UUID → PostScript names) to avoid conflicts
- Version-aware priority naming (highest version gets clean name)
- Multiple fonts with same PS name get `~001`, `~002`, etc. suffixes
- Per-directory isolation (processes each directory independently)
- Cached metadata support (speeds up repeated runs)

**Usage:**
```bash
# Rename fonts in a directory
python FontFiles_Renamer.py /path/to/fonts/

# Recursive processing
python FontFiles_Renamer.py /path/to/fonts/ --recursive

# Preview changes
python FontFiles_Renamer.py /path/to/fonts/ --dry-run

# Rename even fonts with invalid PostScript names
python FontFiles_Renamer.py /path/to/fonts/ --rename-all
```

**Options:**
- `-R, --recursive` - Process directories recursively
- `--dry-run` - Preview changes without renaming
- `--rename-all` - Rename even fonts with invalid PostScript names
- `-V, --verbose` - Show detailed processing information

### `FontFiles_Organizer.py`
Organize font files into family-based directory structures.

### `FontFiles_SHA256_Deduplicator.py`
Detect and remove duplicate font files using SHA256 hashing.

**Usage:**
```bash
python FontFiles_SHA256_Deduplicator.py /path/to/fonts -R
```

### `FontFiles_Byte_Comparator.py`
Compare font files byte-by-byte to detect duplicates.

### `FontFiles_Variation_Analyzer.py`
Analyze variable font variations and instances.

## Common Patterns

### Renaming to PostScript Names

The most common use case is renaming fonts to match their PostScript names:

```bash
cd FileRenamer
python FontFiles_Renamer.py /path/to/fonts -R --dry-run  # Preview first
python FontFiles_Renamer.py /path/to/fonts -R           # Apply changes
```

### Deduplication Workflow

1. Find duplicates:
```bash
python FontFiles_SHA256_Deduplicator.py /path/to/fonts -R --dry-run
```

2. Review the output, then remove duplicates:
```bash
python FontFiles_SHA256_Deduplicator.py /path/to/fonts -R
```

## Dependencies

See `requirements.txt`:
- Core dependencies (fonttools, rich) provided by included `core/` library
- No additional dependencies required

## Installation

### Option 1: Install with pipx (Recommended)

pipx installs the tool in an isolated environment:

```bash
# Install directly from GitHub
pipx install git+https://github.com/andrewsipe/FileRenamer.git
```

After installation, run scripts:
```bash
python FontFiles_Renamer.py /path/to/fonts/ --recursive
```

**Upgrade:** `pipx upgrade font-filerenamer`  
**Uninstall:** `pipx uninstall font-filerenamer`

### Option 2: Manual Installation

1. Clone this repository:
```bash
git clone https://github.com/andrewsipe/FileRenamer.git
cd FileRenamer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run scripts:
```bash
python FontFiles_Renamer.py /path/to/fonts/ --recursive
```

## Related Tools

- [Filename_Tools](https://github.com/andrewsipe/Filename_Tools) - Clean and normalize filenames before renaming
- [FontNameID](https://github.com/andrewsipe/FontNameID) - Update PostScript names in font metadata

