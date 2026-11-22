# FontRenamer

**Version 0.2.2**

Rename font files from their own name tables, and keep the best copy when two fonts want the same name. Two smaller tools in the same package organize a collection into folders and remove exact duplicates.

Typical pipeline: clean filenames ([Filename_Tools](https://github.com/andrewsipe/Filename_Tools)) → fix name-table metadata ([FontNameID](https://github.com/andrewsipe/FontNameID)) → **fontrenamer** → `fontorganizer`.

## Install

Preferred: [uv](https://docs.astral.sh/uv/):

```bash
uv tool install "git+https://github.com/andrewsipe/FileRenamer.git"
# later: uv tool upgrade fontrenamer
```

Or [pipx](https://pipx.pypa.io/): `pipx install "git+https://github.com/andrewsipe/FileRenamer.git"`.
From a clone: `uv tool install --editable .` (edits take effect immediately).

Requires Python 3.9+. FontCore is vendored in `FontCore/` (see `FontCore/VENDOR.md`); there is nothing else to install.

## Commands

| Command | What it does |
|---|---|
| `fontrenamer` | Rename fonts from the name table. Picks the best file when names collide (quality-scored). |
| `fontorganizer` | Sort fonts into family / vendor folders, with filename, vendor and name filters. |
| `fontdeduper` | Remove byte-for-byte duplicates (SHA256). |

## Quick start

```bash
fontrenamer /path/to/fonts -r -n      # preview, recursive
fontrenamer /path/to/fonts -r         # rename for real
fontrenamer /path/to/fonts -N typo    # name stem from nameIDs 16+17

fontorganizer /path/to/fonts -n
fontdeduper /path/to/fonts -r -n
```

`-N` picks the filename stem: `ps` (nameID 6, default), `typo` (16+17), `full` (4), `legacy` (1+2), `uid` (3), or `best` (per file). `--show-quality` and `--explain-quality` show why one file won a collision. Run any command with `--help` for the full list.

## Development

```bash
uv pip install -e . pytest && pytest
```

Refresh the vendored FontCore from a FontCore checkout: `python tools/vendor_fontcore.py sync ../FileRenamer`.
