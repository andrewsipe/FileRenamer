# Vendored FontCore subset

fontrenamer ships a minimal copy of [FontCore](https://github.com/andrewsipe/FontCore)
so it installs without the full library, a submodule, or a symlink.

**Source:** FontCore 0.11.11 (`v0.11.6-5-g69d803f`)

**Do not edit these files here.** Fix them in FontCore, then re-sync.

## Declared modules

- `core_console_styles.py`
- `core_file_collector.py`
- `core_font_extension.py`
- `core_font_metadata.py`
- `core_font_sorter.py`
- `core_font_utils.py`
- `core_name_policies.py`
- `core_variable_filename_parser.py`
- `core_variable_font_detection.py`
- `core_variable_traits.py`

## Pulled in as imports of the above

- `core_console_config.py`
- `core_filename_parts_parser.py`
- `core_font_style_dictionaries.py`
- `core_logging_config.py`
- `core_string_utils.py`

## Refresh

From a FontCore checkout:

```bash
python tools/vendor_fontcore.py sync ../fontrenamer
```

The module list lives in `[tool.fontcore-vendor]` in this project's `pyproject.toml`.
