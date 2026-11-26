"""A real rename (not a preview): variable fonts end up with the house-form name, never a _tmp_ stem."""

import subprocess
import sys
from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

ROOT = Path(__file__).resolve().parents[1]


def _font(path: Path, *, variable: bool, style="Regular", family="Probe", axes=(("wght", 100, 400, 900),)) -> Path:
    fb = FontBuilder(1000, isTTF=True)
    order = [".notdef", "H"]
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap({72: "H"})
    pen = TTGlyphPen(None)
    for i, pt in enumerate(((50, 0), (50, 700), (450, 700), (450, 0))):
        (pen.moveTo if i == 0 else pen.lineTo)(pt)
    pen.closePath()
    fb.setupGlyf({".notdef": pen.glyph(), "H": pen.glyph()})
    fb.setupHorizontalMetrics({g: (500, 50) for g in order})
    fb.setupHorizontalHeader(ascent=900, descent=-250)
    fb.setupNameTable({"familyName": family, "styleName": style, "psName": f"{family.replace(' ', '')}-{style}"})
    fb.setupOS2()
    fb.setupPost()
    if variable:
        fb.setupFvar([(t, lo, d, hi, t) for t, lo, d, hi in axes], [])
    fb.font["head"].modified = 3_000_000_000
    fb.save(path)
    return path


def _rename(folder: Path):
    result = subprocess.run(
        [sys.executable, "-m", "fontrenamer.renamer", "--no-preview", "--no-progress", str(folder)],
        capture_output=True, text=True, cwd=ROOT, timeout=300, stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 0, result.stdout[-600:] + result.stderr[-600:]
    return sorted(p.name for p in folder.iterdir() if not p.name.startswith("."))


def test_a_variable_font_gets_its_house_name_in_a_real_rename(tmp_path):
    _font(tmp_path / "Probe-Variable.ttf", variable=True)
    assert _rename(tmp_path) == ["Probe-VariableUpright.ttf"]


def test_the_family_text_of_the_original_filename_survives_the_temp_rename(tmp_path):
    _font(tmp_path / "BT_Avance-Variable.ttf", variable=True, family="BT Avance", axes=(("wght", 100, 400, 900), ("wdth", 75, 100, 125)))
    assert _rename(tmp_path) == ["BT_Avance-WidthVariableUpright.ttf"]


def test_a_mixed_folder_renames_both_kinds(tmp_path):
    _font(tmp_path / "Probe-Variable.ttf", variable=True)
    _font(tmp_path / "Other-Regular.ttf", variable=False, family="Other")
    names = _rename(tmp_path)
    assert "Probe-VariableUpright.ttf" in names and not any(n.startswith("_tmp_") for n in names) and len(names) == 2


# ---- fonts collected from all over: the metadata is often missing or wrong -------------------------------------------

def _break_names(path: Path, *, ps=None, drop_ps=False, drop_name_table=False) -> Path:
    from fontTools.ttLib import TTFont

    f = TTFont(path, recalcTimestamp=False)
    if drop_name_table:
        del f["name"]
    else:
        names = f["name"].names
        f["name"].names = [n for n in names if n.nameID != 6]
        if ps is not None and not drop_ps:
            f["name"].setName(ps, 6, 3, 1, 0x409)
    f.save(path)
    return path


def test_a_variable_font_with_no_postscript_name_is_still_renamed(tmp_path):
    _break_names(_font(tmp_path / "Probe-Variable.ttf", variable=True), drop_ps=True)
    assert _rename(tmp_path) == ["Probe-VariableUpright.ttf"]


def test_a_variable_font_with_an_invalid_postscript_name_is_still_renamed(tmp_path):
    _break_names(_font(tmp_path / "Probe-Variable.ttf", variable=True), ps="Probe Variable (copy)")
    assert _rename(tmp_path) == ["Probe-VariableUpright.ttf"]


def test_a_variable_font_with_no_name_table_is_still_renamed(tmp_path):
    _break_names(_font(tmp_path / "Probe-Variable.ttf", variable=True), drop_name_table=True)
    assert _rename(tmp_path) == ["Probe-VariableUpright.ttf"]


def test_upright_and_italic_files_that_share_one_postscript_name_each_get_their_own_name(tmp_path):
    from fontTools.ttLib import TTFont

    _font(tmp_path / "Probe-Variable.ttf", variable=True)
    italic = _font(tmp_path / "Probe-VariableItalic.ttf", variable=True, style="Italic")
    f = TTFont(italic, recalcTimestamp=False)
    f["OS/2"].fsSelection = (f["OS/2"].fsSelection & ~0x40) | 0x01
    f["head"].macStyle |= 2
    f.save(italic)
    for p in (tmp_path / "Probe-Variable.ttf", italic):
        _break_names(p, ps="Probe-Variable")  # the same, wrong, PostScript name on both
    assert _rename(tmp_path) == ["Probe-VariableItalic.ttf", "Probe-VariableUpright.ttf"]


def test_true_duplicates_keep_one_clean_name_and_number_the_rest(tmp_path):
    _font(tmp_path / "Probe-Variable.ttf", variable=True)
    _font(tmp_path / "Probe-Variable copy.ttf", variable=True)
    names = _rename(tmp_path)
    assert "Probe-VariableUpright.ttf" in names and "Probe-VariableUpright~001.ttf" in names and len(names) == 2


def test_a_variable_font_with_every_axis_locked_is_named_as_a_static_font(tmp_path):
    _font(tmp_path / "Probe-Regular.ttf", variable=True, axes=(("wght", 400, 400, 400),))
    assert _rename(tmp_path) == ["Probe-Regular.ttf"]


# ---- --variable-only: rename the variable fonts of a mixed folder and leave the static ones alone ---------------------

def _rename_with(folder: Path, *flags):
    result = subprocess.run(
        [sys.executable, "-m", "fontrenamer.renamer", *flags, "--no-progress", str(folder)],
        capture_output=True, text=True, cwd=ROOT, timeout=300, stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 0, result.stdout[-600:] + result.stderr[-600:]
    return sorted(p.name for p in folder.iterdir() if not p.name.startswith(".")), result.stdout


def test_variable_only_renames_the_variable_fonts_and_leaves_the_static_ones(tmp_path):
    _font(tmp_path / "Probe-Variable.ttf", variable=True)
    _font(tmp_path / "messy name.ttf", variable=False, family="Other")
    names, _ = _rename_with(tmp_path, "--no-preview", "--variable-only")
    assert names == ["Probe-VariableUpright.ttf", "messy name.ttf"]


def test_without_the_flag_both_kinds_are_renamed(tmp_path):
    _font(tmp_path / "Probe-Variable.ttf", variable=True)
    _font(tmp_path / "messy name.ttf", variable=False, family="Other")
    names, _ = _rename_with(tmp_path, "--no-preview")
    assert "messy name.ttf" not in names and "Probe-VariableUpright.ttf" in names


def test_variable_only_preview_lists_only_variable_fonts(tmp_path):
    _font(tmp_path / "Probe-Variable.ttf", variable=True)
    _font(tmp_path / "messy name.ttf", variable=False, family="Other")
    names, out = _rename_with(tmp_path, "-n", "--variable-only")
    assert "messy name.ttf" in names and "Probe-Variable.ttf" in names  # a preview changes nothing
    assert "Probe-VariableUpright" in out and "Other-Regular" not in out


def test_the_help_lists_the_flag():
    result = subprocess.run([sys.executable, "-m", "fontrenamer.renamer", "-h"], capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert "--variable-only" in result.stdout


# ---- the preview says what it is doing -------------------------------------------------------------------------------

def _preview(folder: Path, *flags, answer=None):
    result = subprocess.run(
        [sys.executable, "-m", "fontrenamer.renamer", *flags, "--no-progress", str(folder)],
        capture_output=True, text=True, cwd=ROOT, timeout=300, input=answer, stdin=None if answer else subprocess.DEVNULL,
    )
    return " ".join(result.stdout.split())  # the terminal wraps long lines


def test_the_preview_counts_variable_fonts_and_says_the_static_ones_were_left_alone(tmp_path):
    _font(tmp_path / "Probe-Variable.ttf", variable=True)
    _font(tmp_path / "Done-VariableUpright.ttf", variable=True, family="Done")
    _font(tmp_path / "messy name.ttf", variable=False, family="Other")
    out = _preview(tmp_path, "-n", "--variable-only")
    assert "Variable fonts: 2 (1 to rename, 1 already named correctly)" in out
    assert "Static fonts left alone: 1" in out
    assert "Variable fonts only" in out


def test_one_method_for_every_row_is_said_once_not_repeated_in_a_column(tmp_path):
    _font(tmp_path / "Probe-Variable.ttf", variable=True)
    out = _preview(tmp_path, "-n", "--variable-only")
    assert "Variable fonts are named from the filename and the font tables" in out
    assert "Method" not in out


def test_a_mixed_result_keeps_the_method_column(tmp_path):
    _font(tmp_path / "Probe-Variable.ttf", variable=True)
    _font(tmp_path / "messy name.ttf", variable=False, family="Other")
    out = _preview(tmp_path, "-n")
    assert "Method" in out and "Variable (tables)" in out and "PostScript (6)" in out


def test_the_prompt_names_where_the_new_names_come_from(tmp_path):
    _font(tmp_path / "Probe-Variable.ttf", variable=True)
    out = _preview(tmp_path, "--variable-only", answer="n\n")
    assert "variable font files (from filename and font tables)" in out and "to PostScript names" not in out
    _font(tmp_path / "messy name.ttf", variable=False, family="Other")
    out = _preview(tmp_path, answer="n\n")
    assert "variable fonts from filename and font tables, the rest to PostScript names" in out
