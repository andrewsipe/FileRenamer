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
