"""End to end: the renamer applies the shared conflict model to real files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
OLD_DATE = 3_000_000_000
NEW_DATE = 3_700_000_000


def make_font(path: Path, *, ps="Probe-Regular", revision=1.0, version="Version 1.000", weight=400, modified=OLD_DATE, glyphs=0):
    fb = FontBuilder(1000, isTTF=True)
    order = [".notdef", "H"] + [f"g{i}" for i in range(glyphs)]
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap({72: "H"})
    pen = TTGlyphPen(None)
    pen.moveTo((50, 0)); pen.lineTo((50, 700)); pen.lineTo((450, 700)); pen.lineTo((450, 0)); pen.closePath()
    glyph = pen.glyph()
    fb.setupGlyf({g: glyph for g in order})
    fb.setupHorizontalMetrics({g: (500, 50) for g in order})
    fb.setupHorizontalHeader(ascent=900, descent=-250)
    fb.setupNameTable({"familyName": "Probe", "styleName": "Regular"})
    fb.setupOS2(usWeightClass=weight)
    fb.setupPost()
    f = fb.font
    f["head"].fontRevision = revision
    f["head"].created = OLD_DATE
    f["head"].modified = modified
    f["name"].setName(ps, 6, 3, 1, 0x409)
    f["name"].setName(version, 5, 3, 1, 0x409)
    fb.save(path)
    return path


def run_renamer(folder: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "fontrenamer.renamer", "--no-preview", "--no-progress", str(folder)],
        capture_output=True, text=True, cwd=ROOT, timeout=300, stdin=subprocess.DEVNULL,
    )


def stems(folder: Path):
    return sorted(p.name for p in folder.iterdir() if p.suffix == ".ttf")


def revision_of(path: Path) -> float:
    return round(TTFont(path, recalcTimestamp=False)["head"].fontRevision, 3)


def test_the_higher_revision_keeps_the_clean_name(tmp_path):
    make_font(tmp_path / "x.ttf", revision=1.0)
    make_font(tmp_path / "y.ttf", revision=2.0)
    run_renamer(tmp_path)
    assert stems(tmp_path) == ["Probe-Regular.ttf", "Probe-Regular~001.ttf"]
    assert revision_of(tmp_path / "Probe-Regular.ttf") == 2.0


def test_a_newer_file_date_does_not_beat_a_higher_version(tmp_path):
    # same revision; one is a re-save of an older version with a newer date
    make_font(tmp_path / "resaved.ttf", version="Version 1.000", modified=NEW_DATE)
    make_font(tmp_path / "original.ttf", version="Version 1.100", modified=OLD_DATE)
    run_renamer(tmp_path)
    winner = TTFont(tmp_path / "Probe-Regular.ttf", recalcTimestamp=False)
    assert winner["name"].getName(5, 3, 1, 0x409).toUnicode() == "Version 1.100"


def test_the_date_decides_when_the_version_is_the_same(tmp_path):
    make_font(tmp_path / "older.ttf", modified=OLD_DATE)
    make_font(tmp_path / "newer.ttf", modified=NEW_DATE)
    run_renamer(tmp_path)
    assert TTFont(tmp_path / "Probe-Regular.ttf", recalcTimestamp=False)["head"].modified == NEW_DATE


def test_fonts_of_different_weights_sharing_a_name_are_not_renamed_to_it(tmp_path):
    make_font(tmp_path / "Thin-One.ttf", weight=100)
    make_font(tmp_path / "Bold-One.ttf", weight=700)
    result = run_renamer(tmp_path)
    assert stems(tmp_path) == ["Bold-One.ttf", "Thin-One.ttf"]
    assert "different fonts" in result.stdout + result.stderr


def test_true_duplicates_among_a_mixed_group_are_not_ranked_either(tmp_path):
    for n, w in (("a", 100), ("b", 700), ("c", 700)):
        make_font(tmp_path / f"{n}.ttf", weight=w)
    run_renamer(tmp_path)
    assert stems(tmp_path) == ["a.ttf", "b.ttf", "c.ttf"]  # the name cannot identify one face: leave all alone


def test_a_font_alone_is_still_renamed(tmp_path):
    make_font(tmp_path / "whatever.ttf")
    run_renamer(tmp_path)
    assert stems(tmp_path) == ["Probe-Regular.ttf"]
