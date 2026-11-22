"""The organizer uses the same conflict model: copies are ranked, different fonts that share a name are left alone."""

import subprocess
import sys
from pathlib import Path

from test_conflict_e2e import make_font

ROOT = Path(__file__).resolve().parents[1]


def organize(src: Path, out: Path):
    return subprocess.run(
        [sys.executable, "-m", "fontrenamer.organizer", str(src), "-o", str(out), "--no-preview"],
        capture_output=True, text=True, cwd=ROOT, timeout=300, stdin=subprocess.DEVNULL,
    )


def organized_names(out: Path):
    return sorted(p.name for p in out.rglob("*.ttf") if "QUARANTINE" not in p.parts)


def test_true_copies_are_ranked_and_suffixed(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    make_font(src / "x.ttf", revision=1.0)
    make_font(src / "y.ttf", revision=2.0)
    organize(src, out)
    assert organized_names(out) == ["Probe-Regular.ttf", "Probe-Regular~001.ttf"]


def test_different_fonts_sharing_a_name_keep_their_own_names(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    make_font(src / "Thin-One.ttf", weight=100)
    make_font(src / "Bold-One.ttf", weight=700)
    result = organize(src, out)
    assert organized_names(out) == ["Bold-One.ttf", "Thin-One.ttf"], result.stdout[-800:]
    assert "different fonts" in result.stdout + result.stderr


def test_a_lower_version_with_a_newer_date_does_not_win(tmp_path):
    from test_conflict_e2e import NEW_DATE, OLD_DATE

    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    make_font(src / "resaved.ttf", version="Version 1.000", modified=NEW_DATE)
    make_font(src / "original.ttf", version="Version 1.100", modified=OLD_DATE)
    organize(src, out)
    from fontTools.ttLib import TTFont

    clean = next(p for p in out.rglob("Probe-Regular.ttf"))
    assert TTFont(clean, recalcTimestamp=False)["name"].getName(5, 3, 1, 0x409).toUnicode() == "Version 1.100"
