"""Every command ends with one sentence and exits 1 when any font had a problem."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from test_conflict_e2e import ROOT, make_font


def run(module: str, *args):
    return subprocess.run(
        [sys.executable, "-m", f"fontrenamer.{module}", *map(str, args)],
        capture_output=True, text=True, cwd=ROOT, timeout=300, stdin=subprocess.DEVNULL,
    )


def flat(r):
    return " ".join((r.stdout + r.stderr).split())


def test_renamer_clean_run(tmp_path):
    make_font(tmp_path / "x.ttf", ps="Probe-Regular")
    r = run("renamer", "--no-preview", "--no-progress", tmp_path)
    assert r.returncode == 0, flat(r)
    assert "no problems." in flat(r)


def test_renamer_unreadable_font_is_a_problem(tmp_path):
    good = make_font(tmp_path / "x.ttf")
    (tmp_path / "Broken.ttf").write_bytes(good.read_bytes()[:120])
    r = run("renamer", "--no-preview", "--no-progress", tmp_path)
    assert r.returncode == 1, flat(r)
    assert "1 problem." in flat(r)
