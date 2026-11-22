"""The three commands say what they do in plain words, and say that they change files."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ["Process directories recursively", "Preview changes without", "Skip preflight", "Show detailed", "quality-aware", "sanitize_postscript", "Recurse into"]


def help_of(module):
    r = subprocess.run([sys.executable, "-m", f"fontrenamer.{module}", "-h"], capture_output=True, text=True, cwd=ROOT, timeout=120, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr
    return " ".join(r.stdout.split())


@pytest.mark.parametrize("module", ["renamer", "organizer", "deduper"])
def test_standard_flag_words_and_no_old_wording(module):
    text = help_of(module)
    assert "preview only; change nothing" in text and "include subfolders" in text
    assert not [p for p in FORBIDDEN if p in text]


@pytest.mark.parametrize("module", ["renamer", "organizer", "deduper"])
def test_each_command_has_a_heads_up(module):
    text = help_of(module)
    assert "Heads up:" in text and "-n" in text


def test_the_renamer_says_how_conflicts_are_decided():
    assert "best copy keeps it" in help_of("renamer")
