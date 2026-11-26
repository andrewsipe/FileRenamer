"""What a variable font's tables say: which axes are free, whether the slope is free, fixed upright or fixed italic.

The tables decide, not the filename (DECISIONS.md, Area 5). Slope is decided like this:

1. A free ``ital`` or ``slnt`` axis in ``fvar`` (min below max): the slope is free and counts as an axis.
2. Otherwise it is fixed, and these vote: STAT (``ital`` value 1 only = italic, 0 only = upright), the italic bits
   (``fsSelection`` bit 0 or 9, ``macStyle`` bit 1), name cues (ID2, ID17, ID4, ID6 and the filename), and the
   instance names when there are any (all end in an italic word = italic, none = upright, mixed = no vote).
   The majority decides, a tie goes to the name cue, and STAT contradicting the bits and the names is a warning.
3. ``post.italicAngle`` and the caret slope never vote. Surveyed on the vault, they were written wrongly on upright fonts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

_COMMON_AXIS_WORDS = {"wdth": "Width", "opsz": "OpticalSize", "slnt": "Slope", "ital": "Slope"}

# (search word, canonical word). Longest first: "backslanted" contains "slanted".
_ANGLE_WORDS = (
    ("backslanted", "Backslanted"),
    ("backslant", "Backslanted"),
    ("slanted", "Slanted"),
    ("slant", "Slanted"),
    ("oblique", "Oblique"),
    ("inclined", "Inclined"),
    ("reverse", "Reverse"),
    ("kursiv", "Kursiv"),
    ("cursive", "Cursive"),
    ("italic", "Italic"),
)
_ANGLE_ENDING = re.compile("(" + "|".join(w for w, _ in _ANGLE_WORDS) + ")$", re.I)

_WEIGHT_BY_CLASS = {
    100: "Thin", 200: "Extralight", 300: "Light", 400: "Regular", 500: "Medium",
    600: "Semibold", 700: "Bold", 800: "Extrabold", 900: "Black", 1000: "Extrablack",
}


@dataclass(frozen=True)
class VariableTraits:
    is_variable: bool
    free_axes: tuple = ()
    weight_free: bool = False
    slope: Optional[str] = None  # "free" | "upright" | "italic" (None when not variable)
    slope_word: Optional[str] = None  # Italic, Oblique, ... when fixed non-upright
    slope_votes: dict = field(default_factory=dict)
    locked_weight: Optional[str] = None
    warnings: tuple = ()

    @property
    def axis_count(self) -> int:
        return len(self.free_axes)

    @property
    def axes_word(self) -> Optional[str]:
        """Blank for weight only; one common extra axis is named; anything else is a count."""
        n = len(self.free_axes)
        if n == 0:
            return None
        others = [t for t in self.free_axes if t != "wght"]
        if self.weight_free and not others:
            return None
        if len(others) == 1 and others[0] in _COMMON_AXIS_WORDS:
            return _COMMON_AXIS_WORDS[others[0]]
        return f"{n}Axis"


def _name(font: Any, name_id: int) -> str:
    try:
        return font["name"].getDebugName(name_id) or ""
    except Exception:
        return ""


def _angle_word(text: str) -> Optional[str]:
    low = text.lower()
    for word, canonical in _ANGLE_WORDS:
        if word in low:
            return canonical
    return None


def _stat_vote(font: Any) -> Optional[str]:
    if "STAT" not in font:
        return None
    try:
        stat = font["STAT"].table
        axes = stat.DesignAxisRecord.Axis if stat.DesignAxisRecord else []
        tags = [a.AxisTag for a in axes]
        if "ital" not in tags:
            return None
        idx = tags.index("ital")
        values = set()
        for av in (stat.AxisValueArray.AxisValue if stat.AxisValueArray else []):
            if av.Format in (1, 3):
                pairs = [(av.AxisIndex, av.Value)]
            elif av.Format == 2:
                pairs = [(av.AxisIndex, av.NominalValue)]
            elif av.Format == 4:
                pairs = [(r.AxisIndex, r.Value) for r in av.AxisValueRecord]
            else:
                pairs = []
            values.update(v for i, v in pairs if i == idx)
        if values == {1.0} or values == {1}:
            return "italic"
        if values == {0.0} or values == {0}:
            return "upright"
    except Exception:
        return None
    return None


def _instance_vote(font: Any) -> Optional[str]:
    instances = list(font["fvar"].instances)
    if not instances:
        return None
    ending = []
    for inst in instances:
        words = (_name(font, inst.subfamilyNameID)).split()
        ending.append(bool(words) and bool(_ANGLE_ENDING.search(words[-1])))
    if all(ending):
        return "italic"
    if not any(ending):
        return "upright"
    return None


def _bits_vote(font: Any) -> str:
    sel = font["OS/2"].fsSelection if "OS/2" in font else 0
    mac = font["head"].macStyle if "head" in font else 0
    return "italic" if (sel & 1 or sel & (1 << 9) or mac & 2) else "upright"


def analyze_variable_traits(font: Any, filename: Optional[str] = None) -> VariableTraits:
    """Read free axes and slope from the font's tables. ``filename`` (a name or stem) is one name cue."""
    if "fvar" not in font:
        return VariableTraits(is_variable=False)
    fvar = font["fvar"]
    free = tuple(a.axisTag for a in fvar.axes if a.minValue < a.maxValue)
    weight_free = "wght" in free
    warnings: list[str] = []

    locked_weight = None
    if not weight_free:
        weight_class = int(getattr(font["OS/2"], "usWeightClass", 400)) if "OS/2" in font else 400
        word = _WEIGHT_BY_CLASS.get(max(100, min(1000, round(weight_class / 100) * 100)), "Regular")
        if word != "Regular":  # a Regular weight class adds no word (DECISIONS.md, Area 5)
            locked_weight = word
            warnings.append(f"no free weight axis: weight word {word!r} taken from usWeightClass {weight_class}")

    if any(t in free for t in ("ital", "slnt")):
        return VariableTraits(True, free, weight_free, "free", None, {}, locked_weight, tuple(warnings))

    cues = " ".join([_name(font, i) for i in (2, 17, 4, 6)] + ([str(filename)] if filename else []))
    name_italic = _angle_word(cues) is not None
    votes = {
        "stat": _stat_vote(font),
        "bits": _bits_vote(font),
        "name": "italic" if name_italic else "upright",
        "instances": _instance_vote(font),
    }
    cast = [v for v in votes.values() if v]
    italic, upright = cast.count("italic"), cast.count("upright")
    if italic > upright:
        slope = "italic"
    elif upright > italic:
        slope = "upright"
    else:
        slope = votes["name"]
        warnings.append("slope evidence tied; the name decided")

    if votes["stat"] and votes["bits"] == votes["name"] and votes["stat"] != votes["bits"]:
        warnings.append(f"STAT says {votes['stat']} but the italic bits and names say {votes['bits']}")

    word = None
    if slope == "italic":
        word = _angle_word(cues)
        if word is None and "OS/2" in font and font["OS/2"].fsSelection & (1 << 9):
            word = "Oblique"
        if word is None:
            word = "Italic"
            warnings.append("no slope word in the names; Italic assumed")
        elif word != "Italic" and "OS/2" in font and font["OS/2"].fsSelection & (1 << 9) and _angle_word(cues) is None:
            word = "Oblique"
    return VariableTraits(True, free, weight_free, slope, word, votes, locked_weight, tuple(warnings))


__all__ = ["VariableTraits", "analyze_variable_traits"]
