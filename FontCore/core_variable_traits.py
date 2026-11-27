"""What a variable font's tables say: which axes are free, whether the slope is free, fixed upright or fixed italic.

The tables decide, not the filename (DECISIONS.md, Area 5). Slope is decided like this:

1. A free ``ital`` or ``slnt`` axis in ``fvar`` (min below max): the slope is free and counts as an axis.
2. Otherwise it is fixed, and these vote: STAT (``ital`` value 1 only = italic, 0 only = upright), the italic bits
   (``fsSelection`` bit 0 or 9, ``macStyle`` bit 1), name cues (ID2, ID17, ID4, ID6 and the filename), and the
   instance names when there are any (all end in an italic word = italic, none = upright, mixed = no vote).
   The majority decides, a tie goes to the name cue, and STAT contradicting the bits and the names is a warning.
3. ``post.italicAngle`` and the caret slope never vote. Surveyed on the vault, they were written wrongly on upright fonts.

The baseline is weight plus a free slope. ``axes_word`` names only the extras: Width, OpticalSize, then Custom for any
unregistered axis. All four registered axes free make a ``collection`` (the word goes in the family name).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

_REGISTERED = {"wght", "wdth", "opsz", "slnt", "ital"}
_EXTRA_AXIS_WORDS = (("wdth", "Width"), ("opsz", "OpticalSize"))

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
    ("reclined", "Reclined"),
    ("rotalic", "Rotalic"),
    ("tiltalic", "Tiltalic"),
    ("corsiva", "Corsiva"),
    ("cursive", "Cursive"),
    ("italic", "Italic"),
)
_ANGLE_ENDING = re.compile("(" + "|".join(w for w, _ in _ANGLE_WORDS) + ")$", re.I)


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
    italic_angle: float = 0.0  # post.italicAngle as stored (the measured lean, when the font carries one)

    @property
    def axis_count(self) -> int:
        return len(self.free_axes)

    @property
    def no_free_axes(self) -> bool:
        """True when every axis is locked: static in effect, to be named as a static font."""
        return len(self.free_axes) == 0

    @property
    def slope_axis_free(self) -> bool:
        return any(t in self.free_axes for t in ("slnt", "ital"))

    @property
    def collection(self) -> bool:
        """All four registered axes free (weight, width, optical size, slope): a collection or super family."""
        return self.weight_free and "wdth" in self.free_axes and "opsz" in self.free_axes and self.slope_axis_free

    @property
    def axes_word(self) -> Optional[str]:
        """The extras on top of weight and a free slope: Width, OpticalSize, then Custom (None when there are none)."""
        words = []
        if not self.collection:
            words += [word for tag, word in _EXTRA_AXIS_WORDS if tag in self.free_axes]
        if any(t not in _REGISTERED for t in self.free_axes):
            words.append("Custom")
        return "".join(words) or None


def _name(font: Any, name_id: int) -> str:
    try:
        return font["name"].getDebugName(name_id) or ""
    except Exception:
        return ""


def _angle_word(text: str) -> Optional[str]:
    """The slope word in ``text`` as the designer spelled it (Slant stays Slant), first letter capitalised."""
    for word, _canonical in _ANGLE_WORDS:
        m = re.search(word, text, re.I)
        if m:
            return m.group(0).capitalize()
    return None


def _instance_angle_word(font: Any) -> Optional[str]:
    """The most common slope word that ends the named-instance names (Light Backslant -> Backslant)."""
    found: dict = {}
    for inst in font["fvar"].instances:
        words = _name(font, inst.subfamilyNameID).split()
        if words and _ANGLE_ENDING.search(words[-1]):
            w = _angle_word(words[-1])
            if w:
                found[w] = found.get(w, 0) + 1
    return max(found, key=found.get) if found else None


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
    """Italic when every named instance ends in a slope word. Anything else is no vote: a missing slope word is not
    evidence of upright (66 of 69 tied vault fonts were italics whose instances only named weights)."""
    instances = list(font["fvar"].instances)
    if not instances:
        return None
    for inst in instances:
        words = _name(font, inst.subfamilyNameID).split()
        if not (words and _ANGLE_ENDING.search(words[-1])):
            return None
    return "italic"


def _weight_word(text: str) -> Optional[str]:
    squashed = re.sub(r"[\s_-]+", "", text or "").lower()
    for word, canonical in _WEIGHT_WORDS:
        if word in squashed:
            return canonical
    return None


def _bits_vote(font: Any) -> str:
    sel = font["OS/2"].fsSelection if "OS/2" in font else 0
    mac = font["head"].macStyle if "head" in font else 0
    return "italic" if (sel & 1 or sel & (1 << 9) or mac & 2) else "upright"


def _ital_axis_default_leans(font: Any) -> bool:
    """True when a font with a free `ital` axis is italic at its default: it leans right (italicAngle at or beyond -1.5 degrees)
    and either the axis never reaches 0 or the lean is real (5 degrees) and every named instance ends in a slope word. A positive
    italicAngle (the old wrong value) and a plain upright default are not leaning, so those stay free."""
    angle = font["post"].italicAngle if "post" in font else 0
    if angle > -1.5:
        return False
    axis = next((a for a in font["fvar"].axes if a.axisTag == "ital"), None)
    if axis is None:
        return False
    if axis.minValue > 0:
        return True  # the axis never reaches 0, so the font is never upright
    return angle <= -5 and _instance_vote(font) == "italic"


def _without_family(font: Any, cues: str, filename: Optional[str]) -> str:
    """``cues`` with the family text taken out: a slope word that is part of the family (NeulisCursive) is not the slope."""
    families = {_name(font, 16) or _name(font, 1)}
    stem = re.sub(r"\.(ttf|otf|ttc|woff2?)$", "", str(filename or ""), flags=re.I)
    if re.search("variable", stem, re.I):
        families.add(re.split(r"variable", stem, flags=re.I)[0])
    text = cues
    for family in sorted((f for f in families if f), key=lambda f: -len(re.sub(r"[\s_-]+", "", f))):  # the longest first
        squashed = re.sub(r"[\s_-]+", "", family or "")
        if len(squashed) >= 3:
            text = re.sub(r"[\s_-]*".join(re.escape(c) for c in squashed), " ", text, flags=re.I)
    return text


def _filename_says_upright(filename: Optional[str]) -> bool:
    stem = re.sub(r"\.(ttf|otf|ttc|woff2?)$", "", str(filename or ""), flags=re.I)
    return bool(re.search("variable", stem, re.I)) and "upright" in re.split(r"variable", stem, flags=re.I)[-1].lower()


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
        warnings.append("no free weight axis: only a weight the filename states is written (the names and usWeightClass are not evidence of one)")

    if any(t in free for t in ("ital", "slnt")):
        if "ital" in free and _ital_axis_default_leans(font):
            # an italic-only file whose `ital` axis is a degree of italic (it never reaches upright): its slope is Italic, not free
            cues = " ".join([_name(font, i) for i in (2, 17, 4, 6)] + ([str(filename)] if filename else []))
            word = _instance_angle_word(font) or _angle_word(cues) or "Italic"
            return VariableTraits(True, free, weight_free, "italic", word, {}, locked_weight, tuple(warnings))
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

    angle = float(font["post"].italicAngle) if "post" in font else 0.0
    if slope == "italic" and abs(angle) < 0.5 and _filename_says_upright(filename):
        # italic flags and names on a font that does not lean, in a file named Upright (NaraCursive): the lean is the evidence
        slope = "upright"
        warnings.append("italic flags but no lean (italicAngle 0) and an Upright filename: upright")

    word = None
    if slope == "italic":
        word = _instance_angle_word(font) or _angle_word(_without_family(font, cues, filename))
        stem = re.sub(r"\.(ttf|otf|ttc|woff2?)$", "", str(filename or ""), flags=re.I)
        after_variable = re.split(r"variable", stem, flags=re.I)[-1] if re.search("variable", stem, re.I) else ""
        from_file = _angle_word(after_variable) if after_variable else None  # only the slope slot, never the family text (NeulisCursive)
        if from_file and from_file.lower() != "italic":
            word = from_file  # the designer's own word in the filename (Oblique, Slanted) beats the Italic from ID2 and the instances
        if word is None and "OS/2" in font and font["OS/2"].fsSelection & (1 << 9):
            word = "Oblique"
        if word is None:
            word = "Italic"
            warnings.append("no slope word in the names; Italic assumed")
    return VariableTraits(True, free, weight_free, slope, word, votes, locked_weight, tuple(warnings), angle)


__all__ = ["VariableTraits", "analyze_variable_traits"]
