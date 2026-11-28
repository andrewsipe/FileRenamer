"""Measure how far a font's letters lean, from the outlines, and compare it with the angle the font declares.

``post.italicAngle`` is written wrongly often enough that it cannot be trusted either way: an upright font may carry 12 degrees,
an italic may carry 0. The outlines do not lie. The measure takes the long straight edges of the stems of H, I, l, E, h, n and
so on, works out how far each leans from vertical, and takes the length-weighted median. The result uses the same sign as
``italicAngle`` (a right-leaning italic is negative, counter-clockwise from vertical is positive).

Read-only. A font with no usable straight stems (rounded or hand-drawn) gives ``None``: say so, do not guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.pens.statisticsPen import StatisticsPen

# Letters whose long straight edges are stems. Serif brackets and terminals are short, so the length filter drops them.
STEM_CHARS = "HIlEhnmuiLTbdpqkr"
UPRIGHT_LIMIT = 1.0  # a measured lean under this many degrees is upright
AGREE_WITHIN = 1.5  # a declared angle this close to the measured one is right
MIN_EDGE_UNITS = 0.22  # an edge counts when it is at least this fraction of the em long
MAX_LEAN = 40.0  # an edge leaning further than this is not a stem edge
MIN_SAMPLES = 4
SHAPE_CHARS = "HIlEL"  # straight-sided letters: the whole-shape lean of these is the font's lean
STAIR_GAP = 3.0  # edges say upright but the whole shape leans this far: a stair-stepped slant, not an upright


@dataclass(frozen=True)
class SlantMeasure:
    measured: Optional[float]  # degrees, italicAngle sign; None when unmeasurable
    samples: int
    spread: float  # how far the middle half of the edges disagree, in degrees; large means a poor measure


@dataclass(frozen=True)
class SlantCheck:
    verdict: str  # ok | upright-declared-slanted | slanted-declared-upright | sign-flipped | angle-off | unmeasurable
    declared: float
    measured: Optional[float]
    note: str


def _edges(glyph_set: Any, name: str, min_len: float) -> Iterable[tuple[float, float]]:
    """(length, lean in degrees from vertical, italicAngle sign) of each long straight edge in the glyph."""
    pen = DecomposingRecordingPen(glyph_set)
    try:
        glyph_set[name].draw(pen)
    except Exception:
        return
    start = last = None
    for op, args in pen.value:
        if op == "moveTo":
            start = last = args[0]
            continue
        if op == "lineTo":
            point = args[0]
            if last is not None:
                yield from _edge(last, point, min_len)
            last = point
        elif op in ("curveTo", "qCurveTo"):
            point = args[-1]
            last = point if point is not None else last
        elif op in ("closePath", "endPath"):
            if op == "closePath" and last is not None and start is not None:
                yield from _edge(last, start, min_len)
            start = last = None


def _edge(a: tuple, b: tuple, min_len: float) -> Iterable[tuple[float, float]]:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < min_len or dy == 0:
        return
    if dy < 0:
        dx, dy = -dx, -dy
    lean = -math.degrees(math.atan2(dx, dy))  # right-leaning edge: negative, like italicAngle
    if abs(lean) <= MAX_LEAN:
        yield length, lean


def _weighted_quantile(items: list[tuple[float, float]], q: float) -> float:
    items = sorted(items, key=lambda x: x[1])
    total = sum(w for w, _ in items)
    run = 0.0
    for w, v in items:
        run += w
        if run >= q * total:
            return v
    return items[-1][1]


def measure_slant(font: Any) -> SlantMeasure:
    """Lean of the stems at the font's default outlines (a variable font is measured at its default)."""
    try:
        cmap = font.getBestCmap() or {}
        glyph_set = font.getGlyphSet()
        upm = font["head"].unitsPerEm
    except Exception:
        return SlantMeasure(None, 0, 0.0)
    min_len = MIN_EDGE_UNITS * upm
    edges: list[tuple[float, float]] = []
    for ch in STEM_CHARS:
        name = cmap.get(ord(ch))
        if name:
            edges.extend(_edges(glyph_set, name, min_len))
    if len(edges) < MIN_SAMPLES:
        return SlantMeasure(None, len(edges), 0.0)
    median = _weighted_quantile(edges, 0.5)
    spread = _weighted_quantile(edges, 0.75) - _weighted_quantile(edges, 0.25)
    return SlantMeasure(round(median, 2) + 0.0, len(edges), round(spread, 2))


def shape_slant(font: Any) -> Optional[float]:
    """Median whole-shape lean (area statistics, italicAngle sign) of the straight-sided capitals, or None."""
    try:
        cmap = font.getBestCmap() or {}
        glyph_set = font.getGlyphSet()
    except Exception:
        return None
    leans = []
    for ch in SHAPE_CHARS:
        name = cmap.get(ord(ch))
        if not name:
            continue
        try:
            pen = StatisticsPen(glyph_set)
            glyph_set[name].draw(pen)
            if pen.slant:
                leans.append(-math.degrees(math.atan(pen.slant)))
            else:
                leans.append(0.0)
        except Exception:
            continue
    if len(leans) < 2:
        return None
    leans.sort()
    return round(leans[len(leans) // 2], 2) + 0.0


def check_slant(font: Any) -> SlantCheck:
    """Compare ``post.italicAngle`` with the measured lean."""
    declared = float(font["post"].italicAngle) if "post" in font else 0.0
    m = measure_slant(font)
    if m.measured is None:
        return SlantCheck("unmeasurable", declared, None, "no straight stems to measure (rounded or hand-drawn letters)")
    if m.spread > 6:
        return SlantCheck("unmeasurable", declared, m.measured, f"the stem edges disagree by {m.spread} degrees; not a reliable measure")
    upright = abs(m.measured) < UPRIGHT_LIMIT
    if upright:
        shape = shape_slant(font)
        if shape is not None and abs(shape) >= STAIR_GAP:
            return SlantCheck("unmeasurable", declared, None,
                              f"every long edge is vertical but the letters lean {shape:g} degrees as a whole (a stair-stepped slant)")
        if abs(declared) < 0.5:
            return SlantCheck("ok", declared, m.measured, "upright outlines, angle 0")
        return SlantCheck("upright-declared-slanted", declared, m.measured, f"outlines are upright but the font declares {declared:g} degrees")
    if abs(declared - m.measured) <= AGREE_WITHIN:
        return SlantCheck("ok", declared, m.measured, "the declared angle matches the outlines")
    if abs(declared) < 0.5:
        return SlantCheck("slanted-declared-upright", declared, m.measured, f"outlines lean {m.measured:g} degrees but the font declares 0")
    if abs(declared + m.measured) <= AGREE_WITHIN:
        return SlantCheck("sign-flipped", declared, m.measured, f"the declared {declared:g} is the outlines' {m.measured:g} with the sign flipped")
    return SlantCheck("angle-off", declared, m.measured, f"the declared {declared:g} differs from the outlines' {m.measured:g}")
