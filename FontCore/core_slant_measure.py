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
SHALLOW_LEAN = 1.5  # two measures that both lean this far, the same way, outvote edges that read upright
CONFIRM_SPREAD = 0.5  # the stem edges must agree this closely (degrees) before a second measure is asked to confirm them
CONFIRM_SAMPLES = 10
CONFIRM_AGREE = 1.5  # the stem centre-line must land this close to the edge measure
CENTRE_CHARS = "IlH"
CENTRE_ROWS = (0.3, 0.4, 0.6, 0.7)  # heights, as a fraction of the glyph, clear of serifs, feet and crossbars


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


def _flatten(recorded: list, steps: int = 12) -> list[list[tuple[float, float]]]:
    """Closed outlines as polylines (curves cut into ``steps`` pieces)."""
    polys: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []
    last: Optional[tuple] = None

    def curve(p0: tuple, pts: list) -> list[tuple[float, float]]:
        out = []
        for i in range(1, steps + 1):
            t = i / steps
            if len(pts) == 2:
                a, b = pts
                x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * a[0] + t * t * b[0]
                y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * a[1] + t * t * b[1]
            else:
                a, b, c = pts
                x = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * a[0] + 3 * (1 - t) * t * t * b[0] + t**3 * c[0]
                y = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * a[1] + 3 * (1 - t) * t * t * b[1] + t**3 * c[1]
            out.append((x, y))
        return out

    for op, args in recorded:
        if op == "moveTo":
            cur, last = [args[0]], args[0]
        elif op == "lineTo":
            cur.append(args[0])
            last = args[0]
        elif op == "curveTo":
            cur += curve(last, list(args))
            last = args[-1]
        elif op == "qCurveTo":
            pts = list(args)
            if pts[-1] is None:  # a closed outline of off-curve points only: skipped
                continue
            offs, end, p0 = pts[:-1], pts[-1], last
            for i, off in enumerate(offs):
                nxt = end if i == len(offs) - 1 else ((off[0] + offs[i + 1][0]) / 2, (off[1] + offs[i + 1][1]) / 2)
                cur += curve(p0, [off, nxt])
                p0 = nxt
            last = end
        elif op in ("closePath", "endPath"):
            if len(cur) > 2:
                polys.append(cur)
            cur = []
    return polys


def _ink_spans(polys: list, y: float) -> list[tuple[float, float]]:
    """The x ranges that are inked at height ``y`` (non-zero winding), left to right."""
    crossings = []
    for poly in polys:
        for i in range(len(poly)):
            (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % len(poly)]
            if y0 != y1 and (y0 <= y < y1 or y1 <= y < y0):
                crossings.append((x0 + (y - y0) * (x1 - x0) / (y1 - y0), 1 if y1 > y0 else -1))
    crossings.sort()
    spans, winding, start = [], 0, None
    for x, direction in crossings:
        before = winding
        winding += direction
        if before == 0 and winding != 0:
            start = x
        elif before != 0 and winding == 0 and start is not None:
            spans.append((start, x))
    return spans


def stem_centre_lean(font: Any) -> Optional[float]:
    """Lean (italicAngle sign) of the centre line of the first stem of I, l and H, read at heights clear of serifs and
    crossbars; the median over those letters, or None. A second opinion on the stem edges: it follows where the stroke
    runs, not where its edges and terminals point."""
    try:
        cmap = font.getBestCmap() or {}
        glyph_set = font.getGlyphSet()
    except Exception:
        return None
    leans = []
    for ch in CENTRE_CHARS:
        name = cmap.get(ord(ch))
        if not name:
            continue
        pen = DecomposingRecordingPen(glyph_set)
        try:
            glyph_set[name].draw(pen)
        except Exception:
            continue
        polys = _flatten(pen.value)
        ys = [pt[1] for poly in polys for pt in poly]
        if not ys or max(ys) <= min(ys):
            continue
        low, height = min(ys), max(ys) - min(ys)
        points = []
        for row in CENTRE_ROWS:
            y = low + row * height
            spans = _ink_spans(polys, y)
            if not spans or spans[0][1] <= spans[0][0]:
                points = []
                break
            points.append((y, (spans[0][0] + spans[0][1]) / 2))
        if len(points) < 3:
            continue
        mean_y = sum(p[0] for p in points) / len(points)
        mean_x = sum(p[1] for p in points) / len(points)
        spread_y = sum((p[0] - mean_y) ** 2 for p in points)
        if spread_y == 0:
            continue
        slope = sum((p[0] - mean_y) * (p[1] - mean_x) for p in points) / spread_y  # dx/dy
        leans.append(-math.degrees(math.atan(slope)))
    if not leans:
        return None
    leans.sort()
    return round(leans[len(leans) // 2], 2) + 0.0


def confirmed_slant(font: Any) -> Optional[float]:
    """The stem-edge lean when a second, independent measure agrees with it, else None.

    The edges must lean at least ``UPRIGHT_LIMIT`` degrees, agree among themselves (spread at most ``CONFIRM_SPREAD`` over
    ``CONFIRM_SAMPLES`` edges or more), and the centre line of a stem must land within ``CONFIRM_AGREE`` of them. A font
    that fails any of these is not confirmed, which is not the same as wrong: say nothing about it."""
    m = measure_slant(font)
    if m.measured is None or abs(m.measured) < UPRIGHT_LIMIT or m.spread > CONFIRM_SPREAD or m.samples < CONFIRM_SAMPLES:
        return None
    centre = stem_centre_lean(font)
    if centre is None or abs(centre - m.measured) > CONFIRM_AGREE:
        return None
    return m.measured


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
