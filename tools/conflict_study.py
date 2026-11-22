"""Read-only study of the conflict model on a folder of fonts (never writes to it).

Groups fonts by PostScript name + format as the renamer does, then reports:
  - how many groups hold 2+ fonts, and how many of those are NOT duplicates (different weight, slant or size),
  - for the real duplicate groups, how often the shared ranking (fontrenamer/conflict.py) picks the same winner as
    the old quality score and as the old version priority (revision, oldest created, newest modified).

    python tools/conflict_study.py FOLDER
"""

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fontrenamer.conflict import cluster_same_face, explain_winner, rank  # noqa: E402
from fontrenamer.renamer import extract_metadata, sort_by_quality_score  # noqa: E402


def old_version_priority(metas):
    def key(m):
        return (
            -m.font_revision if m.font_revision else float("-inf"),
            m.head_created if m.head_created else float("inf"),
            -m.head_modified if m.head_modified else float("-inf"),
        )

    return sorted(metas, key=key)


root = Path(sys.argv[1]).expanduser()
paths = [p for p in root.rglob("*") if p.suffix.lower() in (".otf", ".ttf") and "_archive" not in p.parts and not any(x.startswith(".") for x in p.parts)]
print("fonts:", len(paths), flush=True)
groups = collections.defaultdict(list)
unreadable = 0
for p in paths:
    m = extract_metadata(p, silent=True)
    if m is None or not m.ps_name:
        unreadable += 1
        continue
    groups[(m.ps_name, m.detected_format or p.suffix.lower().lstrip("."))].append(m)
multi = {k: v for k, v in groups.items() if len(v) > 1}
print(f"unreadable or no PostScript name: {unreadable} | groups: {len(groups)} | groups with 2+ fonts: {len(multi)} ({sum(len(v) for v in multi.values())} fonts)")

mixed = {k: v for k, v in multi.items() if len(cluster_same_face(v)) > 1}
real = {k: v for k, v in multi.items() if k not in mixed}
print(f"not duplicates (different weight, slant or size; the renamer leaves them alone): {len(mixed)} groups, {sum(len(v) for v in mixed.values())} fonts")
print(f"real duplicate groups: {len(real)} ({sum(len(v) for v in real.values())} fonts)")

same_q = same_v = 0
reasons = collections.Counter()
examples = []
for k, v in real.items():
    new = rank(list(v))
    q = sort_by_quality_score(list(v))[0]
    o = old_version_priority(list(v))[0]
    same_q += new[0] is q
    same_v += new[0] is o
    reasons[explain_winner(new[0], new[1], new).split(" (")[0]] += 1
    if new[0] is not q and len(examples) < 4:
        examples.append((k, new[0], q))
print(f"new winner = old quality-score winner in {same_q} of {len(real)}; = old version-priority winner in {same_v} of {len(real)}")
print("what decides the winner:", dict(reasons))
for k, n, q in examples:
    print(f"\n  {k}\n    new picks: {Path(n.file_path).name} (revision {n.font_revision}, {n.version_string!r})\n    old quality score picked: {Path(q.file_path).name} (revision {q.font_revision}, {q.version_string!r})")
