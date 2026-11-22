"""Read-only study of the conflict model: group fonts by PostScript name + format (as the renamer does) and compare
the renamer's quality-score ranking with the organizers' version-priority ranking.

    python tools/conflict_study.py FOLDER

Never writes to FOLDER. Prints counts and a few examples; nothing is saved."""
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fontrenamer.renamer import extract_metadata, sort_by_quality_score
from fontrenamer.version_priority import sort_by_version_priority

root = Path(sys.argv[1]).expanduser()
paths = [p for p in root.rglob("*") if p.suffix.lower() in (".otf", ".ttf") and "_archive" not in p.parts and not any(x.startswith(".") for x in p.parts)]
print("fonts:", len(paths), flush=True)
groups = collections.defaultdict(list)
fail = 0
for i, p in enumerate(paths):
    m = extract_metadata(p, silent=True)
    if m is None or not m.ps_name:
        fail += 1; continue
    groups[(m.ps_name, m.detected_format or p.suffix.lower().lstrip("."))].append(m)
dups = {k: v for k, v in groups.items() if len(v) > 1}
print("unreadable/no PS name:", fail, "| groups:", len(groups), "| groups with 2+ fonts:", len(dups), "| fonts in them:", sum(len(v) for v in dups.values()))
agree = disagree = 0
rows = []
for k, v in dups.items():
    q = sort_by_quality_score(list(v))[0]
    r = sort_by_version_priority(list(v))[0]
    same = q is r
    agree += same; disagree += (not same)
    if not same: rows.append((k, q, r, v))
print("same winner:", agree, "different winner:", disagree)
def desc(m): return f"{Path(m.file_path).parent.name[-28:]}/{Path(m.file_path).name} rev={m.font_revision} glyphs={m.glyph_count} feats={len(m.opentype_features)} created={m.head_created} modified={m.head_modified}"
for k, q, r, v in rows[:12]:
    print("\n", k, f"({len(v)} fonts)"); print("   quality  picks:", desc(q)); print("   version  picks:", desc(r))
# why they differ: what separates the quality winner from the version winner
why = collections.Counter()
for k, q, r, v in rows:
    if (q.font_revision or 0) < (r.font_revision or 0): why["quality winner has a LOWER revision (language/features/glyphs outweigh it)"] += 1
    elif (q.font_revision or 0) > (r.font_revision or 0): why["quality winner has a higher revision (version priority chose on dates, revisions tied or missing)"] += 1
    else: why["same revision: dates vs features/glyphs/languages decide"] += 1
print("\nwhy:", dict(why))

# ---- are these groups really duplicates of one face? ----
import re
def face_key(m):
    t = (m.full_font_name or "").lower()
    return re.sub(r"[^a-z0-9]", "", t)
pure, mixed = [], []
for k, v in dups.items():
    faces = {face_key(m) for m in v}
    (pure if len(faces) == 1 else mixed).append((k, v, faces))
print("\n== group purity (by full name, nameID 4) ==")
print("groups where every font has the same full name (true duplicates):", len(pure), "| fonts:", sum(len(v) for _, v, _ in pure))
print("groups that mix different full names (the PostScript name does not identify one face):", len(mixed), "| fonts:", sum(len(v) for _, v, _ in mixed))
print("fonts that would get a ~NNN suffix because of a mixed group:", sum(len(v) - 1 for _, v, _ in mixed))
print("fonts that would get a ~NNN suffix in true-duplicate groups:", sum(len(v) - 1 for _, v, _ in pure))
for k, v, faces in sorted(mixed, key=lambda x: -len(x[1]))[:6]:
    print("  ", k, f"{len(v)} fonts, {len(faces)} different full names, e.g.", sorted(faces)[:3])
a = d = 0
for k, v, _ in pure:
    q = sort_by_quality_score(list(v))[0]; r = sort_by_version_priority(list(v))[0]
    a += q is r; d += q is not r
print("\ntrue-duplicate groups: same winner", a, "| different winner", d)
# variable vs static inside pure groups

print("\n== the biggest groups ==")
for k, v in sorted(dups.items(), key=lambda kv: -len(kv[1]))[:8]:
    ids4 = collections.Counter((m.full_font_name or "") for m in v)
    st = collections.Counter(Path(m.file_path).stem.split("-")[-1] if "-" in Path(m.file_path).stem else "" for m in v)
    fam = collections.Counter(Path(m.file_path).parent.name for m in v)
    print(f"  {k}: {len(v)} fonts | distinct full names {len(ids4)} (top: {ids4.most_common(2)}) | distinct filenames {len({Path(m.file_path).name for m in v})} | folders {len(fam)}")
