#!/usr/bin/env python3
"""
Independently verify data/catalog.js against the raw .mtx headers.

Checks, per matrix:
  1. dim     == header n
  2. nnz     == header nnz (3rd token of the size line)
  3. density == nnz / (n*(n+1)/2) * 100   (recomputed fresh)
  4. nnz     == "binned N entries" actually streamed by gen_plots.R (if logged)

Header reads are bounded (first 200 KB) so this is cheap and does not
re-stream the GB files.
"""
import os, re, glob, sys

ROOT    = "/home/abdulfe/rinladownload/mtx"
SUITS   = os.path.dirname(os.path.abspath(__file__))
CATALOG = os.path.join(SUITS, "data", "catalog.js")
LOG     = os.path.join(SUITS, "plots_gen.log")


def header(path):
    with open(path, "r", errors="replace") as fh:
        fh.readline()  # %%MatrixMarket line
        for line in fh:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            p = re.split(r"\s+", s)
            return int(p[0]), int(p[2])
    raise ValueError("no size line")


# --- parse catalog.js ---------------------------------------------------
cat = {}
pat = re.compile(r'name:\s*"([^"]+)",\s*group:\s*"([^"]+)",\s*dim:\s*(\d+),'
                 r'\s*nnz:\s*(\d+),\s*density:\s*([\d.]+)')
with open(CATALOG) as fh:
    for m in pat.finditer(fh.read()):
        name, grp, dim, nnz, dens = m.groups()
        cat[name] = dict(group=grp, dim=int(dim), nnz=int(nnz),
                         density=float(dens))

# --- parse "binned N entries" -> matrix name from the plot log ----------
binned = {}
if os.path.exists(LOG):
    pend = None
    for line in open(LOG, errors="replace"):
        mb = re.search(r"binned ([\d,]+) entries", line)
        if mb:
            pend = int(mb.group(1).replace(",", ""))
            continue
        mw = re.search(r"/([^/]+)_sparsity\.png", line)
        if mw and pend is not None:
            binned[mw.group(1)] = pend
            pend = None

# --- compare ------------------------------------------------------------
mtx = {os.path.basename(f)[:-4]: f
       for f in glob.glob(os.path.join(ROOT, "group*", "*.mtx"))}

problems = []
checked = 0
# Matrices present in the .mtx tree but intentionally excluded from the suite
# (keep in sync with gen_catalog.py EXCLUDE).
EXCLUDE = {"lme4_crossed_n42k", "lme4_crossed_n94k", "lme4_crossed_n146k",
           "inla_graph_net549851", "inla_graph_pid6922_Q",
           "bcsstk12"}  # exact duplicate of bcsstk11
not_in_catalog = sorted(set(mtx) - set(cat) - EXCLUDE)
not_on_disk    = sorted(set(cat) - set(mtx))

for name in sorted(cat):
    if name not in mtx:
        continue
    c = cat[name]
    n, nnz = header(mtx[name])
    checked += 1
    if c["dim"] != n:
        problems.append(f"{name}: dim {c['dim']} != header n {n}")
    if c["nnz"] != nnz:
        problems.append(f"{name}: nnz {c['nnz']} != header nnz {nnz}")
    tri = n * (n + 1) / 2.0
    dens = nnz / tri * 100.0
    if abs(dens - c["density"]) > 5e-4:
        problems.append(f"{name}: density {c['density']} != recomputed {dens:.4f}")
    if name in binned and binned[name] != nnz:
        problems.append(f"{name}: streamed {binned[name]} entries != header nnz {nnz}")

print(f"catalog entries : {len(cat)}")
print(f".mtx on disk    : {len(mtx)}")
print(f"checked         : {checked}")
print(f"nnz cross-check : {sum(1 for n in cat if n in binned)} matrices had a streamed-entry count to compare")
if not_in_catalog:
    print(f"\n.mtx present but NOT in catalog: {not_in_catalog}")
if not_on_disk:
    print(f"\ncatalog entry with NO .mtx file: {not_on_disk}")
if problems:
    print(f"\n*** {len(problems)} MISMATCHES ***")
    for p in problems:
        print("  -", p)
    sys.exit(1)
else:
    print("\nALL MATCH: dim, nnz, density consistent with headers"
          + (" and streamed entry counts" if binned else ""))
