#!/usr/bin/env python3
"""
Generate one detail page (matrix_<name>.html) per matrix from data/catalog.js.

Each page shows the matrix facts (n, nnz, density, group, file size, format),
the full-size sparsity plot, and a prominent download link to the matrix on
the r-inla download server. No benchmark data yet (added later).

Download URL layout mirrors the local tree:
    https://esmail.r-inla-download.org/mtx/<group>/<name>.mtx
"""
import os, re, glob

SUITS    = os.path.dirname(os.path.abspath(__file__))
CATALOG  = os.path.join(SUITS, "data", "catalog.js")
MTX_ROOT = "/home/abdulfe/rinladownload/mtx"
DL_BASE  = "https://esmail.r-inla-download.org/mtx"
RES_DIR  = "/home/abdulfe/Documents/ideas/adv_sTiles/docs/paper/Semisparse___sTiles/results-latest"

# Solvers shown on each page; sTiles first (the home solver), then competitors.
SOLVERS = ["sTiles", "PARDISO", "MUMPS", "CHOLMOD", "PaStiX", "symPACK"]
LIB_FILES = {
    "sTiles": "stiles/stiles_auto.csv",
    "PARDISO": "pardiso/pardiso.csv", "MUMPS": "mumps/mumps.csv",
    "CHOLMOD": "cholmod/cholmod.csv", "PaStiX": "pastix/pastix.csv",
    "symPACK": "sympack/sympack.csv",
}


def human(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return (f"{nbytes:.0f} {unit}" if unit == "B"
                    else f"{nbytes/1.0:.1f} {unit}")
        nbytes /= 1024.0


def fmt_pct(x):
    if x >= 1:    return f"{x:.2f}%"
    if x >= 0.01: return f"{x:.3f}%"
    return f"{x:.4f}%"


# --- load catalog -------------------------------------------------------
pat = re.compile(r'name:\s*"([^"]+)",\s*group:\s*"([^"]+)",\s*dim:\s*(\d+),'
                 r'\s*nnz:\s*(\d+),\s*density:\s*([\d.]+),\s*domain:\s*"([^"]+)"')
mats = []
with open(CATALOG) as fh:
    for m in pat.finditer(fh.read()):
        name, grp, dim, nnz, dens, dom = m.groups()
        mats.append(dict(name=name, group=grp, dim=int(dim), nnz=int(nnz),
                         density=float(dens), domain=dom))

# local file sizes (for display)
local = {os.path.basename(f)[:-4]: f
         for f in glob.glob(os.path.join(MTX_ROOT, "group*", "*.mtx"))}


# --- competitor benchmark results --------------------------------------
def _load_agg(path):
    """numeric.csv / symbolic.csv -> dict[matrix][solver] = float | None."""
    out = {}
    if not os.path.exists(path):
        return out
    lines = open(path).read().splitlines()
    solvers = lines[0].split(",")[4:]          # cols 0-3 = matrix,n,nnz,source
    for ln in lines[1:]:
        if not ln.strip():
            continue
        f = ln.split(",")
        d = {}
        for k, sol in enumerate(solvers):
            v = f[4 + k].strip() if 4 + k < len(f) else ""
            d[sol] = float(v) if v not in ("", "nan", "NA") else None
        out[f[0]] = d
    return out


CORES = [1, 2, 4, 8, 16, 32, 40]


def _load_sweeps():
    """Per-library numeric factorization sweep: out[matrix][solver] = {core: s}."""
    out = {}
    for sol, rel in LIB_FILES.items():
        p = os.path.join(RES_DIR, rel)
        if not os.path.exists(p):
            continue
        lines = open(p).read().splitlines()
        hdr = lines[0].split(",")
        cidx = {c: hdr.index("c" + str(c)) for c in CORES if ("c" + str(c)) in hdr}
        for ln in lines[1:]:
            if not ln.strip():
                continue
            f = ln.split(",")
            d = {}
            for c, ci in cidx.items():
                v = f[ci].strip() if ci < len(f) else ""
                d[c] = float(v) if v not in ("", "nan", "NA") else None
            out.setdefault(f[0], {})[sol] = d
    return out


SYMBOLIC = _load_agg(os.path.join(RES_DIR, "symbolic.csv"))
SWEEPS   = _load_sweeps()


def fmt_t(v):
    return "n/a" if v is None else f"{v:.6g}"


def _label(s):
    return (s + ' <span class="tag">this work</span>') if s == "sTiles" else s


# NOTE: benchmark tables (numerical factorization + symbolic/bitset preprocessing)
# are intentionally NOT rendered on the matrix pages. Each page keeps only the
# matrix properties, sparsity plot, and .mtx download link. build_benchmarks() is
# kept for reference; to restore the tables, add its output back into TEMPLATE.
def build_benchmarks(name):
    sweeps = SWEEPS.get(name)
    sym = SYMBOLIC.get(name) or {}
    if not sweeps:
        return ('<div class="bench"><div class="lbl">Solver benchmarks</div>'
                '<div class="bench-pending">Benchmark results pending for this '
                'matrix.</div></div>')
    solver_list = [s for s in SOLVERS if s in sweeps]

    # fastest single cell across all solvers and all thread counts
    allv = [v for s in solver_list for v in sweeps[s].values() if v is not None]
    gmin = min(allv) if allv else None

    head = "".join("<th>" + str(c) + "</th>" for c in CORES)
    rows = ""
    for s in solver_list:
        d = sweeps[s]
        rv = [d.get(c) for c in CORES if d.get(c) is not None]
        rmin = min(rv) if rv else None          # fastest thread count for this solver
        cells = ""
        for c in CORES:
            v = d.get(c)
            cls = ""
            if v is not None and gmin is not None and v == gmin:
                cls = ' class="gmin"'
            elif v is not None and rmin is not None and v == rmin:
                cls = ' class="rmin"'
            cells += "<td" + cls + ">" + fmt_t(v) + "</td>"
        home = ' class="home"' if s == "sTiles" else ''
        rows += '<tr' + home + '><td class="solver">' + _label(s) + '</td>' + cells + '</tr>'

    numeric_tbl = (
        '<div class="lbl">Numerical factorization time (s) by thread count'
        ' &middot; Intel 2&times;Xeon Gold, 40 cores</div>'
        '<div class="bench-table-wrap"><table class="bench-table"><thead><tr>'
        '<th>Solver</th>' + head + '</tr></thead><tbody>' + rows + '</tbody></table></div>'
        '<div class="bench-legend"><span class="sw rmin"></span>fastest thread count for that '
        'solver<span class="sw gmin"></span>fastest overall</div>')

    # Symbolic analysis (a.k.a. bitset preprocessing) table is intentionally
    # hidden from the public matrix pages. The data is still loaded above; only
    # its rendering is suppressed. Re-enable by appending symbolic_tbl below.
    return '<div class="bench">' + numeric_tbl + '</div>'

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{name} | sTiles Matrix</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --lime:#c5d86d; --lime-dark:#a8bd4d; --orange:#e08c5a; --orange-dark:#c97442;
            --teal:#4a9a8f; --teal-light:#6bb3a8; --teal-dark:#3a7a71;
            --black:#1a1a1a; --charcoal:#2d2d2d; --bg-primary:#fafafa; --bg-card:#ffffff;
            --text-primary:#1a1a1a; --text-secondary:#555; --text-muted:#888;
            --border:#e0e0e0; --border-light:#f0f0f0;
            --gradient-hero:linear-gradient(180deg, var(--black) 0%, var(--charcoal) 100%);
        }}
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        body {{ font-family:'Inter',-apple-system,sans-serif; background:var(--bg-primary);
               color:var(--text-primary); line-height:1.7; font-size:16px; }}
        .container {{ max-width:1100px; margin:0 auto; padding:0 24px; }}
        nav {{ position:fixed; top:0; left:0; right:0; z-index:1000;
              background:rgba(255,255,255,0.95); backdrop-filter:blur(12px);
              border-bottom:1px solid var(--border); }}
        .nav-content {{ display:flex; align-items:center; justify-content:space-between;
              padding:14px 24px; max-width:1100px; margin:0 auto; }}
        .nav-brand {{ display:flex; align-items:center; gap:10px; text-decoration:none; }}
        .nav-brand img {{ height:30px; border-radius:4px; display:block; }}
        .nav-sub {{ font-size:0.85rem; color:var(--text-muted); }}
        .nav-links a {{ font-size:0.85rem; color:var(--teal); text-decoration:none;
              font-weight:600; margin-left:18px; }}
        .hero {{ padding:104px 0 44px; background:var(--gradient-hero); color:#fff;
              text-align:center; position:relative; overflow:hidden; }}
        .hero::after {{ content:''; position:absolute; bottom:0; left:0; right:0; height:60px;
              background:linear-gradient(to bottom, transparent, var(--bg-primary)); }}
        .hero-content {{ position:relative; z-index:1; }}
        .hero .grp {{ display:inline-block; font-family:'JetBrains Mono',monospace;
              font-size:0.72rem; font-weight:600; text-transform:uppercase;
              letter-spacing:0.1em; color:var(--lime); background:rgba(197,216,109,0.14);
              border:1px solid rgba(197,216,109,0.3); padding:4px 12px; border-radius:100px;
              margin-bottom:14px; }}
        .hero h1 {{ font-family:'JetBrains Mono',monospace; font-size:2.1rem; font-weight:700;
              letter-spacing:-0.01em; word-break:break-all; }}
        .hero .sub {{ color:rgba(255,255,255,0.6); font-size:0.95rem; margin-top:8px; }}
        section {{ padding:36px 0 56px; }}
        .dl-row {{ display:flex; justify-content:center; margin:-12px 0 30px; }}
        .dl-btn {{ display:inline-flex; align-items:center; gap:10px;
              background:var(--teal); color:#fff; text-decoration:none; font-weight:600;
              padding:13px 26px; border-radius:12px; font-size:0.95rem;
              box-shadow:0 6px 20px rgba(74,154,143,0.3); transition:transform .12s, box-shadow .12s; }}
        .dl-btn:hover {{ transform:translateY(-2px); box-shadow:0 10px 26px rgba(74,154,143,0.4); }}
        .dl-btn .sz {{ font-family:'JetBrains Mono',monospace; font-size:0.82rem;
              font-weight:500; opacity:0.85; }}
        .cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(150px,1fr));
              gap:14px; margin-bottom:34px; }}
        .card {{ background:var(--bg-card); border:1px solid var(--border); border-radius:12px;
              padding:16px 18px; box-shadow:0 2px 10px rgba(0,0,0,0.04); }}
        .card .lbl {{ font-size:0.7rem; font-weight:700; text-transform:uppercase;
              letter-spacing:0.07em; color:var(--text-muted); margin-bottom:6px; }}
        .card .val {{ font-family:'JetBrains Mono',monospace; font-size:1.15rem; font-weight:600;
              color:var(--teal-dark); }}
        .plot-wrap {{ background:var(--bg-card); border:1px solid var(--border); border-radius:14px;
              padding:18px; box-shadow:0 4px 20px rgba(0,0,0,0.05); text-align:center; }}
        .plot-wrap .lbl {{ font-size:0.7rem; font-weight:700; text-transform:uppercase;
              letter-spacing:0.07em; color:var(--text-muted); margin-bottom:14px; }}
        .plot-wrap img {{ width:100%; max-width:680px; height:auto; border:1px solid var(--border-light);
              border-radius:8px; background:#fff; }}
        .plot-wrap .no {{ color:var(--text-muted); font-style:italic; padding:60px 0; }}
        .bench {{ margin-top:38px; }}
        .bench .lbl {{ font-size:0.7rem; font-weight:700; text-transform:uppercase;
              letter-spacing:0.07em; color:var(--text-muted); margin-bottom:14px; }}
        .bench-table-wrap {{ border:1px solid var(--border); border-radius:12px; overflow-x:auto;
              box-shadow:0 2px 10px rgba(0,0,0,0.04); }}
        .bench-table {{ width:100%; border-collapse:collapse; background:var(--bg-card); font-size:0.84rem; }}
        .bench-table thead th {{ background:var(--black); color:#fff; text-align:right;
              padding:10px 13px; font-size:0.7rem; font-weight:600; text-transform:uppercase;
              letter-spacing:0.04em; white-space:nowrap; }}
        .bench-table thead th:first-child {{ text-align:left; }}
        .bench-table td {{ padding:9px 13px; border-bottom:1px solid var(--border-light);
              font-family:'JetBrains Mono',monospace; font-size:0.8rem; text-align:right;
              white-space:nowrap; color:var(--text-secondary); }}
        .bench-table td.solver {{ font-family:'Inter',sans-serif; font-weight:600;
              color:var(--text-primary); text-align:left; }}
        .bench-table tbody tr:nth-child(even) {{ background:#f8f9fa; }}
        .bench-table tbody tr:last-child td {{ border-bottom:none; }}
        .bench-table td.rmin {{ color:var(--teal-dark); font-weight:700; background:rgba(74,154,143,0.16); }}
        .bench-table td.gmin {{ color:#46570a; font-weight:800; background:rgba(197,216,109,0.62); }}
        .bench-table tr.home td.solver {{ border-left:3px solid var(--lime-dark); }}
        .bench-table td.solver .tag {{ font-family:'Inter',sans-serif; font-size:0.58rem;
              font-weight:700; text-transform:uppercase; letter-spacing:0.04em; color:var(--lime-dark);
              background:rgba(197,216,109,0.22); border:1px solid rgba(168,189,77,0.45);
              padding:1px 6px; border-radius:100px; margin-left:7px; vertical-align:middle; }}
        .bench-legend {{ display:flex; align-items:center; flex-wrap:wrap;
              font-size:0.78rem; color:var(--text-muted); margin-top:12px; }}
        .bench-legend .sw {{ display:inline-block; width:15px; height:15px; border-radius:4px;
              margin-left:18px; margin-right:6px; vertical-align:middle; border:1px solid var(--border); }}
        .bench-legend .sw:first-child {{ margin-left:0; }}
        .bench-legend .sw.rmin {{ background:rgba(74,154,143,0.32); }}
        .bench-legend .sw.gmin {{ background:rgba(197,216,109,0.72); }}
        .bench-note {{ font-size:0.8rem; color:var(--text-muted); margin-top:12px; line-height:1.6; }}
        .bench-pending {{ background:var(--bg-card); border:1px dashed var(--border); border-radius:12px;
              padding:22px; text-align:center; color:var(--text-muted); font-style:italic; }}
        footer {{ background:var(--black); color:rgba(255,255,255,0.5); padding:28px 0;
              text-align:center; font-size:0.85rem; margin-top:30px; }}
        footer strong {{ color:#fff; }}
    </style>
</head>
<body>
<nav>
    <div class="nav-content">
        <a class="nav-brand" href="index.html">
            <img src="imgs/stiles.jpeg" alt="sTiles"><span class="nav-sub">/ Matrix</span>
        </a>
        <div class="nav-links">
            <a href="index.html">&larr; Catalog</a>
            <a href="results.html">Solver benchmarks</a>
            <a href="dense.html">Dense benchmark</a>
            <a href="../index.html">Home</a>
        </div>
    </div>
</nav>

<div class="hero"><div class="container"><div class="hero-content">
    <div class="grp">{group}</div>
    <h1>{name}</h1>
    <div class="sub">Real symmetric &middot; SPD &middot; MatrixMarket (lower triangle)</div>
</div></div></div>

<section><div class="container">
    <div class="dl-row" style="margin-bottom:34px;">
        <a class="dl-btn" href="{dl_url}" download>
            &#x2193; Download .mtx <span class="sz">{size}</span>
        </a>
    </div>

    <div class="cards">
        <div class="card"><div class="lbl">Dimension</div><div class="val">{dim}</div></div>
        <div class="card"><div class="lbl">NNZ (lower &#9651;)</div><div class="val">{nnz}</div></div>
        <div class="card"><div class="lbl">Density</div><div class="val">{density}</div></div>
        <div class="card"><div class="lbl">Application</div><div class="val" style="font-size:0.92rem;">{domain}</div></div>
        <div class="card"><div class="lbl">File size</div><div class="val">{size}</div></div>
        <div class="card"><div class="lbl">Format</div><div class="val" style="font-size:0.92rem;">real symmetric</div></div>
    </div>

    <div class="plot-wrap">
        <div class="lbl">Sparsity pattern &middot; 1000&times;1000 bins &middot; log-density</div>
        <a href="{plot}" target="_blank">
            <img src="{plot}" alt="{name} sparsity pattern" loading="lazy"
                 onerror="this.parentElement.outerHTML='<div class=no>sparsity plot not generated yet</div>'">
        </a>
    </div>
</div></section>

<footer><div class="container">
    <p><strong>sTiles</strong> &middot; KAUST &middot; Test Suite &middot; &copy; 2026</p>
</div></footer>
</body>
</html>
"""


def main():
    # WARNING: file sizes come from local MTX_ROOT. If that tree is not mounted,
    # sizes render as "&mdash;". The matrices are also hosted at DL_BASE
    # (https://esmail.r-inla-download.org/mtx/<group>/<name>.mtx); mount or sync
    # MTX_ROOT before regenerating so existing file sizes are not blanked.
    if not local:
        print("WARNING: MTX_ROOT is empty; file sizes will render as '&mdash;'.")
    n_written = 0
    for m in mats:
        name, grp = m["name"], m["group"]
        path = local.get(name)
        size = human(os.path.getsize(path)) if path else "&mdash;"
        dl_url = f"{DL_BASE}/{grp}/{name}.mtx"
        html = TEMPLATE.format(
            name=name, group=grp,
            dim=f"{m['dim']:,}", nnz=f"{m['nnz']:,}",
            density=fmt_pct(m["density"]),
            size=size, dl_url=dl_url, domain=m["domain"],
            plot=f"plots/{name}_sparsity.png")
        out = os.path.join(SUITS, f"matrix_{name}.html")
        with open(out, "w") as fh:
            fh.write(html)
        n_written += 1
    print(f"Wrote {n_written} matrix_*.html pages")


if __name__ == "__main__":
    main()
