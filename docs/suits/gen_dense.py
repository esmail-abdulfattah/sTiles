#!/usr/bin/env python3
"""
Generate dense.html: the dense-factorization benchmark page comparing sTiles
against PLASMA and LAPACK (and the sparse solvers) on synthetic dense SPD
matrices. Reads the dense_* CSVs from results-latest. Re-run to refresh.
"""
import os, csv

SUITS = os.path.dirname(os.path.abspath(__file__))
RES   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results-latest")
OUT   = os.path.join(SUITS, "dense.html")

SIZES = ["dense_2000", "dense_5000", "dense_10000"]
NLAB  = {"dense_2000": "2,000", "dense_5000": "5,000", "dense_10000": "10,000"}


def read_csv_skip_comments(path):
    rows = []
    with open(path) as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    rdr = csv.DictReader(lines)
    for r in rdr:
        rows.append(r)
    return {r["matrix"]: r for r in rows}


NUM = read_csv_skip_comments(os.path.join(RES, "dense_numeric_all.csv"))
CMP = read_csv_skip_comments(os.path.join(RES, "dense_compare.csv"))
SYM = read_csv_skip_comments(os.path.join(RES, "dense_symbolic.csv"))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def ft(v):
    return "n/a" if v is None else f"{v:.6g}"


def table(title, note, col_label_each, rows, fmt, best="min", home_row=None):
    """rows: list of (label, [values across SIZES]). best=min|max highlights per column."""
    # per-column best
    colbest = []
    for ci in range(len(SIZES)):
        vals = [r[1][ci] for r in rows if r[1][ci] is not None]
        if not vals:
            colbest.append(None)
        else:
            colbest.append(min(vals) if best == "min" else max(vals))
    head = "".join(f"<th>n = {NLAB[s]}</th>" for s in SIZES)
    body = ""
    for label, vals in rows:
        home = ' class="home"' if label == home_row else ''
        tag = ' <span class="tag">this work</span>' if label == home_row else ''
        cells = ""
        for ci, v in enumerate(vals):
            cls = ' class="gmin"' if (v is not None and colbest[ci] is not None
                                      and v == colbest[ci]) else ''
            cells += f"<td{cls}>{fmt(v)}</td>"
        body += f'<tr{home}><td class="solver">{label}{tag}</td>{cells}</tr>'
    return (
        f'<div class="lbl">{title}</div>'
        + (f'<div class="bench-note" style="margin:0 0 12px;">{note}</div>' if note else '')
        + '<div class="bench-table-wrap"><table class="bench-table"><thead><tr>'
        f'<th>{col_label_each}</th>{head}</tr></thead><tbody>{body}</tbody></table></div>')


# --- GFLOP/s (headline) ---
gflops_rows = [
    ("sTiles",        [fnum(CMP[s]["stiles_Gflops"]) for s in SIZES]),
    ("PLASMA (opt)",  [fnum(CMP[s]["plasma_opt_Gflops"]) for s in SIZES]),
    ("LAPACK",        [fnum(CMP[s]["lapack_Gflops"]) for s in SIZES]),
]
sp_plasma = [fnum(CMP[s]["st_vs_plasma_opt"]) for s in SIZES]
sp_lapack = [fnum(CMP[s]["st_vs_lapack"]) for s in SIZES]
gflops_note = ("Cholesky throughput; higher is better (fastest per size highlighted). "
               "sTiles is "
               + ", ".join(f"{x:.2f}&times;" for x in sp_plasma if x) +
               " faster than optimized PLASMA and "
               + ", ".join(f"{x:.2f}&times;" for x in sp_lapack if x) +
               " faster than LAPACK across n = 2,000 / 5,000 / 10,000.")
gflops_tbl = table("Performance (GFLOP/s)", gflops_note, "Solver",
                   gflops_rows, lambda v: "n/a" if v is None else f"{v:.0f}",
                   best="max", home_row="sTiles")

# --- numeric factorization time ---
num_map = [
    ("sTiles", "stiles"), ("PLASMA (nb 80)", "plasma_nb80"),
    ("PLASMA (nb 120)", "plasma_nb120"), ("LAPACK", "lapack"),
    ("PARDISO", "pardiso"), ("MUMPS", "mumps"), ("CHOLMOD", "cholmod"),
    ("PaStiX", "pastix"), ("symPACK", "sympack"),
]
num_rows = [(lbl, [fnum(NUM[s][key]) for s in SIZES]) for lbl, key in num_map]
num_note = ("Numerical Cholesky factorization, seconds (fastest per size highlighted). "
            "Dense solvers (sTiles, PLASMA, LAPACK) measured on node cn603-04; "
            "the sparse solvers on cn512-12.")
num_tbl = table("Best factorization time, all solvers (s)", num_note, "Solver",
                num_rows, ft, best="min", home_row="sTiles")

# --- symbolic / analysis time (cn512, sparse solvers + sTiles) ---
sym_map = [
    ("sTiles", "stiles_cn512"), ("PARDISO", "pardiso_cn512"),
    ("MUMPS", "mumps_cn512"), ("CHOLMOD", "cholmod_cn512"),
    ("PaStiX", "pastix_cn512"), ("symPACK", "sympack_cn512"),
]
sym_rows = [(lbl, [fnum(SYM[s][key]) for s in SIZES]) for lbl, key in sym_map]
sym_note = ("Analysis / symbolic phase, seconds, all measured on node cn512-12. "
            "PLASMA and LAPACK have no separate analysis phase for dense matrices.")
sym_tbl = table("Analysis (symbolic) time (s)", sym_note, "Solver",
                sym_rows, ft, best="min", home_row="sTiles")

# --- per-core numeric sweeps for the dense libraries -------------------
CORES = [1, 2, 4, 8, 16, 32, 40]
DENSE_SWEEP_FILES = [
    ("sTiles", "dense_stiles.csv"),
    ("PLASMA (nb 80)", "dense_plasma.csv"),
    ("PLASMA (nb 120)", "dense_plasma_nb120.csv"),
    ("LAPACK", "dense_lapack.csv"),
]


def load_sweeps():
    out = {}  # out[matrix][solver] = {core: value}
    for label, fn in DENSE_SWEEP_FILES:
        rows = read_csv_skip_comments(os.path.join(RES, fn))
        for mat, r in rows.items():
            out.setdefault(mat, {})[label] = {c: fnum(r.get("c" + str(c))) for c in CORES}
    return out


SWEEP = load_sweeps()
SWEEP_SOLVERS = [lbl for lbl, _ in DENSE_SWEEP_FILES]


def sweep_table(mat):
    allv = [v for s in SWEEP_SOLVERS for v in SWEEP[mat][s].values() if v is not None]
    gmin = min(allv) if allv else None
    head = "".join(f"<th>{c}</th>" for c in CORES)
    body = ""
    for s in SWEEP_SOLVERS:
        d = SWEEP[mat][s]
        rv = [d[c] for c in CORES if d[c] is not None]
        rmin = min(rv) if rv else None
        home = ' class="home"' if s == "sTiles" else ''
        tag = ' <span class="tag">this work</span>' if s == "sTiles" else ''
        cells = ""
        for c in CORES:
            v = d[c]
            cls = ""
            if v is not None and gmin is not None and v == gmin:
                cls = ' class="gmin"'
            elif v is not None and rmin is not None and v == rmin:
                cls = ' class="rmin"'
            cells += f"<td{cls}>{ft(v)}</td>"
        body += f'<tr{home}><td class="solver">{s}{tag}</td>{cells}</tr>'
    return (f'<div class="lbl">Factorization time (s) by thread count &middot; n = {NLAB[mat]}</div>'
            '<div class="bench-table-wrap"><table class="bench-table"><thead><tr>'
            f'<th>Solver</th>{head}</tr></thead><tbody>{body}</tbody></table></div>')


sweeps_html = (
    '<div class="bench-note" style="margin:34px 0 0;">Full thread sweeps for the dense '
    'libraries (the reference sparse solvers were run at a single configuration; their best '
    'times are in the summary below).</div>'
    + "".join(sweep_table(m) for m in SIZES)
    + '<div class="bench-legend"><span class="sw rmin"></span>fastest thread count for that '
      'solver<span class="sw gmin"></span>fastest overall</div>')

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dense Factorization | sTiles Test Suite</title>
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
        .hero h1 {{ font-size:2.2rem; font-weight:800; letter-spacing:-0.02em; }}
        .hero .sub {{ color:rgba(255,255,255,0.62); font-size:0.98rem; margin-top:10px;
              max-width:640px; margin-left:auto; margin-right:auto; }}
        section {{ padding:36px 0 56px; }}
        .lbl {{ font-size:0.72rem; font-weight:700; text-transform:uppercase;
              letter-spacing:0.07em; color:var(--text-muted); margin:34px 0 6px; }}
        .lbl:first-of-type {{ margin-top:0; }}
        .bench-table-wrap {{ border:1px solid var(--border); border-radius:12px; overflow-x:auto;
              box-shadow:0 2px 10px rgba(0,0,0,0.04); }}
        .bench-table {{ width:100%; border-collapse:collapse; background:var(--bg-card); font-size:0.86rem; }}
        .bench-table thead th {{ background:var(--black); color:#fff; text-align:right;
              padding:11px 18px; font-size:0.72rem; font-weight:600; text-transform:uppercase;
              letter-spacing:0.04em; white-space:nowrap; }}
        .bench-table thead th:first-child {{ text-align:left; }}
        .bench-table td {{ padding:10px 18px; border-bottom:1px solid var(--border-light);
              font-family:'JetBrains Mono',monospace; font-size:0.84rem; text-align:right;
              white-space:nowrap; color:var(--text-secondary); }}
        .bench-table td.solver {{ font-family:'Inter',sans-serif; font-weight:600;
              color:var(--text-primary); text-align:left; }}
        .bench-table tbody tr:nth-child(even) {{ background:#f8f9fa; }}
        .bench-table tbody tr:last-child td {{ border-bottom:none; }}
        .bench-table td.gmin {{ color:#46570a; font-weight:800; background:rgba(197,216,109,0.62); }}
        .bench-table td.rmin {{ color:var(--teal-dark); font-weight:700; background:rgba(74,154,143,0.16); }}
        .bench-table tr.home td.solver {{ border-left:3px solid var(--lime-dark); }}
        .bench-legend {{ display:flex; align-items:center; flex-wrap:wrap; font-size:0.78rem;
              color:var(--text-muted); margin-top:12px; }}
        .bench-legend .sw {{ display:inline-block; width:15px; height:15px; border-radius:4px;
              margin-left:18px; margin-right:6px; vertical-align:middle; border:1px solid var(--border); }}
        .bench-legend .sw:first-child {{ margin-left:0; }}
        .bench-legend .sw.rmin {{ background:rgba(74,154,143,0.32); }}
        .bench-legend .sw.gmin {{ background:rgba(197,216,109,0.72); }}
        .bench-table td.solver .tag {{ font-family:'Inter',sans-serif; font-size:0.58rem;
              font-weight:700; text-transform:uppercase; letter-spacing:0.04em; color:var(--lime-dark);
              background:rgba(197,216,109,0.22); border:1px solid rgba(168,189,77,0.45);
              padding:1px 6px; border-radius:100px; margin-left:7px; vertical-align:middle; }}
        .bench-note {{ font-size:0.82rem; color:var(--text-muted); margin:10px 0 0; line-height:1.6; }}
        .intro {{ color:var(--text-secondary); font-size:1rem; line-height:1.85; max-width:820px;
              margin-bottom:30px; }}
        footer {{ background:var(--black); color:rgba(255,255,255,0.5); padding:28px 0;
              text-align:center; font-size:0.85rem; margin-top:40px; }}
        footer strong {{ color:#fff; }}
    </style>
</head>
<body>
<nav>
    <div class="nav-content">
        <a class="nav-brand" href="index.html">
            <img src="imgs/stiles.jpeg" alt="sTiles"><span class="nav-sub">/ Dense factorization</span>
        </a>
        <div class="nav-links">
            <a href="index.html">&larr; Catalog</a>
            <a href="../index.html">Home</a>
        </div>
    </div>
</nav>

<div class="hero"><div class="container"><div class="hero-content">
    <h1>Dense Factorization Benchmark</h1>
    <div class="sub">sTiles as a dense Cholesky solver, compared with PLASMA and LAPACK
    (and the sparse direct solvers) on dense symmetric positive definite matrices.</div>
</div></div></div>

<section><div class="container">
    <p class="intro">
        Beyond sparse problems, the sTiles tile engine is also a competitive <strong>dense</strong>
        Cholesky solver. The matrices below are fully dense SPD systems of order n = 2,000, 5,000,
        and 10,000. sTiles is compared against the reference dense libraries PLASMA and LAPACK,
        and, for context, against the sparse direct solvers. The fastest result in each column is
        highlighted.
    </p>

    {gflops_tbl}
    {sweeps_html}
    {num_tbl}
    {sym_tbl}
</div></section>

<footer><div class="container">
    <p><strong>sTiles</strong> &middot; KAUST &middot; Test Suite &middot; &copy; 2026</p>
</div></footer>
</body>
</html>
"""

with open(OUT, "w") as fh:
    fh.write(HTML)
print(f"Wrote {OUT}")
