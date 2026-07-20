#!/usr/bin/env python3
"""
Generate results.html: the gated "Solver Benchmarks" page for the suits site.

Contents are restricted to results that appear in the paper (main.tex):
  Figures : results-latest/plots/{all_matrices_time,speedup_dist,scaling}.png
  Table   : results-latest/plots/per_matrix_times.tex   (Intel node)

Figures are copied into docs/suits/results/ and embedded as PNGs; the LaTeX
per-matrix table is parsed into a live HTML table (fastest solver per matrix
highlighted, matching the paper's bold).
"""
import os, re, shutil

SUITS   = os.path.dirname(os.path.abspath(__file__))
RES_DIR = "/home/abdulfe/Documents/ideas/adv_sTiles/docs/paper/Semisparse___sTiles/results-latest"
PLOTS   = os.path.join(RES_DIR, "plots")
OUT_DIR = os.path.join(SUITS, "results")      # where figures are copied
FIGS    = ["all_matrices_time", "speedup_dist", "scaling"]
SOLVERS = ["sTiles", "PARDISO", "MUMPS", "CHOLMOD", "PaStiX", "symPACK"]


# --- copy paper figures -------------------------------------------------
def copy_figures():
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in FIGS:
        src = os.path.join(PLOTS, f + ".png")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUT_DIR, f + ".png"))
        else:
            print(f"WARNING: missing figure {src}")


# --- parse the LaTeX per-matrix table -----------------------------------
def _clean(cell):
    """Return (text, is_fastest) for one LaTeX table cell."""
    cell = cell.strip()
    fastest = False
    m = re.match(r"\\textbf\{(.*)\}$", cell)
    if m:
        cell, fastest = m.group(1).strip(), True
    cell = cell.replace(r"\_", "_").replace(r"\texttt{", "").rstrip("}")
    return cell, fastest


def parse_per_matrix():
    """Parse per_matrix_times.tex -> list of row dicts."""
    path = os.path.join(PLOTS, "per_matrix_times.tex")
    rows = []
    if not os.path.exists(path):
        print(f"WARNING: missing {path}")
        return rows
    for line in open(path):
        line = line.strip()
        if not line.endswith(r"\\"):
            continue
        body = line[:-2].strip()
        cols = [c.strip() for c in body.split("&")]
        if len(cols) != 8:                      # #, matrix, + 6 solvers
            continue
        if not cols[0].isdigit():               # skip header rows
            continue
        num = cols[0]
        name, _ = _clean(cols[1])
        vals = [_clean(c) for c in cols[2:]]     # [(text, fastest), ...]
        rows.append(dict(num=num, name=name, vals=vals))
    return rows


def build_table(rows):
    head = ("<th>#</th><th>matrix</th>"
            + "".join('<th class="stiles-col">sTiles</th>' if s == "sTiles"
                      else "<th>" + s + "</th>" for s in SOLVERS))
    body = ""
    for r in rows:
        cells = ('<td class="num">' + r["num"] + '</td>'
                 '<td class="mtx">' + r["name"] + '</td>')
        for i, (txt, fastest) in enumerate(r["vals"]):
            cls = []
            if SOLVERS[i] == "sTiles":
                cls.append("stiles-col")
            if fastest:
                cls.append("gmin")
            c = (' class="' + " ".join(cls) + '"') if cls else ""
            cells += "<td" + c + ">" + (txt if txt != "--" else "&mdash;") + "</td>"
        body += "<tr>" + cells + "</tr>"
    return ('<div class="bench-table-wrap tall"><table class="bench-table pm">'
            '<thead><tr>' + head + '</tr></thead><tbody>'
            + body + '</tbody></table></div>')


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Solver Benchmarks | sTiles Test Suite</title>
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
              max-width:660px; margin-left:auto; margin-right:auto; }}
        section {{ padding:36px 0 56px; }}
        .intro {{ color:var(--text-secondary); font-size:1rem; line-height:1.85; max-width:820px;
              margin-bottom:14px; }}
        .setup {{ display:flex; flex-wrap:wrap; gap:10px; margin:22px 0 40px; }}
        .chip {{ font-family:'JetBrains Mono',monospace; font-size:0.78rem; font-weight:500;
              color:var(--text-secondary); background:var(--bg-card); border:1px solid var(--border);
              border-radius:100px; padding:6px 14px; }}
        .fig {{ margin:40px 0 6px; }}
        .fig-lbl {{ font-size:0.72rem; font-weight:700; text-transform:uppercase;
              letter-spacing:0.07em; color:var(--text-muted); margin-bottom:12px; }}
        .fig-wrap {{ background:var(--bg-card); border:1px solid var(--border); border-radius:14px;
              padding:20px; box-shadow:0 4px 20px rgba(0,0,0,0.05); text-align:center; }}
        .fig-wrap img {{ width:100%; max-width:900px; height:auto; }}
        .cap {{ font-size:0.86rem; color:var(--text-secondary); line-height:1.65;
              max-width:860px; margin:14px auto 0; }}
        .bench-table-wrap {{ border:1px solid var(--border); border-radius:12px; overflow:auto;
              box-shadow:0 2px 10px rgba(0,0,0,0.04); }}
        .bench-table-wrap.tall {{ max-height:620px; }}
        .bench-table {{ width:100%; border-collapse:collapse; background:var(--bg-card); font-size:0.84rem; }}
        .bench-table thead th {{ position:sticky; top:0; z-index:2; background:var(--black); color:#fff;
              text-align:right; padding:10px 14px; font-size:0.7rem; font-weight:600;
              text-transform:uppercase; letter-spacing:0.04em; white-space:nowrap; }}
        .bench-table thead th:first-child, .bench-table thead th:nth-child(2) {{ text-align:left; }}
        .bench-table thead th.stiles-col {{ background:#2c3a1a; color:var(--lime); }}
        .bench-table td {{ padding:8px 14px; border-bottom:1px solid var(--border-light);
              font-family:'JetBrains Mono',monospace; font-size:0.8rem; text-align:right;
              white-space:nowrap; color:var(--text-secondary); }}
        .bench-table td.num {{ color:var(--text-muted); text-align:left; }}
        .bench-table td.mtx {{ font-family:'JetBrains Mono',monospace; color:var(--text-primary);
              text-align:left; }}
        .bench-table td.stiles-col {{ background:rgba(197,216,109,0.10); color:var(--text-primary);
              font-weight:600; }}
        .bench-table tbody tr:hover td {{ background:rgba(74,154,143,0.06); }}
        .bench-table td.gmin {{ color:#46570a; font-weight:800; background:rgba(197,216,109,0.62); }}
        .bench-legend {{ display:flex; align-items:center; flex-wrap:wrap; font-size:0.78rem;
              color:var(--text-muted); margin-top:12px; }}
        .bench-legend .sw {{ display:inline-block; width:15px; height:15px; border-radius:4px;
              margin-right:6px; vertical-align:middle; border:1px solid var(--border); }}
        .bench-legend .sw.gmin {{ background:rgba(197,216,109,0.72); }}
        footer {{ background:var(--black); color:rgba(255,255,255,0.5); padding:28px 0;
              text-align:center; font-size:0.85rem; margin-top:40px; }}
        footer strong {{ color:#fff; }}
    </style>
</head>
<body>
<nav>
    <div class="nav-content">
        <a class="nav-brand" href="index.html">
            <img src="imgs/stiles.jpeg" alt="sTiles"><span class="nav-sub">/ Solver benchmarks</span>
        </a>
        <div class="nav-links">
            <a href="index.html">&larr; Catalog</a>
            <a href="dense.html">Dense benchmark</a>
            <a href="../index.html">Home</a>
        </div>
    </div>
</nav>

<div class="hero"><div class="container"><div class="hero-content">
    <h1>Solver Benchmarks</h1>
    <div class="sub">sTiles against five sparse direct solvers across the full 88-matrix
    suite. All results shown here are from the sTiles paper.</div>
</div></div></div>

<section><div class="container">
    <p class="intro">
        Every matrix in the suite was factored by sTiles and five reference sparse direct
        solvers. For each matrix and solver we report the <strong>best factorization time</strong>
        over a sweep of thread counts. Times below are the same measurements reported in the paper.
    </p>
    <div class="setup">
        <span class="chip">88 matrices</span>
        <span class="chip">6 solvers</span>
        <span class="chip">Intel 2&times;Xeon Gold &middot; 40 cores</span>
        <span class="chip">best over 1&ndash;40 core sweep</span>
        <span class="chip">Cholesky factorization</span>
    </div>

    <div class="fig">
        <div class="fig-lbl">Factorization time across all matrices</div>
        <div class="fig-wrap"><img src="results/all_matrices_time.png" alt="Best factorization time for every matrix and solver" loading="lazy"></div>
        <p class="cap">Best factorization time (minimum over the swept core counts) for every matrix
        in the 88-matrix suite and every solver, on the Intel node. Matrices are sorted along the
        horizontal axis by sTiles' time, so the sTiles curve is monotonic; a competitor marker below
        it is faster than sTiles on that matrix, above it slower. PARDISO tracks sTiles closely, ahead
        on the inexpensive matrices and behind on the expensive ones; MUMPS, PaStiX, CHOLMOD, and
        symPACK are slower across most of the suite.</p>
    </div>

    <div class="fig">
        <div class="fig-lbl">Per-matrix best factorization time (s) &middot; Intel node</div>
        {table}
        <div class="bench-legend"><span class="sw gmin"></span>fastest solver on that matrix &middot; &mdash; marks a matrix a solver does not factor</div>
        <p class="cap">Best factorization time in seconds on the Intel node (minimum over the swept
        core counts) for every matrix and solver, sorted by sTiles' time to match the figure above.
        The fastest solver on each matrix is highlighted.</p>
    </div>

    <div class="fig">
        <div class="fig-lbl">Speedup of sTiles over PARDISO</div>
        <div class="fig-wrap"><img src="results/speedup_dist.png" alt="Per-matrix speedup of sTiles over PARDISO" loading="lazy"></div>
        <p class="cap">Per-matrix speedup of sTiles over PARDISO against matrix cost (both axes
        logarithmic; a value above the 1&times; line means sTiles is faster). By count the contest is
        a near-tie: sTiles is slower on 54 matrices and faster on 34, yet the geometric-mean ratio
        (1.03&times;) already favors sTiles, because its fewer wins are the larger ones. Every loss is
        on a cheap matrix (all but one under 0.5&nbsp;s), while the wins are on the expensive matrices
        and reach minutes.</p>
    </div>

    <div class="fig">
        <div class="fig-lbl">Strong scaling to 40 cores</div>
        <div class="fig-wrap"><img src="results/scaling.png" alt="Strong scaling of sTiles vs PARDISO" style="max-width:680px;" loading="lazy"></div>
        <p class="cap">Strong scaling (speedup over one core) of a single factorization on three large
        finite-element matrices, sTiles (solid) against PARDISO (dashed), with the ideal line. Both
        flatten near 32 cores (shaded band), the socket's memory-bandwidth ceiling for sparse Cholesky.</p>
    </div>
</div></section>

<footer><div class="container">
    <p><strong>sTiles</strong> &middot; KAUST &middot; Test Suite &middot; &copy; 2026</p>
</div></footer>
</body>
</html>
"""


def main():
    copy_figures()
    rows = parse_per_matrix()
    html = TEMPLATE.format(table=build_table(rows))
    out = os.path.join(SUITS, "results.html")
    with open(out, "w") as fh:
        fh.write(html)
    print(f"Wrote results.html ({len(rows)} matrices in table); "
          f"figures copied to {OUT_DIR}")


if __name__ == "__main__":
    main()
