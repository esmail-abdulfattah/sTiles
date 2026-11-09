#!/usr/bin/env python3
"""Build docs/examples.html from the programs in examples/.

The page shows each program exactly as it is in the repository, so the site
and the files cannot drift apart: edit the program, then rerun this script.

    python3 docs/_build_examples.py          # page from the saved outputs
    python3 docs/_build_examples.py --run    # rerun every example first

--run executes all 24 programs and saves what they print to
docs/_examples_output.json. It needs python (with sTiles), Rscript (with
sTiles), julia (with sTiles; set JULIA to override the command) and, for C,
STILES pointing at an unpacked release archive.
"""
import html
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
OUTPUTS = ROOT / "docs" / "_examples_output.json"
PAGE = ROOT / "docs" / "examples.html"
GITHUB = "https://github.com/esmail-abdulfattah/sTiles/blob/main/examples"

LANGS = [  # key, label, folder, extension, highlight.js language
    ("python", "Python", "python", ".py", "python"),
    ("r", "R", "r", ".R", "r"),
    ("julia", "Julia", "julia", ".jl", "julia"),
    ("c", "C", "c", ".c", "c"),
]

EXAMPLES_META = [
    dict(slug="01_quickstart", title="Quickstart",
         text="Factorize Q, then compute log|Q|, solve Q x = b and get the marginal variances diag(Q<sup>-1</sup>)."),
    dict(slug="02_solve", title="Solving linear systems",
         text="Solve Q x = b for one right-hand side, and for several at once by passing a matrix "
              "with one column per right-hand side. In C the solve overwrites its input."),
    dict(slug="03_selected_inverse", title="Selected inverse",
         text="Entries of Q<sup>-1</sup> without the dense inverse. Values are exact on the pattern "
              "of the Cholesky factor (this includes the diagonal and the non-zeros of Q) and 0 outside it."),
    dict(slug="04_reuse_analysis", title="Reusing the analysis",
         text="The ordering and tile layout depend only on the sparsity pattern. Compute them once "
              "and refactorize for each new set of values with the same pattern. Timings depend on the machine."),
    dict(slug="05_sampling", title="Sampling from N(0, Q<sup>-1</sup>)",
         text="With Q = C C<sup>T</sup>, x = C<sup>-T</sup> z has covariance Q<sup>-1</sup> when z ~ N(0, I). "
              "The backward solve computes this. Sample variances differ slightly between languages "
              "because each uses its own random numbers."),
    dict(slug="06_matrix_market", title="Reading a Matrix Market file",
         text="Factorize a matrix read from a .mtx file. Without an argument it reads "
              "<code>examples/data/grid40.mtx</code> (a 1600 &times; 1600 grid Laplacian)."),
]


# --------------------------------------------------------------------------
# Running the examples
# --------------------------------------------------------------------------

def run_all():
    out = {}
    stiles = os.environ.get("STILES")
    if not stiles:
        sys.exit("--run needs STILES=/path/to/unpacked/archive for the C examples")
    subprocess.run(["make", "-s", f"STILES={stiles}"], cwd=EXAMPLES / "c", check=True)
    commands = {
        "python": lambda f: [sys.executable, f],
        "r": lambda f: ["Rscript", f],
        "julia": lambda f: os.environ.get("JULIA", "julia").split() + [f],
        "c": lambda f: ["./" + Path(f).stem],
    }
    for key, _, folder, ext, _ in LANGS:
        out[key] = {}
        for ex in EXAMPLES_META:
            name = ex["slug"] + ext
            print(f"running {folder}/{name}", file=sys.stderr)
            res = subprocess.run(commands[key](name), cwd=EXAMPLES / folder,
                                 capture_output=True, text=True)
            if res.returncode != 0:
                sys.exit(f"{folder}/{name} failed:\n{res.stdout}\n{res.stderr}")
            out[key][ex["slug"]] = res.stdout.rstrip()
    subprocess.run(["make", "-s", "clean"], cwd=EXAMPLES / "c", check=True)
    OUTPUTS.write_text(json.dumps(out, indent=1) + "\n")
    return out


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

def esc(s):
    return html.escape(s, quote=False)


def only(key):
    """Attributes for a block shown only for one language (Python by default)."""
    return f'data-lang="{key}"' + ("" if key == "python" else " hidden")


def code(text, hl="bash"):
    return f'<pre><code class="language-{hl}">{esc(text.strip(chr(10)))}</code></pre>'


CLONE = "git clone --depth 1 https://github.com/esmail-abdulfattah/sTiles.git"

CC_LINUX = """gcc -O2 app.c -I$STILES/include \\
    -L$STILES/lib -Wl,-rpath,$STILES/lib -lstiles -lm -o app"""
CC_MACOS = """clang -O2 app.c -I$STILES/include \\
    -L$STILES/lib -Wl,-rpath,$STILES/lib -lstiles -o app"""
CC_WINDOWS = """gcc -O2 app.c -I$STILES/include -L$STILES/lib -lstiles -o app.exe
PATH="$STILES/lib:$PATH" ./app.exe"""
CC_PKGCONFIG = """export PKG_CONFIG_PATH=$STILES/lib/pkgconfig
gcc -O2 app.c $(pkg-config --cflags --libs stiles) -Wl,-rpath,$STILES/lib -o app"""
CMAKE = """cmake_minimum_required(VERSION 3.16)
project(app C)

find_path(STILES_INCLUDE stiles.h PATHS ${STILES}/include REQUIRED)
find_library(STILES_LIB stiles PATHS ${STILES}/lib REQUIRED)

add_executable(app app.c)
target_include_directories(app PRIVATE ${STILES_INCLUDE})
target_link_libraries(app PRIVATE ${STILES_LIB} m)"""

INSTALL = {
    "python": f"""
<p>Requires Python 3.8 or later. NumPy and SciPy are installed as dependencies.</p>
{code("pip install sTiles")}
<p>The solver library for your platform is downloaded on first use and cached in
<code>~/.cache/sTiles</code>. <code>python -m sTiles info</code> shows the version and
location of the solver in use.</p>
<p>To run the examples:</p>
{code(CLONE + chr(10) + "cd sTiles/examples/python" + chr(10) + "python 01_quickstart.py")}""",

    "r": f"""
<p>sTiles is on CRAN. The solver library is a separate download, installed once:</p>
{code('install.packages("sTiles")' + chr(10) + "sTiles::sTiles_install_library()", "r")}
<p>The development version is on r-universe:
<code>install.packages("sTiles", repos = c("https://esmail-abdulfattah.r-universe.dev", "https://cloud.r-project.org"))</code>.</p>
<p>To run the examples:</p>
{code(CLONE + chr(10) + "cd sTiles/examples/r" + chr(10) + "Rscript 01_quickstart.R")}
<p>From RStudio, set the working directory to <code>examples/r</code> before sourcing a file.</p>""",

    "julia": f"""
<p>Requires Julia 1.10 or later. In package mode (press <code>]</code>):</p>
{code("pkg> add https://github.com/esmail-abdulfattah/sTiles:julia", "julia")}
<p>The solver library is downloaded when the package first loads and cached in
<code>~/.cache/sTiles</code>.</p>
<p>To run the examples:</p>
{code(CLONE + chr(10) + "cd sTiles/examples/julia" + chr(10) + "julia 01_quickstart.jl")}
<p>If Julia crashes on <code>using SparseArrays</code>, an <code>LD_LIBRARY_PATH</code> set in
your shell (often by Intel oneAPI) is shadowing Julia's own libraries; run
<code>env -u LD_LIBRARY_PATH julia ...</code>.</p>""",

    "c": f"""
<p>Download the archive for your platform from the <a href="download.html">download page</a>
and unpack it. It contains:</p>
<pre class="plain">include/stiles.h          header
lib/libstiles.so          library (.dylib on macOS, .dll on Windows)
lib/pkgconfig/stiles.pc   pkg-config file
BUILDINFO                 compiler, flags and BLAS used for the build</pre>
<p>Below, <code>STILES</code> is the path of the unpacked folder. To build and run the examples:</p>
{code(CLONE + chr(10) + "cd sTiles/examples/c" + chr(10) + "make STILES=$HOME/libstiles-linux-x86_64" + chr(10) + "./01_quickstart")}
<p><code>make STILES=... run</code> builds and runs all six. <code>common.h</code> contains
helpers shared by the examples (error check, test matrix, residual, timer); it is not part of sTiles.</p>

<h3>Compiling your own program</h3>
<p>Linux:</p>
{code(CC_LINUX)}
<p>macOS:</p>
{code(CC_MACOS)}
<p>Windows, from an MSYS2 UCRT64 shell (the DLLs are found through <code>PATH</code>):</p>
{code(CC_WINDOWS)}
<p>With pkg-config:</p>
{code(CC_PKGCONFIG)}
<p>With CMake (<code>cmake -DSTILES=/path/to/archive -B build &amp;&amp; cmake --build build</code>):</p>
{code(CMAKE, "cmake")}
<p>The <code>rpath</code> lets the program find the library at run time without
<code>LD_LIBRARY_PATH</code>. The header has <code>extern "C"</code> guards, so C++ code
includes and links it the same way.</p>

<h3>Order of calls</h3>
<p>For one matrix (group 0, call 0):</p>
<table class="calls">
<tr><td><code>sTiles_create</code></td><td>create the handle: number of matrices, cores, whether the selected inverse is needed</td></tr>
<tr><td><code>sTiles_assign_graph_one_call</code></td><td>give the sparsity pattern (lower triangle, 0-based)</td></tr>
<tr><td><code>sTiles_init_group</code></td><td>analysis: ordering and tile layout</td></tr>
<tr><td><code>sTiles_assign_values</code></td><td>give the values, in the same order as the pattern</td></tr>
<tr><td><code>sTiles_bind</code></td><td>start the worker threads</td></tr>
<tr><td><code>sTiles_chol</code>, <code>sTiles_selinv</code>, <code>sTiles_solve_*</code></td><td>compute</td></tr>
<tr><td><code>sTiles_unbind</code></td><td>stop the worker threads</td></tr>
<tr><td><code>sTiles_get_logdet</code>, <code>sTiles_get_selinv_*</code></td><td>read results</td></tr>
<tr><td><code>sTiles_quit</code></td><td>free everything</td></tr>
</table>
<p>To factorize new values with the same pattern, repeat from <code>sTiles_assign_values</code>
to <code>sTiles_unbind</code> (example 04). All functions are documented in the
<a href="index.html#api">API reference</a>.</p>""",
}

NOTES = """
<ul>
<li>sTiles reads only the lower triangle of Q.
  <span data-lang="c" hidden>Pass it in coordinate form (row &ge; col), diagonal included.</span></li>
<li>Indices are <span data-lang="python">0-based</span><span data-lang="r" hidden>1-based</span><span data-lang="julia" hidden>1-based</span><span data-lang="c" hidden>0-based</span>
  and refer to the original ordering of Q.</li>
<li>The number of threads defaults to 1. The examples use 2.</li>
<li>The selected inverse must be requested when the factorization is created
  (<span data-lang="python"><code>inverse=True</code></span><span data-lang="r" hidden><code>inverse = TRUE</code></span><span data-lang="julia" hidden><code>inverse = true</code></span><span data-lang="c" hidden><code>get_inverse[0] = true</code></span>).</li>
<li>Only one factorization can be open per process. Close it
  (<span data-lang="python"><code>s.close()</code> or a <code>with</code> block</span><span data-lang="r" hidden><code>sTiles_close(s)</code></span><span data-lang="julia" hidden><code>close(F)</code></span><span data-lang="c" hidden><code>sTiles_quit()</code></span>)
  before creating the next one.</li>
<li>Examples 01&ndash;05 use a tridiagonal matrix with 4 on the diagonal and &minus;1 off the diagonal.</li>
</ul>"""

CSS = """
:root { --teal: #3a7a71; --text: #1a1a1a; --muted: #666; --border: #ddd; --code-bg: #f6f8fa; }
* { box-sizing: border-box; }
body { margin: 0; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; font-size: 16px;
       line-height: 1.65; color: var(--text); background: #fff; }
[hidden] { display: none !important; }
a { color: var(--teal); }
code, pre { font-family: 'JetBrains Mono', monospace; }
code { font-size: 0.88em; }
.container { max-width: 860px; margin: 0 auto; padding: 0 20px; }

nav { background: #fff; border-bottom: 1px solid var(--border); }
.nav-content { display: flex; align-items: center; justify-content: space-between; gap: 16px;
               max-width: 1100px; margin: 0 auto; padding: 12px 20px; }
.nav-brand { display: flex; align-items: center; gap: 10px; text-decoration: none; color: var(--text); font-weight: 700; }
.nav-brand img { height: 28px; display: block; }
.nav-links { display: flex; gap: 20px; flex-wrap: wrap; }
.nav-links a { color: var(--muted); text-decoration: none; font-size: 0.92rem; }
.nav-links a:hover, .nav-links a.current { color: var(--teal); }

main { padding: 12px 0 40px; }
h1 { font-size: 1.9rem; margin: 0.6em 0 0.3em; }
h2 { font-size: 1.35rem; margin: 2.2em 0 0.6em; padding-bottom: 0.25em; border-bottom: 1px solid var(--border); }
h3 { font-size: 1.08rem; margin: 1.8em 0 0.4em; }
p, ul { margin: 0.6em 0; }
ul { padding-left: 1.3em; }

.tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border); margin: 1.4em 0 0; }
.tabs button { font: inherit; font-size: 0.95rem; background: none; border: 0; border-bottom: 2px solid transparent;
               margin-bottom: -1px; padding: 8px 16px; color: var(--muted); cursor: pointer; }
.tabs button:hover { color: var(--text); }
.tabs button[aria-pressed="true"] { color: var(--text); border-bottom-color: var(--teal); font-weight: 600; }

pre { background: var(--code-bg); border: 1px solid #e5e7eb; border-radius: 4px; padding: 12px 14px;
      overflow-x: auto; font-size: 0.84rem; line-height: 1.5; margin: 0.6em 0; }
pre code { font-size: inherit; }
pre code.hljs { padding: 0; background: transparent; }
pre.plain { background: #fff; }
.file { font-size: 0.82rem; color: var(--muted); margin: 1em 0 -0.3em; }
.file a { font-family: 'JetBrains Mono', monospace; }
.label { font-size: 0.85rem; color: var(--muted); margin: 0.8em 0 -0.3em; }
pre.output { background: #fff; }

table.calls { border-collapse: collapse; font-size: 0.9rem; margin: 0.6em 0; }
table.calls td { border-top: 1px solid #eee; padding: 5px 16px 5px 0; vertical-align: top; }
table.calls td:first-child { white-space: nowrap; }

footer { border-top: 1px solid var(--border); margin-top: 48px; padding: 20px 0; font-size: 0.85rem; color: var(--muted); }

@media (max-width: 640px) {
    .nav-links { gap: 12px; }
    .tabs button { padding: 8px 10px; }
    table.calls td:first-child { white-space: normal; }
}
"""

JS = """
(function () {
    var KEY = 'stiles-examples-lang', LANGS = ['python', 'r', 'julia', 'c'];
    function setLang(lang, remember) {
        if (LANGS.indexOf(lang) < 0) lang = 'python';
        document.querySelectorAll('[data-lang]').forEach(function (el) { el.hidden = el.dataset.lang !== lang; });
        document.querySelectorAll('.tabs button').forEach(function (b) {
            b.setAttribute('aria-pressed', b.dataset.set === lang ? 'true' : 'false');
        });
        if (remember) { try { localStorage.setItem(KEY, lang); } catch (e) {} }
    }
    var fromUrl = new URLSearchParams(location.search).get('lang'), saved = null;
    try { saved = localStorage.getItem(KEY); } catch (e) {}
    setLang(fromUrl || saved || 'python', !!fromUrl);
    document.querySelectorAll('.tabs button').forEach(function (b) {
        b.addEventListener('click', function () { setLang(b.dataset.set, true); });
    });
    if (window.hljs) document.querySelectorAll('pre code[class^="language-"]').forEach(function (el) {
        hljs.highlightElement(el);
    });
})();
"""


def build(outputs):
    tabs = "".join(f'<button type="button" data-set="{k}" aria-pressed="{"true" if k == "python" else "false"}">{label}</button>'
                   for k, label, *_ in LANGS)
    install = "\n".join(f"<div {only(k)}>{INSTALL[k]}</div>" for k, *_ in LANGS)
    contents = "\n".join(f'<li><a href="#{e["slug"]}">{e["slug"][:2]} &nbsp;{e["title"]}</a></li>'
                         for e in EXAMPLES_META)

    articles = []
    for e in EXAMPLES_META:
        parts = [f'<h3 id="{e["slug"]}">{e["slug"][:2]}. {e["title"]}</h3>', f'<p>{e["text"]}</p>']
        for key, _, folder, ext, hl in LANGS:
            fname = e["slug"] + ext
            src = (EXAMPLES / folder / fname).read_text().rstrip()
            out = outputs.get(key, {}).get(e["slug"], "")
            parts.append(f'<div {only(key)}>')
            parts.append(f'<p class="file"><a href="{GITHUB}/{folder}/{fname}">examples/{folder}/{fname}</a></p>')
            parts.append(code(src, hl))
            if out:
                parts.append(f'<p class="label">Output:</p>\n<pre class="output">{esc(out)}</pre>')
            parts.append("</div>")
        articles.append("\n".join(parts))

    page = f"""<!DOCTYPE html>
<!-- Generated by docs/_build_examples.py from examples/. Edit those, then rerun the script. -->
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Examples | sTiles</title>
<meta name="description" content="sTiles examples in Python, R, Julia and C, with installation and compilation instructions.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
<style>{CSS}</style>
</head>
<body>
<nav><div class="nav-content">
  <a class="nav-brand" href="index.html"><img src="images/stiles.jpeg" alt="sTiles"> sTiles</a>
  <div class="nav-links">
    <a href="index.html">Home</a>
    <a href="examples.html" class="current">Examples</a>
    <a href="index.html#api">API</a>
    <a href="download.html">Download</a>
  </div>
</div></nav>

<main class="container">
<h1>Examples</h1>
<p>Six examples, each written in Python, R, Julia and C. The source files are in
<a href="https://github.com/esmail-abdulfattah/sTiles/tree/main/examples"><code>examples/</code></a>
in the repository; the output shown under each one is what it prints.</p>

<div class="tabs" role="group" aria-label="Language">{tabs}</div>

<ul>
<li><a href="#installation">Installation</a></li>
<li><a href="#notes">Notes</a></li>
<li><a href="#examples">Examples</a>
<ul>
{contents}
</ul></li>
</ul>

<h2 id="installation">Installation</h2>
{install}

<h2 id="notes">Notes</h2>
{NOTES}

<h2 id="examples">Examples</h2>
{chr(10).join(articles)}
</main>

<footer><div class="container">
  <a href="index.html">Home</a> &middot; <a href="index.html#api">API reference</a> &middot;
  <a href="https://github.com/esmail-abdulfattah/sTiles/issues">Issues</a>
</div></footer>

<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/r.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/julia.min.js"></script>
<script>{JS}</script>
</body>
</html>
"""
    PAGE.write_text(page)
    print(f"wrote {PAGE.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    if "--run" in sys.argv:
        outputs = run_all()
    else:
        outputs = json.loads(OUTPUTS.read_text()) if OUTPUTS.exists() else {}
    build(outputs)
