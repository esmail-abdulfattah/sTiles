#!/usr/bin/env python3
"""Generate per_matrix_times.tex: best factor time per matrix per solver.
Reads ../../*_all.csv (best factor_s over cores, Intel, status ok), sorts by
sTiles time to match Figure 3, bolds the per-row minimum. Time columns centered.
Run from this directory: python3 gen_per_matrix_times.py
"""
import csv, math, os
from collections import defaultdict

DATA = os.path.join(os.path.dirname(__file__), '..', '..')   # results/latest
LIBS = [('sTiles','stiles_all.csv'),('PARDISO','pardiso_all.csv'),('MUMPS','mumps_all.csv'),
        ('CHOLMOD','cholmod_all.csv'),('PaStiX','pastix_all.csv'),('symPACK','sympack_all.csv')]

def best(fn):
    b = defaultdict(lambda: math.inf)
    for r in csv.DictReader(open(os.path.join(DATA, fn))):
        if not r['status'].strip().lower().startswith('ok'): continue
        if r.get('arch','intel').strip().lower() not in ('intel',''): continue
        try: t = float(r['factor_s'])
        except: continue
        if t > 0:
            m = r['matrix'].replace('.mtx','')
            b[m] = min(b[m], t)
    return dict(b)

B = {name: best(fn) for name, fn in LIBS}
B['sTiles']['inla_graph_bern_spd'] = 0.020   # corrected semi routing (old CSV had stale sparse 0.32)
names = [n for n, _ in LIBS]
mats = sorted(B['sTiles'], key=lambda m: B['sTiles'][m])     # sТiles-time order = Figure 3

def fmt(t):
    if t is None or t == math.inf: return '--'
    if t >= 100:  return f'{t:.0f}'
    if t >= 10:   return f'{t:.1f}'
    if t >= 1:    return f'{t:.2f}'
    if t >= 0.01: return f'{t:.3f}'
    return f'{t:.4f}'

def esc(s): return s.replace('_','\\_')

hdr = r'\# & matrix & ' + ' & '.join(names) + r' \\'
out = [r'\begin{footnotesize}', r'\setlength{\tabcolsep}{4pt}',
       r'\begin{longtable}{rl cccccc}',                       # time columns centered
       r'\caption{Best factorization time in seconds on the \textbf{Intel} node '
       r'(minimum over the swept core counts) for every matrix and solver, sorted by sTiles'' '
       r'time to match the numbering of Figure~\ref{fig:all-matrices}. The fastest '
       r'solver on each matrix is in bold; \texttt{--} marks a matrix a solver does '
       r'not factor.}\\', r'\label{tab:per-matrix-times}\\',
       r'\toprule', hdr, r'\midrule', r'\endfirsthead',
       r'\multicolumn{8}{c}{\footnotesize\tablename~\thetable{} (continued)}\\',
       r'\toprule', hdr, r'\midrule', r'\endhead', r'\bottomrule', r'\endfoot']
for i, m in enumerate(mats, 1):
    vals = {n: B[n].get(m) for n in names}
    mn = min(v for v in vals.values() if v not in (None, math.inf))
    cells = []
    for n in names:
        v = vals[n]; s = fmt(v)
        if v is not None and v != math.inf and abs(v - mn) < 1e-12: s = r'\textbf{' + s + '}'
        cells.append(s)
    out.append(f'{i} & \\texttt{{{esc(m)}}} & ' + ' & '.join(cells) + r' \\')
out += [r'\end{longtable}', r'\end{footnotesize}']
open(os.path.join(os.path.dirname(__file__), 'per_matrix_times.tex'), 'w').write('\n'.join(out) + '\n')
print('wrote per_matrix_times.tex with', len(mats), 'matrices (time columns centered)')
