#!/usr/bin/env python3
"""Generate per_matrix_times_amd.tex: best factor time per matrix per solver on
the AMD EPYC node. Reads ../amd/*_amd.csv (per-core-best 'best' column, cores
1..128), sorts by sTiles time, bolds the per-row minimum. Mirrors the Intel
gen_per_matrix_times.py. Run from this directory: python3 gen_per_matrix_times_amd.py
"""
import csv, math, os

DATA = os.path.join(os.path.dirname(__file__), '..')   # results-latest
LIBS = [('sTiles','amd/stiles_auto_amd.csv'),('PARDISO','amd/pardiso_amd.csv'),
        ('MUMPS','amd/mumps_amd.csv'),('CHOLMOD','amd/cholmod_amd.csv'),
        ('PaStiX','amd/pastix_amd.csv'),('symPACK','amd/sympack_amd.csv')]

def best(fn):
    b = {}
    for r in csv.DictReader(open(os.path.join(DATA, fn))):
        m = r['matrix'].replace('.mtx','')
        try: v = float(r['best'])
        except: continue
        if v > 0: b[m] = v
    return b

B = {name: best(fn) for name, fn in LIBS}
names = [n for n, _ in LIBS]
mats = sorted(B['sTiles'], key=lambda m: B['sTiles'][m])     # sТiles-time order

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
       r'\begin{longtable}{rl cccccc}',
       r'\caption{Best factorization time in seconds on the \textbf{AMD}~EPYC node '
       r'(minimum over cores $\{1,\dots,128\}$; sTiles is the best over its two '
       r'thread-affinity settings) for every matrix and solver, sorted by sTiles '
       r'time. The fastest solver on each matrix is in bold; \texttt{--} marks a '
       r'matrix a solver does not factor.}\\', r'\label{tab:per-matrix-times-amd}\\',
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
open(os.path.join(os.path.dirname(__file__), 'per_matrix_times_amd.tex'), 'w').write('\n'.join(out) + '\n')
print('wrote per_matrix_times_amd.tex with', len(mats), 'matrices')
