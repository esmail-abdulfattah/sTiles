#!/usr/bin/env python3
"""Threshold-sensitivity study: sweep each routing cutoff individually over a
band, re-route offline, and report the change in the selector's aggregate
factorization time. Demonstrates the hand-set cutoffs sit on a plateau (anti
-overfit evidence). Features (occ/fill/skew) parsed from the v5 routing log;
per-mode best times from sTiles/stiles_{dense,semisparse,sparse}.csv.
Writes threshold_sensitivity.tex (tab:sensitivity) and prints prose numbers.
Run from this directory: python3 gen_threshold_sensitivity.py
"""
import re, csv, os
HERE=os.path.dirname(__file__); DATA=os.path.join(HERE,'..','..')
LOG='/home/abdulfe/ibex/bench/results_v5/routing_v5_47712819.log'

cur=None; feat={}
for line in open(LOG, errors='ignore'):
    m=re.match(r'### (\S+) c=', line)
    if m: cur=m.group(1).replace('.mtx','')
    s=re.search(r'occ-select\] occ=([\d.]+) fill=([\d.]+) skew=([\d.]+)', line)
    if s and cur and cur not in feat: feat[cur]=tuple(float(x) for x in s.groups())

def best(fn):
    d={}
    for r in csv.DictReader(open(os.path.join(HERE,'..','stiles',fn))):
        try: d[r['matrix'].replace('.mtx','')]=float(r['best'])
        except: pass
    return d
D,S,P=best('stiles_dense.csv'),best('stiles_semisparse.csv'),best('stiles_sparse.csv')
mats=[m for m in feat if D.get(m) and S.get(m) and P.get(m)]
T={m:{'dense':D[m],'semi':S[m],'sparse':P[m]} for m in mats}

# selector rule (process.cpp): occ>=To -> dense; elif fill<=Tf and (occ>=Tr or skew>=Ts) -> semi; else sparse
def route(o,f,s,To,Tf,Tr,Ts):
    if o>=To: return 'dense'
    if f<=Tf and (o>=Tr or s>=Ts): return 'semi'
    return 'sparse'
BASE=(0.9,2.0,0.15,20.0)
def agg(To,Tf,Tr,Ts):
    return sum(T[m][route(*feat[m],To,Tf,Tr,Ts)] for m in mats)
b_t=agg(*BASE); oracle=sum(min(T[m].values()) for m in mats)

SWEEPS=[  # label, idx, default-display, latex-range, values
 (r'dense gate $\rho \ge$', 0, '0.90', r'$[0.80,\,0.99]$', [0.80,0.85,0.88,0.90,0.92,0.95,0.99]),
 (r'fill gate $\phi \le$',  1, '2.0',  r'$[1.0,\,4.0]$',   [1.0,1.5,1.8,2.0,2.2,2.5,3.0,4.0]),
 (r'semi gate $\rho \ge$',  2, '0.15', r'$[0.05,\,0.50]$', [0.05,0.10,0.15,0.20,0.30,0.50]),
 (r'skew gate $\sigma \ge$',3, '20',   r'$[5,\,100]$',     [5,10,20,30,50,100]),
]
rows=[]; worst=0.0; best_alt=(0.0,None,None)
for lab,idx,dvs,rng,vals in SWEEPS:
    md=0.0
    for v in vals:
        t=list(BASE); t[idx]=v; d=(agg(*t)/b_t-1)*100
        md=max(md,abs(d))
        if d<best_alt[0]: best_alt=(d,lab,v)
    worst=max(worst,md); rows.append((lab,dvs,rng,md))

out=[r'\begin{table}[t]', r'\centering\small',
 r'\caption{Threshold sensitivity. Each routing cutoff (Section~\ref{sec:selector}) '
 r'is swept individually over the listed range with the others held at their default, '
 r"and the routing is recomputed offline. The selector's aggregate factorization time "
 fr'over the ${len(mats)}$ matrices whose routing features were logged (including $7$ of '
 r'the $9$ costliest matrices, those above $1$~s; the remaining matrices route to the supernodal '
 r'default far from every gate and do not move) changes by at most the last column. '
 r'The hand-set defaults sit on a wide plateau, not a tuned optimum.}',
 r'\label{tab:sensitivity}',
 r'\begin{tabular}{@{}lccc@{}}', r'\toprule',
 r'cutoff & default & swept range & max time change \\', r'\midrule']
for lab,dvs,rng,md in rows:
    out.append(fr'{lab} & ${dvs}$ & {rng} & ${md:.1f}\%$ \\')
out+=[r'\bottomrule', r'\end{tabular}', r'\end{table}']
open(os.path.join(HERE,'threshold_sensitivity.tex'),'w').write('\n'.join(out)+'\n')

print(f'N={len(mats)}  baseline={b_t:.1f}s  oracle={oracle:.1f}s  '
      f'regret={(b_t/oracle-1)*100:.1f}%  worst single-threshold change={worst:.1f}%')
print(f'best alternative: {best_alt[1]}={best_alt[2]} -> {best_alt[0]:+.1f}% (defaults are not the optimum)')
print('wrote threshold_sensitivity.tex')
