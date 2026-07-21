#!/usr/bin/env python3
"""Feature scatter: each matrix at (rho=occ, phi=fill), colored by its
empirically-best mode, square markers for the bordered class (skew>=20),
with the selector's decision boundaries drawn. Features parsed from the v5
routing log ([occ-select] occ/fill/skew), best mode from the per-mode CSVs.
"""
import csv, math, re, os
from collections import defaultdict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=os.path.dirname(__file__); DATA=os.path.join(HERE,'..','..')
LOG='/home/abdulfe/ibex/bench/results_v5/routing_v5_47712819.log'

# --- features per matrix from the routing log ---
cur=None; feat={}
for line in open(LOG, errors='ignore'):
    m=re.match(r'### (\S+) c=', line)
    if m: cur=m.group(1).replace('.mtx','')
    s=re.search(r'occ-select\] occ=([\d.]+) fill=([\d.]+) skew=([\d.]+)', line)
    if s and cur and cur not in feat:
        feat[cur]=(float(s.group(1)), float(s.group(2)), float(s.group(3)))

# --- empirically-best mode from per-mode best times ---
def best(fn):
    d={}
    for r in csv.DictReader(open(os.path.join(HERE,'..','stiles',fn))):
        try: d[r['matrix'].replace('.mtx','')]=float(r['best'])
        except: pass
    return d
D,S,P=best('stiles_dense.csv'),best('stiles_semisparse.csv'),best('stiles_sparse.csv')
col={'dense':'#c0392b','semi':'#e67e22','sparse':'#2980b9'}
def bestmode(m):
    c={k:v for k,v in [('dense',D.get(m)),('semi',S.get(m)),('sparse',P.get(m))] if v is not None}
    return min(c,key=c.get) if c else None

fig,ax=plt.subplots(figsize=(6.4,4.2))
# decision boundaries (process.cpp): occ>=0.9 -> dense; else fill<=2 & (occ>=0.15 or skew>=20)-> semi
ax.axvspan(0.9,1.02, color=col['dense'], alpha=0.06)
ax.axhline(2.0, color='gray', ls=':', lw=0.8)
ax.axvline(0.9, color=col['dense'], ls='--', lw=0.9)
ax.axvline(0.15, color='gray', ls=':', lw=0.8)
def besttime(m):
    c=[v for v in [D.get(m),S.get(m),P.get(m)] if v is not None]
    return min(c) if c else None
n=0
for m,(rho,phi,skew) in feat.items():
    bm=bestmode(m)
    if bm is None: continue
    t=besttime(m)
    sz=12+50*max(0.0,(math.log10(t)+4))/5.6   # marker area grows with factor time
    mk='s' if skew>=20 else 'o'
    ax.scatter(rho, phi, c=col[bm], marker=mk, s=sz,
               edgecolor='black', linewidth=0.4, alpha=0.85, zorder=3); n+=1
ax.set_yscale('log')
ax.set_xlabel(r'$\rho$  (mean off-diagonal block density)')
ax.set_ylabel(r'$\phi$  (fill ratio $\mathrm{nnz}(L)/\mathrm{nnz}(A)$)')
ax.set_xlim(-0.02,1.02)
from matplotlib.lines import Line2D
leg=[Line2D([],[],marker='o',ls='',mfc=col['sparse'],mec='k',label='best: sparse'),
     Line2D([],[],marker='o',ls='',mfc=col['semi'],mec='k',label='best: semisparse'),
     Line2D([],[],marker='o',ls='',mfc=col['dense'],mec='k',label='best: dense'),
     Line2D([],[],marker='s',ls='',mfc='white',mec='k',label=r'bordered ($\sigma\geq 20$)'),
     Line2D([],[],ls='--',c=col['dense'],label=r'dense gate $\rho=0.9$'),
     Line2D([],[],ls=':',c='gray',label=r'$\phi=2$, $\rho=0.15$'),
     Line2D([],[],marker='o',ls='',mfc='gray',mec='k',ms=3,label='small = cheap matrix'),
     Line2D([],[],marker='o',ls='',mfc='gray',mec='k',ms=8,label='large = expensive')]
ax.legend(handles=leg, fontsize=7.5, frameon=False, loc='upper left', ncol=2)
for s in ('top','right'): ax.spines[s].set_visible(False)
fig.tight_layout()
out=os.path.join(HERE,'feature_scatter.pdf')
fig.savefig(out); fig.savefig(out.replace('.pdf','.png'),dpi=150)
print(f"wrote {out}: {n} matrices ({len(feat)} with features)")
