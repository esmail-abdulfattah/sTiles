#!/usr/bin/env python3
"""Grouped bar chart: aggregate factor time per solver, split by the three
router regimes (sparse / semisparse / dense), log-y. sTiles highlighted.
Reads ../../*_all.csv + modes.csv (+ the 4 stragglers by v5 routing).
Run: python3 gen_per_regime_bars.py  ->  per_regime_bars.pdf
"""
import csv, math, os
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, '..')

grp = {r['matrix'].replace('.mtx',''): r['mode']
       for r in csv.DictReader(open(os.path.join(DATA,'modes.csv')))}
for m, g in {'inla_graph_lgm_50400_bw2':'sparse','inla_graph_lgm_50000_bw15000':'sparse',
             'inla_graph_lgm_10010_bw2':'sparse','inla_graph_bern_spd':'semi'}.items():
    grp[m] = g

LIBS = [('sTiles','stiles_all.csv'),('PARDISO','pardiso_all.csv'),('MUMPS','mumps_all.csv'),
        ('CHOLMOD','cholmod_all.csv'),('PaStiX','pastix_all.csv'),('symPACK','sympack_all.csv')]
def best(fn):
    b = defaultdict(lambda: math.inf)
    for r in csv.DictReader(open(os.path.join(DATA, fn))):
        if not r['status'].strip().lower().startswith('ok'): continue
        if r.get('arch','intel').strip().lower() not in ('intel',''): continue
        try: t = float(r['factor_s'])
        except: continue
        if t > 0: b[r['matrix'].replace('.mtx','')] = min(b[r['matrix'].replace('.mtx','')], t)
    return b
B = {n: best(f) for n, f in LIBS}

regimes = ['sparse','semi','dense']
labels  = ['sparse (77)','semisparse (7)','dense (5)']
agg = {n: [sum(B[n][m] for m in grp if grp[m]==g and m in B[n]) for g in regimes]
       for n,_ in LIBS}

names = [n for n,_ in LIBS]
colors = {'sTiles':'#c0392b','PARDISO':'#2c3e50','MUMPS':'#2980b9',
          'CHOLMOD':'#27ae60','PaStiX':'#8e44ad','symPACK':'#7f8c8d'}
x = np.arange(len(regimes)); w = 0.13
fig, ax = plt.subplots(figsize=(7.2, 3.4))
for i, n in enumerate(names):
    bars = ax.bar(x + (i - 2.5)*w, agg[n], w, label=n, color=colors[n],
                  edgecolor='black', linewidth=0.4,
                  zorder=3, hatch='' if n!='sTiles' else '//')
ax.set_yscale('log')
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('aggregate factor time (s)')
ax.set_axisbelow(True); ax.grid(axis='y', which='both', ls=':', lw=0.5, alpha=0.6)
ax.legend(ncol=6, fontsize=8, frameon=False, loc='upper center',
          bbox_to_anchor=(0.5, 1.16), columnspacing=1.0, handletextpad=0.4)
for s in ('top','right'): ax.spines[s].set_visible(False)
fig.tight_layout()
out = os.path.join(HERE, 'per_regime_bars.pdf')
fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.pdf','.png'), dpi=150, bbox_inches='tight')
print('wrote', out)
for n in names: print(f'  {n:8s}', [round(v,2) for v in agg[n]])
