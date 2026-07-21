#!/usr/bin/env python3
"""Per-matrix speedup of sTiles over PARDISO vs matrix cost, the asymmetry
behind the aggregate: cheap matrices scatter around break-even (losses are
sub-second), expensive matrices climb well above it (wins reach minutes).
Reads draft/latest/{stiles,pardiso}_all.csv (Intel, status ok, best over cores).
Run from this directory: python3 gen_speedup_dist.py
"""
import csv, os, math
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=os.path.dirname(__file__)
# locate draft/latest by walking up to the paper root (handles the symlinked plots dir)
d=HERE
ALL=os.path.join(os.path.dirname(__file__),'..')   # results-latest (self-contained)

def best(fn):
    b={}
    for r in csv.DictReader(open(os.path.join(ALL,fn))):
        if not r['status'].strip().lower().startswith('ok'): continue
        if r.get('arch','intel').strip().lower() not in ('intel',''): continue
        try: t=float(r['factor_s'])
        except: continue
        if t>0: m=r['matrix'].replace('.mtx',''); b[m]=min(b.get(m,math.inf),t)
    return b
S=best('stiles_all.csv')
P=best('pardiso_all.csv')
mats=[m for m in S if m in P]
x=[S[m] for m in mats]                 # cost = sTiles factor time
y=[P[m]/S[m] for m in mats]            # speedup over PARDISO (>1 = sTiles faster)
gm=math.exp(sum(math.log(v) for v in y)/len(y))

fig,ax=plt.subplots(figsize=(6.2,3.8))
ax.axhspan(min(y)*0.8,1.0,color='#c0392b',alpha=0.05,zorder=0)   # loss band
ax.axhline(1.0,color='gray',lw=1,ls='-')
win=[(xi,yi) for xi,yi in zip(x,y) if yi>=1]
los=[(xi,yi) for xi,yi in zip(x,y) if yi<1]
maxloss=max(xi*(1-yi) for xi,yi in los)   # absolute loss (s) = cost*(1 - speedup)
ax.scatter([a for a,_ in los],[b for _,b in los],s=22,c='#c0392b',alpha=0.7,
           edgecolor='none',label=f'sTiles slower ({len(los)}, max loss {maxloss:.2f} s)')
ax.scatter([a for a,_ in win],[b for _,b in win],s=22,c='#27ae60',alpha=0.8,
           edgecolor='none',label=f'sTiles faster ({len(win)})')
ax.axhline(gm,color='black',lw=0.9,ls=':',label=f'geo. mean {gm:.2f}$\\times$')
# annotate the expensive wins
lab={'inla_graph_animal2':'animal2','Emilia_923':'Emilia','Fault_639':'Fault',
     'audikw_1':'audikw','inla_graph_lgm_50400_bw2':'lgm'}
for m,t in lab.items():
    if m in S and m in P: ax.annotate(t,(S[m],P[m]/S[m]),fontsize=7,
        xytext=(3,2),textcoords='offset points',color='#1e6b3a')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('matrix cost: sTiles factorization time (s)')
ax.set_ylabel(r'speedup over PARDISO ($t_{\mathrm{PARDISO}}/t_{\mathrm{sTiles}}$)')
ax.legend(fontsize=8,frameon=False,loc='upper left')
for s in ('top','right'): ax.spines[s].set_visible(False)
fig.tight_layout()
out=os.path.join(HERE,'speedup_dist.pdf')
fig.savefig(out); fig.savefig(out.replace('.pdf','.png'),dpi=150)
print(f'wrote {out}: {len(win)} wins / {len(los)} losses, geomean {gm:.2f}x')
