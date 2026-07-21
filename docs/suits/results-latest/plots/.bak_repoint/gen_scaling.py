#!/usr/bin/env python3
"""Strong-scaling: speedup vs cores (1..40), sTiles (solid) vs PARDISO
(dashed), on three large FEM matrices, with the ideal line. Shows the
~32-core bandwidth flattening. Reads sTiles/stiles_auto.csv + pardiso_all.csv.
"""
import csv, os
from collections import defaultdict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=os.path.dirname(__file__); DATA=os.path.join(HERE,'..','..')
cores=[1,2,4,8,16,32,40]
st={}
for r in csv.DictReader(open(os.path.join(HERE,'..','stiles','stiles_auto.csv'))):
    try: st[r['matrix'].replace('.mtx','')]={c:float(r['c%d'%c]) for c in cores}
    except: pass
pa=defaultdict(dict)
for r in csv.DictReader(open(os.path.join(DATA,'pardiso_all.csv'))):
    if not r['status'].startswith('ok'): continue
    try: pa[r['matrix'].replace('.mtx','')][int(r['cores'])]=float(r['factor_s'])
    except: pass

mats=[('audikw_1','#c0392b'),('Emilia_923','#2980b9'),('bone010','#27ae60')]
fig,ax=plt.subplots(figsize=(5.6,3.8))
ax.plot(cores,cores,ls=':',c='gray',lw=1,label='ideal')
for m,c in mats:
    sp=[st[m][1]/st[m][k] for k in cores]
    ax.plot(cores,sp,'-o',c=c,ms=4,lw=1.6,label=f'{m} (sTiles)')
    if all(k in pa[m] for k in cores):
        pp=[pa[m][1]/pa[m][k] for k in cores]
        ax.plot(cores,pp,'--s',c=c,ms=3,lw=1.1,alpha=0.7,mfc='white')
ax.set_xscale('log',base=2); ax.set_xticks(cores); ax.set_xticklabels(cores)
ax.set_xlabel('cores'); ax.set_ylabel(r'speedup vs $1$ core')
ax.axvline(32,ls='-',c='lightgray',lw=6,alpha=0.4,zorder=0)
from matplotlib.lines import Line2D
extra=[Line2D([],[],c='k',ls='-',label='sTiles (solid)'),
       Line2D([],[],c='k',ls='--',mfc='white',marker='s',ms=4,label='PARDISO (dashed)')]
h,l=ax.get_legend_handles_labels()
ax.legend(h[:1]+[Line2D([],[],c=c,marker='o',ls='') for _,c in mats]+extra,
          ['ideal']+[m for m,_ in mats]+['sTiles','PARDISO'],
          fontsize=8,frameon=False,loc='upper left',ncol=2)
for s in ('top','right'): ax.spines[s].set_visible(False)
fig.tight_layout()
out=os.path.join(HERE,'scaling.pdf')
fig.savefig(out); fig.savefig(out.replace('.pdf','.png'),dpi=150)
print('wrote',out)
for m,_ in mats: print(f'  {m}: sTiles {st[m][1]/st[m][40]:.1f}x  PARDISO {pa[m][1]/pa[m][40]:.1f}x')
