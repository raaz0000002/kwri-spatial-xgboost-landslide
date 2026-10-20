import numpy as np, pandas as pd, pickle, matplotlib.pyplot as plt, sys
from figstyle import *
from common import *
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import os
TT=next(x for x in ['_rx2','_rx','_nospec'] if os.path.exists(f'out_metrics_summary{x}.csv'))
TAGS={TT:'Model T (terrain, soil, rainfall, proximity)','':'Model S (Model T + spectral indices)'}
MC={'RF':CAT[0],'XGBoost':CAT[1],'GBM':CAT[2],'SVM':CAT[6],'Ensemble-equal':'#0b0b0b'}
YT=pd.read_csv('samples_raster.csv').Landslide.values if TT.startswith('_rx') else y
YS=pd.read_csv('samples_proj.csv').Landslide.values
def pooled(res,rep,kind):
    n=max(max(r['test_idx']) for r in res)+1
    o={};
    for r in res:
        if r['rep']!=rep: continue
        for m,v in r[kind].items():
            o.setdefault(m,np.full(n,np.nan))[r['test_idx']]=v
    return o
R={t:pickle.load(open(f'out_nested{t}.pkl','rb')) for t in TAGS}
# ---- reliability diagrams: rows = raw / calibrated ; cols = models (primary tag)
for t,name in TAGS.items():
    y=YT if t==TT else YS
    fig,axs=plt.subplots(1,2,figsize=(6.9,3.2),sharey=True)
    for ax,kind,ttl in zip(axs,['raw','cal'],['(a) Uncalibrated scores','(b) After inner-fold Platt calibration']):
        o=pooled(R[t],0,kind)
        ax.plot([0,1],[0,1],color='#52514e',lw=0.8,ls=':')
        for m,c in MC.items():
            p=o[m]; bins=np.linspace(0,1,11); idx=np.clip(np.digitize(p,bins)-1,0,9)
            xs=[p[idx==k].mean() for k in range(10) if (idx==k).sum()>=10]; ys=[y[idx==k].mean() for k in range(10) if (idx==k).sum()>=10]
            ax.plot(xs,ys,marker='o',ms=3.5,color=c,lw=1.4 if m!='Ensemble-equal' else 2.2,label=m.replace('Ensemble-equal','Ensemble (equal)'))
        ax.set_title(ttl,loc='left'); ax.set_xlabel('Mean predicted score (bin)'); ax.grid(alpha=.6); ax.set_xlim(0,1); ax.set_ylim(0,1)
    axs[0].set_ylabel('Observed landslide fraction'); axs[1].legend(frameon=False,fontsize=7,loc='lower right')
    save(fig,f"Fig_reliability{'_T' if t==TT else '_S'}")
# ---- ROC + PR for both models (calibrated ensemble + members) rep0
fig,axs=plt.subplots(2,2,figsize=(6.9,6.2))
for row,(t,name) in enumerate(TAGS.items()):
    y=YT if t==TT else YS
    o=pooled(R[t],0,'cal')
    for m,c in MC.items():
        f,tp,_=roc_curve(y,o[m]); axs[row,0].plot(f,tp,color=c,lw=1.3 if m!='Ensemble-equal' else 2.2,label=f"{m.replace('Ensemble-equal','Ensemble')} ({roc_auc_score(y,o[m]):.3f})")
        pr,rc,_=precision_recall_curve(y,o[m]); axs[row,1].plot(rc,pr,color=c,lw=1.3 if m!='Ensemble-equal' else 2.2,label=f"{m.replace('Ensemble-equal','Ensemble')} ({average_precision_score(y,o[m]):.3f})")
    axs[row,0].plot([0,1],[0,1],ls=':',color='#52514e',lw=.8); axs[row,1].axhline(0.5,ls=':',color='#52514e',lw=.8)
    axs[row,0].set_xlabel('False-positive rate'); axs[row,0].set_ylabel('True-positive rate'); axs[row,1].set_xlabel('Recall'); axs[row,1].set_ylabel('Precision')
    axs[row,0].set_title(f"({'ac'[row]}) ROC – {name.split(' (')[0]}",loc='left'); axs[row,1].set_title(f"({'bd'[row]}) Precision–recall – {name.split(' (')[0]}",loc='left')
    for a in axs[row]: a.legend(frameon=False,fontsize=6.8,loc='lower right' if a is axs[row,0] else 'lower left'); a.grid(alpha=.6)
fig.tight_layout(); save(fig,'Fig_roc_pr')
# ---- grouped permutation importance both models
fig,axs=plt.subplots(1,2,figsize=(6.9,3.4))
for ax,(t,name) in zip(axs,TAGS.items()):
    P=pd.read_csv(f'out_perm_importance{t}.csv'); P=P[(P.model=='Ensemble-equal')&P.unit.str.startswith('FAMILY:')].copy()
    P['fam']=P.unit.str.replace('FAMILY:',''); P=P.sort_values('mean_drop')
    ax.barh(P.fam,P.mean_drop,xerr=P.sd_drop,color=CAT[0] if t==TT else CAT[1],height=0.6,error_kw=dict(lw=1,capsize=2,ecolor='#52514e'))
    ax.set_title(f"({'ab'[list(TAGS).index(t)]}) {name.split(' (')[0]}",loc='left'); ax.set_xlabel('Drop in outer-fold AUC when permuted'); ax.grid(axis='x',alpha=.6)
fig.tight_layout(); save(fig,'Fig_importance')
# ---- fold-level AUC dot plot
fig,ax=plt.subplots(figsize=(6.9,3.0))
meths=['RF','XGBoost','GBM','SVM','Ensemble-equal','Ensemble-weighted','Ensemble-stacked','Best single (inner-CV)']
for k,(t,name) in enumerate(TAGS.items()):
    F=pd.read_csv(f'out_metrics_by_fold{t}.csv'); F=F[F.kind=='cal']
    for i,m in enumerate(meths):
        v=F[F.method==m].AUC.values; xx=i+(k-0.5)*0.32+np.random.RandomState(i).uniform(-.06,.06,len(v))
        ax.scatter(xx,v,s=9,color=CAT[0] if t==TT else CAT[1],alpha=.7,label=name.split(' (')[0] if i==0 else None)
        ax.plot([i+(k-0.5)*0.32-0.12,i+(k-0.5)*0.32+0.12],[v.mean()]*2,color='#0b0b0b',lw=2)
ax.set_xticks(range(len(meths))); ax.set_xticklabels([m.replace('Ensemble-','Ens. ').replace('Best single (inner-CV)','Best single\n(inner CV)') for m in meths],fontsize=7.5)
ax.set_ylabel('Outer-fold AUC'); ax.grid(axis='y',alpha=.6); ax.legend(frameon=False,fontsize=7.5,loc='lower left')
save(fig,'Fig_fold_auc')
# ---- sample + block map (projected)
from matplotlib.lines import Line2D
fig,ax=plt.subplots(figsize=(6.0,5.6))
y=YT; g=square_blocks(XY,10000,(0,0)); res0=R[TT]
fold=np.zeros(len(y),int)
for r in res0:
    if r['rep']==0: fold[r['test_idx']]=r['fold']
fc=[CAT[i] for i in [0,1,2,4,6]]
x0,y0=np.floor(XY.min(0)/10000)*10000; 
for gx in np.arange(x0,XY[:,0].max()+10000,10000): ax.axvline(gx,color='#c9c8c3',lw=0.6)
for gy in np.arange(y0,XY[:,1].max()+10000,10000): ax.axhline(gy,color='#c9c8c3',lw=0.6)
for f in range(5):
    for cls,mk in [(1,'^'),(0,'o')]:
        m=(fold==f)&(y==cls); ax.scatter(XY[m,0],XY[m,1],s=10 if cls else 8,marker=mk,facecolor=fc[f] if cls else 'none',edgecolor=fc[f],lw=0.7)
h=[Line2D([],[],marker='s',ls='',color=fc[f],label=f'Fold {f+1}') for f in range(5)]+[Line2D([],[],marker='^',ls='',color='#52514e',label='Landslide'),Line2D([],[],marker='o',ls='',mfc='none',color='#52514e',label='Non-landslide')]
ax.legend(handles=h,frameon=True,fontsize=7,loc='lower right',ncol=1)
ax.set_aspect('equal'); ax.set_xlabel('Easting (m), WGS 84 / UTM 45N'); ax.set_ylabel('Northing (m)')
ax.ticklabel_format(style='plain'); ax.tick_params(axis='x',rotation=0)
xs,ys=ax.get_xlim(),ax.get_ylim(); map_furniture(ax,[xs[0],xs[1],ys[0],ys[1]],corner='ll',fs=7.5)
save(fig,'Fig_samples_folds')
