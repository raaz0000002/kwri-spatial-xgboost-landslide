import pandas as pd, numpy as np, matplotlib.pyplot as plt
from figstyle import *
from common import *
# correlogram
import os
cg=pd.read_csv(next(f for f in ['out_correlogram_rx2.csv','out_correlogram_rx.csv','out_correlogram.csv'] if os.path.exists(f)))
fig,ax=plt.subplots(figsize=(6.7,3.2))
order=[v for v in ['Landslide label','Elevation','Slope','TWI','Rainfall','Sand','EVI','MBI','DistRoad'] if v in set(cg.variable)]
sty={'Landslide label':dict(color='#0b0b0b',lw=2.5)}
for i,v in enumerate([o for o in order if o!='Landslide label']):
    sty[v]=dict(color=CAT[i%8],lw=1.4)
for v in order:
    s=cg[cg.variable==v]; ax.plot(s.dist_km,s.moranI,marker='o',ms=3,label=v,**sty[v])
ax.axhline(0,color='#52514e',lw=0.8); ax.axvline(10,color='#52514e',lw=0.8,ls='--')
ax.text(10.2,1.0,'10-km block',fontsize=8,color='#52514e',va='top')
ax.set_xlabel('Distance class midpoint (km)'); ax.set_ylabel("Moran's I"); ax.grid(axis='y')
ax.legend(ncol=5,frameon=False,loc='upper center',bbox_to_anchor=(0.5,-0.2))
save(fig,'Fig_correlogram')
# scheme sensitivity
R=pd.read_csv('out_scheme_sensitivity.csv'); R['features']=R.features.replace({'Predictors':'Model S predictors'})
import os
RT=pd.read_csv(next(f for f in ['out_scheme_sensitivity_rx2.csv','out_scheme_sensitivity_rx.csv','out_scheme_sensitivity_nospec.csv'] if os.path.exists(f))); RT['features']='Model T predictors'; R=pd.concat([R,RT])
S=R.groupby(['scheme','features','model']).pooled_auc.agg(['mean','std']).reset_index()
NN=R.groupby('scheme').median_nn_test_train_m.mean()
sch=['Random 5-fold','Block 2km','Block 5km','Block 10km','Block 15km','K-means 20','K-means 10']
lab=[f"{s.replace('km',' km').replace(' ','\n',1)}\n({NN[s]/1000:.1f} km)" for s in sch]
fig,ax=plt.subplots(figsize=(6.7,3.3))
x=np.arange(len(sch)); off={'Model T predictors':-0.25,'Model S predictors':0.0,'Coordinates only':0.25}
mk={'XGBoost (default)':'o','RF (default)':'s'}
col={'Model T predictors':CAT[0],'Model S predictors':CAT[6],'Coordinates only':CAT[1]}
for fs in ['Model T predictors','Model S predictors','Coordinates only']:
    for j,(m,mm) in enumerate(mk.items()):
        s=S[(S.features==fs)&(S.model==m)].set_index('scheme').loc[sch]
        ax.errorbar(x+off[fs]+(j-0.5)*0.1,s['mean'],yerr=s['std'],fmt=mm,ms=5,color=col[fs],mfc=col[fs] if j==0 else 'white',lw=1.2,capsize=2,label=f'{fs} – {m.split()[0]}')
ax.axhline(0.5,color='#52514e',lw=0.8,ls=':'); ax.text(6.4,0.51,'no skill',fontsize=7,color='#52514e',ha='right')
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylabel('Pooled out-of-fold AUC'); ax.set_ylim(0.45,1.0); ax.grid(axis='y')
ax.set_xlabel('Validation scheme (median test-to-nearest-training distance)')
ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(0.5,-0.3),fontsize=7)
save(fig,'Fig_scheme_sensitivity')
# Spearman correlation matrix (20 continuous predictors + sin/cos aspect) grouped
X=build_X(soil=False)
fam_order=['Terrain','Hydrology','Soil texture','Rainfall','Spectral indices','Proximity']
cols=sorted(X.columns,key=lambda c:(fam_order.index(FAMILY[c]),c))
C=X[cols].corr(method='spearman')
fig,ax=plt.subplots(figsize=(6.7,5.8))
im=ax.imshow(C.values,cmap='RdBu_r',vmin=-1,vmax=1)
ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols))); ax.set_xticklabels(cols,rotation=90); ax.set_yticklabels(cols)
for i in range(len(cols)):
    for j in range(len(cols)):
        if i!=j and abs(C.values[i,j])>=0.5: ax.text(j,i,f'{C.values[i,j]:.2f}'.replace('0.','.'),ha='center',va='center',fontsize=5.5,color='white' if abs(C.values[i,j])>0.7 else '#0b0b0b')
b=0
for f in fam_order:
    n=sum(FAMILY[c]==f for c in cols); ax.add_patch(plt.Rectangle((b-0.5,b-0.5),n,n,fill=False,ec='#0b0b0b',lw=1.2)); b+=n
cb=fig.colorbar(im,ax=ax,shrink=0.7); cb.set_label("Spearman's ρ")
ax.spines[:].set_visible(False)
save(fig,'Fig_correlation')
print(C.where(np.triu(np.ones(C.shape),1).astype(bool)).stack().loc[lambda s:s.abs()>=0.6].round(2).to_string())
