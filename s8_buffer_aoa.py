import numpy as np, pandas as pd, pickle, warnings; warnings.filterwarnings('ignore')
from common import *
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import os
EXCL=[c for c in os.environ.get('LS_EXCLUDE','MBI,EVI,MNDWI').split(',') if c]; TAG=os.environ.get('LS_TAG','_nospec')
X=build_X(soil=True).drop(columns=EXCL)
# ---- buffered 10-km block CV (default RF), 5 repeats
rows=[]
for buf in [0,1000,2000,3000]:
    for rep in range(5):
        g=square_blocks(XY,10000,tuple(np.random.RandomState(rep).uniform(0,10000,2)))
        oof=np.full(len(y),np.nan); ntr=[]
        for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=rep).split(X,y,g):
            if buf:
                d=cKDTree(XY[te]).query(XY[tr])[0]; tr=tr[d>buf]
            ntr.append(len(tr))
            oof[te]=RandomForestClassifier(300,random_state=SEED,n_jobs=2).fit(X.iloc[tr],y[tr]).predict_proba(X.iloc[te])[:,1]
        rows.append([buf,rep,roc_auc_score(y,oof),np.mean(ntr)])
B=pd.DataFrame(rows,columns=['buffer_m','rep','auc','mean_n_train']).groupby('buffer_m').agg(auc=('auc','mean'),sd=('auc','std'),n_train=('mean_n_train','mean')).reset_index()
B.to_csv(f'out_buffer_sensitivity{TAG}.csv',index=False); print(B.round(3))
# ---- sample-level AOA (Meyer & Pebesma 2021) with Model T, rep 0 folds
res=pickle.load(open(f'out_nested{TAG}.pkl','rb'))
PI=pd.read_csv(f'out_perm_importance{TAG}.csv'); PI=PI[(PI.model=='Ensemble-equal')&~PI.unit.str.startswith('FAMILY:')].set_index('unit').mean_drop.clip(lower=0)
cols=list(X.columns); w=np.array([PI.get('ParentSoil',0)/6 if c.startswith('ParentSoil') else PI.get(c,0) for c in cols])
Z=((X-X.mean())/X.std().replace(0,1)).values*w
dbar=cdist(Z,Z)[np.triu_indices(len(Z),1)].mean()
out=[]
for rep in range(3):
    oof=np.full(len(y),np.nan); di=np.full(len(y),np.nan); thr_list=[]
    for r in res:
        if r['rep']!=rep: continue
        te=r['test_idx']; tr=np.setdiff1d(np.arange(len(y)),te)
        # training-DI threshold from inner CV within training data (10-km blocks)
        g=square_blocks(XY[tr],10000,(0,0)); di_tr=np.empty(len(tr))
        for a,b in StratifiedGroupKFold(4,shuffle=True,random_state=rep).split(tr,y[tr],g):
            di_tr[b]=cdist(Z[tr][b],Z[tr][a]).min(1)/dbar
        q1,q3=np.quantile(di_tr,[.25,.75]); thr=q3+1.5*(q3-q1); thr_list.append(thr)
        di[te]=cdist(Z[te],Z[tr]).min(1)/dbar/thr  # scaled so >1 = outside AOA
        oof[te]=r['cal']['Ensemble-equal']
    inside=di<=1
    err=np.abs(oof-y)
    out.append([rep,inside.mean()*100,roc_auc_score(y[inside],oof[inside]),roc_auc_score(y[~inside],oof[~inside]) if (~inside).sum()>20 and len(set(y[~inside]))>1 else np.nan,
                ((oof[inside]>=.5)!=y[inside]).mean()*100,((oof[~inside]>=.5)!=y[~inside]).mean()*100 if (~inside).sum() else np.nan,(~inside).sum()])
A=pd.DataFrame(out,columns=['rep','pct_test_inside_aoa','auc_inside','auc_outside','err_inside_pct','err_outside_pct','n_outside'])
A.to_csv(f'out_sample_aoa{TAG}.csv',index=False); print(A.round(3))
