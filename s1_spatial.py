import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from common import *
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.spatial import cKDTree

import os
EXCL=[c for c in os.environ.get('LS_EXCLUDE','').split(',') if c]; TAG=os.environ.get('LS_TAG','')
X = build_X().drop(columns=EXCL); D = squareform(pdist(XY))
# ---- Moran's I correlogram
def moran(v, lo, hi):
    z = v - v.mean(); W = ((D > lo) & (D <= hi)).astype(float)
    S0 = W.sum(); return len(v)/S0 * (z @ W @ z)/(z @ z)
bands = [(0,500),(500,1000),(1000,2000),(2000,3000),(3000,4000),(4000,5000),(5000,7500),(7500,10000),(10000,15000),(15000,20000)]
vars_ = {'Landslide label': y.astype(float)}
for c in ['Elevation','Slope','Rainfall','EVI','MBI','Sand','DistRoad','TWI']:
    if c in X: vars_[c]=X[c].values
rows=[]
for name,v in vars_.items():
    for lo,hi in bands: rows.append([name,(lo+hi)/2/1000,moran(v,lo,hi)])
cg=pd.DataFrame(rows,columns=['variable','dist_km','moranI']); cg.to_csv(f'out_correlogram{TAG}.csv',index=False)
print(cg.pivot(index='dist_km',columns='variable',values='moranI').round(3).to_string())

# ---- validation schemes
def folds_for(scheme, rep):
    rng = np.random.RandomState(rep)
    if scheme == 'Random 5-fold':
        return list(StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y))
    if scheme.startswith('Block'):
        s = float(scheme.split()[1].replace('km',''))*1000
        g = square_blocks(XY, s, offset=tuple(rng.uniform(0, s, 2)))
        return list(StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(X, y, g))
    if scheme.startswith('K-means'):
        k = int(scheme.split()[1]); g = KMeans(k, n_init=10, random_state=rep).fit_predict(XY)
        return list(StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(X, y, g))
schemes = ['Random 5-fold','Block 2km','Block 5km','Block 10km','Block 15km','K-means 20','K-means 10']
models = {'XGBoost (default)': lambda: XGBClassifier(eval_metric='logloss', random_state=SEED, n_jobs=2),
          'RF (default)': lambda: RandomForestClassifier(300, random_state=SEED, n_jobs=2)}
featsets = {'Predictors': X.values} if TAG else {'Predictors': X.values, 'Coordinates only': XY}
res=[]
for sch in schemes:
    for rep in range(5):
        F = folds_for(sch, rep)
        nnd = np.concatenate([cKDTree(XY[tr]).query(XY[te])[0] for tr,te in F])
        for fs, M in featsets.items():
            for mn, mk in models.items():
                oof = np.zeros(len(y)); fa=[]
                for tr,te in F:
                    m = mk().fit(M[tr], y[tr]); p = m.predict_proba(M[te])[:,1]; oof[te]=p
                    fa.append(roc_auc_score(y[te], p) if len(set(y[te]))>1 else np.nan)
                res.append([sch,rep,fs,mn,roc_auc_score(y,oof),np.nanmean(fa),np.median(nnd),np.percentile(nnd,10)])
    print(sch,'done',flush=True)
R=pd.DataFrame(res,columns=['scheme','rep','features','model','pooled_auc','mean_fold_auc','median_nn_test_train_m','p10_nn_m'])
R.to_csv(f'out_scheme_sensitivity{TAG}.csv',index=False)
S=R.groupby(['scheme','features','model']).agg(auc_mean=('pooled_auc','mean'),auc_sd=('pooled_auc','std'),foldauc=('mean_fold_auc','mean'),nn_med=('median_nn_test_train_m','mean'),nn_p10=('p10_nn_m','mean')).reset_index()
print(S.round(3).to_string())
