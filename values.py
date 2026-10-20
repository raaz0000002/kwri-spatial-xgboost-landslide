"""Collect every number quoted in the manuscript from the analysis outputs (single source of truth)."""
import numpy as np, pandas as pd, pickle, json, os
from common import *

V = {}
D = RAW
V['n'] = len(D); V['n1'] = int((D.Landslide == 1).sum()); V['n0'] = int((D.Landslide == 0).sum())

# ---- sampling geometry
from scipy.spatial import cKDTree
L, N = XY[y == 1], XY[y == 0]
qLN = cKDTree(N).query(L)[0]
V['ls_nonls_med_km'] = np.median(qLN) / 1000; V['ls_nonls_min_m'] = qLN.min(); V['ls_nonls_p10_m'] = np.percentile(qLN, 10)
qLL = cKDTree(L).query(L, k=2)[0][:, 1]; qNN = cKDTree(N).query(N, k=2)[0][:, 1]
V['ll_med_m'] = np.median(qLL); V['ll_min_m'] = qLL.min(); V['nn_med_m'] = np.median(qNN); V['nn_min_m'] = qNN.min()
V['ll_lt60'] = int(len(cKDTree(L).query_pairs(60))); V['ll_lt100'] = int(len(cKDTree(L).query_pairs(100)))
V['n_blocks10'] = len(set(square_blocks(XY, 10000, (0, 0))))

# ---- soil columns
_TAB = pd.read_csv(_find_data('samples_proj.csv'))
ps = _TAB['Parent_soil']
V['ps_counts'] = ps.value_counts().to_dict(); V['ps_missing'] = int((ps == -9999).sum())
V['psc_levels'] = sorted(_TAB['Parent_soil_class'].unique().tolist())
ct = pd.crosstab(_TAB['Parent_soil'], _TAB['Landslide'])
V['ps_ct'] = ct
tex = _TAB['Sand'] + _TAB['Slit'] + _TAB['Clay']
V['tex_mean'], V['tex_sd'], V['tex_min'], V['tex_max'] = tex.mean(), tex.std(), tex.min(), tex.max()

# ---- VIF
vif = pd.read_csv(_find_data('out_vif_check.csv'), index_col=0)
V['vif'] = vif

# ---- sample vs raster
V['svr'] = pd.read_csv(_find_data('sample_vs_raster.csv'))

# ---- spatial structure
cg = pd.read_csv(next(f for f in ['out_correlogram_rx2.csv', 'out_correlogram_rx.csv', 'out_correlogram.csv'] if os.path.exists(f)))
V['cg'] = cg
lab = cg[cg.variable == 'Landslide label'].set_index('dist_km').moranI
V['moran_lab'] = lab.to_dict()
sch = pd.read_csv('out_scheme_sensitivity.csv'); sch['features'] = sch.features.replace({'Predictors': 'Model S predictors'})
scT = pd.read_csv(next(f for f in ['out_scheme_sensitivity_rx2.csv', 'out_scheme_sensitivity_rx.csv', 'out_scheme_sensitivity_nospec.csv'] if os.path.exists(f))); scT['features'] = 'Model T predictors'
sch = pd.concat([sch, scT])
S = sch.groupby(['scheme', 'features', 'model']).agg(auc=('pooled_auc', 'mean'), sd=('pooled_auc', 'std'), nn=('median_nn_test_train_m', 'mean')).reset_index()
V['sch'] = S


def g(scheme, feats, model):
    r = S[(S.scheme == scheme) & (S.features == feats) & (S.model == model)].iloc[0]; return r.auc, r.sd


V['g'] = g

# ---- nested results for both models
TT = next(t for t in ['_rx2', '_rx', '_nospec'] if os.path.exists(f'out_metrics_summary{t}.csv'))
V['TT'] = TT
for tag, key in [(TT, 'T'), ('', 'S')]:
    M = pd.read_csv(f'out_metrics_summary{tag}.csv').set_index(['kind', 'method'])
    F = pd.read_csv(f'out_fold_auc_summary{tag}.csv').set_index(['kind', 'method'])
    Dl = pd.read_csv(f'out_delong{tag}.csv')
    A = pd.read_csv(f'out_agreement_error{tag}.csv')
    Z = pd.read_csv(f'out_dispersion_error{tag}.csv')
    FD = pd.read_csv(f'out_fold_decisions{tag}.csv')
    PI = pd.read_csv(f'out_perm_importance{tag}.csv')
    CM = pd.read_csv(f'out_confusion_rep0_cal{tag}.csv').set_index('method')
    CI = pd.read_csv(f'out_auc_ci{tag}.csv')
    PR = pd.read_csv(f'out_fold_params{tag}.csv')
    V[key] = dict(M=M, F=F, D=Dl, A=A, Z=Z, FD=FD, PI=PI, CM=CM, CI=CI, PR=PR)

_fmp = next((f for f in ['final_models_rx2.pkl', 'final_models_rx.pkl', 'final_models_nospec.pkl'] if os.path.exists(f)), 'none')
if os.path.exists(_fmp):
    fm = pickle.load(open(_fmp, 'rb'))
    V['final'] = {k: fm[k] for k in ['feature_set', 'columns', 'params', 'cv_auc', 'weights']}

import sklearn, xgboost, scipy, sys, statsmodels
V['versions'] = dict(python=sys.version.split()[0], sklearn=sklearn.__version__, xgboost=xgboost.__version__,
                     numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, statsmodels=statsmodels.__version__)
