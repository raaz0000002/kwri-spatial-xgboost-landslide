"""Summarise nested spatial CV results into manuscript tables."""
import numpy as np, pandas as pd, pickle, json, warnings
warnings.filterwarnings('ignore')
from common import *
from delong import delong_pair, auc_ci, holm
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss, log_loss,
                             balanced_accuracy_score, matthews_corrcoef, f1_score, accuracy_score,
                             precision_score, recall_score, confusion_matrix)
from sklearn.linear_model import LogisticRegression

import os
TAG = os.environ.get('LS_TAG', '')
res = pickle.load(open(f'out_nested{TAG}.pkl', 'rb'))
reps = sorted(set(r['rep'] for r in res))
METHODS = ['RF', 'XGBoost', 'GBM', 'SVM', 'Ensemble-equal', 'Ensemble-weighted', 'Ensemble-stacked', 'Best single (inner-CV)']


def pooled(rep, kind):
    o = {m: np.full(len(y), np.nan) for m in METHODS}
    foldid = np.full(len(y), -1)
    for r in res:
        if r['rep'] != rep:
            continue
        for m in METHODS:
            o[m][r['test_idx']] = r[kind][m]
        foldid[r['test_idx']] = r['fold']
    return o, foldid


def cal_slope_int(yv, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    s = LogisticRegression(C=1e6).fit(lp[:, None], yv).coef_[0, 0]
    # calibration-in-the-large: intercept with slope fixed at 1 (offset model) via Newton
    a = 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(a + lp))); g = (yv - q).sum(); h = (q * (1 - q)).sum(); a += g / h
    return s, a


def ece(yv, p, bins=10):
    e = 0; edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1)
        if m.sum():
            e += m.mean() * abs(p[m].mean() - yv[m].mean())
    return e


def metrics(yv, p):
    yh = (p >= 0.5).astype(int); s, a = cal_slope_int(yv, p)
    return dict(AUC=roc_auc_score(yv, p), PR_AUC=average_precision_score(yv, p), Brier=brier_score_loss(yv, p),
                LogLoss=log_loss(yv, np.clip(p, 1e-6, 1 - 1e-6)), Accuracy=accuracy_score(yv, yh),
                BalAcc=balanced_accuracy_score(yv, yh), MCC=matthews_corrcoef(yv, yh), F1=f1_score(yv, yh),
                Precision=precision_score(yv, yh), Recall=recall_score(yv, yh), CalSlope=s, CalIntercept=a, ECE=ece(yv, p))


rows, frows = [], []
for rep in reps:
    for kind in ['raw', 'cal']:
        o, fid = pooled(rep, kind)
        for m in METHODS:
            rows.append(dict(rep=rep, kind=kind, method=m, **metrics(y, o[m])))
            for f in sorted(set(fid)):
                mm = fid == f
                frows.append(dict(rep=rep, kind=kind, method=m, fold=f, n=mm.sum(), pos=y[mm].sum(),
                                  AUC=roc_auc_score(y[mm], o[m][mm]), Brier=brier_score_loss(y[mm], o[m][mm])))
M = pd.DataFrame(rows); F = pd.DataFrame(frows)
M.to_csv(f'out_metrics_by_rep{TAG}.csv', index=False); F.to_csv(f'out_metrics_by_fold{TAG}.csv', index=False)
agg = M.groupby(['kind', 'method']).agg(['mean', 'std']).drop(columns='rep')
agg.columns = [f'{a}_{b}' for a, b in agg.columns]
agg = agg.reset_index(); agg.to_csv(f'out_metrics_summary{TAG}.csv', index=False)
fagg = F.groupby(['kind', 'method']).AUC.agg(['mean', 'std', 'min', 'max']).reset_index()
fagg.to_csv(f'out_fold_auc_summary{TAG}.csv', index=False)
pd.set_option('display.width', 250)
print(agg[['kind', 'method', 'AUC_mean', 'AUC_std', 'PR_AUC_mean', 'Brier_mean', 'LogLoss_mean', 'BalAcc_mean', 'MCC_mean', 'F1_mean', 'CalSlope_mean', 'CalIntercept_mean', 'ECE_mean']].round(4).to_string())
print(fagg.round(3).to_string())

# ---- DeLong on pooled OOF (raw scores; AUC is invariant to monotone calibration) for each rep
drows = []
for rep in reps:
    o, _ = pooled(rep, 'raw')
    pairs = [('XGBoost', 'RF'), ('XGBoost', 'GBM'), ('XGBoost', 'SVM'), ('RF', 'GBM'), ('RF', 'SVM'), ('GBM', 'SVM'),
             ('Ensemble-equal', 'XGBoost'), ('Ensemble-equal', 'RF'), ('Ensemble-equal', 'GBM'), ('Ensemble-equal', 'SVM'), ('Ensemble-equal', 'Best single (inner-CV)'), ('Ensemble-stacked', 'Ensemble-equal'),
             ('Ensemble-weighted', 'Ensemble-equal')]
    tmp = []
    for a, b in pairs:
        if a == b:
            continue
        r = delong_pair(y, o[a], o[b]); tmp.append([rep, a, b, *r])
    ps = holm([t[-1] for t in tmp])
    for t, ph in zip(tmp, ps):
        drows.append(t + [ph])
D = pd.DataFrame(drows, columns=['rep', 'A', 'B', 'AUC_A', 'AUC_B', 'diff', 'diff_lo', 'diff_hi', 'p', 'p_holm'])
D.to_csv(f'out_delong{TAG}.csv', index=False)
print(D.round(4).to_string())
ci = []
for rep in reps:
    o, _ = pooled(rep, 'raw')
    for m in METHODS:
        a, lo, hi = auc_ci(y, o[m]); ci.append([rep, m, a, lo, hi])
pd.DataFrame(ci, columns=['rep', 'method', 'AUC', 'lo', 'hi']).to_csv(f'out_auc_ci{TAG}.csv', index=False)

# ---- confusion matrices (rep 0, calibrated, 0.5)
cm = []
o, _ = pooled(0, 'cal')
for m in METHODS:
    tn, fp, fn, tp = confusion_matrix(y, (o[m] >= 0.5).astype(int)).ravel(); cm.append([m, tn, fp, fn, tp])
pd.DataFrame(cm, columns=['method', 'TN', 'FP', 'FN', 'TP']).to_csv(f'out_confusion_rep0_cal{TAG}.csv', index=False)

# ---- agreement categories vs error (calibrated scores, 0.5 threshold), pooled over reps
ag = []
for rep in reps:
    o, _ = pooled(rep, 'cal')
    votes = sum((o[m] >= 0.5).astype(int) for m in ['RF', 'XGBoost', 'GBM', 'SVM'])
    ens = (o['Ensemble-equal'] >= 0.5).astype(int)
    for v in range(5):
        mm = votes == v
        ag.append([rep, v, mm.sum(), y[mm].mean() if mm.sum() else np.nan, (ens[mm] != y[mm]).mean() if mm.sum() else np.nan])
A = pd.DataFrame(ag, columns=['rep', 'votes', 'n', 'landslide_share', 'ensemble_error'])
A = A.groupby('votes').agg(n=('n', 'mean'), landslide_share=('landslide_share', 'mean'), ensemble_error=('ensemble_error', 'mean')).reset_index()
A.to_csv(f'out_agreement_error{TAG}.csv', index=False); print(A.round(3))
# dispersion vs error
dz = []
for rep in reps:
    o, _ = pooled(rep, 'cal')
    sd = np.std([o[m] for m in ['RF', 'XGBoost', 'GBM', 'SVM']], 0)
    err = np.abs(o['Ensemble-equal'] - y)
    q = pd.qcut(sd, 4, labels=False)
    for k in range(4):
        dz.append([rep, k, sd[q == k].mean(), err[q == k].mean(), ((o['Ensemble-equal'][q == k] >= .5) != y[q == k]).mean()])
DZ = pd.DataFrame(dz, columns=['rep', 'sd_quartile', 'mean_sd', 'mean_abs_err', 'misclass']).groupby('sd_quartile').mean().reset_index()
DZ.to_csv(f'out_dispersion_error{TAG}.csv', index=False); print(DZ.round(3))

# ---- feature-set choices, best single, params, weights
fs = pd.DataFrame([dict(rep=r['rep'], fold=r['fold'], choice=r['fs_choice'], best=r['best_single'], **{f'fs_{k}': v for k, v in r['fs_scores'].items()},
                        vif_dropped=';'.join(sorted(set(c for c in build_X().columns if c not in os.environ.get('LS_EXCLUDE','').split(',') if not c.startswith('ParentSoil')) - set(r['vif_kept']))),
                        **{f'w_{k}': v for k, v in r['weights_auc'].items()}, **{f'stk_{k}': v for k, v in r['stack_coef'].items()},
                        **{f'inner_{k}': v for k, v in r['inner_auc'].items()}) for r in res])
fs.to_csv(f'out_fold_decisions{TAG}.csv', index=False); print(fs.drop(columns=[c for c in fs.columns if c.startswith(('w_', 'stk_'))]).round(4).to_string())
pr = pd.DataFrame([dict(rep=r['rep'], fold=r['fold'], model=m, **p) for r in res for m, p in r['params'].items()])
pr.to_csv(f'out_fold_params{TAG}.csv', index=False)

# ---- permutation importance
P = pd.DataFrame([x for r in res for x in r['perm']], columns=['rep', 'fold', 'model', 'unit', 'r', 'drop'])
PI = P.groupby(['model', 'unit', 'rep', 'fold'])['drop'].mean().reset_index().groupby(['model', 'unit'])['drop'].agg(['mean', 'std', lambda s: (s > 0).mean()]).reset_index()
PI.columns = ['model', 'unit', 'mean_drop', 'sd_drop', 'share_folds_positive']
PI.to_csv(f'out_perm_importance{TAG}.csv', index=False)
print(PI[PI.model == 'Ensemble-equal'].sort_values('mean_drop', ascending=False).round(4).to_string())
