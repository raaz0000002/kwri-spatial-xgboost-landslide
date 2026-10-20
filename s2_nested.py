"""Nested spatial cross-validation of the full modelling pipeline.
Outer: 10-km square blocks -> 5 folds (StratifiedGroupKFold), repeated with random grid offsets.
Inner: 10-km blocks within the outer-training data -> 4 folds.
Everything data-dependent (feature-set choice, VIF screening, Yeo-Johnson/standardisation,
hyperparameters, Platt calibration, ensemble weights, stacking, best-model choice) is fitted
inside the outer-training data only."""
import numpy as np, pandas as pd, warnings, json, pickle, sys, time
warnings.filterwarnings('ignore')
from common import *
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

_a = [x for x in sys.argv[1:] if x.isdigit()] if __name__ == '__main__' else []
N_REP = int(_a[0]) if len(_a) > 0 else 3
N_ITER = int(_a[1]) if len(_a) > 1 else 15
BLOCK = 10000.0
import os
EXCL = [c for c in os.environ.get('LS_EXCLUDE', '').split(',') if c]
TAG = os.environ.get('LS_TAG', '')
XF = build_X(soil=True).drop(columns=EXCL)
COLS_FULL = list(XF.columns)
COLS_NOSOIL = [c for c in COLS_FULL if not c.startswith('ParentSoil')]

MODELS = {
    'RF': (RandomForestClassifier(random_state=SEED, n_jobs=1),
           {'m__n_estimators': [200, 300, 500], 'm__max_depth': [None, 5, 10, 15, 20],
            'm__min_samples_split': [2, 5, 10], 'm__min_samples_leaf': [1, 2, 4],
            'm__max_features': ['sqrt', 'log2', 0.5]}),
    'XGBoost': (XGBClassifier(eval_metric='logloss', random_state=SEED, n_jobs=1, tree_method='hist'),
                {'m__n_estimators': [100, 200, 300, 500], 'm__max_depth': [2, 3, 4, 5, 6, 8],
                 'm__learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2], 'm__subsample': [0.7, 0.8, 0.9, 1.0],
                 'm__colsample_bytree': [0.6, 0.7, 0.8, 1.0], 'm__min_child_weight': [1, 3, 5, 10],
                 'm__reg_lambda': [0.5, 1, 5, 10]}),
    'GBM': (GradientBoostingClassifier(random_state=SEED),
            {'m__n_estimators': [100, 200, 300], 'm__max_depth': [2, 3, 4, 5],
             'm__learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2], 'm__subsample': [0.7, 0.8, 0.9, 1.0],
             'm__min_samples_leaf': [1, 5, 10]}),
    'SVM': (SVC(kernel='rbf', probability=False, random_state=SEED),
            {'m__C': [0.1, 0.3, 1, 3, 10, 30, 100], 'm__gamma': ['scale', 0.003, 0.01, 0.03, 0.1, 0.3]}),
}


def score(est, X):
    """Continuous score in [0,1] (raw, uncalibrated). SVM decision function -> logistic."""
    if hasattr(est, 'predict_proba'):
        return est.predict_proba(X)[:, 1]
    return 1 / (1 + np.exp(-est.decision_function(X)))


def vif_select(Xtr, cols, thr=10.0):
    cont = [c for c in cols if not c.startswith('ParentSoil')]
    keep = list(cont)
    Z = (Xtr[keep] - Xtr[keep].mean()) / Xtr[keep].std()
    while True:
        v = [variance_inflation_factor(np.c_[np.ones(len(Z)), Z[keep].values], i + 1) for i in range(len(keep))]
        if max(v) <= thr:
            break
        keep.pop(int(np.argmax(v)))
    return keep + [c for c in cols if c.startswith('ParentSoil')]


def inner_splits(idx_tr, rep, fold):
    g = square_blocks(XY[idx_tr], BLOCK, offset=tuple(np.random.RandomState(1000 + rep).uniform(0, BLOCK, 2)))
    return list(StratifiedGroupKFold(4, shuffle=True, random_state=rep * 10 + fold).split(np.zeros(len(idx_tr)), y[idx_tr], g))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def run_fold(rep, fold, tr, te):
    t0 = time.time()
    Xtr_all, Xte_all, ytr = XF.iloc[tr], XF.iloc[te], y[tr]
    isp = inner_splits(tr, rep, fold)
    # ---- 1. feature-set decision inside inner CV (default XGB + RF)
    cand = {'All (incl. parent soil)': COLS_FULL, 'No parent soil': COLS_NOSOIL,
            'VIF<=10 reduced': vif_select(Xtr_all, COLS_FULL)}
    fs_scores = {}
    for name, cols in cand.items():
        s = []
        for est in [XGBClassifier(eval_metric='logloss', random_state=SEED, n_jobs=1),
                    RandomForestClassifier(300, random_state=SEED, n_jobs=1)]:
            p = cross_val_predict(Pipeline([('t', SkewYJ()), ('m', est)]), Xtr_all[cols], ytr, cv=isp, method='predict_proba')[:, 1]
            s.append(roc_auc_score(ytr, p))
        fs_scores[name] = float(np.mean(s))
    fs_choice = max(fs_scores, key=fs_scores.get)
    cols = cand[fs_choice]
    Xtr, Xte = Xtr_all[cols], Xte_all[cols]
    # ---- 2. tuning, inner OOF predictions, calibration
    out = {'rep': rep, 'fold': fold, 'test_idx': te, 'fs_scores': fs_scores, 'fs_choice': fs_choice,
           'n_feat': len(cols), 'vif_kept': cand['VIF<=10 reduced'], 'raw': {}, 'cal': {}, 'inner_auc': {}, 'params': {}}
    inner_oof = {}
    fitted = {}
    for mn, (est, grid) in MODELS.items():
        rs = RandomizedSearchCV(Pipeline([('t', SkewYJ()), ('m', est)]), grid, n_iter=N_ITER, scoring='roc_auc',
                                cv=isp, random_state=SEED + rep * 100 + fold, n_jobs=1, refit=True)
        rs.fit(Xtr, ytr)
        best = rs.best_estimator_
        oof = np.zeros(len(tr))
        for a, b in isp:
            from sklearn.base import clone
            oof[b] = score(clone(best).fit(Xtr.iloc[a], ytr[a]), Xtr.iloc[b])
        inner_oof[mn] = oof
        out['inner_auc'][mn] = roc_auc_score(ytr, oof)
        out['params'][mn] = {k.replace('m__', ''): v for k, v in rs.best_params_.items()}
        fitted[mn] = best
        p_te = score(best, Xte)
        out['raw'][mn] = p_te
        platt = LogisticRegression(C=1e6).fit(logit(oof)[:, None], ytr)  # Platt scaling on inner OOF scores
        out['cal'][mn] = platt.predict_proba(logit(p_te)[:, None])[:, 1]
        inner_oof[mn + '_cal'] = platt.predict_proba(logit(oof)[:, None])[:, 1]
    names = list(MODELS)
    # ---- 3. ensembles (weights/stacker learned on inner OOF only)
    out['raw']['Ensemble-equal'] = np.mean([out['raw'][m] for m in names], 0)
    out['cal']['Ensemble-equal'] = np.mean([out['cal'][m] for m in names], 0)
    w = np.array([max(out['inner_auc'][m] - 0.5, 1e-3) for m in names]); w = w / w.sum()
    out['weights_auc'] = dict(zip(names, w))
    out['raw']['Ensemble-weighted'] = np.sum([wi * out['raw'][m] for wi, m in zip(w, names)], 0)
    out['cal']['Ensemble-weighted'] = np.sum([wi * out['cal'][m] for wi, m in zip(w, names)], 0)
    Zi = np.column_stack([logit(inner_oof[m]) for m in names])
    stk = LogisticRegression(C=1.0).fit(Zi, ytr)
    out['stack_coef'] = dict(zip(names, stk.coef_[0]))
    Zt = np.column_stack([logit(out['raw'][m]) for m in names])
    out['raw']['Ensemble-stacked'] = stk.predict_proba(Zt)[:, 1]
    out['cal']['Ensemble-stacked'] = out['raw']['Ensemble-stacked']  # stacker is itself a calibrated LR
    bestm = max(names, key=lambda m: out['inner_auc'][m])
    out['best_single'] = bestm
    out['raw']['Best single (inner-CV)'] = out['raw'][bestm]
    out['cal']['Best single (inner-CV)'] = out['cal'][bestm]
    # ---- 4. permutation importance on the outer test fold (AUC drop, equal ensemble + each model)
    rng = np.random.RandomState(rep * 100 + fold)
    fams = sorted(set(family_of(c) for c in cols))
    def ens_pred(Xp):
        return np.mean([score(fitted[m], Xp) for m in names], 0)
    preds = {m: (lambda Xp, m=m: score(fitted[m], Xp)) for m in names}
    preds['Ensemble-equal'] = ens_pred
    base = {k: roc_auc_score(y[te], f(Xte)) for k, f in preds.items()}
    imp_rows = []
    units = [(c, [c]) for c in cols if not c.startswith('ParentSoil')] + \
            ([('ParentSoil', [c for c in cols if c.startswith('ParentSoil')])] if any(c.startswith('ParentSoil') for c in cols) else []) + \
            [('FAMILY:' + f, [c for c in cols if family_of(c) == f]) for f in fams]
    for uname, ucols in units:
        for r in range(5):
            Xp = Xte.copy(); perm = rng.permutation(len(Xp))
            Xp[ucols] = Xte[ucols].values[perm]
            for k, f in preds.items():
                imp_rows.append([rep, fold, k, uname, r, base[k] - roc_auc_score(y[te], f(Xp))])
    out['perm'] = imp_rows
    out['secs'] = time.time() - t0
    print(f'rep{rep} fold{fold} done {out["secs"]:.0f}s fs={fs_choice} best={bestm}', flush=True)
    return out


if __name__ == '__main__':
    jobs = []
    for rep in range(N_REP):
        g = square_blocks(XY, BLOCK, offset=tuple(np.random.RandomState(rep).uniform(0, BLOCK, 2)))
        for fold, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(XF, y, g)):
            jobs.append((rep, fold, tr, te))
    res = Parallel(n_jobs=2, verbose=0)(delayed(run_fold)(*j) for j in jobs)
    pickle.dump(res, open(f'out_nested{TAG}.pkl', 'wb'))
    print('ALL DONE')
