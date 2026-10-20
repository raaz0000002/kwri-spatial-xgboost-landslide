"""Fit the locked pipeline on all 1,200 samples for mapping (no performance is reported from this fit).
Feature set = the one chosen most often in the nested outer folds. Hyperparameters tuned by spatial
(10-km block) 5-fold CV on all samples; Platt calibrators fitted on the resulting spatial OOF scores."""
import numpy as np, pandas as pd, pickle, sys, json, warnings
warnings.filterwarnings('ignore')
from common import *
from s2_nested import MODELS, score, logit, vif_select, BLOCK, COLS_FULL, COLS_NOSOIL, TAG, EXCL
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

dec = pd.read_csv(f'out_fold_decisions{TAG}.csv')
choice = dec.choice.value_counts().idxmax()
XF = build_X(soil=True).drop(columns=EXCL)
cols = {'All (incl. parent soil)': COLS_FULL, 'No parent soil': COLS_NOSOIL,
        'VIF<=10 reduced': vif_select(XF, COLS_FULL)}[choice]
X = XF[cols]
g = square_blocks(XY, BLOCK, offset=(0.0, 0.0))
cv = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, g))


def tune(mn):
    est, grid = MODELS[mn]
    rs = RandomizedSearchCV(Pipeline([('t', SkewYJ()), ('m', est)]), grid, n_iter=40, scoring='roc_auc', cv=cv,
                            random_state=SEED, n_jobs=1, refit=True).fit(X, y)
    best = rs.best_estimator_
    oof = np.zeros(len(y))
    for a, b in cv:
        oof[b] = score(clone(best).fit(X.iloc[a], y[a]), X.iloc[b])
    platt = LogisticRegression(C=1e6).fit(logit(oof)[:, None], y)
    return mn, best, oof, platt, {k.replace('m__', ''): v for k, v in rs.best_params_.items()}, rs.best_score_


out = Parallel(n_jobs=2)(delayed(tune)(m) for m in MODELS)
final = {'feature_set': choice, 'columns': cols, 'models': {}, 'platt': {}, 'oof': {}, 'params': {}, 'cv_auc': {}}
for mn, best, oof, platt, params, sc in out:
    final['models'][mn] = best; final['platt'][mn] = platt; final['oof'][mn] = oof
    final['params'][mn] = params; final['cv_auc'][mn] = sc
names = list(MODELS)
Zi = np.column_stack([logit(final['oof'][m]) for m in names])
final['stacker'] = LogisticRegression(C=1.0).fit(Zi, y)
w = np.array([max(roc_auc_score(y, final['oof'][m]) - 0.5, 1e-3) for m in names]); final['weights'] = dict(zip(names, w / w.sum()))
pickle.dump(final, open(f'final_models{TAG}.pkl', 'wb'))
print(choice, len(cols)); print(json.dumps(final['params'], indent=1, default=str)); print(final['cv_auc']); print(final['weights'])
