"""Pseudo-absence sensitivity: replace the supplied non-landslide points by 5 random realizations drawn over the
valid district domain (>= BUF m from any landslide point, >= MINSP m apart), keep the 599 landslide points, and repeat
(a) 10-km spatial CV of the four models (final Model T hyperparameters) and their equal-weight ensemble,
(b) the coordinates-only baseline, (c) family permutation importance, (d) agreement of the resulting map with the
final map. Run with LS_DATA=samples_raster.csv LS_EXCLUDE=... LS_TAG=_rx2."""
import numpy as np, pandas as pd, pickle, rasterio, warnings, os
warnings.filterwarnings('ignore')
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from common import *
from s2_nested import MODELS, score, logit, EXCL, TAG
from s6_extract_from_rasters import FILES, load_aligned, find

BUF, MINSP, NREAL = 300.0, 100.0, 5
final = pickle.load(open(f'final_models{TAG}.pkl', 'rb')); COLS = final['columns']; NAMES = list(MODELS)
ref = rasterio.open(find('Elevation')); H, W = ref.shape
mask = rasterio.open('maps/nodata_mask.tif').read(1) == 1
EXCL_COLS = {CONT[k] for k in EXCL if k in CONT}
arrs = {c: (load_aligned(s, ref)[0] if c not in EXCL_COLS else None) for c, s in FILES.items()}
ens_map = rasterio.open('maps/ensemble_score_calibrated.tif').read(1)
ls = RAW[RAW.Landslide == 1].copy()
lsxy = ls[['E', 'N']].values
vr, vc = np.where(mask)
pxE, pxN = rasterio.transform.xy(ref.transform, vr, vc); pxE, pxN = np.array(pxE), np.array(pxN)
far = cKDTree(lsxy).query(np.c_[pxE, pxN])[0] >= BUF
cand = np.where(far)[0]


def frame(rows, cols_):
    d = {c: (arrs[c][rows, cols_] if arrs[c] is not None else np.zeros(len(rows))) for c in FILES}
    df = pd.DataFrame(d); df['Parent_soil'] = df['Parent_soil'].fillna(-9999)
    return df


rng0 = np.random.RandomState(SEED)
sub = rng0.choice(len(vr), 200000, replace=False)
Xsub = build_X(frame(vr[sub], vc[sub]), soil=True).drop(columns=EXCL)[COLS]
ens_sub = ens_map[vr[sub], vc[sub]]
q80 = np.quantile(ens_sub, 0.8)
fams = sorted(set(family_of(c) for c in COLS))
out, imp = [], []
for r in range(-1, NREAL):
    rng = np.random.RandomState(100 + r)
    order = rng.permutation(cand) if r >= 0 else []; chosen = []
    grid = {}
    for i in order:
        p = (pxE[i], pxN[i]); gx, gy = int(p[0] // MINSP), int(p[1] // MINSP)
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for q in grid.get((gx + dx, gy + dy), []):
                    if np.hypot(p[0] - q[0], p[1] - q[1]) < MINSP:
                        ok = False; break
                if not ok: break
            if not ok: break
        if not ok:
            continue
        grid.setdefault((gx, gy), []).append(p); chosen.append(i)
        if len(chosen) == len(ls):
            break
    chosen = np.array(chosen, dtype=int)
    if r >= 0:
        ab = frame(vr[chosen], vc[chosen]); ab['E'] = pxE[chosen]; ab['N'] = pxN[chosen]; ab['Landslide'] = 0
        lsf = ls[list(FILES) + ['E', 'N', 'Landslide']].copy()
        D = pd.concat([lsf, ab[list(FILES) + ['E', 'N', 'Landslide']]], ignore_index=True)
    else:
        D = RAW[list(FILES) + ['E', 'N', 'Landslide']].copy()
    X = build_X(D, soil=True).drop(columns=EXCL)[COLS]; yy = D.Landslide.values; xy = D[['E', 'N']].values
    g = square_blocks(xy, 10000, tuple(rng.uniform(0, 10000, 2)))
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=r + 1).split(X, yy, g))
    oof = {m: np.zeros(len(yy)) for m in NAMES}
    for tr, te in folds:
        for m in NAMES:
            est = clone(MODELS[m][0]).set_params(**final['params'][m])
            p = Pipeline([('t', SkewYJ()), ('m', est)]).fit(X.iloc[tr], yy[tr])
            oof[m][te] = score(p, X.iloc[te])
    # per-fold Platt using training-fold OOF is costly; ensemble uses mean of within-fold rank-normalised scores
    ens = np.zeros(len(yy))
    for tr, te in folds:
        ens[te] = np.mean([pd.Series(oof[m][te]).rank(pct=True).values for m in NAMES], 0)
    res = dict(realization=('original' if r < 0 else r + 1), n_absence=int((D.Landslide == 0).sum()), **{f'AUC_{m}': roc_auc_score(yy, oof[m]) for m in NAMES},
               AUC_ensemble=roc_auc_score(yy, ens))
    c_rand = np.zeros(len(yy)); c_blk = np.zeros(len(yy))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=r + 1).split(xy, yy):
        c_rand[te] = RandomForestClassifier(300, random_state=SEED, n_jobs=2).fit(xy[tr], yy[tr]).predict_proba(xy[te])[:, 1]
    for tr, te in folds:
        c_blk[te] = RandomForestClassifier(300, random_state=SEED, n_jobs=2).fit(xy[tr], yy[tr]).predict_proba(xy[te])[:, 1]
    res['AUC_coords_random'] = roc_auc_score(yy, c_rand); res['AUC_coords_10km'] = roc_auc_score(yy, c_blk)
    # family permutation importance (ensemble, 10-km folds, 3 permutations)
    fitted = {}
    for k, (tr, te) in enumerate(folds):
        fitted[k] = {m: Pipeline([('t', SkewYJ()), ('m', clone(MODELS[m][0]).set_params(**final['params'][m]))]).fit(X.iloc[tr], yy[tr]) for m in NAMES}
    for f in fams:
        cols_f = [c for c in COLS if family_of(c) == f]; drops = []
        for k, (tr, te) in enumerate(folds):
            Xte = X.iloc[te]
            base = roc_auc_score(yy[te], np.mean([score(fitted[k][m], Xte) for m in NAMES], 0))
            for rep in range(3):
                Xp = Xte.copy(); Xp[cols_f] = Xte[cols_f].values[rng.permutation(len(Xte))]
                drops.append(base - roc_auc_score(yy[te], np.mean([score(fitted[k][m], Xp) for m in NAMES], 0)))
        imp.append([('original' if r < 0 else r + 1), f, np.mean(drops)])
    # map agreement: refit on all data with Platt on OOF, predict pixel subsample
    preds = []
    for m in NAMES:
        p = Pipeline([('t', SkewYJ()), ('m', clone(MODELS[m][0]).set_params(**final['params'][m]))]).fit(X, yy)
        pl = LogisticRegression(C=1e6).fit(logit(oof[m])[:, None], yy)
        preds.append(pl.predict_proba(logit(score(p, Xsub))[:, None])[:, 1])
    e2 = np.mean(preds, 0)
    res['map_spearman'] = spearmanr(e2, ens_sub).correlation
    vh_final = ens_sub >= q80; vh_new = e2 >= np.quantile(e2, 0.8)
    res['VH_overlap_pct'] = (vh_final & vh_new).sum() / vh_final.sum() * 100
    out.append(res); print(res, flush=True)
O = pd.DataFrame(out); O.to_csv(f'out_pseudoabsence{TAG}.csv', index=False)
I = pd.DataFrame(imp, columns=['realization', 'family', 'mean_drop']); I.to_csv(f'out_pseudoabsence_importance{TAG}.csv', index=False)
print(O.round(3).to_string()); print(I.pivot(index='family', columns='realization', values='mean_drop').round(3))
