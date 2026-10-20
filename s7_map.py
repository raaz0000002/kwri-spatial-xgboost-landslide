"""Step 7: district-scale inference with the locked Model T pipeline, reliability layers, AOA and class-wise validation.
Usage: LS_DATA=samples_raster.csv LS_EXCLUDE=MBI,EVI,MNDWI LS_TAG=_nospec python s7_map.py <raster_dir> [boundary_file]"""
import sys, os, glob, json, pickle, numpy as np, pandas as pd, rasterio, warnings
warnings.filterwarnings('ignore')
from rasterio.features import geometry_mask
from common import *
from s6_extract_from_rasters import find, load_aligned, FILES
from s2_nested import score, logit, MODELS, EXCL, TAG, vif_select, COLS_FULL, COLS_NOSOIL
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

RDIR = sys.argv[1]
BND = sys.argv[2] if len(sys.argv) > 2 else os.path.join(RDIR, 'kavre_boundary.gpkg')
os.makedirs('maps', exist_ok=True)
final = pickle.load(open(f'final_models{TAG}.pkl', 'rb'))
COLS = final['columns']; NAMES = list(MODELS)
ref = rasterio.open(find('Elevation')); H, W = ref.shape

# ---- stack
arrs = {}
EXCL_COLS0 = {CONT[k] for k in EXCL if k in CONT}
for col, stem in FILES.items():
    if col in EXCL_COLS0:
        arrs[col] = np.zeros((H, W)); continue   # excluded predictor: placeholder, never used by the model
    arrs[col], _ = load_aligned(stem, ref)
df_cols = {k: v for k, v in CONT.items()}
valid_feat = np.ones((H, W), bool)
EXCL_COLS = {CONT[k] for k in EXCL if k in CONT}
USE = [c for c in FILES if c not in EXCL_COLS]
for c in USE:
    if c != 'Parent_soil':
        valid_feat &= np.isfinite(arrs[c])
if BND is None:
    c = [p for ext in ('shp', 'geojson', 'gpkg') for p in glob.glob(os.path.join(RDIR, '**', f'*.{ext}'), recursive=True)]
    BND = c[0] if c else None
if BND:
    import geopandas as gpd
    b = gpd.read_file(BND).to_crs(ref.crs)
    inside = ~geometry_mask(b.geometry, (H, W), ref.transform)
else:
    inside = np.isfinite(arrs['Elevation'])
valid = valid_feat & inside
audit = [['District polygon (or elevation footprint)', int(inside.sum())]]
for c in USE:
    audit.append([f'missing in {c}', int((inside & ~np.isfinite(arrs[c])).sum())])
audit.append(['Valid complete-case pixels', int(valid.sum())])
pd.DataFrame(audit, columns=['item', 'pixels']).assign(km2=lambda d: d.pixels * 0.0009).to_csv('out_nodata_audit.csv', index=False)
with rasterio.open('maps/nodata_mask.tif', 'w', **{**ref.profile, 'dtype': 'uint8', 'nodata': 255, 'count': 1, 'compress': 'deflate'}) as dst:
    m = np.full((H, W), 255, 'uint8'); m[inside] = 0; m[valid] = 1; dst.write(m, 1)

# ---- pixel feature frame (same construction as build_X)
idx = np.where(valid.ravel())[0]
raw = pd.DataFrame({c: arrs[c].ravel()[idx] for c in FILES})
raw['Parent_soil'] = raw['Parent_soil'].fillna(-9999)
Xpix = build_X(raw, soil=True).drop(columns=EXCL)[COLS]


def predict_all(models, platts, X, batch=300000):
    outs = {m: np.empty(len(X), 'float32') for m in NAMES}
    for s in range(0, len(X), batch):
        xb = X.iloc[s:s + batch]
        for mname in NAMES:
            p = score(models[mname], xb)
            outs[mname][s:s + batch] = platts[mname].predict_proba(logit(p)[:, None])[:, 1]
    return outs


cal = predict_all(final['models'], final['platt'], Xpix)
ens = np.mean([cal[m] for m in NAMES], 0)
sd = np.std([cal[m] for m in NAMES], 0)
votes = np.sum([(cal[m] >= 0.5) for m in NAMES], 0).astype('uint8')
q = np.quantile(ens, [0.2, 0.4, 0.6, 0.8])
cls = (np.digitize(ens, q) + 1).astype('uint8')


def write(name, vals, dtype, nodata):
    a = np.full(H * W, nodata, dtype); a[idx] = vals
    with rasterio.open(f'maps/{name}.tif', 'w', **{**ref.profile, 'dtype': dtype, 'nodata': nodata, 'count': 1, 'compress': 'deflate'}) as dst:
        dst.write(a.reshape(H, W), 1)


write('ensemble_score_calibrated', ens, 'float32', np.nan)
write('intermodel_dispersion_sd', sd, 'float32', np.nan)
write('agreement_votes', votes, 'uint8', 255)
write('susceptibility_quintiles', cls, 'uint8', 255)
for mname in NAMES:
    write(f'score_{mname}', cal[mname], 'float32', np.nan)

# ---- AOA (Meyer & Pebesma 2021): weighted standardized predictor space
PI = pd.read_csv(f'out_perm_importance{TAG}.csv'); PI = PI[(PI.model == 'Ensemble-equal') & ~PI.unit.str.startswith('FAMILY:')].set_index('unit').mean_drop.clip(lower=0)
wts = np.array([PI.get('ParentSoil', 0) / 6 if c.startswith('ParentSoil') else PI.get(c, 0) for c in COLS])
Xtr = build_X(soil=True).drop(columns=EXCL)[COLS].values
mu, sg = Xtr.mean(0), Xtr.std(0) + 1e-12
Ztr = (Xtr - mu) / sg * wts
from scipy.spatial.distance import cdist
dbar = cdist(Ztr, Ztr)[np.triu_indices(len(Ztr), 1)].mean()
g = square_blocks(XY, 10000, (0, 0))
from sklearn.model_selection import StratifiedGroupKFold
di_tr = np.empty(len(Ztr))
for a, bb in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Ztr, y, g):
    di_tr[bb] = cdist(Ztr[bb], Ztr[a]).min(1) / dbar
thr = np.quantile(di_tr, 0.75) + 1.5 * (np.quantile(di_tr, 0.75) - np.quantile(di_tr, 0.25))
di = np.empty(len(Xpix), 'float32')
Zp = (Xpix.values - mu) / sg * wts
for s in range(0, len(Zp), 20000):
    di[s:s + 20000] = cdist(Zp[s:s + 20000], Ztr).min(1) / dbar
write('aoa_dissimilarity_index', di, 'float32', np.nan)
write('aoa_inside', (di <= thr).astype('uint8'), 'uint8', 255)

# ---- univariate coverage: share of pixels below / within / above training range
cov = []
Xtr_df = build_X(soil=True).drop(columns=EXCL)[COLS]
for c in COLS:
    if c.startswith('ParentSoil'):
        continue
    lo, hi = Xtr_df[c].min(), Xtr_df[c].max(); v = Xpix[c].values
    cov.append([c, lo, hi, (v < lo).mean() * 100, ((v >= lo) & (v <= hi)).mean() * 100, (v > hi).mean() * 100])
pd.DataFrame(cov, columns=['predictor', 'train_min', 'train_max', 'pct_below', 'pct_within', 'pct_above']).to_csv('out_pixel_coverage.csv', index=False)

# ---- class-wise validation with fold models (held-out landslides vs fold-specific quintile breaks)
res = pickle.load(open(f'out_nested{TAG}.pkl', 'rb'))
rng = np.random.RandomState(SEED); sub = rng.choice(len(Xpix), min(300000, len(Xpix)), replace=False)
XF = build_X(soil=True).drop(columns=EXCL)
rows_xy = rasterio.transform.rowcol(ref.transform, XY[:, 0], XY[:, 1])
cv_rows = []
for r in res:
    if r['rep'] != 0:
        continue
    te = r['test_idx']; tr = np.setdiff1d(np.arange(len(y)), te)
    cols = {'All (incl. parent soil)': COLS_FULL, 'No parent soil': COLS_NOSOIL, 'VIF<=10 reduced': r['vif_kept']}[r['fs_choice']]
    from sklearn.pipeline import Pipeline
    fm, fp = {}, {}
    from s2_nested import inner_splits
    isp = inner_splits(tr, r['rep'], r['fold'])
    for mname in NAMES:
        est = clone(MODELS[mname][0]).set_params(**{k: v for k, v in r['params'][mname].items()})
        pipe = Pipeline([('t', SkewYJ()), ('m', est)])
        oof = np.zeros(len(tr))
        for a_, b_ in isp:
            oof[b_] = score(clone(pipe).fit(XF.iloc[tr].iloc[a_][cols], y[tr][a_]), XF.iloc[tr].iloc[b_][cols])
        fm[mname] = pipe.fit(XF.iloc[tr][cols], y[tr])
        fp[mname] = LogisticRegression(C=1e6).fit(logit(oof)[:, None], y[tr])
    Xs = build_X(raw.iloc[sub], soil=True).drop(columns=EXCL)[cols]
    ens_sub = np.mean(list(predict_all(fm, fp, Xs).values()), 0)
    br = np.quantile(ens_sub, [0.2, 0.4, 0.6, 0.8])
    Xte = XF.iloc[te][cols]
    ens_te = np.mean(list(predict_all(fm, fp, Xte).values()), 0)
    c_te = np.digitize(ens_te, br) + 1
    for k in range(1, 6):
        cv_rows.append([r['fold'], k, int(((c_te == k) & (y[te] == 1)).sum()), int((y[te] == 1).sum())])
CV = pd.DataFrame(cv_rows, columns=['fold', 'class', 'ls_in_class', 'ls_total']).groupby('class')[['ls_in_class', 'ls_total']].sum().reset_index()
CV['share_pct'] = CV.ls_in_class / CV.ls_in_class.sum() * 100
CV['density_ratio'] = CV.share_pct / 20.0
CV['cum_capture_from_top_pct'] = CV.share_pct[::-1].cumsum()[::-1]
CV.to_csv('out_class_validation.csv', index=False)

area = pd.DataFrame({'class': range(1, 6), 'label': ['Very Low', 'Low', 'Moderate', 'High', 'Very High'],
                     'pixels': [(cls == k).sum() for k in range(1, 6)]})
area['km2'] = area.pixels * 0.0009; area['lower'] = [ens.min()] + list(q); area['upper'] = list(q) + [ens.max()]
area.to_csv('out_area_classes.csv', index=False)
summ = dict(valid_pixels=int(valid.sum()), valid_km2=float(valid.sum() * 0.0009), ens_min=float(ens.min()), ens_max=float(ens.max()),
            ens_mean=float(ens.mean()), sd_min=float(sd.min()), sd_max=float(sd.max()), sd_mean=float(sd.mean()),
            votes_pct={int(k): float((votes == k).mean() * 100) for k in range(5)}, aoa_threshold=float(thr),
            aoa_inside_pct=float((di <= thr).mean() * 100), share_cal_ge05={m: float((cal[m] >= .5).mean() * 100) for m in NAMES},
            ens_ge05_pct=float((ens >= .5).mean() * 100))
json.dump(summ, open('out_raster_summary.json', 'w'), indent=1)
print(json.dumps(summ, indent=1)); print(area); print(CV)
