"""Step 6: rebuild the training table from the final aligned raster stack (single source of truth).
Usage: python s6_extract_from_rasters.py <raster_dir>
Writes samples_raster.csv (same schema as samples_proj.csv) and out_extract_vs_table.csv."""
import sys, glob, os, numpy as np, pandas as pd, rasterio
from rasterio.warp import reproject, Resampling

RDIR = sys.argv[1] if len(sys.argv) > 1 else '../rasters'
FILES = {  # sample-table column : raster file stem
    'Elevation': 'Elevation', 'Slope': 'Slope', 'Aspect': 'Aspect', 'TPI(Topographic Position Index)': 'tpi',
    'Roughness': 'Roughness', 'TRI(Topographic Ruggedness Index)': 'TRI', 'Flow Accumulation': 'Flow_Accumulation',
    'Curvature': 'Curvature', 'TWI': 'TWI', 'Organic': 'Organic', 'Sand': 'Sand', 'Slit': 'Silt', 'Clay': 'Clay',
    'Parent_soil': 'Parent_soil_class', 'Monsoon_Cumulative_Rainfall': 'rainfall', 'MBI': 'MBI', 'MNDWI': 'MNDWI',
    'EVI': 'EVI', 'dist_building_m': 'dist_building', 'dist_river_m': 'dist_river_m', 'dist_road_m': 'dist_road_m'}
CATEG = {'Parent_soil'}


def find(stem):
    c = [p for p in glob.glob(os.path.join(RDIR, '**', '*.tif'), recursive=True) if os.path.splitext(os.path.basename(p))[0].lower() == stem.lower()]
    if not c:
        raise FileNotFoundError(stem)
    return c[0]


def load_aligned(stem, ref):
    """Return array on the reference grid (reprojected if needed) with NoData -> NaN."""
    with rasterio.open(find(stem)) as src:
        same = (src.crs == ref.crs and src.transform == ref.transform and src.shape == ref.shape)
        if same:
            a = src.read(1).astype('float64')
        else:
            a = np.full(ref.shape, np.nan)
            reproject(rasterio.band(src, 1), a, src_transform=src.transform, src_crs=src.crs, dst_transform=ref.transform,
                      dst_crs=ref.crs, src_nodata=src.nodata, dst_nodata=np.nan,
                      resampling=Resampling.nearest if stem == 'Parent_soil_class' else Resampling.bilinear)
        nd = src.nodata
    if nd is not None:
        a[a == nd] = np.nan
    a[(a <= -9999) | ~np.isfinite(a)] = np.nan
    if stem == 'Parent_soil_class':
        a[a == 0] = np.nan  # 0 = unmapped
    return a, same


if __name__ == '__main__':
    T = pd.read_csv('samples_proj.csv')
    ref = rasterio.open(find('Elevation'))
    rows, cols = rasterio.transform.rowcol(ref.transform, T.E.values, T.N.values)
    rows, cols = np.array(rows), np.array(cols)
    inside = (rows >= 0) & (rows < ref.height) & (cols >= 0) & (cols < ref.width)
    out = T.copy(); cmp = []; skipped = []
    for col, stem in FILES.items():
        try:
            a, same = load_aligned(stem, ref)
        except FileNotFoundError:
            cmp.append([col, stem, 'missing', 0, np.nan, np.nan, np.nan, np.nan, np.nan]); skipped.append(col); continue
        if np.nanstd(a) == 0:
            cmp.append([col, stem, 'constant raster', 0, np.nan, np.nan, np.nan, np.nan, np.nan]); skipped.append(col); continue
        v = np.full(len(T), np.nan); v[inside] = a[rows[inside], cols[inside]]
        tab = T[col].replace(-9999, np.nan).astype(float)
        ok = np.isfinite(v) & np.isfinite(tab)
        if col in CATEG:
            agree = (v[ok] == tab[ok]).mean() * 100
            cmp.append([col, stem, same, ok.sum(), np.nan, np.nan, np.nan, np.nan, agree])
        else:
            r = np.corrcoef(v[ok], tab[ok])[0, 1]
            mad = np.median(np.abs(v[ok] - tab[ok]))
            rel = mad / (np.nanstd(tab) + 1e-12)
            ratio = np.nanmedian(v[ok] / np.where(tab[ok] == 0, np.nan, tab[ok]))
            cmp.append([col, stem, same, ok.sum(), r, mad, rel, ratio, np.nan])
        out[col] = v
    C = pd.DataFrame(cmp, columns=['column', 'raster', 'on_reference_grid', 'n_valid', 'pearson_r', 'median_abs_diff', 'median_abs_diff_over_sd', 'median_ratio_raster_to_table', 'categorical_agreement_pct'])
    C.to_csv('out_extract_vs_table.csv', index=False); print(C.round(3).to_string())
    out['Parent_soil'] = out['Parent_soil'].fillna(-9999)
    na = out[[c for c in FILES if c not in skipped]].isna().any(axis=1)
    print('NOT extracted (table values kept, must be excluded from models):', skipped)
    print('samples with missing raster values:', int(na.sum()))
    out[~na].to_csv('samples_raster.csv', index=False)
    print('wrote samples_raster.csv with', int((~na).sum()), 'rows')
