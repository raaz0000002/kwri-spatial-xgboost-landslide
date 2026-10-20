"""Shared data preparation for the Kavrepalanchowk landslide susceptibility analysis."""
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SEED = 42
import os

def _find_data(name):
    if os.path.exists(name):
        return name
    for folder in ['data', os.path.join(os.path.dirname(__file__), 'data')]:
        cand = os.path.join(folder, name)
        if os.path.exists(cand):
            return cand
    return name

RAW = pd.read_csv(_find_data(os.environ.get('LS_DATA', 'samples_proj.csv')))

CONT = {  # manuscript name : column in sample table
    'Elevation': 'Elevation', 'Slope': 'Slope', 'TPI': 'TPI(Topographic Position Index)',
    'Roughness': 'Roughness', 'TRI': 'TRI(Topographic Ruggedness Index)',
    'FlowAcc': 'Flow Accumulation', 'Curvature': 'Curvature', 'TWI': 'TWI',
    'Organic': 'Organic', 'Sand': 'Sand', 'Silt': 'Slit', 'Clay': 'Clay',
    'Rainfall': 'Monsoon_Cumulative_Rainfall', 'MBI': 'MBI', 'MNDWI': 'MNDWI', 'EVI': 'EVI',
    'DistBuilding': 'dist_building_m', 'DistRiver': 'dist_river_m', 'DistRoad': 'dist_road_m',
}
FAMILY = {
    'Elevation': 'Terrain', 'Slope': 'Terrain', 'TPI': 'Terrain', 'Roughness': 'Terrain', 'TRI': 'Terrain',
    'Curvature': 'Terrain', 'Northness': 'Terrain', 'Eastness': 'Terrain',
    'FlowAcc': 'Hydrology', 'TWI': 'Hydrology',
    'Organic': 'Soil texture', 'Sand': 'Soil texture', 'Silt': 'Soil texture', 'Clay': 'Soil texture',
    'ParentSoil': 'Parent soil',
    'Rainfall': 'Rainfall',
    'MBI': 'Spectral indices', 'MNDWI': 'Spectral indices', 'EVI': 'Spectral indices',
    'DistBuilding': 'Proximity', 'DistRiver': 'Proximity', 'DistRoad': 'Proximity',
}
SOIL_LEVELS = [1, 2, 3, 4, 5]


def build_X(df=RAW, soil=True):
    X = pd.DataFrame({k: df[v].astype(float) for k, v in CONT.items()})
    a = np.deg2rad(df['Aspect'].astype(float))
    flat = df['Slope'].astype(float) <= 0  # flat cells: aspect undefined -> 0,0
    X['Northness'] = np.where(flat, 0.0, np.cos(a))
    X['Eastness'] = np.where(flat, 0.0, np.sin(a))
    if soil:
        ps = df['Parent_soil'].replace(-9999, np.nan)
        for lv in SOIL_LEVELS:
            X[f'ParentSoil_{lv}'] = (ps == lv).astype(float)
        X['ParentSoil_missing'] = ps.isna().astype(float)
    return X


def family_of(col):
    return 'Parent soil' if col.startswith('ParentSoil') else FAMILY[col]


y = RAW['Landslide'].values.astype(int)
XY = RAW[['E', 'N']].values


class SkewYJ(BaseEstimator, TransformerMixin):
    """Yeo-Johnson transform applied only to continuous columns whose |skew|>0.5 in the
    *training* data, followed by standardisation of all continuous columns. Indicator
    (one-hot) columns pass through. Fitted inside each training fold only."""
    def __init__(self, skew_thr=0.5):
        self.skew_thr = skew_thr

    def fit(self, X, y=None):
        from sklearn.preprocessing import PowerTransformer, StandardScaler
        X = pd.DataFrame(X).copy()
        self.cols_ = list(X.columns)
        self.ind_ = [c for c in self.cols_ if str(c).startswith('ParentSoil')]
        self.cont_ = [c for c in self.cols_ if c not in self.ind_]
        sk = X[self.cont_].skew()
        self.skewed_ = [c for c in self.cont_ if abs(sk[c]) > self.skew_thr]
        self.pt_ = PowerTransformer(method='yeo-johnson', standardize=False).fit(X[self.skewed_]) if self.skewed_ else None
        Z = X[self.cont_].copy()
        if self.pt_ is not None:
            Z[self.skewed_] = self.pt_.transform(X[self.skewed_])
        self.sc_ = StandardScaler().fit(Z)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.cols_).copy()
        Z = X[self.cont_].copy()
        if self.pt_ is not None:
            Z[self.skewed_] = self.pt_.transform(X[self.skewed_])
        Z = pd.DataFrame(self.sc_.transform(Z), columns=self.cont_, index=X.index)
        return pd.concat([Z, X[self.ind_]], axis=1).values


def square_blocks(xy, size_m, offset=(0.0, 0.0)):
    gx = np.floor((xy[:, 0] - offset[0]) / size_m).astype(int)
    gy = np.floor((xy[:, 1] - offset[1]) / size_m).astype(int)
    return gx * 100000 + gy
