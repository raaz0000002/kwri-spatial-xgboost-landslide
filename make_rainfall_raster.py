"""CHIRPS v2.0 monthly 2024 -> June-September sum -> 30 m grid (bilinear), edge cells filled from nearest valid cell."""
import rasterio, numpy as np
from rasterio.warp import reproject, Resampling
from affine import Affine
from scipy.ndimage import distance_transform_edt
s = rasterio.open('Kavre_Rainfall_2024_Monthly.tif'); a = s.read().astype('float32'); m = a[5:9].sum(0)
idx = distance_transform_edt(~np.isfinite(m), return_distances=False, return_indices=True); m = m[tuple(idx)]
mon = np.pad(m, 1, mode='edge'); t = s.transform; tp = Affine(t.a, t.b, t.c - t.a, t.d, t.e, t.f - t.e)
ref = rasterio.open('rasters/Elevation.tif'); el = ref.read(1)
out = np.full(ref.shape, np.nan, 'float32')
reproject(mon, out, src_transform=tp, src_crs=s.crs, dst_transform=ref.transform, dst_crs=ref.crs, resampling=Resampling.bilinear, dst_nodata=np.nan)
out[el == -9999] = np.nan; out = np.where(np.isfinite(out), out, -9999).astype('float32')
with rasterio.open('rasters/rainfall.tif', 'w', **{**ref.profile, 'dtype': 'float32', 'nodata': -9999.0, 'compress': 'deflate'}) as d:
    d.write(out, 1)
