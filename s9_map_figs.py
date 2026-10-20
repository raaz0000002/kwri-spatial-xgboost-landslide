"""Publication maps of the deployed Model T products (scale bar, north arrow, UTM coordinates)."""
import numpy as np, rasterio, geopandas as gpd, pandas as pd, matplotlib.pyplot as plt, json
from matplotlib.colors import ListedColormap, BoundaryNorm
from figstyle import *

b = gpd.read_file('../rasters/kavre_boundary.gpkg')


def rd(name, nodata=None):
    with rasterio.open(f'maps/{name}.tif') as s:
        a = s.read(1).astype(float); ext = [s.bounds.left, s.bounds.right, s.bounds.bottom, s.bounds.top]
        nd = s.nodata if nodata is None else nodata
    if nd is not None and np.isfinite(nd):
        a[a == nd] = np.nan
    return a, ext


def deco(ax, ext, lab):
    b.boundary.plot(ax=ax, color='k', lw=0.8)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect('equal')
    ax.ticklabel_format(style='plain'); ax.tick_params(labelsize=6)
    ax.set_xticks([350000, 370000]); ax.set_yticks([3030000, 3050000, 3070000])
    ax.set_xlabel('Easting (m)', fontsize=7); ax.set_ylabel('Northing (m)', fontsize=7)
    map_furniture(ax, ext, pad_frac=0.0)
    ax.set_title(lab, loc='left', fontsize=8.5)


summ = json.load(open('out_raster_summary.json'))
# ---- Figure: primary continuous score + quintiles
fig, axs = plt.subplots(1, 2, figsize=(6.9, 3.6))
a, ext = rd('ensemble_score_calibrated')
cm = plt.get_cmap('YlOrRd').copy(); cm.set_bad('white')
im = axs[0].imshow(a, extent=ext, cmap=cm, vmin=0, vmax=1)
deco(axs[0], ext, '(a) Calibrated ensemble score')
cb = fig.colorbar(im, ax=axs[0], shrink=0.7, pad=0.02); cb.set_label('Score (0–1)', fontsize=7); cb.ax.tick_params(labelsize=6)
q, _ = rd('susceptibility_quintiles', 255)
cols = ['#1a9850', '#91cf60', '#fee08b', '#fc8d59', '#d73027']
cmq = ListedColormap(cols); cmq.set_bad('white')
im2 = axs[1].imshow(q, extent=ext, cmap=cmq, norm=BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 5))
deco(axs[1], ext, '(b) Quintile classes')
cb2 = fig.colorbar(im2, ax=axs[1], shrink=0.7, pad=0.02, ticks=[1, 2, 3, 4, 5])
cb2.ax.set_yticklabels(['Very Low', 'Low', 'Moderate', 'High', 'Very High'], fontsize=6)
fig.tight_layout(); save(fig, 'Fig_map_susceptibility')
# ---- Figure: reliability layers
fig, axs = plt.subplots(1, 3, figsize=(6.9, 3.4))
d, ext = rd('intermodel_dispersion_sd')
cmd = plt.get_cmap('Purples').copy(); cmd.set_bad('white')
im = axs[0].imshow(d, extent=ext, cmap=cmd, vmin=0, vmax=np.nanpercentile(d, 99))
deco(axs[0], ext, '(a) Dispersion (SD)'); cb = fig.colorbar(im, ax=axs[0], orientation='horizontal', shrink=0.8, pad=0.18); cb.ax.tick_params(labelsize=6)
v, _ = rd('agreement_votes', 255)
cmv = ListedColormap(['#2166ac', '#92c5de', '#f7f7f7', '#f4a582', '#b2182b']); cmv.set_bad('white')
im = axs[1].imshow(v, extent=ext, cmap=cmv, norm=BoundaryNorm(np.arange(-0.5, 5.5), 5))
deco(axs[1], ext, '(b) Models with score ≥ 0.5'); cb = fig.colorbar(im, ax=axs[1], orientation='horizontal', shrink=0.8, pad=0.18, ticks=range(5)); cb.ax.tick_params(labelsize=6)
ao, _ = rd('aoa_inside', 255)
cma = ListedColormap(['#e34948', '#d9d8d3']); cma.set_bad('white')
im = axs[2].imshow(ao, extent=ext, cmap=cma, norm=BoundaryNorm([-0.5, 0.5, 1.5], 2))
deco(axs[2], ext, '(c) Area of applicability'); cb = fig.colorbar(im, ax=axs[2], orientation='horizontal', shrink=0.8, pad=0.18, ticks=[0, 1]); cb.ax.set_xticklabels(['outside', 'inside'], fontsize=6)
for ax in axs[1:]: ax.set_yticklabels([]); ax.set_ylabel('')
fig.tight_layout(); save(fig, 'Fig_map_reliability')
# ---- Figure: class-wise validation
CV = pd.read_csv('out_class_validation.csv')
fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.8))
lab = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
axs[0].bar(lab, CV.share_pct, color=cols, width=0.6, edgecolor='white')
axs[0].axhline(20, color='#52514e', ls=':', lw=0.8); axs[0].text(4.4, 21, 'random (20%)', fontsize=6.5, ha='right', color='#52514e')
for i, r in CV.iterrows():
    axs[0].text(i, r.share_pct + 1, f"{r.share_pct:.1f}%\n(×{r.density_ratio:.2f})", ha='center', fontsize=6.5)
axs[0].set_ylabel('Held-out landslides in class (%)'); axs[0].set_title('(a) Class-wise held-out capture', loc='left'); axs[0].tick_params(axis='x', labelsize=7)
axs[0].set_ylim(0, max(CV.share_pct) * 1.3)
cum = np.r_[0, CV.share_pct[::-1].cumsum().values]
axs[1].plot([0, 20, 40, 60, 80, 100], cum, marker='o', color=CAT[0]); axs[1].plot([0, 100], [0, 100], ls=':', color='#52514e', lw=0.8)
axs[1].set_xlabel('Share of district, highest score first (%)'); axs[1].set_ylabel('Cumulative held-out landslides (%)')
axs[1].set_title('(b) Success-rate curve (fold maps)', loc='left'); axs[1].grid(alpha=.6)
fig.tight_layout(); save(fig, 'Fig_class_validation')
m, ext = rd('nodata_mask', 255)
fig, ax = plt.subplots(figsize=(4.5, 4.2))
cmn = ListedColormap(['#e34948', '#d9d8d3']); cmn.set_bad('white')
ax.imshow(m, extent=ext, cmap=cmn, norm=BoundaryNorm([-0.5, 0.5, 1.5], 2), interpolation='nearest')
deco(ax, ext, 'NoData mask (red: missing predictor)')
save(fig, 'FigS2_nodata_mask')
print('maps done')
