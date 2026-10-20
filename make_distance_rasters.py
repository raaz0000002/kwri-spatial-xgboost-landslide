import geopandas as gpd, rasterio, numpy as np
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
ref=rasterio.open('rasters/Elevation.tif'); H,W=ref.shape
src={'dist_road_m':'vectors/Kavre_roads.shp','dist_river_m':'vectors/kavre_rivers.shp','dist_building':'vectors/Kavre_Buildings.shp'}
prof={**ref.profile,'dtype':'float32','nodata':-9999.0,'compress':'deflate','count':1}
elev=ref.read(1); valid=elev!=-9999
for name,f in src.items():
    g=gpd.read_file(f).to_crs(ref.crs)
    m=rasterize(((geom,1) for geom in g.geometry if geom is not None),out_shape=(H,W),transform=ref.transform,all_touched=True,dtype='uint8')
    d=(distance_transform_edt(m==0)*30.0).astype('float32')
    d[~valid]=-9999
    with rasterio.open(f'rasters/{name}.tif','w',**prof) as dst: dst.write(d,1); dst.set_band_description(1,name)
    print(name, len(g), m.sum(), float(d[valid].min()), float(d[valid].max()), float(d[valid].mean()))
b=gpd.read_file('vectors/kavre.shp').to_crs(32645); b.to_file('rasters/kavre_boundary.gpkg'); print(b.area.sum()/1e6)
