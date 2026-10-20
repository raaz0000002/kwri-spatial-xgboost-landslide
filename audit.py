import pandas as pd, numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree
d=pd.read_csv('samples.csv')
t=Transformer.from_crs(4326,32645,always_xy=True)
d['E'],d['N']=t.transform(d.Long.values,d.Lat.values)
print('E range',d.E.min(),d.E.max(),'N',d.N.min(),d.N.max())
xy=d[['E','N']].values; tr=cKDTree(xy)
dd,ii=tr.query(xy,k=2); nn=dd[:,1]
print('exact dup coords',(nn==0).sum())
for c in [0,1]:
    m=d.Landslide==c; t2=cKDTree(xy[m]); q,_=t2.query(xy[m],k=2)
    print('class',c,'NN within-class median %.1f m, p10 %.1f, <30m %d, <100m %d'%(np.median(q[:,1]),np.percentile(q[:,1],10),(q[:,1]<30).sum(),(q[:,1]<100).sum()))
# cross-class NN
L=xy[d.Landslide==1]; N=xy[d.Landslide==0]
q,_=cKDTree(N).query(L); print('landslide->nearest nonls median %.0f m min %.1f p10 %.0f'%(np.median(q),q.min(),np.percentile(q,10)))
# cluster landslide points within 60 m -> candidate same-landslide
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
for thr in [30,60,100]:
    pairs=cKDTree(L).query_pairs(thr,output_type='ndarray')
    g=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(L),len(L)))
    n,lab=connected_components(g,directed=False); print('LS clusters at',thr,'m:',n,'max size',np.bincount(lab).max())
# duplicate predictor rows
feat=[c for c in d.columns if c not in ['ID','Landslide','Lat','Long','Year','E','N']]
print('dup feature rows',d.duplicated(subset=feat).sum())
print(pd.crosstab(d.Parent_soil,d.Landslide)); print(pd.crosstab(d.Parent_soil_class,d.Parent_soil).to_string())
# compare to raster diag
r=pd.read_csv('raster_diag.csv').set_index('Feature')
mp={'Elevation':'Elevation','Slope':'Slope','Aspect':'Aspect','tpi':'TPI(Topographic Position Index)','Roughness':'Roughness','TRI(Topographic Ruggedness Index)':'TRI(Topographic Ruggedness Index)','Flow Accumulation':'Flow Accumulation','Curvature':'Curvature','TWI':'TWI','Organic':'Organic','Sand':'Sand','Silt':'Slit','Clay':'Clay','rainfall':'Monsoon_Cumulative_Rainfall','MBI':'MBI','MNDWI':'MNDWI','EVI':'EVI','dist_building':'dist_building_m','dist_river_m':'dist_river_m','dist_road_m':'dist_road_m'}
rows=[]
for rk,sk in mp.items():
    s=d[sk]; rr=r.loc[rk]
    rows.append([rk,s.min(),s.max(),s.mean(),s.median(),rr.Min,rr.Max,rr.Mean,(rr.Mean-s.mean())/s.std(), ((s<rr.Min)|(s>rr.Max)).mean()*100])
o=pd.DataFrame(rows,columns=['feat','s_min','s_max','s_mean','s_med','r_min','r_max','r_mean','z_rmean','pct_samples_outside_raster'])
pd.set_option('display.width',250); print(o.round(3).to_string())
o.to_csv('sample_vs_raster.csv',index=False)
d.to_csv('samples_proj.csv',index=False)
