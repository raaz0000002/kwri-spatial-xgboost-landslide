import numpy as np, geopandas as gpd, rasterio, matplotlib.pyplot as plt, pandas as pd
from matplotlib.colors import LightSource
from figstyle import *
V='../vec/'
ref=rasterio.open('../rasters/Elevation.tif'); el=ref.read(1).astype(float); el[el==-9999]=np.nan
b=gpd.read_file('../rasters/kavre_boundary.gpkg')
riv=gpd.read_file(V+'88f449ac-rivers_and_roads/kavre_rivers.shp').to_crs(32645)
rd=gpd.read_file(V+'88f449ac-rivers_and_roads/Kavre_roads.shp').to_crs(32645)
rdm=rd[rd.highway.isin(['motorway','trunk','primary','secondary','tertiary'])]
S=pd.read_csv('samples_proj.csv')
ne=gpd.read_file('ne/naturalearth_lowres.shp')
fig=plt.figure(figsize=(6.9,5.6))
ax=fig.add_axes([0.0,0.0,0.74,1.0])
ext=[ref.bounds.left,ref.bounds.right,ref.bounds.bottom,ref.bounds.top]
ls=LightSource(azdeg=315,altdeg=45); z=np.nan_to_num(el,nan=np.nanmin(el))
hs=ls.hillshade(z,vert_exag=2,dx=30,dy=30)
hs=np.where(np.isfinite(el),hs,np.nan); cm=plt.get_cmap('gray').copy(); cm.set_bad('white'); ax.imshow(hs,cmap=cm,extent=ext,alpha=1.0)
im=ax.imshow(el,cmap='terrain',extent=ext,alpha=0.55,vmin=200,vmax=3000)
riv.plot(ax=ax,color='#2a78d6',lw=0.25,alpha=0.6)
rdm.plot(ax=ax,color='#4a3aa7',lw=0.6)
b.boundary.plot(ax=ax,color='k',lw=1.2)
ax.scatter(S.E[S.Landslide==1],S.N[S.Landslide==1],s=6,marker='^',c='#e34948',lw=0,label='Landslide (n = 600)')
ax.scatter(S.E[S.Landslide==0],S.N[S.Landslide==0],s=5,marker='o',facecolor='none',edgecolor='#0b0b0b',lw=0.5,label='Non-landslide (n = 600)')
from matplotlib.lines import Line2D
h=ax.get_legend_handles_labels()[0]+[Line2D([],[],color='#4a3aa7',lw=1,label='Main roads (OSM)'),Line2D([],[],color='#2a78d6',lw=1,label='Rivers'),Line2D([],[],color='k',lw=1.2,label='District boundary')]
fig.legend(handles=h,loc='upper left',bbox_to_anchor=(0.745,0.57),fontsize=6.5,frameon=True,framealpha=0.95)
ax.set_xlim(ext[0],ext[1]); ax.set_ylim(ext[2],ext[3]); ax.ticklabel_format(style='plain'); ax.tick_params(labelsize=7)
ax.set_xlabel('Easting (m), WGS 84 / UTM zone 45N',fontsize=8); ax.set_ylabel('Northing (m)',fontsize=8)
map_furniture(ax,ext,fs=7)
ax.set_title('(b) Kavrepalanchowk District',loc='left',fontsize=9)
cax=fig.add_axes([0.77,0.06,0.02,0.27]); cb=fig.colorbar(im,cax=cax); cb.set_label('Elevation (m)',fontsize=8); cb.ax.tick_params(labelsize=7)
ax2=fig.add_axes([0.73,0.62,0.27,0.33])
reg=ne[ne.name.isin(['Nepal','India','China','Bhutan','Bangladesh'])]
reg.plot(ax=ax2,color='#f3f2ef',edgecolor='#9a9893',lw=0.4); ne[ne.name=='Nepal'].plot(ax=ax2,color='#dbe8f8',edgecolor='#2a78d6',lw=0.8)
b.to_crs(4326).plot(ax=ax2,color='#e34948')
ax2.set_xlim(79.5,88.8); ax2.set_ylim(26,31); ax2.set_aspect('equal'); ax2.tick_params(labelsize=6); ax2.set_xlabel(''); ax2.set_ylabel(''); ax2.set_xticks([80,84,88]); ax2.set_xticklabels(['80°E','84°E','88°E']); ax2.set_yticks([26,28,30]); ax2.set_yticklabels(['26°N','28°N','30°N'])
ax2.set_title('(a) Nepal',loc='left',fontsize=8); ax2.text(84.2,28.3,'NEPAL',fontsize=6,color='#2a78d6'); ax2.text(80.2,26.4,'INDIA',fontsize=6,color='#52514e'); ax2.text(84.5,30.3,'CHINA',fontsize=6,color='#52514e')
save(fig,'Fig1_study_area')
