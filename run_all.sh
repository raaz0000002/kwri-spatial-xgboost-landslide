#!/usr/bin/env bash
# Inputs: samples.csv (ML Data - Sheet1.csv), rasters/ (21 GeoTIFFs), vectors/ (roads, rivers, buildings, kavre boundary),
#         Kavre_Rainfall_2024_Monthly.tif (CHIRPS 2024 monthly, 12 bands)
set -e
python audit.py                                    # project points, sample-vs-raster statistics
python make_distance_rasters.py                    # Euclidean distance rasters + boundary (EPSG:32645)
python make_rainfall_raster.py                     # CHIRPS Jun-Sep 2024 sum on 30 m grid
python s6_extract_from_rasters.py rasters          # sample every predictor from the raster stack -> samples_raster.csv
export LS_DATA=samples_raster.csv
python s1_spatial.py                               # correlogram, block-size sensitivity, coordinates-only baseline
LS_EXCLUDE=MBI,EVI,MNDWI,Curvature LS_TAG=_rx2 python s1_spatial.py
LS_DATA=samples_proj.csv python s2_nested.py 3 20  # Model S (leakage test; table values)
LS_DATA=samples_proj.csv python s4_summarize.py
export LS_EXCLUDE=MBI,EVI,MNDWI,Curvature LS_TAG=_rx2   # Model T (primary; curvature = -TPI/100 removed)
python s2_nested.py 3 20 && python s4_summarize.py && python s5_final_fit.py
python s7_map.py rasters                           # maps, AOA, NoData audit, class-wise validation
python s8_buffer_aoa.py
python s10_pseudoabsence.py                        # random non-landslide sensitivity (5 realizations)
unset LS_EXCLUDE LS_TAG
python fig_study_area.py; python figs1.py; python figs2.py; python fig_workflow.py; python s9_map_figs.py
python mk_supp.py; python build_ms.py; python build_response.py; python build_cover.py
