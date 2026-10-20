# Kavrepalanchowk Landslide Susceptibility Datasets

This directory contains the spatial modeling datasets, extracted raster values, and diagnostic tables for the Kavrepalanchowk District landslide susceptibility assessment (Nepal Middle Hills).

---

## 1. Datasets Summary

| File | Rows | Coordinates | Description |
|:---|:---:|:---:|:---|
| `samples_proj.csv` | 1,200 | Projected (UTM 45N: `E`, `N`) | **Primary Modeling Dataset**: 600 landslide points (September 2024 rainstorm event) + 600 non-landslide points with all 21 terrain, soil, rainfall, spectral, and proximity predictors. |
| `samples_raster.csv` | 1,200 | Projected (UTM 45N: `E`, `N`) | Point dataset re-sampled directly from the aligned 30-m GeoTIFF raster stack. |
| `sample_vs_raster.csv` | 20 | Summary Stats | Distribution comparison between sample table values and district raster values (min, max, mean, median, z-scores, percent out of bounds). |
| `raster_diag.csv` | 21 | Summary Stats | Diagnostic summary of the 21 predictor rasters across Kavrepalanchowk. |
| `out_extract_vs_table.csv` | 21 | Comparison Metrics | Point-wise comparison metrics (Pearson $r$, median absolute difference, ratio, categorical agreement). |
| `sample_test_points.csv` | 120 | WGS84 (`POINT_X`, `POINT_Y`) | Legacy synthetic unit test dataset. |

---

## 2. Coordinate Reference System (CRS)

- **Geographic**: WGS 84 (`Lat`, `Long`), EPSG:4326
- **Projected**: WGS 84 / UTM Zone 45N (`E`, `N`), **EPSG:32645**
- **Extent**: Kavrepalanchowk District, Central Nepal Middle Hills:
  - Easting ($E$): 342,142 m to 389,148 m
  - Northing ($N$): 3,025,526 m to 3,069,966 m

---

## 3. Predictor Schema and Attribute Definitions

| Predictor Family | Column Name in Data | Manuscript Name | Native Source & Resolution | Description |
|:---|:---|:---|:---|:---|
| **Target Variable** | `Landslide` | Landslide | Sentinel-2 Pre/Post Event (2024) | Binary classification label: `1` = Landslide failure, `0` = Non-landslide. |
| **Terrain** | `Elevation` | Elevation | SRTM 1-arc-second (~30 m) | Ground elevation above sea level (meters). |
| **Terrain** | `Slope` | Slope | SRTM DEM derivative (~30 m) | Ground slope inclination (degrees). |
| **Terrain** | `Aspect` | Aspect / Northness, Eastness | SRTM DEM derivative (~30 m) | Converted to $\text{Northness} = \cos(\text{Aspect})$ and $\text{Eastness} = \sin(\text{Aspect})$ (flat terrain set to 0.0). |
| **Terrain** | `TPI(Topographic Position Index)` | TPI | SRTM DEM derivative (~30 m) | Topographic Position Index measuring relative elevation vs neighbors. |
| **Terrain** | `Roughness` | Roughness | SRTM DEM derivative (~30 m) | Difference between maximum and minimum elevation in 3x3 window. |
| **Terrain** | `TRI(Topographic Ruggedness Index)` | TRI | SRTM DEM derivative (~30 m) | Mean of absolute elevation differences between focal cell and 8 neighbors. |
| **Terrain** | `Curvature` | Curvature | SRTM DEM derivative (~30 m) | Total surface curvature ($m^{-1} \times 100$). Note: removed in Model T due to redundancy with TPI. |
| **Hydrology** | `Flow Accumulation` | FlowAcc | HydroSHEDS / SRTM DEM | Upslope accumulated contributing area ($km^2$). |
| **Hydrology** | `TWI` | TWI | SRTM DEM derivative | Topographic Wetness Index: $\ln(a / \tan \beta)$. |
| **Soil Texture** | `Organic` | Organic | Survey Department of Nepal | Soil organic carbon content (%). |
| **Soil Texture** | `Sand` | Sand | Survey Department of Nepal | Sand particle fraction (%). |
| **Soil Texture** | `Slit` | Silt | Survey Department of Nepal | Silt particle fraction (%). |
| **Soil Texture** | `Clay` | Clay | Survey Department of Nepal | Clay particle fraction (%). |
| **Soil Class** | `Parent_soil` | ParentSoil | Survey Department of Nepal | Categorical soil class (`1` to `5`; `-9999` for unmapped/missing). |
| **Rainfall** | `Monsoon_Cumulative_Rainfall` | Rainfall | CHIRPS v2.0 (0.05° resampled) | Total cumulative rainfall from June–September 2024 rainstorm (mm). |
| **Spectral (Leakage)** | `MBI` | MBI | Sentinel-2 (Sept 2024) | Modified Bare Soil Index (post-event; dropped in Model T to eliminate target leakage). |
| **Spectral (Leakage)** | `MNDWI` | MNDWI | Sentinel-2 (Sept 2024) | Modified Normalized Difference Water Index (post-event; dropped in Model T). |
| **Spectral (Leakage)** | `EVI` | EVI | Sentinel-2 (Sept 2024) | Enhanced Vegetation Index (post-event; dropped in Model T). |
| **Proximity** | `dist_building_m` | DistBuilding | Google Open Buildings v3 | Euclidean distance to nearest building footprint (meters). |
| **Proximity** | `dist_river_m` | DistRiver | Survey Department Hydrography | Euclidean distance to nearest river / drainage stream (meters). |
| **Proximity** | `dist_road_m` | DistRoad | OpenStreetMap (OSM) | Euclidean distance to nearest road centerline (meters). |

---

## 4. Sampling Design & Quality Checks

1. **Class Balance**: 600 landslide points and 600 non-landslide points (1:1 balanced design).
2. **Nearest-Neighbor Spacing**:
   - Landslide-to-nearest non-landslide median distance: **1,314 m** (minimum 278 m).
   - Within-class landslide nearest-neighbor median: **588.6 m** (only 3 pairs < 60 m, candidate multi-point scars).
   - Within-class non-landslide nearest-neighbor median: **629.9 m**.
3. **No Duplicate Rows**: 0 duplicate coordinates, 0 duplicate predictor vectors.
4. **Leakage Protection**: Post-event spectral indices (`MBI`, `EVI`, `MNDWI`) reflect the barren ground created by the landslide scars themselves rather than predisposing geo-environmental conditions; Model T strictly drops these to prevent target leakage.
