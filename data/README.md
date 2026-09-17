# Dataset Specification and Attribute Dictionary

This directory contains modeling datasets used for Landslide Susceptibility Mapping (LSM) within the Kentucky Water Resources Research Institute (KWRI) spatial study area.

---

## 1. Demo Dataset (`sample_test_points.csv`)

A synthetic, 120-row demo dataset is included in this repository to facilitate immediate dry-run testing, CI/CD pipeline validation, and code exploration without requiring access to proprietary GIS geodatabases.

---

## 2. Required Schema and Attribute Definitions

Every dataset passed into the pipeline must be a CSV file with the following required columns:

| Column Name | Type | Description | Unit / Encoding |
|:---|:---:|:---|:---|
| `Labels` | `int` | **Target Variable**: Binary landslide occurrence label. | `0` = Non-Landslide, `1` = Landslide |
| `POINT_X` | `float` | Projected Easting / Longitude coordinate. | Decimal degrees or UTM Easting |
| `POINT_Y` | `float` | Projected Northing / Latitude coordinate. | Decimal degrees or UTM Northing |
| `Curvature` | `float` | Total topographic surface curvature. | $m^{-1} \times 100$ |
| `TWI` | `float` | Topographic Wetness Index ($\ln(a / \tan \beta)$). | Dimensionless |
| `Dist_Road_KYTC_1` | `float` | Euclidean proximity to KYTC transportation infrastructure. | Meters |
| `Plan_Curv` | `float` | Plan (horizontal) curvature perpendicular to slope direction. | $m^{-1} \times 100$ |
| `Profile_Curv` | `float` | Profile (vertical) curvature in the direction of maximum slope. | $m^{-1} \times 100$ |
| `Distan_to_Rivers` | `float` | Proximity to river and drainage networks. | Meters |
| `Slope_KWRI` | `float` | Ground surface slope steepness. | Degrees |
| `Aspect_KWRI` | `float` | Compass direction that a slope face faces. | Degrees ($0^\circ - 360^\circ$) |
| `Rain_Mean_KWRI` | `float` | Long-term mean annual precipitation. | Millimeters (mm) |
| `NDVI_Mean_KWRI` | `float` | Mean Normalized Difference Vegetation Index. | Dimensionless ($-1.0$ to $+1.0$) |
| `Landcover_KWRI` | `int` / `float` | Categorical NLCD (National Land Cover Database) code. | See NLCD lookup table below |

---

## 3. National Land Cover Database (NLCD) Classification Lookup

The categorical predictor `Landcover_KWRI` follows standard USGS/MRLC NLCD numeric codes:

| NLCD Code | Land Cover Category |
|:---:|:---|
| `-9999` | Missing / Unknown Land Cover Code |
| `0` | NoData / Unclassified |
| `11` | Open Water |
| `12` | Perennial Ice/Snow |
| `21` | Developed, Open Space |
| `22` | Developed, Low Intensity |
| `23` | Developed, Medium Intensity |
| `24` | Developed, High Intensity |
| `31` | Barren Land |
| `41` | Deciduous Forest |
| `42` | Evergreen Forest |
| `43` | Mixed Forest |
| `51` | Dwarf Scrub |
| `52` | Shrub/Scrub |
| `71` | Grassland/Herbaceous |
| `72` | Sedge/Herbaceous |
| `73` | Lichens |
| `74` | Moss |
| `81` | Pasture/Hay |
| `82` | Cultivated Crops |
| `90` | Woody Wetlands |
| `95` | Emergent Herbaceous Wetlands |

---

## 4. Preprocessing & Quality Notes

- **Infinite Values**: Any positive (`+inf`) or negative (`-inf`) infinity values caused by GIS raster division (e.g. curvature calculation over flat terrain) are automatically trapped and replaced by fold-specific medians during preprocessing.
- **Missing Values**: Numeric attributes are imputed using the **median of the active training fold** to prevent data leakage. Categorical land cover missing codes are mapped to `-9999` and encoded with dedicated one-hot columns.
- **Coordinates**: Coordinates (`POINT_X`, `POINT_Y`) are used strictly for spatial cluster blocking (K-Means) and are never used as direct features in the tree splits, avoiding geographic coordinate memorization.
