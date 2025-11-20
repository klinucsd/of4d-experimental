# CH-GEE: Canopy Height Mapping with Google Earth Engine

A Python toolkit for generating wall-to-wall canopy height maps using GEDI LiDAR data and satellite imagery through Google Earth Engine.

## Overview

This repository provides tools to create continuous canopy height maps by training a Random Forest model on GEDI (Global Ecosystem Dynamics Investigation) LiDAR measurements and predicting height across landscapes using Sentinel-1, Sentinel-2, and topographic data.

## Requirements

```bash
pip install earthengine-api geemap rasterio numpy pandas scikit-learn requests
```

You must also authenticate with Google Earth Engine:
```python
import ee
ee.Authenticate()
ee.Initialize()
```

## Files

### 1. `gedi_l2a_download.py`

**Purpose:** Downloads GEDI L2A canopy height data for use as training labels.

**Key Features:**
- Queries the `LARSE/GEDI/GEDI02_A_002_MONTHLY` ImageCollection
- Samples rh98 (98th percentile height) values as point data
- Exports coordinates and height metrics to CSV
- Includes data quality verification

**Usage:**
```python
python gedi_l2a_download.py
```

**Configuration:**
- `bbox_medium`: Bounding box coordinates `[min_lon, min_lat, max_lon, max_lat]`
- `start_date` / `end_date`: Date range (GEDI available from April 2019)

**Output:** `gedi_downloads/GEDI_L2A_rh98_<dates>.csv`

---

### 2. `ch_gee_small_area.py`

**Purpose:** Complete local workflow for small areas (~100-500 MB downloads).

**Key Features:**
- Downloads Sentinel-2 optical imagery (B4, B8, B11, B12, NDVI)
- Downloads Sentinel-1 SAR data (VV, VH polarizations)
- Downloads SRTM topography (elevation, slope)
- Trains scikit-learn Random Forest regressor locally
- Generates wall-to-wall canopy height predictions

**Usage:**
```python
python ch_gee_small_area.py
```

**Configuration:**
```python
bbox_tiny = [-60.0, -3.5, -59.9, -3.4]    # ~120 km² (~50 MB)
bbox_small = [-60.0, -3.5, -59.8, -3.3]   # ~480 km² (~200 MB)
bbox_medium = [-60.0, -3.5, -59.75, -3.25] # ~750 km² (~300 MB)
```

Resolution is automatically selected based on area size (10m, 20m, or 30m).

**Output:** `canopy_height_<scale>m.tif`

---

### 3. `ch_gee_server_side.py`

**Purpose:** Server-side processing using geemap for larger areas.

**Key Features:**
- Performs all computation on Earth Engine servers
- Trains Random Forest using `ee.Classifier.smileRandomForest`
- Uses `geemap.download_ee_image()` for efficient tiled downloads
- Better suited for larger regions that exceed direct download limits

**Usage:**
```python
python ch_gee_server_side.py
```

**Configuration:**
```python
BOUNDING_BOX = [-60.0, -3.5, -59.8, -3.3]
START_DATE = '2019-04-01'
END_DATE = '2024-12-31'
TARGET_SCALE = 30  # meters
N_TREES = 100
MAX_SAMPLES = 5000
```

**Output:** `GEDI_rh98_ML_predicted.tif`

## Workflow

1. **Download GEDI training data:**
   ```bash
   python gedi_l2a_download.py
   ```

2. **Generate canopy height map (choose one):**
   
   For small areas (local processing):
   ```bash
   python ch_gee_small_area.py
   ```
   
   For larger areas (server-side processing):
   ```bash
   python ch_gee_server_side.py
   ```

## Output Visualization

The output GeoTIFF can be visualized using:
- **QGIS:** Layer → Add Raster Layer
- **Python:** `rasterio` + `matplotlib`
- Any GIS software supporting GeoTIFF

## Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | Random Forest Regressor |
| Trees | 100 |
| Target Variable | rh98 (canopy height in meters) |
| Predictors | Sentinel-2 bands, NDVI, Sentinel-1 VV/VH, elevation, slope |
| Output Range | 0-100 meters |

## Troubleshooting

- **No GEDI data found:** Expand bounding box or date range
- **Download fails:** Reduce area size or increase `TARGET_SCALE`
- **Few training points:** GEDI coverage is sparse; use larger regions
- **Memory errors:** Use server-side script or reduce `MAX_SAMPLES`

## References

- GEDI Mission: https://gedi.umd.edu/
- Google Earth Engine: https://earthengine.google.com/
