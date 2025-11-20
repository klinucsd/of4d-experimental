"""
Requirements: pip install earthengine-api rasterio numpy pandas scikit-learn requests
"""

import ee
import requests
import rasterio
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import os
import json

try:
    ee.Initialize()
except:
    ee.Authenticate(auth_mode='notebook')
    ee.Initialize()


def estimate_download_size(bbox, scale):
    """Estimate download size in MB"""
    width_deg = bbox[2] - bbox[0]
    height_deg = bbox[3] - bbox[1]
    
    # Approximate pixels (at equator, ~111km per degree)
    width_m = width_deg * 111000
    height_m = height_deg * 111000
    
    pixels = (width_m / scale) * (height_m / scale)
    
    # Estimate: 4 bytes per pixel per band
    # S2: 5 bands, S1: 2 bands, Topo: 2 bands = 9 bands total
    size_mb = (pixels * 9 * 4) / (1024 * 1024)
    
    return size_mb


def download_sentinel2_simple(bbox, start_date, end_date, scale=30, output_dir='data'):
    """
    Download Sentinel-2 with key bands only
    Using 30m resolution for smaller file size
    """
    os.makedirs(output_dir, exist_ok=True)
    roi = ee.Geometry.Rectangle(bbox)
    
    print("Downloading Sentinel-2...")
    
    # Select only most important bands for height prediction
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .select(['B4', 'B8', 'B11', 'B12'])  # Red, NIR, SWIR1, SWIR2
    
    count = s2.size().getInfo()
    print(f"  Found {count} images")
    
    if count == 0:
        raise Exception("No Sentinel-2 images found! Try longer date range.")
    
    s2_median = s2.median().clip(roi)
    
    # Calculate NDVI
    ndvi = s2_median.normalizedDifference(['B8', 'B4']).rename('NDVI')
    
    # Combine
    features = s2_median.addBands(ndvi).toFloat()  # Convert to Float32
    
    print(f"  Downloading at {scale}m resolution...")
    
    url = features.getDownloadURL({
        'scale': scale,
        'crs': 'EPSG:4326',
        'region': roi.getInfo()['coordinates'],
        'format': 'GEO_TIFF'
    })
    
    filepath = os.path.join(output_dir, f'sentinel2_{scale}m.tif')
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\r  Progress: {percent:.1f}%", end='', flush=True)
    
    print(f"\n  ✓ Downloaded: {os.path.getsize(filepath) / (1024**2):.1f} MB")
    return filepath


def download_sentinel1_simple(bbox, start_date, end_date, scale=30, output_dir='data'):
    """Download Sentinel-1 SAR"""
    os.makedirs(output_dir, exist_ok=True)
    roi = ee.Geometry.Rectangle(bbox)
    
    print("\nDownloading Sentinel-1...")
    
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .select(['VV', 'VH'])
    
    count = s1.size().getInfo()
    print(f"  Found {count} images")
    
    if count == 0:
        print("  ⚠ No Sentinel-1 data, continuing without SAR...")
        return None
    
    s1_median = s1.median().clip(roi)
    
    # Clip extreme SAR values (prevent inf/nan issues)
    s1_clipped = s1_median.clamp(-50, 10).toFloat()  # Typical SAR dB range
    
    print(f"  Downloading at {scale}m resolution...")
    
    url = s1_clipped.getDownloadURL({
        'scale': scale,
        'crs': 'EPSG:4326',
        'region': roi.getInfo()['coordinates'],
        'format': 'GEO_TIFF'
    })
    
    filepath = os.path.join(output_dir, f'sentinel1_{scale}m.tif')
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\r  Progress: {percent:.1f}%", end='', flush=True)
    
    print(f"\n  ✓ Downloaded: {os.path.getsize(filepath) / (1024**2):.1f} MB")
    return filepath


def download_topography_simple(bbox, scale=30, output_dir='data'):
    """Download SRTM DEM"""
    os.makedirs(output_dir, exist_ok=True)
    roi = ee.Geometry.Rectangle(bbox)
    
    print("\nDownloading topography...")
    
    dem = ee.Image('USGS/SRTMGL1_003').clip(roi)
    slope = ee.Terrain.slope(dem)
    
    topo = dem.addBands(slope).rename(['elevation', 'slope']).toFloat()
    
    print(f"  Downloading at {scale}m resolution...")
    
    url = topo.getDownloadURL({
        'scale': scale,
        'crs': 'EPSG:4326',
        'region': roi.getInfo()['coordinates'],
        'format': 'GEO_TIFF'
    })
    
    filepath = os.path.join(output_dir, f'topography_{scale}m.tif')
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\r  Progress: {percent:.1f}%", end='', flush=True)
    
    print(f"\n  ✓ Downloaded: {os.path.getsize(filepath) / (1024**2):.1f} MB")
    return filepath


def load_and_fix_gedi_csv(gedi_csv):
    """Load GEDI CSV and ensure it has latitude/longitude columns"""
    df = pd.read_csv(gedi_csv)
    
    # Check for coordinates
    if 'latitude' in df.columns and 'longitude' in df.columns:
        return df
    
    # Try to extract from .geo column (GeoJSON)
    if '.geo' in df.columns:
        print("  Extracting coordinates from .geo column...")
        def parse_geojson(geo_str):
            try:
                geo = json.loads(geo_str)
                coords = geo['coordinates']
                return pd.Series({'longitude': coords[0], 'latitude': coords[1]})
            except:
                return pd.Series({'longitude': None, 'latitude': None})
        
        coords = df['.geo'].apply(parse_geojson)
        df['longitude'] = coords['longitude']
        df['latitude'] = coords['latitude']
        df = df.drop(columns=['.geo'])
        
        # Save fixed version
        df.to_csv(gedi_csv, index=False)
        print(f"  ✓ Added lat/lon columns, saved updated CSV")
        return df
    
    raise Exception("GEDI CSV missing latitude/longitude columns and no .geo column found!")


def extract_features_at_gedi_points(gedi_csv, s2_tif, s1_tif, topo_tif):
    """
    Extract predictor values at GEDI locations
    """
    print("\n" + "="*60)
    print("Extracting Features at GEDI Points")
    print("="*60 + "\n")
    
    # Load GEDI with coordinate fix
    gedi = load_and_fix_gedi_csv(gedi_csv)
    print(f"Loaded {len(gedi)} GEDI points")
    
    if 'rh98' not in gedi.columns:
        raise Exception("GEDI data must have 'rh98' column (canopy height)")
    
    # Filter for valid coordinates and rh98 > 0
    valid_mask = (gedi['rh98'] > 0) & gedi['latitude'].notna() & gedi['longitude'].notna()
    gedi = gedi[valid_mask]
    print(f"Valid points (rh98 > 0 and has coordinates): {len(gedi)}")
    
    if len(gedi) < 10:
        raise Exception("Not enough valid GEDI points! Need at least 10.")
    
    coords = list(zip(gedi['longitude'], gedi['latitude']))
    
    # Extract S2 values
    print("Extracting Sentinel-2 values...")
    with rasterio.open(s2_tif) as src:
        s2_data = np.array([x for x in src.sample(coords)])
        s2_bands = [f'S2_B{i+1}' for i in range(s2_data.shape[1])]
    
    # Extract S1 values
    if s1_tif and os.path.exists(s1_tif):
        print("Extracting Sentinel-1 values...")
        with rasterio.open(s1_tif) as src:
            s1_data = np.array([x for x in src.sample(coords)])
            s1_bands = [f'S1_B{i+1}' for i in range(s1_data.shape[1])]
    else:
        print("No Sentinel-1 data, skipping...")
        s1_data = np.array([]).reshape(len(coords), 0)
        s1_bands = []
    
    # Extract topography
    print("Extracting topography values...")
    with rasterio.open(topo_tif) as src:
        topo_data = np.array([x for x in src.sample(coords)])
        topo_bands = [f'Topo_B{i+1}' for i in range(topo_data.shape[1])]
    
    # Combine all features
    if s1_data.shape[1] > 0:
        X = np.hstack([s2_data, s1_data, topo_data])
        feature_names = s2_bands + s1_bands + topo_bands
    else:
        X = np.hstack([s2_data, topo_data])
        feature_names = s2_bands + topo_bands
    
    y = gedi['rh98'].values
    
    # Remove NaN
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"\nTraining samples after removing NaN: {len(X)}")
    print(f"Features: {len(feature_names)}")
    print(f"  {', '.join(feature_names[:5])}...")
    
    return X, y, feature_names


def train_and_evaluate(X, y, feature_names):
    """Train Random Forest model"""
    print("\n" + "="*60)
    print("Training Random Forest Model")
    print("="*60 + "\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train
    print("Training...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"Results:")
    print(f"  Training R²: {r2_train:.3f}")
    print(f"  Testing R²: {r2_test:.3f}")
    print(f"  Testing RMSE: {rmse_test:.2f} m")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 5 Important Features:")
    for _, row in importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    return model


def predict_height_map(model, s2_tif, s1_tif, topo_tif, output_tif='canopy_height.tif'):
    """Create wall-to-wall height map"""
    print("\n" + "="*60)
    print("Generating Canopy Height Map")
    print("="*60 + "\n")
    
    # Read all rasters
    with rasterio.open(s2_tif) as s2_src:
        s2_data = s2_src.read()
        profile = s2_src.profile
        
        if s1_tif and os.path.exists(s1_tif):
            with rasterio.open(s1_tif) as s1_src:
                # Resample S1 to match S2
                from rasterio.warp import reproject, Resampling
                s1_data = np.zeros((s1_src.count, s2_data.shape[1], s2_data.shape[2]))
                for i in range(s1_src.count):
                    reproject(
                        source=s1_src.read(i+1),
                        destination=s1_data[i],
                        src_transform=s1_src.transform,
                        src_crs=s1_src.crs,
                        dst_transform=s2_src.transform,
                        dst_crs=s2_src.crs,
                        resampling=Resampling.bilinear
                    )
        else:
            s1_data = np.array([]).reshape(0, s2_data.shape[1], s2_data.shape[2])
        
        with rasterio.open(topo_tif) as topo_src:
            # Resample topo to match S2
            topo_data = np.zeros((topo_src.count, s2_data.shape[1], s2_data.shape[2]))
            for i in range(topo_src.count):
                reproject(
                    source=topo_src.read(i+1),
                    destination=topo_data[i],
                    src_transform=topo_src.transform,
                    src_crs=topo_src.crs,
                    dst_transform=s2_src.transform,
                    dst_crs=s2_src.crs,
                    resampling=Resampling.bilinear
                )
    
    # Stack all features
    if s1_data.shape[0] > 0:
        all_data = np.vstack([s2_data, s1_data, topo_data])
    else:
        all_data = np.vstack([s2_data, topo_data])
    
    # Reshape for prediction
    n_features, height, width = all_data.shape
    data_2d = all_data.reshape(n_features, -1).T
    
    # Predict
    print("Predicting heights...")
    
    # Remove infinite and extreme values
    finite_mask = np.all(np.isfinite(data_2d), axis=1)
    
    # Also check for extreme values that might cause issues
    reasonable_mask = np.all(np.abs(data_2d) < 1e10, axis=1)
    valid_mask = finite_mask & reasonable_mask
    
    print(f"  Valid pixels: {valid_mask.sum()}/{len(valid_mask)} ({valid_mask.sum()/len(valid_mask)*100:.1f}%)")
    
    predictions = np.full(data_2d.shape[0], np.nan)
    if valid_mask.sum() > 0:
        predictions[valid_mask] = model.predict(data_2d[valid_mask])
    else:
        print("  ⚠ Warning: No valid pixels to predict!")
    
    # Clip predictions to reasonable range
    predictions = np.clip(predictions, 0, 100)  # Canopy height 0-100m
    
    # Reshape
    height_map = predictions.reshape(height, width)
    
    # Save
    profile.update(count=1, dtype='float32')
    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(height_map.astype('float32'), 1)
    
    print(f"✓ Saved to {output_tif}")
    print(f"  Mean height: {np.nanmean(height_map):.1f} m")
    print(f"  Max height: {np.nanmax(height_map):.1f} m")
    print(f"  File size: {os.path.getsize(output_tif) / (1024**2):.1f} MB")


if __name__ == "__main__":
    
    # SMALL TEST AREAS (choose one)
    bbox_tiny = [-60.0, -3.5, -59.9, -3.4]      # 0.1° = ~120 km² = ~50 MB total
    bbox_small = [-60.0, -3.5, -59.8, -3.3]     # 0.2° = ~480 km² = ~200 MB total  
    bbox_medium = [-60.0, -3.5, -59.75, -3.25]  # 0.25° = ~750 km² = ~300 MB total
    
    # Choose bbox
    bbox = bbox_small  # Start here!
    scale = 30  # 30m resolution (change to 10 for finer, but larger files)
    
    start_date = '2022-01-01'
    end_date = '2023-12-31'
    
    # Estimate size
    est_size = estimate_download_size(bbox, scale)
    area_km2 = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1]) * 12100
    
    print("="*60)
    print("CH-GEE Small Area Test")
    print("="*60)
    print(f"\nArea: {area_km2:.0f} km²")
    print(f"Resolution: {scale}m")
    print(f"Estimated total download: ~{est_size:.0f} MB")
    print(f"Bbox: {bbox}\n")
    
    if est_size > 1000:
        print("⚠ WARNING: Downloads > 1 GB!")
        print("Consider using smaller bbox or coarser resolution")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            exit()
    
    # Download data
    s2_path = download_sentinel2_simple(bbox, start_date, end_date, scale)
    s1_path = download_sentinel1_simple(bbox, start_date, end_date, scale)
    topo_path = download_topography_simple(bbox, scale)
    
    # Check for GEDI data - try both files
    gedi_files = [
        'gedi_downloads/GEDI_L2A_rh98_2019-04-01_2024-12-31_clean.csv',
        'gedi_downloads/GEDI_L2A_rh98_2019-04-01_2024-12-31.csv'
    ]
    
    gedi_csv = None
    for f in gedi_files:
        if os.path.exists(f):
            gedi_csv = f
            break
    
    if gedi_csv is None:
        print("\n" + "="*60)
        print("ERROR: GEDI L2A data not found!")
        print("="*60)
        print(f"\nExpected files:")
        for f in gedi_files:
            print(f"  {f}")
        print("\nPlease run: python gedi_l2a_download.py")
        exit()
    
    print(f"\nUsing GEDI file: {gedi_csv}")
    
    # Filter GEDI to bbox
    gedi_full = load_and_fix_gedi_csv(gedi_csv)
    gedi_bbox = gedi_full[
        (gedi_full['longitude'] >= bbox[0]) &
        (gedi_full['longitude'] <= bbox[2]) &
        (gedi_full['latitude'] >= bbox[1]) &
        (gedi_full['latitude'] <= bbox[3]) &
        (gedi_full['rh98'] > 0)
    ]
    
    gedi_bbox_csv = 'gedi_downloads/GEDI_bbox_subset.csv'
    gedi_bbox.to_csv(gedi_bbox_csv, index=False)
    
    print(f"\n✓ Filtered GEDI to bbox: {len(gedi_bbox)} points")
    
    if len(gedi_bbox) < 10:
        print("\n⚠ WARNING: Very few GEDI points in this area!")
        print("Model may not be reliable. Consider larger bbox.")
    
    # Extract features & train
    X, y, features = extract_features_at_gedi_points(
        gedi_bbox_csv, s2_path, s1_path, topo_path
    )
    
    if len(X) < 10:
        print("\n✗ ERROR: Not enough training samples!")
        print("Need at least 10 GEDI points with valid data.")
        exit()
    
    model = train_and_evaluate(X, y, features)
    
    # Generate map
    predict_height_map(model, s2_path, s1_path, topo_path, 
                      output_tif=f'canopy_height_{scale}m.tif')
    
    print("\n" + "="*60)
    print("✓ SUCCESS!")
    print("="*60)
    print(f"\nOutput: canopy_height_{scale}m.tif")
    print("\nVisualize in:")
    print("  - QGIS")
    print("  - Python: rasterio/matplotlib")
    print("  - Any GIS software")
