# CH-GEE with direct local download using geemap
# Requirements: pip install earthengine-api geemap

import ee
import geemap
import os

# ---------------------------
# User configuration
# ---------------------------
BOUNDING_BOX = [-60.0, -3.5, -59.8, -3.3]
START_DATE = '2019-04-01'
END_DATE = '2024-12-31'

# Collections
GEDI_MONTHLY = 'LARSE/GEDI/GEDI02_A_002_MONTHLY'
S2_COLLECTION = 'COPERNICUS/S2_SR_HARMONIZED'
S1_COLLECTION = 'COPERNICUS/S1_GRD'
TOPO_COLLECTION = 'USGS/SRTMGL1_003'

# Export settings
TARGET_CRS = 'EPSG:4326'
TARGET_SCALE = 30  # meters
OUTPUT_FILENAME = 'GEDI_rh98_ML_predicted.tif'

# Random Forest parameters
N_TREES = 100
MAX_SAMPLES = 5000

def initialize_ee():
    try:
        ee.Initialize()
        print("✓ Earth Engine initialized.")
    except Exception as e:
        print("Earth Engine initialization error:", e)
        print("Run ee.Authenticate() in interactive environment, then rerun.")
        raise

def region_geom(bbox):
    return ee.Geometry.Rectangle(bbox)

# ---------------------------
# Data loaders
# ---------------------------

def mask_s2_clouds(img):
    qa = img.select('QA60')
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(
        qa.bitwiseAnd(cirrusBitMask).eq(0))
    return img.updateMask(mask)

def add_ndvi(img):
    ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return img.addBands(ndvi)

def load_sentinel2(region):
    s2_coll = (ee.ImageCollection(S2_COLLECTION)
               .filterBounds(region)
               .filterDate(START_DATE, END_DATE)
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
               .map(mask_s2_clouds)
               .map(add_ndvi))
    
    n = s2_coll.size().getInfo()
    print(f"✓ Sentinel-2 images found: {n}")
    
    if n == 0:
        return None
    
    s2_med = s2_coll.median().clip(region)
    return s2_med.select(['B4', 'B8', 'B11', 'NDVI'])

def load_sentinel1(region):
    s1 = (ee.ImageCollection(S1_COLLECTION)
          .filterBounds(region)
          .filterDate(START_DATE, END_DATE)
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
          .select(['VV', 'VH']))
    
    n = s1.size().getInfo()
    print(f"✓ Sentinel-1 images found: {n}")
    
    if n == 0:
        return None
    
    return s1.median().clip(region)

def load_topography(region):
    dem = ee.Image(TOPO_COLLECTION).select('elevation').clip(region)
    slope = ee.Terrain.slope(dem)
    return dem.addBands(slope).rename(['elevation', 'slope'])

def load_gedi_training_points(region):
    """Load GEDI monthly data and sample to points for training"""
    gedi_coll = (ee.ImageCollection(GEDI_MONTHLY)
                 .select('rh98')
                 .filterBounds(region)
                 .filterDate(START_DATE, END_DATE))
    
    n = gedi_coll.size().getInfo()
    print(f"✓ GEDI monthly images found: {n}")
    
    if n == 0:
        raise RuntimeError("No GEDI data in region/date-range.")
    
    gedi_composite = gedi_coll.mean()
    
    # Sample points - limit to MAX_SAMPLES to avoid memory issues
    points = gedi_composite.sample(
        region=region,
        scale=1000,
        numPixels=MAX_SAMPLES,
        seed=42,
        geometries=True
    )
    
    # Filter valid heights
    points = points.filter(ee.Filter.gt('rh98', 0))
    points = points.filter(ee.Filter.lt('rh98', 100))
    
    # Limit collection size to avoid memory issues
    points = points.limit(MAX_SAMPLES)
    
    print(f"✓ Training points prepared (max: {MAX_SAMPLES})")
    
    return points

# ---------------------------
# Machine Learning
# ---------------------------

def extract_features_at_points(points, feature_image):
    """Extract predictor values at GEDI point locations"""
    training = feature_image.sampleRegions(
        collection=points,
        properties=['rh98'],
        scale=TARGET_SCALE,
        geometries=False
    )
    
    training = training.filter(
        ee.Filter.notNull(feature_image.bandNames())
    )
    
    # Note: Don't call .size().getInfo() here - it can exceed memory limits
    # The training will work fine without knowing the exact count
    print(f"✓ Training samples extracted (count computed during model training)")
    
    return training

def train_random_forest(training_data, feature_bands):
    """Train Random Forest REGRESSOR in Earth Engine"""
    print("Training Random Forest regressor...")
    
    regressor = ee.Classifier.smileRandomForest(
        numberOfTrees=N_TREES,
        variablesPerSplit=None,
        minLeafPopulation=5,
        bagFraction=0.5,
        maxNodes=None,
        seed=42
    ).setOutputMode('REGRESSION')
    
    trained = regressor.train(
        features=training_data,
        classProperty='rh98',
        inputProperties=feature_bands
    )
    
    print("✓ Model trained successfully")
    
    return trained

# ---------------------------
# Main workflow
# ---------------------------

def main():
    initialize_ee()
    region = region_geom(BOUNDING_BOX)
    
    print("\n" + "="*60)
    print("CH-GEE with Direct Local Download (geemap)")
    print("="*60 + "\n")
    
    # 1. Load predictor data
    print("1) Loading predictor data...")
    s2 = load_sentinel2(region)
    s1 = load_sentinel1(region)
    topo = load_topography(region)
    
    # 2. Build feature image
    print("\n2) Building feature stack...")
    feature_list = []
    band_names = []
    
    if s2 is not None:
        feature_list.append(s2)
        band_names.extend(['B4', 'B8', 'B11', 'NDVI'])
    
    if s1 is not None:
        feature_list.append(s1)
        band_names.extend(['VV', 'VH'])
    
    feature_list.append(topo)
    band_names.extend(['elevation', 'slope'])
    
    feature_image = ee.Image.cat(feature_list).float()
    print(f"   Features: {', '.join(band_names)}")
    
    # 3. Load GEDI training points
    print("\n3) Loading GEDI training data...")
    gedi_points = load_gedi_training_points(region)
    
    # 4. Extract features at GEDI locations
    print("\n4) Extracting features at GEDI points...")
    training = extract_features_at_points(gedi_points, feature_image)
    
    # 5. Train Random Forest model
    print("\n5) Training model...")
    model = train_random_forest(training, band_names)
    
    # 6. Predict wall-to-wall
    print("\n6) Generating predictions...")
    prediction = feature_image.classify(model, 'predicted_rh98')
    prediction = prediction.select('predicted_rh98')
    
    # Post-process: clip to reasonable range
    prediction = prediction.clamp(0, 100)
    
    # Optional: apply light smoothing to reduce noise
    prediction = prediction.focal_median(radius=1, kernelType='square', units='pixels')
    
    final = prediction.rename('Canopy_Height_m').float()
    
    # 7. Direct download to local machine using geemap
    print("\n7) Downloading directly to local disk...")
    
    out_file = os.path.join(os.getcwd(), OUTPUT_FILENAME)
    print(f"   Target file: {out_file}")
    print("   Starting download (this may take a few minutes)...")
    
    try:
        geemap.download_ee_image(
            image=final,
            filename=out_file,
            region=region,
            scale=TARGET_SCALE,
            crs=TARGET_CRS,
            dtype='float32',
            max_tile_dim=256,  # CRITICAL: Use small tiles to avoid memory limits
            overwrite=True
        )
        
        # Get file size
        file_size = os.path.getsize(out_file)
        
        print("\n" + "="*60)
        print("✓ SUCCESS!")
        print("="*60)
        print(f"\nOutput downloaded to local machine:")
        print(f"  File: {out_file}")
        print(f"  Size: {file_size / (1024**2):.2f} MB")
        print(f"\nVisualize with:")
        print(f"  - QGIS: Layer > Add Raster Layer")
        print(f"  - Python: rasterio.open('{OUTPUT_FILENAME}')")
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ DOWNLOAD FAILED")
        print("="*60)
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("  1. Try smaller max_tile_dim (e.g., 256)")
        print("  2. Use smaller bounding box")
        print("  3. Increase TARGET_SCALE (e.g., 50m instead of 30m)")
        print("  4. Check your internet connection")

if __name__ == '__main__':
    main()
