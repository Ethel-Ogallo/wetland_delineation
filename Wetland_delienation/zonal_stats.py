#!/usr/bin/env python3
import os
import rasterio
import geopandas as gpd
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
from rasterstats import zonal_stats

# ----------------------------
# Paths
# ----------------------------
base_dir = "/home/eouser/wetland_classification"
data_dir = os.path.join(base_dir, "Data")
year = "2024"

training_polygons = os.path.join(data_dir, "training_polygons.shp") 
feature_stack     = os.path.join(data_dir, f"feature_stack_{year}.tif")
segments_raster   = os.path.join(data_dir, f"segmentation_{year}.tif")
segments_shp      = os.path.join(data_dir, f"segments_{year}.shp") 
output_csv        = os.path.join(data_dir, f"segments_features_{year}.csv")
print(f"Running pipeline for year {year}")

# ----------------------------
# Load segments shapefile
# ----------------------------
print("Loading segments shapefile...")
gdf_segments = gpd.read_file(segments_shp)
print(f"Number of segments: {len(gdf_segments)}")

# ----------------------------
# Function: zonal stats batch
# ----------------------------
def zonal_stats_batch_allbands(args):
    gdf_batch, raster_path, nodata_value = args
    
    with rasterio.open(raster_path) as src:
        raster_data = src.read()  # (bands, height, width)
        transform = src.transform

    all_stats = []
    for geom in gdf_batch.geometry:
        polygon_stats = []
        for b in range(raster_data.shape[0]):
            zs = zonal_stats(
                [geom],
                raster_data[b],
                affine=transform,
                stats="mean",
                nodata=nodata_value,
                all_touched=False,
                geojson_out=False
            )
            val = zs[0]["mean"] if isinstance(zs[0], dict) else zs[0]
            polygon_stats.append(val)
        all_stats.append(polygon_stats)

    return np.array(all_stats)


# ----------------------------
# Function: label segments using polygons
# ----------------------------
def label_segments_with_polygons(gdf_segments, gdf_train):
    """
    Assign class labels to segments based on intersecting training polygons.
    """
    # Ensure same CRS
    if gdf_segments.crs != gdf_train.crs:
        gdf_train = gdf_train.to_crs(gdf_segments.crs)

    # Spatial join: assign class to segments intersecting training polygons
    joined = gpd.sjoin(
        gdf_segments[['segment_id', 'geometry']],
        gdf_train[['class', 'geometry']],
        how='left',
        predicate='intersects'
    )

    # Keep first match per segment (or handle multiple overlaps differently)
    joined = joined.groupby('segment_id').first().reset_index()
    return joined[['segment_id', 'class']]


# ----------------------------
# Function: extract features + label
# ----------------------------
def extract_features_and_label(
    gdf_segments,
    feature_stack_path,
    training_polygons_path,
    batch_size=500,
    num_processes=None,
    nodata_value=-9999
):
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)
    print(f"Using {num_processes} parallel processes for zonal stats...")

    # Split polygons into batches
    total_segments = len(gdf_segments)
    batches = [gdf_segments.iloc[i:i + batch_size] for i in range(0, total_segments, batch_size)]
    args = [(batch, feature_stack_path, nodata_value) for batch in batches]

    results = []
    with Pool(num_processes) as pool:
        for r in tqdm(pool.imap(zonal_stats_batch_allbands, args), total=len(args), desc="Computing zonal means"):
            results.append(r)

    stats_array = np.vstack(results)

    # Load raster to get band names
    with rasterio.open(feature_stack_path) as src:
        num_bands = src.count
        band_names = src.descriptions if src.descriptions else [f"band_{i+1}" for i in range(num_bands)]

    df_segments = pd.DataFrame(stats_array, columns=band_names)
    df_segments["segment_id"] = gdf_segments["segment_id"].values

    # Load training polygons
    gdf_train = gpd.read_file(training_polygons_path)

    # Label segments using polygons
    df_labels = label_segments_with_polygons(gdf_segments, gdf_train)

    # Merge labels into segment features
    df_segments = df_segments.merge(df_labels, on="segment_id", how="left")

    return df_segments, band_names


# ----------------------------
# Run extraction
# ----------------------------
df_segments, band_names = extract_features_and_label(
    gdf_segments,
    feature_stack,
    training_polygons
)

# ----------------------------
# Export to CSV
# ----------------------------
df_segments.to_csv(output_csv, index=False)
print(f"Segment features + labels saved to: {output_csv}")
