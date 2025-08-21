#!/usr/bin/env python3
import os
import geopandas as gpd
import pandas as pd
import odc.stac
import xarray as xr
import numpy as np
import rioxarray
from scipy.ndimage import uniform_filter
from shapely.geometry import mapping, shape
from pystac_client import Client
from pystac import ItemCollection
from skimage.filters import threshold_otsu

# -----------------------------
# USER SETTINGS
# -----------------------------
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
BASE_DIR = "/home/eouser/inundation"
AOI_FILE = os.path.join(BASE_DIR, "baringo_bbox.shp")
OUTFILE = os.path.join(BASE_DIR, "inundation_freq_2023.tif")

STAC_API = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-1-grd"
DATE_RANGE = "2023-01-01/2023-12-31"

# -----------------------------
# FUNCTIONS
# -----------------------------
def speckle_filter(ds, size=7):
    """Apply simple Lee speckle filtering in linear scale, return filtered dB bands."""
    filtered_vars = {}
    for band in ds.data_vars:
        arr_db = ds[band]
        valid_mask = np.isfinite(arr_db)

        arr_linear = 10 ** (arr_db / 10.0)
        arr_linear_filled = arr_linear.fillna(0)

        img_mean = uniform_filter(arr_linear_filled, size)
        img_sqr_mean = uniform_filter(arr_linear_filled ** 2, size)
        img_variance = img_sqr_mean - img_mean ** 2

        overall_variance = np.mean(img_variance)
        img_weights = img_variance / (img_variance + overall_variance)
        filtered_linear = img_mean + img_weights * (arr_linear_filled - img_mean)

        filtered_db = 10 * np.log10(
            xr.where(filtered_linear > 0, filtered_linear, 1e-10)
        )
        filtered_db = filtered_db.where(valid_mask)

        filtered_vars[f"{band}_filtered"] = filtered_db

    return ds.assign(**filtered_vars)


def classify_water(ds_filtered, vv_thresh=None, vh_thresh=None):
    """Classify water pixels with Otsu thresholds."""
    nodata_value = 255

    def compute_otsu_threshold(arr):
        vals = arr.where(arr > 0).values.flatten()
        vals = vals[np.isfinite(vals)]
        return threshold_otsu(vals)

    vv = ds_filtered["vv_filtered"].where(ds_filtered["vv_filtered"] > 0)
    vh = ds_filtered["vh_filtered"].where(ds_filtered["vh_filtered"] > 0)

    if vv_thresh is None:
        vv_thresh = compute_otsu_threshold(vv)
    if vh_thresh is None:
        vh_thresh = compute_otsu_threshold(vh)

    print(f"[INFO] Otsu thresholds: VV={vv_thresh:.2f} dB, VH={vh_thresh:.2f} dB")

    water_mask = ((vv < vv_thresh) | (vh < vh_thresh)).astype("uint8")
    nodata_mask = vv.isnull() | vh.isnull()
    water_mask = water_mask.where(~nodata_mask, other=nodata_value).astype("uint8")

    return water_mask


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print("[INFO] Reading AOI...")
    aoi_gdf = gpd.read_file(AOI_FILE)
    aoi_geom = mapping(aoi_gdf.to_crs("EPSG:4326").geometry.iloc[0])

    print("[INFO] Querying STAC API...")
    client = Client.open(STAC_API)
    search = client.search(
        collections=[COLLECTION], intersects=aoi_geom, datetime=DATE_RANGE
    )
    items = list(search.items_as_dicts())
    print(f"[INFO] {len(items)} Sentinel-1 scenes found for 2024")

    if len(items) == 0:
        raise SystemExit("No Sentinel-1 GRD items found for 2024 in AOI")

    # Load data
    item_collection = ItemCollection(items)
    ds = odc.stac.load(
        item_collection,
        bands=["vv", "vh"],
        crs="EPSG:32637",  # UTM Zone 37N
        resolution=10,
        group_by="solar_day",
    )

    print("[INFO] Converting to dB scale...")
    ds_dB = ds.copy()
    ds_dB["vv"] = 10 * np.log10(ds["vv"].where(ds["vv"] > 0))
    ds_dB["vh"] = 10 * np.log10(ds["vh"].where(ds["vh"] > 0))

    print("[INFO] Clipping to AOI...")
    ds_clipped = ds_dB.rio.clip(
        aoi_gdf.to_crs(ds_dB.rio.crs).geometry, ds_dB.rio.crs, drop=True
    )

    print("[INFO] Applying speckle filter...")
    ds_filtered = speckle_filter(ds_clipped, size=7)

    print("[INFO] Classifying water...")
    water_mask = classify_water(ds_filtered)

    print("[INFO] Computing inundation frequency...")
    water_mask_valid = water_mask.where(water_mask != 255)
    inundation_freq = water_mask_valid.sum(dim="time")

    print(f"[INFO] Saving annual inundation raster -> {OUTFILE}")
    inundation_freq.rio.to_raster(OUTFILE, compress="deflate")

    print("[INFO] Done.")
