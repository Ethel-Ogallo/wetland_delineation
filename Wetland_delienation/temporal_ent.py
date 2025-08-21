import os
import numpy as np
import rasterio
from scipy.stats import entropy

# === CONFIGURATION ===
input_folder = "/home/eouser/temporal_entropy"      # folder with all Monthly_*_Stack_<YEAR>.tif files
output_folder = "/home/eouser/temporal_entropy/output"  # folder where results will be saved
indices = ["NDVI", "NDWI", "BSI"]

# Set this to a year string like "2018" to process only that year,
# or None to process all years
year_to_process = None  # e.g., "2018" or None

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Helper function to compute temporal entropy for a single raster stack
def compute_temporal_entropy(raster_path, output_path):
    with rasterio.open(raster_path) as src:
        stack = src.read()  # shape: (bands, rows, cols)
        profile = src.profile

    print(f"Processing {os.path.basename(raster_path)} with shape {stack.shape}")
    
    bands, rows, cols = stack.shape
    reshaped = stack.reshape(bands, -1).T  # (pixels, bands)
    
    valid_mask = ~np.all(np.isnan(reshaped), axis=1)
    temporal_entropy = np.full(reshaped.shape[0], np.nan)
    
    bins = np.linspace(-1, 1, 21)  # adjust for NDVI/NDWI/BSI range if needed

    def pixel_entropy(pixel_timeseries):
        pixel_timeseries = pixel_timeseries[~np.isnan(pixel_timeseries)]
        if len(pixel_timeseries) == 0:
            return np.nan
        hist, _ = np.histogram(pixel_timeseries, bins=bins, density=True)
        hist = hist + 1e-10  # avoid log(0)
        return entropy(hist, base=2)

    for i in np.where(valid_mask)[0]:
        temporal_entropy[i] = pixel_entropy(reshaped[i])
    
    entropy_map = temporal_entropy.reshape(rows, cols)
    
    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw',
        nodata=np.nan
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(entropy_map.astype(rasterio.float32), 1)
    
    print(f"Saved temporal entropy to {output_path}")

# === MAIN LOOP ===
for index in indices:
    # Find all files for this index
    files = [f for f in os.listdir(input_folder) if f.startswith(f"Monthly_{index}_Stack") and f.endswith(".tif")]
    files.sort()  # sort by year
    
    for f in files:
        year = f.split("_")[-1].split(".")[0]  # extract year from filename
        
        # Skip if a specific year is selected
        if year_to_process is not None and year != year_to_process:
            continue
        
        input_path = os.path.join(input_folder, f)
        output_path = os.path.join(output_folder, index, f"Temporal_Entropy_{index}_{year}.tif")
        compute_temporal_entropy(input_path, output_path)
