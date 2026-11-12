import os
import numpy as np
import rasterio

BASE_DIR = r"D:\Intelligent_Land_Use_Change_Analysis"

CLASSIFIED_DIR = os.path.join(
    BASE_DIR, "data", "classified", "mangaluru"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "data", "results", "water_investigation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

path_2022 = os.path.join(
    CLASSIFIED_DIR, "land_use_2022.tif"
)

path_2025 = os.path.join(
    CLASSIFIED_DIR, "land_use_2025.tif"
)

with rasterio.open(path_2022) as src:
    lulc_2022 = src.read(1)
    profile = src.profile.copy()

with rasterio.open(path_2025) as src:
    lulc_2025 = src.read(1)

valid = (lulc_2022 > 0) & (lulc_2025 > 0)

transition = np.zeros(lulc_2022.shape, dtype=np.uint8)

# 1 = Built-up to Water
transition[
    valid & (lulc_2022 == 4) & (lulc_2025 == 1)
] = 1

# 2 = Water to Built-up
transition[
    valid & (lulc_2022 == 1) & (lulc_2025 == 4)
] = 2

profile.update(
    count=1,
    dtype="uint8",
    nodata=0,
    compress="lzw"
)

output_path = os.path.join(
    OUTPUT_DIR,
    "builtup_water_transitions_2022_2025.tif"
)

with rasterio.open(output_path, "w", **profile) as dst:
    dst.write(transition, 1)

print("Built-up to Water:",
      np.sum(transition == 1) * 0.0001, "km²")

print("Water to Built-up:",
      np.sum(transition == 2) * 0.0001, "km²")

print("Saved:", output_path)