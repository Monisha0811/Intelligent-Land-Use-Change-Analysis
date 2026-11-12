import os
import glob
import numpy as np
import rasterio
from rasterio.enums import Resampling


# =========================================================
# Project paths
# =========================================================

BASE_DIR = r"D:\Intelligent_Land_Use_Change_Analysis"

RAW_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

YEAR = 2025


# =========================================================
# Find Sentinel-2 band
# =========================================================

def find_band(year, band):

    pattern = os.path.join(
        RAW_DIR,
        str(year),
        "**",
        f"*_{band}.jp2"
    )

    files = glob.glob(
        pattern,
        recursive=True
    )

    if not files:
        raise FileNotFoundError(
            f"{band} band not found for {year}"
        )

    if len(files) > 1:
        print(f"WARNING: {len(files)} files found for {band}")
        print("Using:")
        print(files[0])

    return files[0]


# =========================================================
# Process 2025
# =========================================================

print(f"Processing {YEAR}...")

b02 = find_band(YEAR, "B02_10m")
b03 = find_band(YEAR, "B03_10m")
b04 = find_band(YEAR, "B04_10m")
b08 = find_band(YEAR, "B08_10m")
b11 = find_band(YEAR, "B11_20m")

print("B02:", b02)
print("B03:", b03)
print("B04:", b04)
print("B08:", b08)
print("B11:", b11)


# =========================================================
# Read 10 m reference band
# =========================================================

with rasterio.open(b02) as src:

    profile = src.profile.copy()

    blue = src.read(1)

    height = src.height
    width = src.width

    transform = src.transform
    crs = src.crs


# =========================================================
# Read remaining 10 m bands
# =========================================================

with rasterio.open(b03) as src:
    green = src.read(1)

with rasterio.open(b04) as src:
    red = src.read(1)

with rasterio.open(b08) as src:
    nir = src.read(1)


# =========================================================
# Read 20 m SWIR and resample to 10 m
# =========================================================

with rasterio.open(b11) as src:

    swir = src.read(
        1,
        out_shape=(height, width),
        resampling=Resampling.bilinear
    )


# =========================================================
# Convert DN to reflectance
# =========================================================

blue = blue.astype(np.float32) / 10000.0
green = green.astype(np.float32) / 10000.0
red = red.astype(np.float32) / 10000.0
nir = nir.astype(np.float32) / 10000.0
swir = swir.astype(np.float32) / 10000.0


# =========================================================
# Output file
# =========================================================

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)

output_file = os.path.join(
    PROCESSED_DIR,
    "sentinel2_2025.tif"
)


# =========================================================
# Update GeoTIFF profile
# =========================================================

profile.update(
    driver="GTiff",
    dtype="float32",
    count=5,
    height=height,
    width=width,
    transform=transform,
    crs=crs,
    compress="lzw"
)


# =========================================================
# Write output
# =========================================================

with rasterio.open(
    output_file,
    "w",
    **profile
) as dst:

    dst.write(blue, 1)
    dst.write(green, 2)
    dst.write(red, 3)
    dst.write(nir, 4)
    dst.write(swir, 5)

    dst.set_band_description(
        1, "Blue"
    )

    dst.set_band_description(
        2, "Green"
    )

    dst.set_band_description(
        3, "Red"
    )

    dst.set_band_description(
        4, "NIR"
    )

    dst.set_band_description(
        5, "SWIR"
    )


print("\nSaved:")
print(output_file)

print("Size:", width, "x", height)
print("CRS:", crs)

print("\n2025 preprocessing completed successfully.")