import os
import glob
import numpy as np
import rasterio
from rasterio.enums import Resampling


# Project paths
BASE_DIR = r"D:\Intelligent_Land_Use_Change_Analysis"

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


# Years to process
YEARS = [2018, 2022, 2025]


def find_band(year, band):
    """
    Find a particular Sentinel-2 band inside the year's .SAFE folder.
    """

    pattern = os.path.join(
    RAW_DIR,
    str(year),
    "**",
    f"*_{band}.jp2"
)

    files = glob.glob(pattern, recursive=True)

    if not files:
        raise FileNotFoundError(
            f"{band} band not found for {year}"
        )

    return files[0]


def process_year(year):

    print(f"\nProcessing {year}...")

    # Find required bands
    b02 = find_band(year, "B02_10m")
    b03 = find_band(year, "B03_10m")
    b04 = find_band(year, "B04_10m")
    b08 = find_band(year, "B08_10m")
    b11 = find_band(year, "B11_20m")

    print("B02:", os.path.basename(b02))
    print("B03:", os.path.basename(b03))
    print("B04:", os.path.basename(b04))
    print("B08:", os.path.basename(b08))
    print("B11:", os.path.basename(b11))

    # Read the 10m reference band
    with rasterio.open(b02) as src:

        profile = src.profile.copy()

        blue = src.read(1)
        height = src.height
        width = src.width
        transform = src.transform
        crs = src.crs

    # Read remaining 10m bands
    with rasterio.open(b03) as src:
        green = src.read(1)

    with rasterio.open(b04) as src:
        red = src.read(1)

    with rasterio.open(b08) as src:
        nir = src.read(1)

    # Read B11 (20m) and resample to 10m
    with rasterio.open(b11) as src:

        swir = src.read(
            1,
            out_shape=(height, width),
            resampling=Resampling.bilinear
        )

    # Convert to floating point
    blue = blue.astype(np.float32)
    green = green.astype(np.float32)
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    swir = swir.astype(np.float32)

    # Sentinel-2 reflectance scaling
    blue /= 10000.0
    green /= 10000.0
    red /= 10000.0
    nir /= 10000.0
    swir /= 10000.0

    # Create output filename
    output_file = os.path.join(
        PROCESSED_DIR,
        f"sentinel2_{year}.tif"
    )

    # Update GeoTIFF profile
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

    # Write the five bands
    with rasterio.open(output_file, "w", **profile) as dst:

        dst.write(blue, 1)
        dst.write(green, 2)
        dst.write(red, 3)
        dst.write(nir, 4)
        dst.write(swir, 5)

        dst.set_band_description(1, "Blue")
        dst.set_band_description(2, "Green")
        dst.set_band_description(3, "Red")
        dst.set_band_description(4, "NIR")
        dst.set_band_description(5, "SWIR")

    print(f"Saved: {output_file}")


# Create processed directory
os.makedirs(PROCESSED_DIR, exist_ok=True)


# Process all years
for year in YEARS:
    process_year(year)

print("\nAll years processed successfully!")