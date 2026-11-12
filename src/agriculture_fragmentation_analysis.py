import os
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

# ---------------------------------------------------------
# Agriculture Fragmentation Analysis
# Class ID:
# 1 = Water
# 2 = Vegetation/Forest
# 3 = Agriculture
# 4 = Built-up
# 5 = Barren/Open
# ---------------------------------------------------------

BASE_DIR = r"D:\Intelligent_Land_Use_Change_Analysis"

CLASSIFIED_DIR = os.path.join(
    BASE_DIR, "data", "classified", "mangaluru"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "data", "results", "agriculture_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

YEARS = [2018, 2022, 2025]

AGRICULTURE_CLASS = 3

# Sentinel-2 classified raster resolution is approximately 10 m
PIXEL_AREA_KM2 = 0.0001


def analyze_fragmentation(year):
    input_file = os.path.join(
        CLASSIFIED_DIR,
        f"land_use_{year}.tif"
    )

    print(f"\nAnalyzing agriculture fragmentation for {year}...")
    print(f"Input: {input_file}")

    with rasterio.open(input_file) as src:
        raster = src.read(1)
        profile = src.profile

    agriculture = raster == AGRICULTURE_CLASS

    agriculture_pixels = int(np.sum(agriculture))
    agriculture_area = agriculture_pixels * PIXEL_AREA_KM2

    # 8-connected neighborhood
    structure = np.ones((3, 3), dtype=np.uint8)

    labeled, number_of_patches = ndimage.label(
        agriculture,
        structure=structure
    )

    if number_of_patches > 0:
        patch_sizes = np.bincount(labeled.ravel())[1:]
    else:
        patch_sizes = np.array([])

    patch_areas = patch_sizes * PIXEL_AREA_KM2

    largest_patch = (
        float(np.max(patch_areas))
        if len(patch_areas) > 0 else 0
    )

    mean_patch = (
        float(np.mean(patch_areas))
        if len(patch_areas) > 0 else 0
    )

    median_patch = (
        float(np.median(patch_areas))
        if len(patch_areas) > 0 else 0
    )

    # Patch density per 100 km²
    valid_pixels = np.sum(raster > 0)
    valid_area_km2 = valid_pixels * PIXEL_AREA_KM2

    patch_density = (
        number_of_patches / valid_area_km2 * 100
        if valid_area_km2 > 0 else 0
    )

    # Largest patch proportion
    largest_patch_proportion = (
        largest_patch / agriculture_area * 100
        if agriculture_area > 0 else 0
    )

    results = {
        "year": year,
        "agriculture_pixels": agriculture_pixels,
        "agriculture_area_km2": agriculture_area,
        "agriculture_percentage": (
            agriculture_pixels / valid_pixels * 100
            if valid_pixels > 0 else 0
        ),
        "number_of_patches": number_of_patches,
        "largest_patch_km2": largest_patch,
        "mean_patch_km2": mean_patch,
        "median_patch_km2": median_patch,
        "patch_density_per_100km2": patch_density,
        "largest_patch_percentage_of_agriculture": (
            largest_patch_proportion
        )
    }

    print(f"Agriculture area: {agriculture_area:.2f} km²")
    print(f"Number of patches: {number_of_patches}")
    print(f"Largest patch: {largest_patch:.4f} km²")
    print(f"Mean patch size: {mean_patch:.4f} km²")
    print(f"Median patch size: {median_patch:.4f} km²")
    print(f"Patch density: {patch_density:.2f} per 100 km²")

    return results


# ---------------------------------------------------------
# Run analysis
# ---------------------------------------------------------

all_results = []

for year in YEARS:
    result = analyze_fragmentation(year)
    all_results.append(result)


df = pd.DataFrame(all_results)

output_csv = os.path.join(
    OUTPUT_DIR,
    "agriculture_fragmentation_statistics.csv"
)

df.to_csv(output_csv, index=False)

print("\n---------------------------------------------")
print("Agriculture fragmentation analysis completed.")
print("---------------------------------------------")
print(f"Output: {output_csv}")

print("\nSummary:")
print(df.to_string(index=False))