import os
import csv
import numpy as np
import rasterio
from scipy.ndimage import label


# ============================================================
# BUILT-UP EXPANSION AND FRAGMENTATION ANALYSIS
# ============================================================

BASE_DIR = r"D:\Intelligent_Land_Use_Change_Analysis"

CLASSIFIED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "classified",
    "mangaluru"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "results",
    "builtup_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


YEARS = [2018, 2022, 2025]

BUILT_UP_CLASS = 4

PIXEL_AREA_KM2 = 0.0001


# ============================================================
# LOAD RASTER
# ============================================================

def load_lulc(year):

    filepath = os.path.join(
        CLASSIFIED_DIR,
        f"land_use_{year}.tif"
    )

    with rasterio.open(filepath) as src:
        data = src.read(1)
        profile = src.profile.copy()

    return data, profile


# ============================================================
# BASIC BUILT-UP STATISTICS
# ============================================================

def calculate_basic_statistics(data, year):

    valid = data > 0
    builtup = data == BUILT_UP_CLASS

    valid_pixels = np.sum(valid)
    builtup_pixels = np.sum(builtup)

    area = builtup_pixels * PIXEL_AREA_KM2

    percentage = (
        builtup_pixels /
        valid_pixels *
        100
    )

    return [
        year,
        valid_pixels,
        builtup_pixels,
        percentage,
        area
    ]


# ============================================================
# PATCH FRAGMENTATION
# ============================================================

def calculate_fragmentation(data, year):

    valid = data > 0
    builtup = data == BUILT_UP_CLASS

    # 8-neighbour connectivity
    structure = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    labeled, patch_count = label(
        builtup,
        structure=structure
    )

    if patch_count == 0:
        return [
            year,
            0,
            0,
            0,
            0,
            0
        ]

    patch_sizes = np.bincount(
        labeled.ravel()
    )[1:]

    total_builtup_pixels = np.sum(
        patch_sizes
    )

    largest_patch_pixels = np.max(
        patch_sizes
    )

    mean_patch_pixels = np.mean(
        patch_sizes
    )

    median_patch_pixels = np.median(
        patch_sizes
    )

    largest_patch_area = (
        largest_patch_pixels *
        PIXEL_AREA_KM2
    )

    mean_patch_area = (
        mean_patch_pixels *
        PIXEL_AREA_KM2
    )

    median_patch_area = (
        median_patch_pixels *
        PIXEL_AREA_KM2
    )

    # Simple patch-density indicator:
    # number of built-up patches per 100 km2
    valid_area_km2 = (
        np.sum(valid) *
        PIXEL_AREA_KM2
    )

    patch_density = (
        patch_count /
        valid_area_km2 *
        100
    )

    return [
        year,
        patch_count,
        largest_patch_pixels,
        largest_patch_area,
        mean_patch_area,
        median_patch_area,
        patch_density
    ]


# ============================================================
# BUILT-UP CHANGE BETWEEN TWO YEARS
# ============================================================

def calculate_expansion(
    year1,
    year2,
    lulc1,
    lulc2
):

    valid = (
        (lulc1 > 0) &
        (lulc2 > 0)
    )

    builtup1 = (
        lulc1 == BUILT_UP_CLASS
    )

    builtup2 = (
        lulc2 == BUILT_UP_CLASS
    )

    # New built-up:
    # non-built-up -> built-up
    expansion = (
        valid &
        (~builtup1) &
        builtup2
    )

    # Lost built-up:
    # built-up -> non-built-up
    contraction = (
        valid &
        builtup1 &
        (~builtup2)
    )

    expansion_pixels = np.sum(
        expansion
    )

    contraction_pixels = np.sum(
        contraction
    )

    expansion_area = (
        expansion_pixels *
        PIXEL_AREA_KM2
    )

    contraction_area = (
        contraction_pixels *
        PIXEL_AREA_KM2
    )

    # --------------------------------------------------------
    # Create change map
    #
    # 0 = NoData
    # 1 = No built-up change
    # 2 = Built-up expansion
    # 3 = Built-up contraction
    # --------------------------------------------------------

    change_map = np.zeros(
        lulc1.shape,
        dtype=np.uint8
    )

    change_map[
        valid &
        ~(expansion | contraction)
    ] = 1

    change_map[expansion] = 2
    change_map[contraction] = 3

    output_file = os.path.join(
        OUTPUT_DIR,
        f"builtup_change_{year1}_{year2}.tif"
    )

    profile = {
        "driver": "GTiff",
        "height": lulc1.shape[0],
        "width": lulc1.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:32643",
        "transform": None,
        "nodata": 0,
        "compress": "lzw"
    }

    # Reuse transform/CRS from original raster
    original_file = os.path.join(
        CLASSIFIED_DIR,
        f"land_use_{year1}.tif"
    )

    with rasterio.open(
        original_file
    ) as src:

        profile = src.profile.copy()

    profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=0,
        compress="lzw"
    )

    with rasterio.open(
        output_file,
        "w",
        **profile
    ) as dst:

        dst.write(
            change_map,
            1
        )

    # --------------------------------------------------------
    # Source classes converted to built-up
    # --------------------------------------------------------

    source_classes = {
        1: "Water",
        2: "Vegetation/Forest",
        3: "Agriculture",
        5: "Barren/Open"
    }

    transition_results = []

    for class_id, class_name in source_classes.items():

        transition = (
            valid &
            (lulc1 == class_id) &
            (lulc2 == BUILT_UP_CLASS)
        )

        count = np.sum(transition)

        area = (
            count *
            PIXEL_AREA_KM2
        )

        transition_results.append([
            class_name,
            count,
            area
        ])

    return (
        expansion_pixels,
        expansion_area,
        contraction_pixels,
        contraction_area,
        transition_results,
        output_file
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("BUILT-UP EXPANSION AND FRAGMENTATION ANALYSIS")
print("=" * 80)


rasters = {}

for year in YEARS:

    print()
    print(f"Loading {year}...")

    rasters[year], profile = load_lulc(
        year
    )


# ============================================================
# YEAR-WISE BUILT-UP STATISTICS
# ============================================================

basic_results = []

for year in YEARS:

    result = calculate_basic_statistics(
        rasters[year],
        year
    )

    basic_results.append(result)


print()
print("=" * 80)
print("BUILT-UP AREA SUMMARY")
print("=" * 80)

print(
    f"{'Year':<10}"
    f"{'Pixels':>15}"
    f"{'Area (km2)':>15}"
    f"{'Study Area %':>15}"
)

for row in basic_results:

    print(
        f"{row[0]:<10}"
        f"{row[2]:>15,}"
        f"{row[4]:>15.2f}"
        f"{row[3]:>15.2f}%"
    )


# ============================================================
# FRAGMENTATION STATISTICS
# ============================================================

fragmentation_results = []

print()
print("=" * 80)
print("BUILT-UP FRAGMENTATION")
print("=" * 80)

print(
    f"{'Year':<10}"
    f"{'Patches':>15}"
    f"{'Largest (km2)':>18}"
    f"{'Mean (km2)':>15}"
    f"{'Median (km2)':>17}"
    f"{'Patch Density':>17}"
)

for year in YEARS:

    result = calculate_fragmentation(
        rasters[year],
        year
    )

    fragmentation_results.append(result)

    print(
        f"{result[0]:<10}"
        f"{result[1]:>15,}"
        f"{result[3]:>18.4f}"
        f"{result[4]:>15.4f}"
        f"{result[5]:>17.4f}"
        f"{result[6]:>17.4f}"
    )


# ============================================================
# SAVE YEAR-WISE STATISTICS
# ============================================================

basic_csv = os.path.join(
    OUTPUT_DIR,
    "builtup_area_statistics.csv"
)

with open(
    basic_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Year",
        "Valid Pixels",
        "Built-up Pixels",
        "Built-up Percentage",
        "Built-up Area (km2)"
    ])

    writer.writerows(
        basic_results
    )


fragmentation_csv = os.path.join(
    OUTPUT_DIR,
    "builtup_fragmentation_statistics.csv"
)

with open(
    fragmentation_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Year",
        "Patch Count",
        "Largest Patch Pixels",
        "Largest Patch Area (km2)",
        "Mean Patch Area (km2)",
        "Median Patch Area (km2)",
        "Patch Density per 100 km2"
    ])

    writer.writerows(
        fragmentation_results
    )


# ============================================================
# BUILT-UP EXPANSION
# ============================================================

expansion_csv = os.path.join(
    OUTPUT_DIR,
    "builtup_expansion_statistics.csv"
)

with open(
    expansion_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Period",
        "Expansion Pixels",
        "Expansion Area (km2)",
        "Contraction Pixels",
        "Contraction Area (km2)"
    ])

    for year1, year2 in [
        (2018, 2022),
        (2022, 2025),
        (2018, 2025)
    ]:

        result = calculate_expansion(
            year1,
            year2,
            rasters[year1],
            rasters[year2]
        )

        (
            expansion_pixels,
            expansion_area,
            contraction_pixels,
            contraction_area,
            transition_results,
            output_file
        ) = result

        print()
        print("=" * 80)
        print(f"BUILT-UP CHANGE: {year1} -> {year2}")
        print("=" * 80)

        print(
            f"Expansion:   "
            f"{expansion_pixels:,} pixels "
            f"({expansion_area:.2f} km2)"
        )

        print(
            f"Contraction: "
            f"{contraction_pixels:,} pixels "
            f"({contraction_area:.2f} km2)"
        )

        print()
        print("SOURCES CONVERTED TO BUILT-UP")
        print("-" * 80)

        for (
            class_name,
            count,
            area
        ) in transition_results:

            print(
                f"{class_name:<22}"
                f"{count:>12,} pixels "
                f"{area:>10.2f} km2"
            )

        writer.writerow([
            f"{year1}-{year2}",
            expansion_pixels,
            expansion_area,
            contraction_pixels,
            contraction_area
        ])


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 80)
print("BUILT-UP ANALYSIS COMPLETED")
print("=" * 80)

print()
print("Output folder:")
print(OUTPUT_DIR)

print()
print("Generated:")
print(basic_csv)
print(fragmentation_csv)
print(expansion_csv)