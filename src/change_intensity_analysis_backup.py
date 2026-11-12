import os
import numpy as np
import rasterio
import csv


# ============================================================
# CHANGE INTENSITY ANALYSIS
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
    "change_intensity"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD CLASSIFIED RASTER
# ============================================================

def load_raster(year):

    filepath = os.path.join(
        CLASSIFIED_DIR,
        f"land_use_{year}.tif"
    )

    with rasterio.open(filepath) as src:

        data = src.read(1)

        profile = src.profile.copy()

    return data, profile


# ============================================================
# TWO-PERIOD INTENSITY
# ============================================================

def calculate_period_intensity(
    year1,
    year2,
    lulc1,
    lulc2,
    profile
):

    print()
    print("=" * 80)
    print(f"CHANGE INTENSITY: {year1} -> {year2}")
    print("=" * 80)

    valid = (
        (lulc1 > 0) &
        (lulc2 > 0)
    )

    changed = (
        valid &
        (lulc1 != lulc2)
    )

    # --------------------------------------------------------
    # Intensity:
    # 0 = No change
    # 1 = Change detected
    # --------------------------------------------------------

    intensity = np.zeros(
        lulc1.shape,
        dtype=np.uint8
    )

    intensity[changed] = 1

    valid_count = np.sum(valid)
    changed_count = np.sum(changed)

    print(f"Valid pixels:   {valid_count:,}")
    print(f"Changed pixels: {changed_count:,}")

    # --------------------------------------------------------
    # Save raster
    # --------------------------------------------------------

    output_file = os.path.join(
        OUTPUT_DIR,
        f"change_intensity_{year1}_{year2}.tif"
    )

    output_profile = profile.copy()

    output_profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=0,
        compress="lzw"
    )

    with rasterio.open(
        output_file,
        "w",
        **output_profile
    ) as dst:

        dst.write(intensity, 1)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    unchanged_count = valid_count - changed_count

    pixel_area_km2 = 0.0001

    statistics = [
        (
            0,
            "No Change",
            unchanged_count,
            unchanged_count / valid_count * 100,
            unchanged_count * pixel_area_km2
        ),
        (
            1,
            "Changed",
            changed_count,
            changed_count / valid_count * 100,
            changed_count * pixel_area_km2
        )
    ]

    print()
    print("INTENSITY SUMMARY")
    print("-" * 80)

    for row in statistics:

        print(
            f"{row[1]:12s}"
            f"{row[2]:12,} pixels "
            f"{row[3]:7.2f}% "
            f"{row[4]:8.2f} km2"
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_file = os.path.join(
        OUTPUT_DIR,
        f"change_intensity_statistics_{year1}_{year2}.csv"
    )

    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Intensity",
            "Description",
            "Pixel Count",
            "Percentage",
            "Area (km2)"
        ])

        writer.writerows(statistics)

    print()
    print("Saved:")
    print(output_file)
    print(csv_file)


# ============================================================
# MULTI-TEMPORAL INTENSITY
# ============================================================

def calculate_multi_temporal_intensity(
    lulc2018,
    lulc2022,
    lulc2025,
    profile
):

    print()
    print("=" * 80)
    print("MULTI-TEMPORAL CHANGE INTENSITY")
    print("2018 -> 2022 -> 2025")
    print("=" * 80)

    valid = (
        (lulc2018 > 0) &
        (lulc2022 > 0) &
        (lulc2025 > 0)
    )

    change_2018_2022 = (
        lulc2018 != lulc2022
    )

    change_2022_2025 = (
        lulc2022 != lulc2025
    )

    # --------------------------------------------------------
    # Count number of change events
    #
    # 0 = stable in both periods
    # 1 = changed in one period
    # 2 = changed in both periods
    # --------------------------------------------------------

    intensity = np.zeros(
        lulc2018.shape,
        dtype=np.uint8
    )

    intensity[valid] = (
        change_2018_2022[valid].astype(np.uint8)
        +
        change_2022_2025[valid].astype(np.uint8)
    )

    valid_count = np.sum(valid)

    print(f"Common valid pixels: {valid_count:,}")

    pixel_area_km2 = 0.0001

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    class_names = {
        0: "Stable",
        1: "Changed Once",
        2: "Changed Twice"
    }

    statistics = []

    print()
    print("MULTI-TEMPORAL INTENSITY SUMMARY")
    print("-" * 80)

    for class_id, name in class_names.items():

        count = np.sum(
            intensity[valid] == class_id
        )

        percentage = (
            count / valid_count * 100
        )

        area = (
            count * pixel_area_km2
        )

        print(
            f"{name:15s}"
            f"{count:12,} pixels "
            f"{percentage:7.2f}% "
            f"{area:8.2f} km2"
        )

        statistics.append([
            class_id,
            name,
            count,
            percentage,
            area
        ])

    # --------------------------------------------------------
    # Save raster
    # --------------------------------------------------------

    output_file = os.path.join(
        OUTPUT_DIR,
        "multi_temporal_change_intensity_2018_2025.tif"
    )

    output_profile = profile.copy()

    output_profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=0,
        compress="lzw"
    )

    with rasterio.open(
        output_file,
        "w",
        **output_profile
    ) as dst:

        dst.write(intensity, 1)

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_file = os.path.join(
        OUTPUT_DIR,
        "multi_temporal_change_intensity_summary.csv"
    )

    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Intensity",
            "Description",
            "Pixel Count",
            "Percentage",
            "Area (km2)"
        ])

        writer.writerows(statistics)

    print()
    print("Saved:")
    print(output_file)
    print(csv_file)


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("LULC CHANGE INTENSITY ANALYSIS")
print("=" * 80)


lulc2018, profile = load_raster(2018)
lulc2022, _ = load_raster(2022)
lulc2025, _ = load_raster(2025)


# Consecutive periods

calculate_period_intensity(
    2018,
    2022,
    lulc2018,
    lulc2022,
    profile
)

calculate_period_intensity(
    2022,
    2025,
    lulc2022,
    lulc2025,
    profile
)


# Multi-temporal intensity

calculate_multi_temporal_intensity(
    lulc2018,
    lulc2022,
    lulc2025,
    profile
)


print()
print("=" * 80)
print("CHANGE INTENSITY ANALYSIS COMPLETED")
print("=" * 80)