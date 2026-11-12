
import os
import csv
import numpy as np
import rasterio


# ============================================================
# CHANGE INTENSITY ANALYSIS
# ============================================================

BASE_DIR = r"D:\Intelligent_Land_Use_Change_Analysis"

CLASSIFIED_DIR = os.path.join(
    BASE_DIR, "data", "classified", "mangaluru"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "data", "results", "change_intensity"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

NODATA_VALUE = 255
PIXEL_AREA_KM2 = 0.0001


# ============================================================
# LOAD AND VALIDATE RASTERS
# ============================================================

def load_raster(year):
    filepath = os.path.join(
        CLASSIFIED_DIR, f"land_use_{year}.tif"
    )

    with rasterio.open(filepath) as src:
        data = src.read(1)
        profile = src.profile.copy()
        metadata = {
            "crs": src.crs,
            "transform": src.transform,
            "shape": data.shape,
        }

    return data, profile, metadata


def validate_alignment(metadata1, metadata2):
    if (
        metadata1["shape"] != metadata2["shape"]
        or metadata1["crs"] != metadata2["crs"]
        or metadata1["transform"] != metadata2["transform"]
    ):
        raise ValueError(
            "Input rasters are not aligned. "
            "Check dimensions, CRS and transform before analysis."
        )


def write_raster(path, array, profile):
    output_profile = profile.copy()
    output_profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=NODATA_VALUE,
        compress="lzw"
    )

    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(array.astype(np.uint8), 1)


# ============================================================
# SAVE STATISTICS
# ============================================================

def save_statistics(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def summarize_classes(intensity, valid, class_names, valid_count):
    statistics = []

    for class_id, name in class_names.items():
        count = int(np.count_nonzero(valid & (intensity == class_id)))
        percentage = count / valid_count * 100 if valid_count else 0
        area = count * PIXEL_AREA_KM2

        statistics.append([
            class_id, name, count, percentage, area
        ])

        print(
            f"{name:15s} {count:12,} pixels "
            f"{percentage:7.2f}% {area:8.2f} km²"
        )

    return statistics


# ============================================================
# TWO-PERIOD CHANGE INTENSITY
# ============================================================

def calculate_period_intensity(
    year1, year2, lulc1, lulc2, profile
):
    print()
    print("=" * 75)
    print(f"CHANGE INTENSITY: {year1} -> {year2}")
    print("=" * 75)

    valid = (
        np.isin(lulc1, [1, 2, 3, 4, 5])
        & np.isin(lulc2, [1, 2, 3, 4, 5])
    )

    changed = valid & (lulc1 != lulc2)

    valid_count = int(valid.sum())
    changed_count = int(changed.sum())
    unchanged_count = valid_count - changed_count

    print(f"Valid pixels:   {valid_count:,}")
    print(f"Changed pixels: {changed_count:,}")

    # 0 = No Change; 1 = Changed; 255 = NoData
    intensity = np.full(
        lulc1.shape, NODATA_VALUE, dtype=np.uint8
    )
    intensity[valid] = 0
    intensity[changed] = 1

    output_file = os.path.join(
        OUTPUT_DIR, f"change_intensity_{year1}_{year2}.tif"
    )
    write_raster(output_file, intensity, profile)

    class_names = {
        0: "No Change",
        1: "Changed"
    }

    print("\nINTENSITY SUMMARY")
    print("-" * 75)

    statistics = summarize_classes(
        intensity, valid, class_names, valid_count
    )

    csv_file = os.path.join(
        OUTPUT_DIR,
        f"change_intensity_statistics_{year1}_{year2}.csv"
    )

    save_statistics(
        csv_file,
        ["Intensity", "Description", "Pixel Count",
         "Percentage", "Area (km2)"],
        statistics
    )

    print("\nSaved:")
    print(output_file)
    print(csv_file)


# ============================================================
# MULTI-TEMPORAL CHANGE INTENSITY
# ============================================================

def calculate_multi_temporal_intensity(
    lulc2018, lulc2022, lulc2025, profile
):
    print()
    print("=" * 75)
    print("MULTI-TEMPORAL CHANGE INTENSITY: 2018 -> 2022 -> 2025")
    print("=" * 75)

    valid = (
        np.isin(lulc2018, [1, 2, 3, 4, 5])
        & np.isin(lulc2022, [1, 2, 3, 4, 5])
        & np.isin(lulc2025, [1, 2, 3, 4, 5])
    )

    change_2018_2022 = lulc2018 != lulc2022
    change_2022_2025 = lulc2022 != lulc2025

    valid_count = int(valid.sum())
    print(f"Common valid pixels: {valid_count:,}")

    # 0 = Stable; 1 = Changed Once; 2 = Changed Twice
    # 255 = NoData
    intensity = np.full(
        lulc2018.shape, NODATA_VALUE, dtype=np.uint8
    )

    change_count = (
        change_2018_2022.astype(np.uint8)
        + change_2022_2025.astype(np.uint8)
    )

    intensity[valid] = change_count[valid]

    class_names = {
        0: "Stable",
        1: "Changed Once",
        2: "Changed Twice"
    }

    print("\nMULTI-TEMPORAL INTENSITY SUMMARY")
    print("-" * 75)

    statistics = summarize_classes(
        intensity, valid, class_names, valid_count
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        "multi_temporal_change_intensity_2018_2025.tif"
    )
    write_raster(output_file, intensity, profile)

    csv_file = os.path.join(
        OUTPUT_DIR,
        "multi_temporal_change_intensity_summary.csv"
    )

    save_statistics(
        csv_file,
        ["Intensity", "Description", "Pixel Count",
         "Percentage", "Area (km2)"],
        statistics
    )

    print("\nSaved:")
    print(output_file)
    print(csv_file)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 75)
    print("LULC CHANGE INTENSITY ANALYSIS")
    print("=" * 75)

    lulc2018, profile, meta2018 = load_raster(2018)
    lulc2022, _, meta2022 = load_raster(2022)
    lulc2025, _, meta2025 = load_raster(2025)

    validate_alignment(meta2018, meta2022)
    validate_alignment(meta2022, meta2025)

    calculate_period_intensity(
        2018, 2022, lulc2018, lulc2022, profile
    )

    calculate_period_intensity(
        2022, 2025, lulc2022, lulc2025, profile
    )

    calculate_multi_temporal_intensity(
        lulc2018, lulc2022, lulc2025, profile
    )

    print()
    print("=" * 75)
    print("CHANGE INTENSITY ANALYSIS COMPLETED")
    print("=" * 75)
