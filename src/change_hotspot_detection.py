
import os
import csv
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter


# ============================================================
# RELATIVE SPATIAL CHANGE-DENSITY ANALYSIS
# ============================================================

BASE_DIR = r"D:\Intelligent_Land_Use_Change_Analysis"

CLASSIFIED_DIR = os.path.join(
    BASE_DIR, "data", "classified", "mangaluru"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "data", "results", "hotspots"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

NODATA_VALUE = 255
SIGMA_PIXELS = 15
PIXEL_AREA_KM2 = 0.0001

PERIODS = {
    "2018_2022": (2018, 2022),
    "2022_2025": (2022, 2025),
    "2018_2025": (2018, 2025),
}


# ============================================================
# LOAD AND VALIDATE RASTERS
# ============================================================

def load_raster(year):
    path = os.path.join(
        CLASSIFIED_DIR, f"land_use_{year}.tif"
    )

    with rasterio.open(path) as src:
        data = src.read(1)
        profile = src.profile.copy()
        metadata = {
            "shape": data.shape,
            "crs": src.crs,
            "transform": src.transform,
        }

    return data, profile, metadata


def validate_alignment(meta1, meta2):
    if (
        meta1["shape"] != meta2["shape"]
        or meta1["crs"] != meta2["crs"]
        or meta1["transform"] != meta2["transform"]
    ):
        raise ValueError(
            "Input rasters are not aligned. "
            "Check dimensions, CRS and transform."
        )


def save_raster(path, array, profile, dtype, nodata):
    output_profile = profile.copy()
    output_profile.update(
        dtype=dtype,
        count=1,
        nodata=nodata,
        compress="lzw"
    )

    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(array.astype(dtype), 1)


# ============================================================
# DETECT RELATIVE CHANGE-DENSITY AREAS
# ============================================================

def detect_hotspots(year1, year2, lulc1, lulc2, profile):
    print()
    print("=" * 75)
    print(f"RELATIVE CHANGE DENSITY: {year1} -> {year2}")
    print("=" * 75)

    valid = (
        np.isin(lulc1, [1, 2, 3, 4, 5])
        & np.isin(lulc2, [1, 2, 3, 4, 5])
    )

    changed = valid & (lulc1 != lulc2)

    valid_count = int(valid.sum())
    changed_count = int(changed.sum())

    print(f"Valid pixels:   {valid_count:,}")
    print(f"Changed pixels: {changed_count:,}")

    if valid_count == 0:
        raise ValueError("No common valid pixels found.")

    # No changes: save valid zero scores and NoData outside valid area.
    score = np.full(
        lulc1.shape, NODATA_VALUE, dtype=np.float32
    )
    classes = np.full(
        lulc1.shape, NODATA_VALUE, dtype=np.uint8
    )

    if changed_count == 0:
        score[valid] = 0
        classes[valid] = 1
    else:
        # Smooth only the change signal, accounting for valid support.
        change_float = changed.astype(np.float32)
        valid_float = valid.astype(np.float32)

        numerator = gaussian_filter(
            change_float, sigma=SIGMA_PIXELS
        )
        denominator = gaussian_filter(
            valid_float, sigma=SIGMA_PIXELS
        )

        density = np.zeros(lulc1.shape, dtype=np.float32)

        supported = valid & (denominator > 1e-6)
        density[supported] = (
            numerator[supported] / denominator[supported]
        )

        valid_density = density[valid]
        min_value = float(valid_density.min())
        max_value = float(valid_density.max())

        score_values = np.zeros(
            lulc1.shape, dtype=np.float32
        )

        if max_value > min_value:
            score_values[valid] = (
                (density[valid] - min_value)
                / (max_value - min_value)
                * 100.0
            )

        score[valid] = score_values[valid]

        # 1 = Low; 2 = Moderate; 3 = High; 4 = Very High
        classes[valid & (score_values < 25)] = 1
        classes[
            valid & (score_values >= 25) & (score_values < 50)
        ] = 2
        classes[
            valid & (score_values >= 50) & (score_values < 75)
        ] = 3
        classes[valid & (score_values >= 75)] = 4

    # --------------------------------------------------------
    # SAVE OUTPUT RASTERS
    # --------------------------------------------------------

    score_file = os.path.join(
        OUTPUT_DIR, f"change_hotspot_score_{year1}_{year2}.tif"
    )

    class_file = os.path.join(
        OUTPUT_DIR, f"change_hotspot_class_{year1}_{year2}.tif"
    )

    save_raster(
        score_file, score, profile,
        rasterio.float32, NODATA_VALUE
    )

    save_raster(
        class_file, classes, profile,
        rasterio.uint8, NODATA_VALUE
    )

    # --------------------------------------------------------
    # STATISTICS OVER VALID PIXELS ONLY
    # --------------------------------------------------------

    class_names = {
        1: "Low",
        2: "Moderate",
        3: "High",
        4: "Very High",
    }

    statistics = []

    print("\nRELATIVE CHANGE-DENSITY SUMMARY")
    print("-" * 75)

    for class_id, name in class_names.items():
        count = int(np.count_nonzero(valid & (classes == class_id)))
        percentage = count / valid_count * 100
        area = count * PIXEL_AREA_KM2

        print(
            f"{name:12s} {count:12,} pixels "
            f"{percentage:7.2f}% {area:8.2f} km²"
        )

        statistics.append([
            class_id, name, count, percentage, area
        ])

    csv_file = os.path.join(
        OUTPUT_DIR,
        f"change_hotspot_statistics_{year1}_{year2}.csv"
    )

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Hotspot Class", "Class Name", "Pixel Count",
            "Percentage", "Area (km2)"
        ])
        writer.writerows(statistics)

    print("\nSaved:")
    print(score_file)
    print(class_file)
    print(csv_file)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 75)
    print("RELATIVE SPATIAL LULC CHANGE-DENSITY ANALYSIS")
    print("=" * 75)

    rasters = {}
    profiles = {}
    metadata = {}

    for year in [2018, 2022, 2025]:
        rasters[year], profiles[year], metadata[year] = load_raster(year)

    validate_alignment(metadata[2018], metadata[2022])
    validate_alignment(metadata[2022], metadata[2025])

    for _, (year1, year2) in PERIODS.items():
        detect_hotspots(
            year1,
            year2,
            rasters[year1],
            rasters[year2],
            profiles[year1]
        )

    print()
    print("=" * 75)
    print("RELATIVE CHANGE-DENSITY ANALYSIS COMPLETED")
    print("=" * 75)
