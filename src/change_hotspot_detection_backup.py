import os
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter, label


# ============================================================
# CHANGE HOTSPOT DETECTION
# ============================================================

BASE_DIR = r"D:\Intelligent_Land_Use_Change_Analysis"

CLASSIFIED_DIR = os.path.join(
    BASE_DIR, "data", "classified", "mangaluru"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "data", "results", "hotspots"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


PERIODS = {
    "2018_2022": (2018, 2022),
    "2022_2025": (2022, 2025),
    "2018_2025": (2018, 2025),
}


def detect_hotspots(year1, year2):

    print()
    print("=" * 80)
    print(f"CHANGE HOTSPOT DETECTION: {year1} -> {year2}")
    print("=" * 80)

    file1 = os.path.join(
        CLASSIFIED_DIR,
        f"land_use_{year1}.tif"
    )

    file2 = os.path.join(
        CLASSIFIED_DIR,
        f"land_use_{year2}.tif"
    )

    # --------------------------------------------------------
    # Load rasters
    # --------------------------------------------------------

    with rasterio.open(file1) as src1:
        lulc1 = src1.read(1)
        profile = src1.profile.copy()

    with rasterio.open(file2) as src2:
        lulc2 = src2.read(1)

    print(f"Raster shape: {lulc1.shape}")

    # --------------------------------------------------------
    # Common valid area
    # --------------------------------------------------------

    valid = (
        (lulc1 > 0) &
        (lulc2 > 0)
    )

    print(f"Valid pixels: {np.sum(valid):,}")

    # --------------------------------------------------------
    # Detect changed pixels
    # --------------------------------------------------------

    changed = (
        valid &
        (lulc1 != lulc2)
    )

    changed_count = np.sum(changed)

    print(f"Changed pixels: {changed_count:,}")

    if changed_count == 0:
        print("No changes detected.")
        return

    # --------------------------------------------------------
    # Create binary change raster
    # --------------------------------------------------------

    change_binary = np.zeros(lulc1.shape, dtype=np.uint8)
    change_binary[changed] = 1

    # --------------------------------------------------------
    # Calculate local change density
    #
    # Gaussian smoothing converts individual changed pixels
    # into a continuous spatial density surface.
    # --------------------------------------------------------

    change_float = change_binary.astype(np.float32)

    density = gaussian_filter(
        change_float,
        sigma=15
    )

    # Keep invalid area out of the analysis
    density[~valid] = 0

    # --------------------------------------------------------
    # Normalize density to 0-100
    # --------------------------------------------------------

    valid_density = density[valid]

    min_value = valid_density.min()
    max_value = valid_density.max()

    if max_value > min_value:
        hotspot_score = (
            (density - min_value)
            / (max_value - min_value)
            * 100
        )
    else:
        hotspot_score = np.zeros_like(density)

    hotspot_score[~valid] = 0

    # --------------------------------------------------------
    # Classify hotspot intensity
    #
    # 0 = NoData
    # 1 = Low
    # 2 = Moderate
    # 3 = High
    # 4 = Very High
    # --------------------------------------------------------

    hotspot_class = np.zeros(
        lulc1.shape,
        dtype=np.uint8
    )

    hotspot_class[
        valid & (hotspot_score < 25)
    ] = 1

    hotspot_class[
        valid &
        (hotspot_score >= 25) &
        (hotspot_score < 50)
    ] = 2

    hotspot_class[
        valid &
        (hotspot_score >= 50) &
        (hotspot_score < 75)
    ] = 3

    hotspot_class[
        valid & (hotspot_score >= 75)
    ] = 4

    # --------------------------------------------------------
    # Save hotspot score
    # --------------------------------------------------------

    score_file = os.path.join(
        OUTPUT_DIR,
        f"change_hotspot_score_{year1}_{year2}.tif"
    )

    score_profile = profile.copy()

    score_profile.update(
        dtype=rasterio.float32,
        count=1,
        nodata=0,
        compress="lzw"
    )

    with rasterio.open(
        score_file,
        "w",
        **score_profile
    ) as dst:

        dst.write(
            hotspot_score.astype(np.float32),
            1
        )

    # --------------------------------------------------------
    # Save hotspot classes
    # --------------------------------------------------------

    class_file = os.path.join(
        OUTPUT_DIR,
        f"change_hotspot_class_{year1}_{year2}.tif"
    )

    class_profile = profile.copy()

    class_profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=0,
        compress="lzw"
    )

    with rasterio.open(
        class_file,
        "w",
        **class_profile
    ) as dst:

        dst.write(
            hotspot_class,
            1
        )

    # --------------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------------

    pixel_area_km2 = 0.0001

    print()
    print("HOTSPOT CLASS SUMMARY")
    print("-" * 80)

    statistics = []

    class_names = {
        1: "Low",
        2: "Moderate",
        3: "High",
        4: "Very High"
    }

    for class_id, class_name in class_names.items():

        count = np.sum(
            hotspot_class == class_id
        )

        percentage = (
            count / np.sum(valid) * 100
        )

        area = count * pixel_area_km2

        print(
            f"{class_name:10s} "
            f"{count:10,} pixels "
            f"{percentage:7.2f}% "
            f"{area:8.2f} km2"
        )

        statistics.append(
            (
                class_id,
                class_name,
                count,
                percentage,
                area
            )
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_file = os.path.join(
        OUTPUT_DIR,
        f"change_hotspot_statistics_{year1}_{year2}.csv"
    )

    with open(csv_file, "w", encoding="utf-8") as f:

        f.write(
            "Hotspot Class,Class Name,Pixel Count,"
            "Percentage,Area (km2)\n"
        )

        for row in statistics:

            f.write(
                f"{row[0]},"
                f"{row[1]},"
                f"{row[2]},"
                f"{row[3]:.4f},"
                f"{row[4]:.4f}\n"
            )

    print()
    print("Saved:")
    print(score_file)
    print(class_file)
    print(csv_file)


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("SPATIAL LULC CHANGE HOTSPOT ANALYSIS")
print("=" * 80)

for period, (year1, year2) in PERIODS.items():

    detect_hotspots(year1, year2)

print()
print("=" * 80)
print("CHANGE HOTSPOT ANALYSIS COMPLETED")
print("=" * 80)