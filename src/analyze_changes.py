import rasterio
import numpy as np
import os
import csv


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CLASSIFIED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "classified",
    "mangaluru"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "data",
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LAND-USE CLASSES
# ============================================================

CLASS_NAMES = {
    1: "Water",
    2: "Vegetation/Forest",
    3: "Agriculture",
    4: "Built-up",
    5: "Barren/Open"
}

YEARS = [2018, 2022, 2025]


# ============================================================
# ANALYZE EACH YEAR
# ============================================================

all_results = []

print("\n" + "=" * 65)
print("LAND-USE AREA ANALYSIS")
print("=" * 65)


for year in YEARS:

    path = os.path.join(
        CLASSIFIED_DIR,
        f"land_use_{year}.tif"
    )

    print(f"\n{'=' * 65}")
    print(f"Analyzing {year}")
    print(f"{'=' * 65}")

    with rasterio.open(path) as src:

        data = src.read(1)

        # Count only valid land-use classes
        valid = np.isin(data, list(CLASS_NAMES.keys()))
        valid_pixels = int(np.sum(valid))

        # Calculate pixel area from the raster's actual transform
        pixel_area_m2 = abs(
            src.transform.a * src.transform.e
        )
        pixel_area_km2 = pixel_area_m2 / 1_000_000

        total_area_km2 = valid_pixels * pixel_area_km2

        print(
            f"Valid pixels: {valid_pixels:,}"
        )

        print(
            f"Valid area: {total_area_km2:.2f} km²"
        )

        year_results = []

        for class_id, class_name in CLASS_NAMES.items():

            count = int(
                np.sum(data == class_id)
            )

            percentage = (
                count / valid_pixels * 100
                if valid_pixels > 0
                else 0
            )

            area_km2 = (
                count * pixel_area_km2
            )

            print(
                f"Class {class_id} - {class_name}: "
                f"{count:,} pixels | "
                f"{area_km2:.2f} km² | "
                f"{percentage:.2f}%"
            )

            year_results.append({
                "Year": year,
                "Class_ID": class_id,
                "Class_Name": class_name,
                "Pixels": count,
                "Area_km2": round(area_km2, 4),
                "Percentage": round(percentage, 4)
            })

            all_results.append(
                year_results[-1]
            )


# ============================================================
# SAVE YEARLY RESULTS TO CSV
# ============================================================

csv_file = os.path.join(
    RESULTS_DIR,
    "land_use_area_statistics.csv"
)

with open(
    csv_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Year",
            "Class_ID",
            "Class_Name",
            "Pixels",
            "Area_km2",
            "Percentage"
        ]
    )

    writer.writeheader()
    writer.writerows(all_results)


# ============================================================
# SUMMARY TABLE
# ============================================================

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)

print(
    f"{'Class':<22}"
    f"{'2018 (%)':>12}"
    f"{'2022 (%)':>12}"
    f"{'2025 (%)':>12}"
)

print("-" * 58)

for class_id, class_name in CLASS_NAMES.items():

    percentages = []

    for year in YEARS:

        result = next(
            r for r in all_results
            if r["Year"] == year
            and r["Class_ID"] == class_id
        )

        percentages.append(
            result["Percentage"]
        )

    print(
        f"{class_name:<22}"
        f"{percentages[0]:>12.2f}"
        f"{percentages[1]:>12.2f}"
        f"{percentages[2]:>12.2f}"
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 65)
print("Analysis completed successfully!")
print("=" * 65)

print(
    f"\nResults saved to:\n{csv_file}"
)