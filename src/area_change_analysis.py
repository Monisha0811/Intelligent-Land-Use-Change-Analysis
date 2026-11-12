import os
import numpy as np
import pandas as pd
import rasterio


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
# LOAD CLASSIFIED MAP
# ============================================================

def load_map(year):

    file_path = os.path.join(
        CLASSIFIED_DIR,
        f"land_use_{year}.tif"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Classified file not found: {file_path}"
        )

    with rasterio.open(file_path) as src:

        data = src.read(1)

        profile = src.profile.copy()

        transform = src.transform

        pixel_width = abs(transform.a)
        pixel_height = abs(transform.e)

        pixel_area_m2 = (
            pixel_width * pixel_height
        )

    return data, pixel_area_m2


# ============================================================
# COMMON VALID AREA ANALYSIS
# ============================================================

def analyze_pair(year1, year2):

    print("\n")
    print("=" * 70)
    print(f"COMMON-AREA ANALYSIS: {year1} -> {year2}")
    print("=" * 70)

    map1, pixel_area = load_map(year1)
    map2, _ = load_map(year2)

    if map1.shape != map2.shape:

        raise ValueError(
            f"Raster dimensions do not match: "
            f"{map1.shape} vs {map2.shape}"
        )

    # Only pixels classified in BOTH years
    valid = (
        (map1 >= 1) &
        (map1 <= 5) &
        (map2 >= 1) &
        (map2 <= 5)
    )

    valid_pixels = int(np.sum(valid))

    pixel_area_km2 = pixel_area / 1_000_000

    valid_area_km2 = (
        valid_pixels * pixel_area_km2
    )

    print(
        f"Common valid pixels: "
        f"{valid_pixels:,}"
    )

    print(
        f"Common valid area: "
        f"{valid_area_km2:.2f} km²"
    )

    # ========================================================
    # CLASS AREA FOR EACH YEAR
    # ========================================================

    rows = []

    print("\nClass areas:")

    for class_id, class_name in CLASS_NAMES.items():

        count1 = int(
            np.sum(
                (map1 == class_id) & valid
            )
        )

        count2 = int(
            np.sum(
                (map2 == class_id) & valid
            )
        )

        area1 = (
            count1 * pixel_area_km2
        )

        area2 = (
            count2 * pixel_area_km2
        )

        percentage1 = (
            count1 / valid_pixels * 100
            if valid_pixels > 0
            else 0
        )

        percentage2 = (
            count2 / valid_pixels * 100
            if valid_pixels > 0
            else 0
        )

        area_change = area2 - area1

        if area1 > 0:

            percentage_change = (
                area_change / area1
            ) * 100

        else:

            percentage_change = 0

        rows.append({
            "Class": class_name,
            f"{year1}_Area_km2": round(area1, 2),
            f"{year2}_Area_km2": round(area2, 2),
            "Change_km2": round(
                area_change,
                2
            ),
            "Percentage_Change": round(
                percentage_change,
                2
            )
        })

        print(
            f"{class_name:20s} "
            f"{year1}: {area1:8.2f} km² | "
            f"{year2}: {area2:8.2f} km² | "
            f"Change: {area_change:+8.2f} km²"
        )

    # ========================================================
    # SAVE AREA CHANGE CSV
    # ========================================================

    area_df = pd.DataFrame(rows)

    area_file = os.path.join(
        RESULTS_DIR,
        f"area_change_{year1}_{year2}.csv"
    )

    area_df.to_csv(
        area_file,
        index=False
    )

    print(
        f"\nArea statistics saved:"
        f"\n{area_file}"
    )

    # ========================================================
    # TRANSITION MATRIX
    # ========================================================

    print("\nLand-use transition matrix:")

    transition_matrix = np.zeros(
        (5, 5),
        dtype=np.int64
    )

    old_values = map1[valid]
    new_values = map2[valid]

    for old_class in range(1, 6):

        for new_class in range(1, 6):

            count = np.sum(
                (old_values == old_class) &
                (new_values == new_class)
            )

            transition_matrix[
                old_class - 1,
                new_class - 1
            ] = count

    transition_df = pd.DataFrame(
        transition_matrix,
        index=[
            f"{name} ->"
            for name in CLASS_NAMES.values()
        ],
        columns=list(
            CLASS_NAMES.values()
        )
    )

    print(transition_df)

    transition_file = os.path.join(
        RESULTS_DIR,
        f"transition_matrix_{year1}_{year2}.csv"
    )

    transition_df.to_csv(
        transition_file
    )

    print(
        f"\nTransition matrix saved:"
        f"\n{transition_file}"
    )

    # ========================================================
    # DETAILED TRANSITIONS
    # ========================================================

    transition_rows = []

    for old_class in range(1, 6):

        for new_class in range(1, 6):

            if old_class == new_class:
                continue

            count = transition_matrix[
                old_class - 1,
                new_class - 1
            ]

            area = (
                count * pixel_area_km2
            )

            if count > 0:

                transition_rows.append({
                    "From": CLASS_NAMES[
                        old_class
                    ],
                    "To": CLASS_NAMES[
                        new_class
                    ],
                    "Pixels": int(count),
                    "Area_km2": round(
                        area,
                        2
                    )
                })

    transition_details_df = pd.DataFrame(
        transition_rows
    )

    transition_details_file = os.path.join(
        RESULTS_DIR,
        f"transition_details_{year1}_{year2}.csv"
    )

    transition_details_df.to_csv(
        transition_details_file,
        index=False
    )

    print(
        f"Detailed transitions saved:"
        f"\n{transition_details_file}"
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("LAND-USE COMMON-AREA CHANGE ANALYSIS")
    print("=" * 70)

    analyze_pair(2018, 2022)

    analyze_pair(2022, 2025)

    analyze_pair(2018, 2025)

    print("\n")
    print("=" * 70)
    print("COMMON-AREA ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    print(
        f"\nResults saved in:\n{RESULTS_DIR}"
    )