import os
import csv
import numpy as np
import rasterio


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CLASSIFIED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "classified",
    "mangaluru"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "change_maps"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LAND-USE CLASS NAMES
# ============================================================

CLASS_NAMES = {
    1: "Water",
    2: "Vegetation/Forest",
    3: "Agriculture",
    4: "Built-up",
    5: "Barren/Open"
}


# ============================================================
# COMPARE TWO YEARS
# ============================================================

def compare_years(year1, year2):

    file1 = os.path.join(
        CLASSIFIED_DIR,
        f"land_use_{year1}.tif"
    )

    file2 = os.path.join(
        CLASSIFIED_DIR,
        f"land_use_{year2}.tif"
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        f"change_{year1}_{year2}.tif"
    )

    csv_file = os.path.join(
        OUTPUT_DIR,
        f"transitions_{year1}_{year2}.csv"
    )

    print()
    print("=" * 60)
    print(f"Comparing {year1} -> {year2}")
    print("=" * 60)

    with rasterio.open(file1) as src1, \
         rasterio.open(file2) as src2:

        map1 = src1.read(1)
        map2 = src2.read(1)

        if map1.shape != map2.shape:
            raise ValueError(
                "Raster dimensions do not match."
            )

        # ----------------------------------------------------
        # Valid pixels
        # ----------------------------------------------------
        #
        # 0 = NoData
        # 1-5 = Land-use classes
        #
        # Only pixels where BOTH years contain valid
        # land-use classes are considered.
        # ----------------------------------------------------

        valid = (
            (map1 >= 1) &
            (map1 <= 5) &
            (map2 >= 1) &
            (map2 <= 5)
        )

        # ----------------------------------------------------
        # Create change map
        # ----------------------------------------------------
        #
        # 0 = NoData / ignored
        # 1 = Unchanged
        # 2 = Changed
        # ----------------------------------------------------

        change_map = np.zeros(
            map1.shape,
            dtype=np.uint8
        )

        unchanged = (
            valid &
            (map1 == map2)
        )

        changed = (
            valid &
            (map1 != map2)
        )

        change_map[unchanged] = 1
        change_map[changed] = 2

        # ----------------------------------------------------
        # Save change raster
        # ----------------------------------------------------

        profile = src1.profile.copy()

        profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress="lzw",
            nodata=0
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

            dst.set_band_description(
                1,
                "Land-use change: 0=NoData, 1=Unchanged, 2=Changed"
            )

    # ========================================================
    # BASIC STATISTICS
    # ========================================================

    valid_pixels = int(
        np.sum(valid)
    )

    changed_pixels = int(
        np.sum(changed)
    )

    unchanged_pixels = int(
        np.sum(unchanged)
    )

    print(
        f"Valid pixels:     {valid_pixels:,}"
    )

    print(
        f"Changed pixels:   {changed_pixels:,}"
    )

    print(
        f"Unchanged pixels: {unchanged_pixels:,}"
    )

    if valid_pixels > 0:

        changed_percentage = (
            changed_pixels /
            valid_pixels
        ) * 100

        unchanged_percentage = (
            unchanged_pixels /
            valid_pixels
        ) * 100

        print(
            f"Changed:   {changed_percentage:.2f}%"
        )

        print(
            f"Unchanged: {unchanged_percentage:.2f}%"
        )

    # ========================================================
    # TRANSITION MATRIX
    # ========================================================

    print()
    print("Land-use transitions:")

    transition_rows = []

    for from_class in range(1, 6):

        for to_class in range(1, 6):

            count = int(
                np.sum(
                    valid &
                    (map1 == from_class) &
                    (map2 == to_class)
                )
            )

            if count > 0:

                transition_rows.append([
                    year1,
                    year2,
                    from_class,
                    CLASS_NAMES[from_class],
                    to_class,
                    CLASS_NAMES[to_class],
                    count
                ])

                if from_class != to_class:

                    print(
                        f"{CLASS_NAMES[from_class]} "
                        f"-> "
                        f"{CLASS_NAMES[to_class]}: "
                        f"{count:,} pixels"
                    )

    # ========================================================
    # SAVE TRANSITION CSV
    # ========================================================

    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "From Year",
            "To Year",
            "From Class ID",
            "From Class",
            "To Class ID",
            "To Class",
            "Pixels"
        ])

        writer.writerows(
            transition_rows
        )

    print()
    print(
        f"Change map saved: {output_file}"
    )

    print(
        f"Transition CSV saved: {csv_file}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    compare_years(
        2018,
        2022
    )

    compare_years(
        2022,
        2025
    )

    compare_years(
        2018,
        2025
    )

    print()
    print("=" * 60)
    print(
        "Change detection completed successfully!"
    )
    print("=" * 60)