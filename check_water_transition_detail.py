import os
import numpy as np
import rasterio

BASE_DIR = r"D:\Intelligent_Land_Use_Change_Analysis"

CLASSIFIED_DIR = os.path.join(
    BASE_DIR, "data", "classified", "mangaluru"
)

CLASS_NAMES = {
    1: "Water",
    2: "Vegetation/Forest",
    3: "Agriculture",
    4: "Built-up",
    5: "Barren/Open"
}

for start_year, end_year in [
    (2018, 2022),
    (2022, 2025),
    (2018, 2025)
]:

    with rasterio.open(
        os.path.join(CLASSIFIED_DIR, f"land_use_{start_year}.tif")
    ) as src:
        start = src.read(1)

    with rasterio.open(
        os.path.join(CLASSIFIED_DIR, f"land_use_{end_year}.tif")
    ) as src:
        end = src.read(1)

    valid = (start > 0) & (end > 0)

    print(f"\n{start_year} to {end_year}")
    print("=" * 60)
    print("Changes TO Water:")

    for class_id, class_name in CLASS_NAMES.items():

        count = np.sum(
            valid & (start == class_id) & (end == 1)
        )

        print(
            f"{class_name} -> Water: "
            f"{count:,} pixels "
            f"({count * 0.0001:.2f} km²)"
        )

    print("\nChanges FROM Water:")

    for class_id, class_name in CLASS_NAMES.items():

        count = np.sum(
            valid & (start == 1) & (end == class_id)
        )

        print(
            f"Water -> {class_name}: "
            f"{count:,} pixels "
            f"({count * 0.0001:.2f} km²)"
        )