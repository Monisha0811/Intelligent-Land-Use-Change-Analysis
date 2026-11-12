import os
import numpy as np
import pandas as pd
import rasterio

BASE_DIR = r"D:\Intelligent_Land_Use_Change_Analysis"

CLASS_NAMES = {
    1: "Water",
    2: "Vegetation/Forest",
    3: "Agriculture",
    4: "Built-up",
    5: "Barren/Open"
}

YEARS = [2018, 2022, 2025]

CONFIDENCE_DIR = os.path.join(
    BASE_DIR, "data", "results", "confidence"
)

CLASSIFIED_DIR = os.path.join(
    BASE_DIR, "data", "classified", "mangaluru"
)

results = []

for year in YEARS:

    lulc_path = os.path.join(
        CLASSIFIED_DIR,
        f"land_use_{year}.tif"
    )

    confidence_path = os.path.join(
        CONFIDENCE_DIR,
        f"lulc_confidence_{year}.tif"
    )

    print(f"\nProcessing {year}...")

    with rasterio.open(lulc_path) as lulc_src, \
         rasterio.open(confidence_path) as conf_src:

        lulc = lulc_src.read(1)
        confidence = conf_src.read(1)

    for class_id, class_name in CLASS_NAMES.items():

        mask = (
            (lulc == class_id) &
            (confidence > 0)
        )

        values = confidence[mask]

        if len(values) == 0:
            continue

        low = np.sum(values < 0.60)
        medium = np.sum(
            (values >= 0.60) &
            (values < 0.80)
        )
        high = np.sum(values >= 0.80)

        total = len(values)

        results.append({
            "Year": year,
            "Class ID": class_id,
            "Class": class_name,
            "Pixel Count": total,
            "Mean Confidence": np.mean(values),
            "Median Confidence": np.median(values),
            "Minimum Confidence": np.min(values),
            "Maximum Confidence": np.max(values),
            "Low < 0.60 (%)": (low / total) * 100,
            "Medium 0.60-<0.80 (%)": (medium / total) * 100,
            "High >= 0.80 (%)": (high / total) * 100
        })

results_df = pd.DataFrame(results)

output_path = os.path.join(
    CONFIDENCE_DIR,
    "confidence_by_class.csv"
)

results_df.to_csv(output_path, index=False)

print("\n" + "=" * 90)
print("CONFIDENCE BY LULC CLASS")
print("=" * 90)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nSaved to:")
print(output_path)