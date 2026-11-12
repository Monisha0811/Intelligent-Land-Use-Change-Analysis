import os
import numpy as np
import pandas as pd
import rasterio

BASE_DIR = r"D:\Intelligent_Land_Use_Change_Analysis"

YEARS = [2018, 2022, 2025]

CLASSIFIED_DIR = os.path.join(
    BASE_DIR, "data", "classified", "mangaluru"
)

CONFIDENCE_DIR = os.path.join(
    BASE_DIR, "data", "results", "confidence"
)

for year in YEARS:

    lulc_path = os.path.join(
        CLASSIFIED_DIR, f"land_use_{year}.tif"
    )

    confidence_path = os.path.join(
        CONFIDENCE_DIR, f"lulc_confidence_{year}.tif"
    )

    with rasterio.open(lulc_path) as lulc_src, \
         rasterio.open(confidence_path) as conf_src:

        lulc = lulc_src.read(1)
        confidence = conf_src.read(1)

    water_mask = (lulc == 1) & (confidence > 0)
    water_confidence = confidence[water_mask]

    print(f"\nYEAR: {year}")
    print("Water pixels:", len(water_confidence))
    print("Mean confidence:", np.mean(water_confidence))
    print("Median confidence:", np.median(water_confidence))
    print("Low confidence (<0.60):",
          np.mean(water_confidence < 0.60) * 100, "%")
    print("Medium confidence (0.60-0.80):",
          np.mean((water_confidence >= 0.60) &
                  (water_confidence < 0.80)) * 100, "%")
    print("High confidence (>=0.80):",
          np.mean(water_confidence >= 0.80) * 100, "%")