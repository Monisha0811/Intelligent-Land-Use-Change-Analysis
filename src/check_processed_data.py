import os
import rasterio
import numpy as np

BASE_DIR = r"D:\Intelligent_Land_Use_Change_Analysis"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

YEARS = [2018, 2022, 2025]

for year in YEARS:

    file = os.path.join(
        PROCESSED_DIR,
        f"sentinel2_{year}.tif"
    )

    print("\n" + "=" * 60)
    print(f"YEAR: {year}")
    print("=" * 60)

    with rasterio.open(file) as src:

        print("Size:", src.width, "x", src.height)
        print("Bands:", src.count)
        print("CRS:", src.crs)
        print("Pixel size:", src.res)
        print("Bounds:", src.bounds)
        print("NoData:", src.nodata)

        data = src.read(1)

        valid = np.isfinite(data) & (data > 0)

        print("Valid pixels:", np.sum(valid))
        print("Total pixels:", data.size)

        percentage = (
            np.sum(valid) / data.size
        ) * 100

        print(f"Valid percentage: {percentage:.2f}%")

        print("Band 1 minimum:", np.min(data[valid]))
        print("Band 1 maximum:", np.max(data[valid]))
        print("Band 1 mean:", np.mean(data[valid]))
