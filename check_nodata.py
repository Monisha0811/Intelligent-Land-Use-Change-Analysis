import rasterio
import numpy as np
from pathlib import Path

base = Path(__file__).resolve().parent
folder = base / "data" / "processed" / "mangaluru"
classified_folder = base / "data" / "classified" / "mangaluru"

for year in [2018, 2022, 2025]:
    raster_path = folder / f"sentinel2_{year}_mangaluru.tif"
    class_path = classified_folder / f"land_use_{year}.tif"

    with rasterio.open(raster_path) as src, rasterio.open(class_path) as cls:
        classification = cls.read(1)
        study_mask = classification > 0

        print(f"\nYear: {year}")

        for band in range(1, 6):
            data = src.read(band)
            zero_count = np.sum(study_mask & (data == 0))
            print(f"Band {band} zero pixels inside study area: {zero_count:,}")