import rasterio
from pathlib import Path

base = Path(__file__).resolve().parent

for year in [2018, 2022, 2025]:

    input_path = base / "data" / "processed" / "mangaluru" / f"sentinel2_{year}_mangaluru.tif"
    classified_path = base / "data" / "classified" / "mangaluru" / f"land_use_{year}.tif"

    with rasterio.open(input_path) as src, rasterio.open(classified_path) as cls:

        print(f"\nYear: {year}")
        print("CRS match:", src.crs == cls.crs)
        print("Transform match:", src.transform == cls.transform)
        print("Dimensions match:", (src.width, src.height) == (cls.width, cls.height))