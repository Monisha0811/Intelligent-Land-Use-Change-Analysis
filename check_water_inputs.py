import os
import rasterio

BASE_DIR = r"D:\Intelligent_Land_Use_Change_Analysis"

for year in [2018, 2022, 2025]:

    path = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "mangaluru",
        f"sentinel2_{year}_mangaluru.tif"
    )

    print("\n" + "=" * 60)
    print("YEAR:", year)
    print("=" * 60)

    with rasterio.open(path) as src:

        print("CRS:", src.crs)
        print("Dimensions:", src.width, "x", src.height)
        print("Band count:", src.count)
        print("Pixel size:", src.res)
        print("Bounds:", src.bounds)
        print("NoData:", src.nodata)
        print("Transform:", src.transform)

        print("Band descriptions:", src.descriptions)

        print("Tags:", src.tags())