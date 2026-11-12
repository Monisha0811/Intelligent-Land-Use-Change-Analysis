from pathlib import Path
import rasterio

root = Path("data/raw/2022_backup")
files = list(root.rglob("*_B02_10m.jp2"))

if not files:
    print("B02 10m band not found.")
else:
    for path in files:
        print("File:", path)
        with rasterio.open(path) as ds:
            print("CRS:", ds.crs)
            print("Dimensions:", ds.width, "x", ds.height)
            print("Resolution:", ds.res)
            print("Bounds:", ds.bounds)
            print("Data type:", ds.dtypes[0])
            print("NoData:", ds.nodata)
