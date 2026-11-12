from pathlib import Path
import rasterio
import numpy as np

root = Path(
    "data/raw/2022_backup/"
    "S2B_MSIL2A_20221225T053239_N0510_R105_T43PDQ_20240805T142214.SAFE/"
    "GRANULE/L2A_T43PDQ_A030304_20221225T054420/IMG_DATA"
)

bands = [
    ("B02", root / "R10m/T43PDQ_20221225T053239_B02_10m.jp2"),
    ("B03", root / "R10m/T43PDQ_20221225T053239_B03_10m.jp2"),
    ("B04", root / "R10m/T43PDQ_20221225T053239_B04_10m.jp2"),
    ("B08", root / "R10m/T43PDQ_20221225T053239_B08_10m.jp2"),
    ("B11", root / "R20m/T43PDQ_20221225T053239_B11_20m.jp2"),
]

for name, path in bands:
    print(f"\n{name}: {path.name}")
    with rasterio.open(path) as ds:
        print("dtype:", ds.dtypes[0], "nodata:", ds.nodata)
        print("scale:", ds.scales, "offset:", ds.offsets)
        arr = ds.read(1, masked=True)
        valid = arr.compressed()
        if valid.size:
            print("valid pixels:", valid.size)
            print("min/max:", float(valid.min()), float(valid.max()))
            print("percentiles:", np.percentile(valid, [1, 50, 99]))
        else:
            print("No valid pixels")
