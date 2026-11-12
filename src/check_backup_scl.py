import rasterio
import numpy as np
from pathlib import Path
from collections import Counter

granule = Path(
    "data/raw/2022_backup/"
    "S2B_MSIL2A_20221225T053239_N0510_R105_T43PDQ_20240805T142214.SAFE/"
    "GRANULE/L2A_T43PDQ_A030304_20221225T054420/IMG_DATA"
)

files = list(granule.rglob("*SCL*20m.jp2"))
if not files:
    raise FileNotFoundError("No SCL 20m JP2 file found.")

scl_path = files[0]
print("SCL file:", scl_path)

x, y = (476660 + 501340) / 2, (1418190 + 1442480) / 2

with rasterio.open(scl_path) as src:
    print("CRS:", src.crs)
    print("Resolution:", src.res)
    print("Bounds:", src.bounds)

    row, col = src.index(x, y)
    sample = src.read(
        1,
        window=((row - 50, row + 50), (col - 50, col + 50))
    )

    print("Study-area center sample shape:", sample.shape)
    print("Unique SCL values and counts:", Counter(sample.ravel().tolist()))
    print("Min:", sample.min(), "Max:", sample.max())
