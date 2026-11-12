import rasterio
import numpy as np
from pathlib import Path

base = Path(
    "data/raw/2022_backup/"
    "S2B_MSIL2A_20221225T053239_N0510_R105_T43PDQ_20240805T142214.SAFE/"
    "GRANULE/L2A_T43PDQ_A030304_20221225T054420/IMG_DATA"
)

bands = [
    ("Blue", base / "R10m/T43PDQ_20221225T053239_B02_10m.jp2"),
    ("Green", base / "R10m/T43PDQ_20221225T053239_B03_10m.jp2"),
    ("Red", base / "R10m/T43PDQ_20221225T053239_B04_10m.jp2"),
    ("NIR", base / "R10m/T43PDQ_20221225T053239_B08_10m.jp2"),
    ("SWIR", base / "R20m/T43PDQ_20221225T053239_B11_20m.jp2"),
]

# Study-area center in EPSG:32643.
x, y = (476660 + 501340) / 2, (1418190 + 1442480) / 2

for name, path in bands:
    with rasterio.open(path) as src:
        row, col = src.index(x, y)
        data = src.read(
            1,
            window=((row - 50, row + 50), (col - 50, col + 50))
        )
        print(
            f"{name}: CRS={src.crs}, resolution={src.res}, "
            f"sample min={data.min()}, max={data.max()}, "
            f"nonzero={np.count_nonzero(data)}/{data.size}"
        )
