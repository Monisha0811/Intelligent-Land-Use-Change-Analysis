import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
from pathlib import Path

src_path = Path(
    "data/raw/2022_backup/"
    "S2B_MSIL2A_20221225T053239_N0510_R105_T43PDQ_20240805T142214.SAFE/"
    "GRANULE/L2A_T43PDQ_A030304_20221225T054420/IMG_DATA/R10m/"
    "T43PDQ_20221225T053239_B02_10m.jp2"
)

target_path = Path("data/processed/mangaluru/sentinel2_2022_mangaluru.tif")

with rasterio.open(src_path) as src, rasterio.open(target_path) as target:
    # A 10x10 target window near the study area's upper-left corner.
    window = rasterio.windows.Window(0, 0, 10, 10)
    dst_transform = rasterio.windows.transform(window, target.transform)

    destination = np.zeros((10, 10), dtype=np.float32)

    reproject(
        source=rasterio.band(src, 1),
        destination=destination,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs=target.crs,
        resampling=Resampling.nearest,
        init_dest_nodata=True
    )

    print("Source bounds:", src.bounds)
    print("Target window bounds:", rasterio.windows.bounds(window, target.transform))
    print("\nReprojected sample:")
    print(destination)
    print("Min:", destination.min(), "Max:", destination.max())
    print("Unique values:", np.unique(destination)[:20])
