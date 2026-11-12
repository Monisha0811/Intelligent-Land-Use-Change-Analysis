import rasterio
import numpy as np
from rasterio.windows import from_bounds

src_path = (
    "data/raw/2022_backup/"
    "S2B_MSIL2A_20221225T053239_N0510_R105_T43PDQ_20240805T142214.SAFE/"
    "GRANULE/L2A_T43PDQ_A030304_20221225T054420/IMG_DATA/R10m/"
    "T43PDQ_20221225T053239_B02_10m.jp2"
)
target_path = "data/processed/mangaluru/sentinel2_2022_mangaluru.tif"

with rasterio.open(target_path) as target, rasterio.open(src_path) as src:
    window = from_bounds(
        *target.bounds,
        transform=src.transform
    ).round_offsets().round_lengths()

    print("Source window:", window)
    print("Source window bounds:", rasterio.windows.bounds(window, src.transform))

    sample = src.read(1, window=window)
    print("Window shape:", sample.shape)
    print("Min:", sample.min(), "Max:", sample.max())
    print("Unique values (first 20):", np.unique(sample)[:20])
    print("Nonzero pixels:", np.count_nonzero(sample))
