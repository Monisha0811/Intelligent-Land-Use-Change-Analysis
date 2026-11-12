import rasterio
import numpy as np
from pathlib import Path

src_path = Path(
    "data/raw/2022_backup/"
    "S2B_MSIL2A_20221225T053239_N0510_R105_T43PDQ_20240805T142214.SAFE/"
    "GRANULE/L2A_T43PDQ_A030304_20221225T054420/IMG_DATA/R10m/"
    "T43PDQ_20221225T053239_B02_10m.jp2"
)

with rasterio.open(src_path) as src:
    print("Source dimensions:", src.width, src.height)
    print("Source transform:", src.transform)
    print("Source CRS:", src.crs)

    # Read a small source window directly.
    sample = src.read(1, window=((1000, 1010), (1000, 1010)))
    print("\nSource 10x10 sample:")
    print(sample)
    print("Source sample min/max:", sample.min(), sample.max())
    print("Unique sample values:", np.unique(sample)[:20])

    # Test a normal direct read of the same area.
    full_sample = src.read(1)[1000:1010, 1000:1010]
    print("\nDirect-read test matches:", np.array_equal(sample, full_sample))
