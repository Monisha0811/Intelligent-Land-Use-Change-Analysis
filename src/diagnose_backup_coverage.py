import rasterio
import numpy as np

path = (
    "data/raw/2022_backup/"
    "S2B_MSIL2A_20221225T053239_N0510_R105_T43PDQ_20240805T142214.SAFE/"
    "GRANULE/L2A_T43PDQ_A030304_20221225T054420/IMG_DATA/R10m/"
    "T43PDQ_20221225T053239_B02_10m.jp2"
)

with rasterio.open(path) as src:
    print("Tile bounds:", src.bounds)

    sample = src.read(1, window=((1000, 1010), (1000, 1010)))
    print("\nPreviously tested sample:", sample.min(), sample.max())

    # Check a few windows across the source tile.
    for row, col in [(5752, 7670), (5000, 7000), (7000, 8000), (1000, 1000)]:
        data = src.read(1, window=((row, row + 100), (col, col + 100)))
        print(
            f"Window row={row}, col={col}: "
            f"min={data.min()}, max={data.max()}, "
            f"nonzero={np.count_nonzero(data)}"
        )

    # Count nonzero pixels in manageable blocks.
    nonzero = 0
    total = 0
    for _, window in src.block_windows(1):
        data = src.read(1, window=window)
        nonzero += np.count_nonzero(data)
        total += data.size

    print(f"\nWhole tile nonzero pixels: {nonzero:,} / {total:,}")
