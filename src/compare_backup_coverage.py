import rasterio
import numpy as np

paths = {
    "Primary 2022": "data/processed/mangaluru/sentinel2_2022_mangaluru.tif",
    "Backup B02": (
        "data/raw/2022_backup/"
        "S2B_MSIL2A_20221225T053239_N0510_R105_T43PDQ_20240805T142214.SAFE/"
        "GRANULE/L2A_T43PDQ_A030304_20221225T054420/IMG_DATA/R10m/"
        "T43PDQ_20221225T053239_B02_10m.jp2"
    )
}

for label, path in paths.items():
    with rasterio.open(path) as src:
        print(f"\n--- {label} ---")
        print("CRS:", src.crs)
        print("Bounds:", src.bounds)
        print("Dimensions:", src.width, "x", src.height)

        if label == "Primary 2022":
            data = src.read(1)
            print("Band 1 min/median/max:", np.min(data), np.median(data), np.max(data))
            print("Band 1 nonzero:", np.count_nonzero(data))
        else:
            # Examine the backup around the primary raster's geographic center.
            x = (476660 + 501340) / 2
            y = (1418190 + 1442480) / 2
            row, col = src.index(x, y)
            print("Study-area center pixel:", row, col)
            for dr, dc in [(0, 0), (-500, 0), (500, 0), (0, -500), (0, 500)]:
                r, c = row + dr, col + dc
                sample = src.read(1, window=((r, r + 100), (c, c + 100)))
                print(
                    f"Center offset ({dr},{dc}): "
                    f"min={sample.min()}, max={sample.max()}, "
                    f"nonzero={np.count_nonzero(sample)}"
                )
