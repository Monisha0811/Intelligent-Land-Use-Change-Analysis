from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

root = Path(
    "data/raw/2022_backup/"
    "S2B_MSIL2A_20221225T053239_N0510_R105_T43PDQ_20240805T142214.SAFE/"
    "GRANULE/L2A_T43PDQ_A030304_20221225T054420/IMG_DATA"
)

band_files = [
    root / "R10m/T43PDQ_20221225T053239_B02_10m.jp2",
    root / "R10m/T43PDQ_20221225T053239_B03_10m.jp2",
    root / "R10m/T43PDQ_20221225T053239_B04_10m.jp2",
    root / "R10m/T43PDQ_20221225T053239_B08_10m.jp2",
    root / "R20m/T43PDQ_20221225T053239_B11_20m.jp2",
]
names = ["Blue", "Green", "Red", "NIR", "SWIR"]

primary_path = Path("data/processed/mangaluru/sentinel2_2022_mangaluru.tif")
output_path = Path("data/processed/mangaluru/sentinel2_2022_backup_mangaluru.tif")

with rasterio.open(primary_path) as target:
    profile = target.profile.copy()
    profile.update(
        count=5,
        dtype="float32",
        nodata=0.0,
        compress="deflate"
    )

    result = np.zeros((5, target.height, target.width), dtype=np.float32)

    for i, band_path in enumerate(band_files):
        if not band_path.exists():
            raise FileNotFoundError(f"Missing band: {band_path}")

        with rasterio.open(band_path) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=result[i],
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=target.transform,
                dst_crs=target.crs,
                dst_nodata=0.0,
                resampling=Resampling.bilinear,
                init_dest_nodata=True
            )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(result)
        for i, name in enumerate(names, 1):
            dst.set_band_description(i, name)

print("Rebuilt experimental raster:", output_path)
for i, name in enumerate(names):
    band = result[i]
    valid = band > 0
    print(
        f"{name}: valid={int(valid.sum())}, "
        f"min={float(band[valid].min()) if valid.any() else 'N/A'}, "
        f"median={float(np.median(band[valid])) if valid.any() else 'N/A'}, "
        f"max={float(band[valid].max()) if valid.any() else 'N/A'}"
    )
