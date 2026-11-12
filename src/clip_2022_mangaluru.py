import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask


BASE_DIR = r"D:\Intelligent_Land_Use_Change_Analysis"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "sentinel2_2022.tif"
)

BOUNDARY_FILE = os.path.join(
    BASE_DIR,
    "data",
    "boundary",
    "mangaluru_taluk.gpkg"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "mangaluru"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "sentinel2_2022_mangaluru.tif"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

boundary = gpd.read_file(BOUNDARY_FILE)

with rasterio.open(INPUT_FILE) as src:

    boundary = boundary.to_crs(src.crs)

    clipped, transform = mask(
        src,
        boundary.geometry,
        crop=True,
        nodata=0
    )

    profile = src.profile.copy()

    profile.update(
        height=clipped.shape[1],
        width=clipped.shape[2],
        transform=transform,
        nodata=0,
        compress="lzw"
    )

    with rasterio.open(
        OUTPUT_FILE,
        "w",
        **profile
    ) as dst:

        dst.write(clipped)

        for i, description in enumerate(
            src.descriptions,
            start=1
        ):
            if description:
                dst.set_band_description(
                    i,
                    description
                )

print("Saved:")
print(OUTPUT_FILE)
print("Size:", clipped.shape[2], "x", clipped.shape[1])
print("Bands:", clipped.shape[0])