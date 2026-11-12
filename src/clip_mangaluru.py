import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask


BASE_DIR = r"D:\Intelligent_Land_Use_Change_Analysis"

PROCESSED_DIR = os.path.join(
    BASE_DIR, "data", "processed"
)

OUTPUT_DIR = os.path.join(
    PROCESSED_DIR, "mangaluru"
)

BOUNDARY_FILE = os.path.join(
    BASE_DIR,
    "data",
    "boundary",
    "mangaluru_taluk.gpkg"
)

YEARS = [2018, 2022, 2025]

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("Loading Mangaluru boundary...")

boundary = gpd.read_file(BOUNDARY_FILE)

print("Boundary CRS:", boundary.crs)


for year in YEARS:

    input_file = os.path.join(
        PROCESSED_DIR,
        f"sentinel2_{year}.tif"
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        f"sentinel2_{year}_mangaluru.tif"
    )

    print(f"\nClipping {year}...")

    with rasterio.open(input_file) as src:

        # Make sure boundary uses the same CRS
        boundary_year = boundary.to_crs(src.crs)

        clipped, transform = mask(
            src,
            boundary_year.geometry,
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
            output_file,
            "w",
            **profile
        ) as dst:

            dst.write(clipped)

            for i, description in enumerate(src.descriptions, start=1):
                if description:
                    dst.set_band_description(
                        i,
                        description
                    )

    print("Saved:", output_file)
    print("Size:", clipped.shape[2], "x", clipped.shape[1])


print("\nMangaluru clipping completed successfully!")