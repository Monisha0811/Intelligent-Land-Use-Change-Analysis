import os
import numpy as np
import pandas as pd
import rasterio
import joblib

from rasterio.windows import Window


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

IMAGE_2022 = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "mangaluru",
    "sentinel2_2022_mangaluru.tif"
)

LULC_2022 = os.path.join(
    PROJECT_ROOT,
    "data",
    "classified",
    "mangaluru",
    "land_use_2022.tif"
)


MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "crop_random_forest_2022.joblib"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "classified",
    "mangaluru"
)

OUTPUT_RASTER = os.path.join(
    OUTPUT_DIR,
    "crop_2022_experimental.tif"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "results",
    "crop_reference"
)

OUTPUT_STATS = os.path.join(
    RESULT_DIR,
    "crop_area_statistics_2022_experimental.csv"
)


FEATURES = [
    "blue",
    "green",
    "red",
    "nir",
    "swir",
    "ndvi",
    "ndwi",
    "ndbi"
]

CROP_NAMES = {
    1: "Coconut",
    2: "Arecanut",
    3: "Paddy"
}


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

print("Loading crop Random Forest model...")

saved_model = joblib.load(MODEL_PATH)
model = saved_model["model"] if isinstance(saved_model, dict) else saved_model

print("Model loaded successfully.")


# ---------------------------------------------------------
# Open input rasters
# ---------------------------------------------------------

print("\nOpening 2022 Sentinel-2 image...")

with rasterio.open(IMAGE_2022) as src_img, \
     rasterio.open(LULC_2022) as src_lulc:

    print("Image size:", src_img.width, "x", src_img.height)
    print("Number of bands:", src_img.count)
    print("CRS:", src_img.crs)

    if (
        src_img.width != src_lulc.width
        or src_img.height != src_lulc.height
        or src_img.transform != src_lulc.transform
    ):
        raise RuntimeError(
            "Sentinel-2 and LULC rasters are not aligned."
        )

    profile = src_img.profile.copy()

    profile.update(
        count=1,
        dtype="uint8",
        nodata=0,
        compress="lzw"
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    total_pixels = src_img.width * src_img.height
    agriculture_pixels = 0
    classified_crop_pixels = 0

    crop_counts = {
        1: 0,
        2: 0,
        3: 0
    }

    with rasterio.open(
        OUTPUT_RASTER,
        "w",
        **profile
    ) as dst:

        # Process in row blocks to avoid excessive memory use
        block_height = 256

        for row_start in range(
            0,
            src_img.height,
            block_height
        ):

            height = min(
                block_height,
                src_img.height - row_start
            )

            window = Window(
                col_off=0,
                row_off=row_start,
                width=src_img.width,
                height=height
            )

            data = src_img.read(
                indexes=[1, 2, 3, 4, 5],
                window=window
            ).astype(np.float32)

            lulc = src_lulc.read(
                1,
                window=window
            )

            blue = data[0]
            green = data[1]
            red = data[2]
            nir = data[3]
            swir = data[4]

            # Agriculture mask
            agriculture_mask = (
                lulc == 3
            )

            # Valid spectral pixels
            valid_mask = (
                np.isfinite(blue)
                & np.isfinite(green)
                & np.isfinite(red)
                & np.isfinite(nir)
                & np.isfinite(swir)
                & (
                    (blue != 0)
                    | (green != 0)
                    | (red != 0)
                    | (nir != 0)
                    | (swir != 0)
                )
            )

            crop_mask = (
                agriculture_mask
                & valid_mask
            )

            n_agriculture = int(
                np.count_nonzero(agriculture_mask)
            )

            n_valid_crop = int(
                np.count_nonzero(crop_mask)
            )

            agriculture_pixels += n_agriculture
            classified_crop_pixels += n_valid_crop

            output = np.zeros(
                (height, src_img.width),
                dtype=np.uint8
            )

            if n_valid_crop > 0:

                # Calculate spectral indices
                ndvi = (
                    (nir - red)
                    / (nir + red + 1e-9)
                )

                ndwi = (
                    (green - nir)
                    / (green + nir + 1e-9)
                )

                ndbi = (
                    (swir - nir)
                    / (swir + nir + 1e-9)
                )

                indices = np.where(
                    crop_mask
                )

                X = np.column_stack([
                    blue[indices],
                    green[indices],
                    red[indices],
                    nir[indices],
                    swir[indices],
                    ndvi[indices],
                    ndwi[indices],
                    ndbi[indices]
                ]).astype(np.float32)

              
		
                predictions = model.predict(X).astype(np.uint8)

                output[indices] = predictions

                unique, counts = np.unique(
                    predictions,
                    return_counts=True
                )

                for cls, count in zip(unique, counts):
                    cls = int(cls)
                    if cls in crop_counts:
                        crop_counts[cls] += int(count)

            dst.write(
                output,
                1,
                window=window
            )

            print(
                f"Processed rows "
                f"{row_start + height}/{src_img.height}"
            )


# ---------------------------------------------------------
# Save crop area statistics
# ---------------------------------------------------------

pixel_area_km2 = 0.0001
# 10 m x 10 m = 100 m² = 0.0001 km²

stats_rows = []

for crop_id, name in CROP_NAMES.items():

    pixels = crop_counts[crop_id]

    stats_rows.append({
        "crop_class_id": crop_id,
        "crop_class": name,
        "pixels": pixels,
        "area_km2": pixels * pixel_area_km2
    })

stats_df = pd.DataFrame(stats_rows)

total_crop_area = stats_df["area_km2"].sum()

if total_crop_area > 0:
    stats_df["percentage_of_mapped_crop_area"] = (
        stats_df["area_km2"]
        / total_crop_area
        * 100
    )
else:
    stats_df["percentage_of_mapped_crop_area"] = 0.0

stats_df.to_csv(
    OUTPUT_STATS,
    index=False
)


# ---------------------------------------------------------
# Print summary
# ---------------------------------------------------------

print("\n========================================")
print("2022 CROP CLASSIFICATION COMPLETE")
print("========================================")

print(
    f"Total raster pixels: {total_pixels:,}"
)

print(
    f"Agriculture pixels: {agriculture_pixels:,}"
)

print(
    f"Crop pixels classified: {classified_crop_pixels:,}"
)

print("\nCrop distribution:")

for _, row in stats_df.iterrows():

    print(
        f"{row['crop_class']}: "
        f"{int(row['pixels']):,} pixels | "
        f"{row['area_km2']:.4f} km² | "
        f"{row['percentage_of_mapped_crop_area']:.2f}%"
    )

print("\nSaved crop map:")
print(OUTPUT_RASTER)

print("\nSaved crop statistics:")
print(OUTPUT_STATS)

