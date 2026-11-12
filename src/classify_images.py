import os
import numpy as np
import rasterio
import joblib


BASE_DIR = r"D:\Intelligent_Land_Use_Change_Analysis"

PROCESSED_DIR = os.path.join(
    BASE_DIR, "data", "processed"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "data", "classified"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "model",
    "land_use_random_forest.joblib"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


YEARS = [2018, 2022, 2025]


def calculate_indices(data):
    """Calculate NDVI, NDWI and NDBI."""

    blue = data[0].astype(np.float32)
    green = data[1].astype(np.float32)
    red = data[2].astype(np.float32)
    nir = data[3].astype(np.float32)
    swir = data[4].astype(np.float32)

    ndvi = (nir - red) / (nir + red + 1e-10)
    ndwi = (green - nir) / (green + nir + 1e-10)
    ndbi = (swir - nir) / (swir + nir + 1e-10)

    return ndvi, ndwi, ndbi


def classify_year(year, model):

    input_file = os.path.join(
        PROCESSED_DIR,
        f"sentinel2_{year}.tif"
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        f"land_use_{year}.tif"
    )

    print(f"\nClassifying {year}...")
    print(f"Input: {input_file}")

    with rasterio.open(input_file) as src:

        profile = src.profile.copy()

        # Classification contains one band.
        profile.update(
            count=1,
            dtype="uint8",
            compress="lzw",
            nodata=0
        )

        with rasterio.open(
            output_file,
            "w",
            **profile
        ) as dst:

            total_windows = 0

            for _, window in src.block_windows(1):

                data = src.read(window=window)

                if data.size == 0:
                    continue

                ndvi, ndwi, ndbi = calculate_indices(data)

                # Flatten bands
                blue = data[0].astype(np.float32).ravel()
                green = data[1].astype(np.float32).ravel()
                red = data[2].astype(np.float32).ravel()
                nir = data[3].astype(np.float32).ravel()
                swir = data[4].astype(np.float32).ravel()

                ndvi = ndvi.ravel()
                ndwi = ndwi.ravel()
                ndbi = ndbi.ravel()

                # Valid pixels
                valid = (
                    np.isfinite(blue) &
                    np.isfinite(green) &
                    np.isfinite(red) &
                    np.isfinite(nir) &
                    np.isfinite(swir) &
                    np.isfinite(ndvi) &
                    np.isfinite(ndwi) &
                    np.isfinite(ndbi) &
                    (blue > 0) &
                    (green > 0) &
                    (red > 0) &
                    (nir > 0) &
                    (swir > 0)
                )

                # Create output classification
                result = np.zeros(
                    len(blue),
                    dtype=np.uint8
                )

                if np.any(valid):

                    features = np.column_stack([
                        blue[valid],
                        green[valid],
                        red[valid],
                        nir[valid],
                        swir[valid],
                        ndvi[valid],
                        ndwi[valid],
                        ndbi[valid]
                    ])

                    predictions = model.predict(
                        features
                    )

                    result[valid] = predictions

                # Restore window shape
                result = result.reshape(
                    window.height,
                    window.width
                )

                dst.write(
                    result,
                    1,
                    window=window
                )

                total_windows += 1

                if total_windows % 100 == 0:
                    print(
                        f"Processed {total_windows} windows..."
                    )

    print(f"Saved: {output_file}")


def main():

    print("Loading Random Forest model...")

    model = joblib.load(MODEL_FILE)

    print("Model loaded successfully.")

    for year in YEARS:
        classify_year(year, model)

    print("\nAll years classified successfully!")


if __name__ == "__main__":
    main()