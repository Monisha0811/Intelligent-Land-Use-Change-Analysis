import os
import numpy as np
import rasterio
import joblib


BASE_DIR = r"D:\Intelligent_Land_Use_Change_Analysis"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "mangaluru",
    "sentinel2_2025_mangaluru.tif"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "classified",
    "mangaluru"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "model",
    "land_use_random_forest.joblib"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "land_use_2025.tif"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_indices(data):

    blue = data[0].astype(np.float32)
    green = data[1].astype(np.float32)
    red = data[2].astype(np.float32)
    nir = data[3].astype(np.float32)
    swir = data[4].astype(np.float32)

    ndvi = (nir - red) / (nir + red + 1e-10)
    ndwi = (green - nir) / (green + nir + 1e-10)
    ndbi = (swir - nir) / (swir + nir + 1e-10)

    return ndvi, ndwi, ndbi


print("Loading Random Forest model...")

model = joblib.load(MODEL_FILE)

print("Model loaded:", type(model).__name__)

print("\nInput:", INPUT_FILE)
print("Output:", OUTPUT_FILE)


with rasterio.open(INPUT_FILE) as src:

    profile = src.profile.copy()

    profile.update(
        count=1,
        dtype="uint8",
        compress="lzw",
        nodata=0
    )

    with rasterio.open(
        OUTPUT_FILE,
        "w",
        **profile
    ) as dst:

        windows = 0

        for _, window in src.block_windows(1):

            data = src.read(window=window)

            ndvi, ndwi, ndbi = calculate_indices(data)

            blue = data[0].astype(np.float32).ravel()
            green = data[1].astype(np.float32).ravel()
            red = data[2].astype(np.float32).ravel()
            nir = data[3].astype(np.float32).ravel()
            swir = data[4].astype(np.float32).ravel()

            ndvi = ndvi.ravel()
            ndwi = ndwi.ravel()
            ndbi = ndbi.ravel()

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

                predictions = model.predict(features)

                result[valid] = predictions

            result = result.reshape(
                window.height,
                window.width
            )

            dst.write(
                result,
                1,
                window=window
            )

            windows += 1

            if windows % 20 == 0:
                print(
                    f"Processed {windows} windows..."
                )


print("\n2025 Mangaluru classification completed!")
print("Saved:", OUTPUT_FILE)