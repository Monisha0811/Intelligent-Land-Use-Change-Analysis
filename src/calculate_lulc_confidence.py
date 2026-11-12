from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import joblib


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE / "model" / "land_use_random_forest.joblib"

PROCESSED_DIR = BASE / "data" / "processed" / "mangaluru"

CLASSIFIED_DIR = BASE / "data" / "classified" / "mangaluru"

OUTPUT_DIR = BASE / "data" / "results" / "confidence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

YEARS = [2018, 2022, 2025]

FEATURE_NAMES = [
    "blue",
    "green",
    "red",
    "nir",
    "swir",
    "ndvi",
    "ndwi",
    "ndbi",
]


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading Random Forest model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")
print("Number of trees:", len(model.estimators_))
print("Expected features:", model.n_features_in_)

if model.n_features_in_ != 8:
    raise ValueError(
        f"Model expects {model.n_features_in_} features, "
        f"but this script is configured for 8."
    )


# ============================================================
# PROCESS EACH YEAR
# ============================================================

for year in YEARS:

    print()
    print("=" * 60)
    print(f"Processing {year}")
    print("=" * 60)

    raster_path = (
        PROCESSED_DIR /
        f"sentinel2_{year}_mangaluru.tif"
    )

    classified_path = (
        CLASSIFIED_DIR /
        f"land_use_{year}.tif"
    )

    print("Raster:", raster_path)
    print("Classification mask:", classified_path)

    # --------------------------------------------------------
    # Open Sentinel-2 raster
    # --------------------------------------------------------

    with rasterio.open(raster_path) as src:

        print("Shape:", src.height, "x", src.width)
        print("Bands:", src.count)
        print("CRS:", src.crs)

        if src.count != 5:
            raise ValueError(
                f"Expected 5 spectral bands "
                f"(Blue, Green, Red, NIR, SWIR), "
                f"but found {src.count}"
            )

        # ----------------------------------------------------
        # Read existing LULC classification mask
        # ----------------------------------------------------

        with rasterio.open(classified_path) as cls_src:

            if (
                cls_src.height != src.height or
                cls_src.width != src.width
            ):
                raise ValueError(
                    f"Raster dimensions do not match for {year}"
                )

            classification = cls_src.read(1)

        # Existing classified pixels are > 0
        study_mask = classification > 0

        study_pixel_count = int(np.sum(study_mask))

        print(
            "Valid study-area pixels:",
            f"{study_pixel_count:,}"
        )

        if study_pixel_count == 0:
            raise ValueError(
                f"No valid study-area pixels found for {year}"
            )

        # ----------------------------------------------------
        # Read five spectral bands
        # ----------------------------------------------------

        blue = src.read(1).astype(np.float32)
        green = src.read(2).astype(np.float32)
        red = src.read(3).astype(np.float32)
        nir = src.read(4).astype(np.float32)
        swir = src.read(5).astype(np.float32)

        profile = src.profile.copy()

      
        # ----------------------------------------------------
        # Calculate NDVI
        # ----------------------------------------------------

        ndvi_den = nir + red

        ndvi = np.divide(
            nir - red,
            ndvi_den,
            out=np.zeros_like(nir, dtype=np.float32),
            where=ndvi_den != 0
        )

        # ----------------------------------------------------
        # Calculate NDWI
        # ----------------------------------------------------

        ndwi_den = green + nir

        ndwi = np.divide(
            green - nir,
            ndwi_den,
            out=np.zeros_like(green, dtype=np.float32),
            where=ndwi_den != 0
        )

        # ----------------------------------------------------
        # Calculate NDBI
        # ----------------------------------------------------

        ndbi_den = swir + nir

        ndbi = np.divide(
            swir - nir,
            ndbi_den,
            out=np.zeros_like(swir, dtype=np.float32),
            where=ndbi_den != 0
        )

        # ----------------------------------------------------
        # Find valid study-area pixels
        # ----------------------------------------------------

        valid_mask = (
            study_mask &
            np.isfinite(blue) &
            np.isfinite(green) &
            np.isfinite(red) &
            np.isfinite(nir) &
            np.isfinite(swir) &
            np.isfinite(ndvi) &
            np.isfinite(ndwi) &
            np.isfinite(ndbi)
        )

        rows, cols = np.where(valid_mask)

        print(
            "Pixels used for confidence:",
            f"{len(rows):,}"
        )

        # ----------------------------------------------------
        # Build feature matrix
        # ----------------------------------------------------

        X = pd.DataFrame(
            {
                "blue": blue[rows, cols],
                "green": green[rows, cols],
                "red": red[rows, cols],
                "nir": nir[rows, cols],
                "swir": swir[rows, cols],
                "ndvi": ndvi[rows, cols],
                "ndwi": ndwi[rows, cols],
                "ndbi": ndbi[rows, cols],
            }
        )

        X = X[FEATURE_NAMES]

        # ----------------------------------------------------
        # Random Forest probabilities
        # ----------------------------------------------------

        print("Calculating Random Forest probabilities...")

        probabilities = model.predict_proba(X)

        # Classes stored in the existing LULC map
        predicted_classes = classification[rows, cols]

        # Compare saved map with Random Forest predictions
        model_predictions = model.predict(X)

        agreement = np.mean(model_predictions == predicted_classes)

        print(
            f"Model and saved map agreement: "
            f"{agreement * 100:.2f}%"
        )

        class_to_index = {
            class_id: index
            for index, class_id in enumerate(model.classes_)
        }

        # Confidence for the class stored in the saved LULC map
        confidence = np.array([
            probabilities[i, class_to_index[class_id]]
            for i, class_id in enumerate(predicted_classes)
        ], dtype=np.float32)
        # ----------------------------------------------------
        # Full-size output arrays
        # ----------------------------------------------------

        confidence_array = np.zeros(
            (src.height, src.width),
            dtype=np.float32
        )

        confidence_class_array = np.zeros(
            (src.height, src.width),
            dtype=np.uint8
        )

        confidence_array[rows, cols] = confidence

        # ----------------------------------------------------
        # Confidence categories
        #
        # 0 = NoData
        # 1 = Low       < 0.60
        # 2 = Medium    0.60 - <0.80
        # 3 = High      >=0.80
        # ----------------------------------------------------

        confidence_categories = np.zeros(
            len(confidence),
            dtype=np.uint8
        )

        confidence_categories[
            confidence < 0.60
        ] = 1

        confidence_categories[
            (confidence >= 0.60) &
            (confidence < 0.80)
        ] = 2

        confidence_categories[
            confidence >= 0.80
        ] = 3

        confidence_class_array[
            rows,
            cols
        ] = confidence_categories

        # ----------------------------------------------------
        # Save continuous confidence raster
        # ----------------------------------------------------

        confidence_profile = profile.copy()

        confidence_profile.update(
            dtype="float32",
            count=1,
            nodata=0
        )

        confidence_path = (
            OUTPUT_DIR /
            f"lulc_confidence_{year}.tif"
        )

        with rasterio.open(
            confidence_path,
            "w",
            **confidence_profile
        ) as dst:

            dst.write(
                confidence_array,
                1
            )

            dst.set_band_description(
                1,
                "LULC Classification Confidence"
            )

        # ----------------------------------------------------
        # Save confidence category raster
        # ----------------------------------------------------

        category_profile = profile.copy()

        category_profile.update(
            dtype="uint8",
            count=1,
            nodata=0
        )

        category_path = (
            OUTPUT_DIR /
            f"lulc_confidence_class_{year}.tif"
        )

        with rasterio.open(
            category_path,
            "w",
            **category_profile
        ) as dst:

            dst.write(
                confidence_class_array,
                1
            )

            dst.set_band_description(
                1,
                "Confidence Category"
            )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        low_count = int(
            np.sum(confidence_categories == 1)
        )

        medium_count = int(
            np.sum(confidence_categories == 2)
        )

        high_count = int(
            np.sum(confidence_categories == 3)
        )

        total = len(confidence_categories)

        stats = pd.DataFrame(
            {
                "year": [year],
                "total_valid_pixels": [total],
                "low_pixels": [low_count],
                "medium_pixels": [medium_count],
                "high_pixels": [high_count],
                "low_percent": [
                    low_count / total * 100
                ],
                "medium_percent": [
                    medium_count / total * 100
                ],
                "high_percent": [
                    high_count / total * 100
                ],
                "mean_confidence": [
                    float(np.mean(confidence))
                ],
                "median_confidence": [
                    float(np.median(confidence))
                ],
                "minimum_confidence": [
                    float(np.min(confidence))
                ],
                "maximum_confidence": [
                    float(np.max(confidence))
                ],
            }
        )

        stats_path = (
            OUTPUT_DIR /
            f"confidence_statistics_{year}.csv"
        )

        stats.to_csv(
            stats_path,
            index=False
        )

        # ----------------------------------------------------
        # Print summary
        # ----------------------------------------------------

        print()
        print("Confidence summary:")

        print(
            f"Low:    {low_count:,} "
            f"({low_count / total * 100:.2f}%)"
        )

        print(
            f"Medium: {medium_count:,} "
            f"({medium_count / total * 100:.2f}%)"
        )

        print(
            f"High:   {high_count:,} "
            f"({high_count / total * 100:.2f}%)"
        )

        print(
            f"Mean confidence: "
            f"{np.mean(confidence):.4f}"
        )

        print(
            f"Median confidence: "
            f"{np.median(confidence):.4f}"
        )

        print("Saved:", confidence_path)
        print("Saved:", category_path)
        print("Saved:", stats_path)


print()
print("=" * 60)
print("LULC CONFIDENCE ANALYSIS COMPLETED")
print("=" * 60)