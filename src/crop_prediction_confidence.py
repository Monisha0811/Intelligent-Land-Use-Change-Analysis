from pathlib import Path
import csv
import joblib
import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parent.parent
IMAGE = ROOT / "data/processed/mangaluru/sentinel2_2022_mangaluru.tif"
LULC = ROOT / "data/classified/mangaluru/land_use_2022.tif"
MODEL_PATH = ROOT / "models" / "crop_random_forest_2022.joblib"
OUT = ROOT / "data/results/crop_reference"
OUT.mkdir(parents=True, exist_ok=True)

CONF_RASTER = OUT / "crop_confidence_2022_experimental.tif"
SUMMARY = OUT / "crop_confidence_summary_2022_experimental.csv"

saved = joblib.load(MODEL_PATH)
model = saved["model"] if isinstance(saved, dict) else saved
features = ["blue", "green", "red", "nir", "swir", "ndvi", "ndwi", "ndbi"]

if hasattr(model, "feature_names_in_") and list(model.feature_names_in_) != features:
    raise RuntimeError("Model feature order does not match the script.")
if not hasattr(model, "predict_proba"):
    raise RuntimeError("Model does not support predict_proba.")

classes = list(model.classes_)
bands = {"High": 0, "Medium": 0, "Low": 0}
class_counts = {str(c): 0 for c in classes}
count = 0
probability_sum = 0.0

with rasterio.open(IMAGE) as img, rasterio.open(LULC) as lulc:
    if (img.shape != lulc.shape or img.transform != lulc.transform
            or img.crs != lulc.crs):
        raise RuntimeError("Input rasters are not aligned.")

    profile = img.profile.copy()
    profile.update(count=1, dtype="float32", nodata=-1.0, compress="lzw")

    with rasterio.open(CONF_RASTER, "w", **profile) as dst:
        for row in range(0, img.height, 256):
            height = min(256, img.height - row)
            win = Window(0, row, img.width, height)
            data = img.read([1, 2, 3, 4, 5], window=win).astype(np.float32)
            land = lulc.read(1, window=win)
            blue, green, red, nir, swir = data

            valid = (
                np.isfinite(blue) & np.isfinite(green) & np.isfinite(red)
                & np.isfinite(nir) & np.isfinite(swir)
                & ((blue != 0) | (green != 0) | (red != 0)
                   | (nir != 0) | (swir != 0))
            )
            mask = (land == 3) & valid
            result = np.full((height, img.width), -1.0, dtype=np.float32)

            if np.any(mask):
                ndvi = (nir - red) / (nir + red + 1e-9)
                ndwi = (green - nir) / (green + nir + 1e-9)
                ndbi = (swir - nir) / (swir + nir + 1e-9)
                idx = np.where(mask)

                X = np.column_stack([
                    blue[idx], green[idx], red[idx], nir[idx], swir[idx],
                    ndvi[idx], ndwi[idx], ndbi[idx]
                ]).astype(np.float32)

                probs = model.predict_proba(X)
                best_idx = np.argmax(probs, axis=1)
                confidence = probs[np.arange(len(best_idx)), best_idx]
                labels = np.asarray(classes)[best_idx]
                result[idx] = confidence

                bands["High"] += int(np.count_nonzero(confidence >= 0.80))
                bands["Medium"] += int(np.count_nonzero(
                    (confidence >= 0.60) & (confidence < 0.80)))
                bands["Low"] += int(np.count_nonzero(confidence < 0.60))
                for label in classes:
                    class_counts[str(label)] += int(np.count_nonzero(labels == label))
                count += len(confidence)
                probability_sum += float(confidence.sum())

            dst.write(result, 1, window=win)
            print(f"Processed rows {min(row + height, img.height)}/{img.height}")

with open(SUMMARY, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Classified pixels", count])
    writer.writerow(["Mean maximum class probability",
                     round(probability_sum / count, 4) if count else "N/A"])
    writer.writerow(["High confidence (>=0.80)", bands["High"]])
    writer.writerow(["Medium confidence (0.60 to <0.80)", bands["Medium"]])
    writer.writerow(["Low confidence (<0.60)", bands["Low"]])
    for label, value in class_counts.items():
        writer.writerow([f"Predicted {label} pixels", value])

print("\\nConfidence analysis complete.")
print("Confidence raster:", CONF_RASTER)
print("Summary CSV:", SUMMARY)
print("Reminder: model probability is not verified classification accuracy.")
