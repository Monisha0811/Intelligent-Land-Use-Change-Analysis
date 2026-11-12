from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.ndimage import generic_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parent.parent

RASTER_PATH = ROOT / "data/processed/mangaluru/sentinel2_2022_mangaluru.tif"
INPUT_CSV = ROOT / "data/results/crop_reference/crop_training_samples_2022_clean.csv"
OUTPUT_DIR = ROOT / "data/results/crop_reference"

REPORT_PATH = OUTPUT_DIR / "crop_neighborhood_experiment_2022.txt"
PREDICTIONS_PATH = OUTPUT_DIR / "crop_neighborhood_predictions_2022.csv"

BASE_FEATURES = [
    "blue", "green", "red", "nir",
    "swir", "ndvi", "ndwi", "ndbi"
]
BANDS = ["blue", "green", "red", "nir", "swir"]
FEATURES = BASE_FEATURES + [f"{band}_local_std" for band in BANDS]

CLASS_MAPPING = {"Coconut": 1, "Arecanut": 2, "Paddy": 3}
CLASS_NAMES = ["Coconut", "Arecanut", "Paddy"]


def local_std(window):
    return float(np.std(window))


print("Reading cleaned crop samples...")
df = pd.read_csv(INPUT_CSV)
df = df.drop_duplicates(
    subset=["latitude", "longitude", "crop_class"]
).reset_index(drop=True)

df = df[df["crop_class"].isin(CLASS_MAPPING)].copy()
df["target"] = df["crop_class"].map(CLASS_MAPPING).astype(np.uint8)

print("Samples:", len(df))
print(df["crop_class"].value_counts().to_string())


print("\nCalculating local standard deviation rasters...")
with rasterio.open(RASTER_PATH) as src:
    raster = src.read().astype(np.float32)
    transform = src.transform
    crs = src.crs
    width, height = src.width, src.height
    nodata = src.nodata

    valid = np.all(np.isfinite(raster), axis=0)
    if nodata is not None:
        valid &= np.all(raster != nodata, axis=0)
    valid &= np.any(raster != 0, axis=0)

    local_std_bands = []
    for band in raster:
        # Invalid pixels are excluded from statistics by replacing them
        # with NaN and using a NaN-aware local standard deviation.
        work = band.copy()
        work[~valid] = np.nan

        std_band = generic_filter(
            work,
            function=lambda values: (
                float(np.nanstd(values))
                if np.isfinite(values).any()
                else np.nan
            ),
            size=3,
            mode="nearest"
        )
        local_std_bands.append(std_band.astype(np.float32))

    local_std_raster = np.stack(local_std_bands)

    transformer = Transformer.from_crs(
        "EPSG:4326", crs, always_xy=True
    )

    rows, cols = [], []
    for _, row in df.iterrows():
        x, y = transformer.transform(
            float(row["longitude"]), float(row["latitude"])
        )
        r, c = rasterio.transform.rowcol(transform, x, y)
        rows.append(r)
        cols.append(c)

    df["row"] = rows
    df["col"] = cols

    inside = (
        (df["row"] >= 1) & (df["row"] < height - 1) &
        (df["col"] >= 1) & (df["col"] < width - 1)
    )
    df = df.loc[inside].copy()

    feature_rows = []
    keep_indices = []

    for idx, row in df.iterrows():
        r, c = int(row["row"]), int(row["col"])

        # Skip sample points whose centre pixel is invalid.
        if not valid[r, c]:
            continue

        base_values = row[BASE_FEATURES].to_numpy(dtype=np.float32)
        std_values = local_std_raster[:, r, c]

        if not np.all(np.isfinite(base_values)):
            continue
        if not np.all(np.isfinite(std_values)):
            continue

        feature_rows.append(
            np.concatenate([base_values, std_values]).astype(np.float32)
        )
        keep_indices.append(idx)

df = df.loc[keep_indices].reset_index(drop=True)
X = pd.DataFrame(feature_rows, columns=FEATURES)
y = df["target"].astype(np.uint8).reset_index(drop=True)

print("\nUsable samples with neighborhood features:", len(df))
print("Class counts:")
print(df["crop_class"].value_counts().to_string())


# Spatial groups: approximate local east/north coordinates in metres.
lat0 = np.radians(df["latitude"].mean())
lat = np.radians(df["latitude"].to_numpy())
lon = np.radians(df["longitude"].to_numpy())
lon0 = np.radians(df["longitude"].mean())

earth_radius = 6371000.0
east_m = earth_radius * (lon - lon0) * np.cos(lat0)
north_m = earth_radius * (lat - lat0)

block_size_m = 500
block_x = np.floor(east_m / block_size_m).astype(int)
block_y = np.floor(north_m / block_size_m).astype(int)
groups = pd.Series([
    f"{x}_{yy}" for x, yy in zip(block_x, block_y)
])

splitter = GroupShuffleSplit(
    n_splits=200, test_size=0.25, random_state=42
)

chosen_split = None
for train_idx, test_idx in splitter.split(X, y, groups):
    if (
        len(np.unique(y.iloc[train_idx])) == 3
        and len(np.unique(y.iloc[test_idx])) == 3
    ):
        chosen_split = (train_idx, test_idx)
        break

if chosen_split is None:
    raise RuntimeError(
        "Could not find a spatial split with all three classes in both sets."
    )

train_idx, test_idx = chosen_split

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
model.fit(X.iloc[train_idx], y.iloc[train_idx])
pred = model.predict(X.iloc[test_idx])

accuracy = accuracy_score(y.iloc[test_idx], pred)
report = classification_report(
    y.iloc[test_idx],
    pred,
    labels=[1, 2, 3],
    target_names=CLASS_NAMES,
    zero_division=0
)
cm = confusion_matrix(y.iloc[test_idx], pred, labels=[1, 2, 3])

results = df.iloc[test_idx][
    ["crop_class", "latitude", "longitude"]
].copy()
results["actual_class"] = y.iloc[test_idx].to_numpy()
results["predicted_class"] = pred
results["predicted_crop"] = pd.Series(pred).map({
    1: "Coconut", 2: "Arecanut", 3: "Paddy"
}).to_numpy()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
results.to_csv(PREDICTIONS_PATH, index=False)

summary = (
    "Crop Neighborhood Feature Experiment - Mangaluru, 2022\n"
    + "=" * 58 + "\n"
    + f"Input samples after exact duplicate removal: {len(df)}\n"
    + f"Feature count: {len(FEATURES)}\n"
    + f"Spatial block size: {block_size_m} metres\n"
    + f"Training samples: {len(train_idx)}\n"
    + f"Testing samples: {len(test_idx)}\n"
    + f"Training blocks: {groups.iloc[train_idx].nunique()}\n"
    + f"Testing blocks: {groups.iloc[test_idx].nunique()}\n\n"
    + f"Spatial hold-out accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n"
    + "Classification report\n" + "-" * 58 + "\n"
    + report + "\n"
    + "Confusion matrix (Coconut, Arecanut, Paddy)\n"
    + "-" * 58 + "\n" + str(cm) + "\n"
)

REPORT_PATH.write_text(summary, encoding="utf-8")
print("\n" + summary)
print("Saved report:", REPORT_PATH)
print("Saved predictions:", PREDICTIONS_PATH)
