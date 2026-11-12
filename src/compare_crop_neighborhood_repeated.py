from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.ndimage import generic_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
RASTER_PATH = ROOT / "data/processed/mangaluru/sentinel2_2022_mangaluru.tif"
CSV_PATH = ROOT / "data/results/crop_reference/crop_training_samples_2022_clean.csv"
OUT_DIR = ROOT / "data/results/crop_reference"

BASE = ["blue", "green", "red", "nir", "swir", "ndvi", "ndwi", "ndbi"]
BANDS = ["blue", "green", "red", "nir", "swir"]
EXTENDED = BASE + [f"{b}_local_std" for b in BANDS]
LABELS = ["Coconut", "Arecanut", "Paddy"]
MAPPING = {"Coconut": 1, "Arecanut": 2, "Paddy": 3}


def calculate_std(values):
    finite = values[np.isfinite(values)]
    return float(np.std(finite)) if finite.size else np.nan


print("Loading cleaned samples...")
df = pd.read_csv(CSV_PATH)
df = df.drop_duplicates(
    subset=["latitude", "longitude", "crop_class"]
).reset_index(drop=True)
df = df[df["crop_class"].isin(MAPPING)].copy()
df["target"] = df["crop_class"].map(MAPPING).astype(int)

with rasterio.open(RASTER_PATH) as src:
    raster = src.read().astype(np.float32)
    transform = src.transform
    crs = src.crs
    height, width = src.height, src.width
    nodata = src.nodata

    valid = np.all(np.isfinite(raster), axis=0)
    if nodata is not None:
        valid &= np.all(raster != nodata, axis=0)
    valid &= np.any(raster != 0, axis=0)

    print("Calculating five neighborhood features...")
    std_rasters = []
    for band in raster:
        work = band.copy()
        work[~valid] = np.nan
        std_rasters.append(
            generic_filter(
                work,
                calculate_std,
                size=3,
                mode="nearest"
            ).astype(np.float32)
        )
    std_rasters = np.stack(std_rasters)

    transformer = Transformer.from_crs(
        "EPSG:4326", crs, always_xy=True
    )

    features, keep = [], []
    for i, row in df.iterrows():
        x, y = transformer.transform(
            float(row["longitude"]), float(row["latitude"])
        )
        r, c = rasterio.transform.rowcol(transform, x, y)

        if not (1 <= r < height - 1 and 1 <= c < width - 1):
            continue
        if not valid[r, c]:
            continue

        base_values = row[BASE].to_numpy(dtype=np.float32)
        std_values = std_rasters[:, r, c]

        if not np.isfinite(base_values).all():
            continue
        if not np.isfinite(std_values).all():
            continue

        features.append(np.concatenate([base_values, std_values]))
        keep.append(i)

df = df.loc[keep].reset_index(drop=True)
X_all = np.asarray(features, dtype=np.float32)
y = df["target"].to_numpy(dtype=np.uint8)

X_base = X_all[:, :len(BASE)]
X_extended = X_all

# Assign approximate 500 m spatial blocks.
lat0 = np.radians(df["latitude"].mean())
lon0 = np.radians(df["longitude"].mean())
lat = np.radians(df["latitude"].to_numpy())
lon = np.radians(df["longitude"].to_numpy())
radius = 6371000.0

east = radius * (lon - lon0) * np.cos(lat0)
north = radius * (lat - lat0)
bx = np.floor(east / 500).astype(int)
by = np.floor(north / 500).astype(int)
groups = np.array([f"{x}_{yy}" for x, yy in zip(bx, by)])

splitter = GroupShuffleSplit(
    n_splits=100, test_size=0.25, random_state=2026
)

rows = []
split_number = 0

for train_idx, test_idx in splitter.split(X_base, y, groups):
    # Require all classes in train and test for a comparable evaluation.
    if len(np.unique(y[train_idx])) != 3 or len(np.unique(y[test_idx])) != 3:
        continue

    split_number += 1

    for model_name, X in [
        ("Original_8_features", X_base),
        ("Neighborhood_13_features", X_extended),
    ]:
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])

        recalls = recall_score(
            y[test_idx], pred, labels=[1, 2, 3],
            average=None, zero_division=0
        )

        rows.append({
            "split": split_number,
            "model": model_name,
            "accuracy": accuracy_score(y[test_idx], pred),
            "coconut_recall": recalls[0],
            "arecanut_recall": recalls[1],
            "paddy_recall": recalls[2],
            "test_samples": len(test_idx),
            "test_blocks": len(np.unique(groups[test_idx])),
        })

if split_number == 0:
    raise RuntimeError("No eligible spatial splits found.")

results = pd.DataFrame(rows)
OUT_DIR.mkdir(parents=True, exist_ok=True)

detail_path = OUT_DIR / "crop_neighborhood_repeated_splits_2022.csv"
summary_path = OUT_DIR / "crop_neighborhood_repeated_summary_2022.csv"
results.to_csv(detail_path, index=False)

metrics = [
    "accuracy", "coconut_recall", "arecanut_recall", "paddy_recall"
]
summary = results.groupby("model")[metrics].agg(["mean", "std"])
summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
summary = summary.reset_index()
summary.to_csv(summary_path, index=False)

print(f"\nEligible spatial splits: {split_number}")
print("\nMean and standard deviation across eligible splits:")
print(summary.round(4).to_string(index=False))
print("\nDetailed results:", detail_path)
print("Summary results:", summary_path)
