from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "data/results/crop_reference/crop_training_samples_2022_clean.csv"
REPORT_PATH = ROOT / "data/results/crop_reference/crop_spatial_validation_2022.txt"
PREDICTIONS_PATH = ROOT / "data/results/crop_reference/crop_spatial_predictions_2022.csv"

FEATURES = ["blue", "green", "red", "nir", "swir", "ndvi", "ndwi", "ndbi"]
CLASS_MAPPING = {"Coconut": 1, "Arecanut": 2, "Paddy": 3}
CLASS_NAMES = ["Coconut", "Arecanut", "Paddy"]

df = pd.read_csv(INPUT_CSV)
df = df.reset_index(drop=True)
X = df[FEATURES].astype(np.float32)
y = df["crop_class"].map(CLASS_MAPPING).astype(np.uint8)

# Approximate local east/north coordinates in metres.
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
    f"{x}_{y}" for x, y in zip(block_x, block_y)
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
        "Could not find a spatial split containing all three classes "
        "in both sets. With only 24 Paddy samples, this may be difficult. "
        "Do not interpret this as a model failure."
    )

train_idx, test_idx = chosen_split
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(
    y_test,
    y_pred,
    labels=[1, 2, 3],
    target_names=CLASS_NAMES,
    zero_division=0,
)
cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3])

results = df.iloc[test_idx][
    ["crop_class", "latitude", "longitude"]
].copy()
results["actual_class"] = y_test.to_numpy()
results["predicted_class"] = y_pred
results["predicted_crop"] = pd.Series(y_pred).map({
    1: "Coconut",
    2: "Arecanut",
    3: "Paddy",
}).to_numpy()
results.to_csv(PREDICTIONS_PATH, index=False)

summary_lines = [
    "Spatial Crop Validation - Mangaluru, 2022",
    "=" * 48,
    f"Samples after exact duplicate removal: {len(df)}",
    f"Approximate spatial block size: {block_size_m} metres",
    f"Training samples: {len(train_idx)}",
    f"Testing samples: {len(test_idx)}",
    f"Training blocks: {groups.iloc[train_idx].nunique()}",
    f"Testing blocks: {groups.iloc[test_idx].nunique()}",
    "",
    f"Spatial hold-out accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)",
    "",
    "Classification report",
    "-" * 48,
    report,
    "",
    "Confusion matrix (Coconut, Arecanut, Paddy)",
    "-" * 48,
    str(cm),
]

summary = "\n".join(summary_lines) + "\n"
REPORT_PATH.write_text(summary, encoding="utf-8")

print(summary)
print("Saved report:", REPORT_PATH)
print("Saved predictions:", PREDICTIONS_PATH)
