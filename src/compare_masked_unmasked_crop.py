from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "data/results/crop_reference"

DATASETS = {
    "Masked": BASE / "crop_training_samples_2022_clean.csv",
    "Unmasked": BASE / "crop_training_samples_2022_unmasked_clean.csv",
}

FEATURES = ["blue", "green", "red", "nir", "swir", "ndvi", "ndwi", "ndbi"]
LABELS = ["Coconut", "Arecanut", "Paddy"]
CLASS_MAPPING = {"Coconut": 1, "Arecanut": 2, "Paddy": 3}
BLOCK_SIZE_M = 500
N_SPLITS = 200

def load_data(path):
    df = pd.read_csv(path).reset_index(drop=True)
    df = df.dropna(subset=FEATURES + ["crop_class", "latitude", "longitude"])
    df = df[df["crop_class"].isin(LABELS)].reset_index(drop=True)

    # Remove exact-coordinate conflicts and repeated coordinates.
    conflict_counts = df.groupby(["latitude", "longitude"])["crop_class"].nunique()
    conflict_coords = conflict_counts[conflict_counts > 1].index
    coord_index = pd.MultiIndex.from_frame(df[["latitude", "longitude"]])
    df = df.loc[~coord_index.isin(conflict_coords)].copy()
    df = df.drop_duplicates(["latitude", "longitude"]).reset_index(drop=True)

    return df

def make_groups(df, origin_lat, origin_lon):
    radius = 6371000.0
    lat0 = np.radians(origin_lat)
    lat = np.radians(df["latitude"].to_numpy())
    lon = np.radians(df["longitude"].to_numpy())
    lon0 = np.radians(origin_lon)

    east = radius * (lon - lon0) * np.cos(lat0)
    north = radius * (lat - lat0)

    bx = np.floor(east / BLOCK_SIZE_M).astype(int)
    by = np.floor(north / BLOCK_SIZE_M).astype(int)
    return np.array([f"{x}_{y}" for x, y in zip(bx, by)])

# Use the same geographic origin for both datasets.
all_frames = {name: load_data(path) for name, path in DATASETS.items()}
all_points = pd.concat(
    [df[["latitude", "longitude"]] for df in all_frames.values()],
    ignore_index=True
)
origin_lat = all_points["latitude"].mean()
origin_lon = all_points["longitude"].mean()

prepared = {}
for name, df in all_frames.items():
    prepared[name] = {
        "df": df,
        "X": df[FEATURES].astype(np.float32),
        "y": df["crop_class"].map(CLASS_MAPPING).astype(int).to_numpy(),
        "groups": make_groups(df, origin_lat, origin_lon),
    }

# Build candidate spatial hold-outs from the combined geographic blocks.
combined_groups = np.unique(np.concatenate(
    [item["groups"] for item in prepared.values()]
))
rng = np.random.RandomState(42)
candidate_splits = []

for _ in range(N_SPLITS):
    shuffled = rng.permutation(combined_groups)
    n_test_groups = max(1, round(0.25 * len(shuffled)))
    test_groups = set(shuffled[:n_test_groups])
    candidate_splits.append(test_groups)

records = []
split_id = 0

for test_groups in candidate_splits:
    current = {}
    eligible = True

    for name, item in prepared.items():
        groups = item["groups"]
        y = item["y"]

        test_mask = np.isin(groups, list(test_groups))
        train_mask = ~test_mask

        if (
            len(np.unique(y[train_mask])) < 3
            or len(np.unique(y[test_mask])) < 3
            or train_mask.sum() == 0
            or test_mask.sum() == 0
        ):
            eligible = False
            break

        current[name] = (train_mask, test_mask)

    if not eligible:
        continue

    split_id += 1

    for name, item in prepared.items():
        df, X, y = item["df"], item["X"], item["y"]
        train_mask, test_mask = current[name]

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        model.fit(X.loc[train_mask], y[train_mask])
        pred = model.predict(X.loc[test_mask])

        records.append({
            "split": split_id,
            "dataset": name,
            "accuracy": accuracy_score(y[test_mask], pred),
            "coconut_recall": recall_score(
                y[test_mask], pred, labels=[1], average=None, zero_division=0
            )[0],
            "arecanut_recall": recall_score(
                y[test_mask], pred, labels=[2], average=None, zero_division=0
            )[0],
            "paddy_recall": recall_score(
                y[test_mask], pred, labels=[3], average=None, zero_division=0
            )[0],
            "train_samples": int(train_mask.sum()),
            "test_samples": int(test_mask.sum()),
            "train_blocks": int(np.unique(item["groups"][train_mask]).size),
            "test_blocks": int(np.unique(item["groups"][test_mask]).size),
        })

if not records:
    raise RuntimeError(
        "No common spatial splits contained all three classes in both "
        "training and test sets for both datasets. Do not compare results "
        "using different splits; we will adjust the validation design."
    )

detail = pd.DataFrame(records)
detail_path = BASE / "crop_masked_unmasked_spatial_comparison_detail.csv"
summary_path = BASE / "crop_masked_unmasked_spatial_comparison_summary.csv"
report_path = BASE / "crop_masked_unmasked_spatial_comparison.txt"

detail.to_csv(detail_path, index=False)

metrics = [
    "accuracy", "coconut_recall", "arecanut_recall", "paddy_recall"
]
summary = detail.groupby("dataset")[metrics].agg(["mean", "std"])
summary.to_csv(summary_path)

lines = [
    "Masked vs Unmasked Crop Classification - Spatial Comparison",
    "=" * 62,
    f"Eligible common spatial splits: {detail['split'].nunique()}",
    f"Block size: {BLOCK_SIZE_M} m",
    "The same candidate test blocks are used for both datasets.",
    "Values are mean and standard deviation across eligible splits.",
    "",
    summary.to_string(),
    "",
    "Caution: this is a diagnostic experiment, not independent ground-truth validation.",
    "The datasets have different spatial sample coverage and class balance.",
    "Reference surveys are from 2021; imagery is from December 2022.",
]
report = "\n".join(lines) + "\n"
report_path.write_text(report, encoding="utf-8")

print(report)
print("\nSaved detail:", detail_path)
print("Saved summary:", summary_path)
print("Saved report:", report_path)

