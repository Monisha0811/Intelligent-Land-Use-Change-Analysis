from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "data/results/crop_reference"

DATASETS = {
    "Masked": BASE / "crop_training_samples_2022_clean.csv",
    "Unmasked": BASE / "crop_training_samples_2022_unmasked_clean.csv",
}

FEATURES = ["blue", "green", "red", "nir", "swir", "ndvi", "ndwi", "ndbi"]
CLASS_TO_ID = {"Coconut": 1, "Arecanut": 2, "Paddy": 3}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
IDS = [1, 2, 3]
N_SPLITS = 200
BLOCK_SIZE = 500
RADIUS = 6371000.0

def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=FEATURES + ["crop_class", "latitude", "longitude"])
    df = df[df["crop_class"].isin(CLASS_TO_ID)].copy()

    # Exclude conflicting coordinates and keep one row per coordinate.
    label_counts = df.groupby(["latitude", "longitude"])["crop_class"].nunique()
    conflicts = label_counts[label_counts > 1].index
    coord_index = pd.MultiIndex.from_frame(df[["latitude", "longitude"]])
    df = df.loc[~coord_index.isin(conflicts)].copy()
    df = df.drop_duplicates(["latitude", "longitude"]).reset_index(drop=True)
    return df

frames = {name: load_data(path) for name, path in DATASETS.items()}

# A common geographic origin ensures block IDs have the same meaning.
all_coords = pd.concat(
    [df[["latitude", "longitude"]] for df in frames.values()],
    ignore_index=True
)
lat0 = np.radians(all_coords["latitude"].mean())
lon0 = np.radians(all_coords["longitude"].mean())

def block_ids(df):
    lat = np.radians(df["latitude"].to_numpy())
    lon = np.radians(df["longitude"].to_numpy())
    east = RADIUS * (lon - lon0) * np.cos(lat0)
    north = RADIUS * (lat - lat0)
    bx = np.floor(east / BLOCK_SIZE).astype(int)
    by = np.floor(north / BLOCK_SIZE).astype(int)
    return np.array([f"{x}_{y}" for x, y in zip(bx, by)])

prepared = {}
for name, df in frames.items():
    prepared[name] = {
        "df": df,
        "X": df[FEATURES].astype(np.float32),
        "y": df["crop_class"].map(CLASS_TO_ID).to_numpy(),
        "groups": block_ids(df),
    }

# Candidate spatial splits are generated from the shared block grid.
all_blocks = np.unique(np.concatenate(
    [p["groups"] for p in prepared.values()]
))
rng = np.random.RandomState(2026)
candidate_splits = []

for _ in range(N_SPLITS):
    shuffled = rng.permutation(all_blocks)
    n_test = max(1, round(0.25 * len(shuffled)))
    candidate_splits.append(set(shuffled[:n_test]))

pooled = {name: [] for name in DATASETS}
split_count = 0

for test_blocks in candidate_splits:
    masks = {}
    valid = True

    for name, p in prepared.items():
        test = np.isin(p["groups"], list(test_blocks))
        train = ~test

        if (
            train.sum() == 0 or test.sum() == 0
            or len(np.unique(p["y"][train])) < 3
            or len(np.unique(p["y"][test])) < 3
        ):
            valid = False
            break
        masks[name] = (train, test)

    if not valid:
        continue

    split_count += 1

    for name, p in prepared.items():
        train, test = masks[name]
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        model.fit(p["X"].loc[train], p["y"][train])
        pred = model.predict(p["X"].loc[test])

        out = p["df"].loc[test, [
            "crop_class", "latitude", "longitude"
        ]].copy()
        out["actual_id"] = p["y"][test]
        out["predicted_id"] = pred
        out["actual_crop"] = out["actual_id"].map(ID_TO_CLASS)
        out["predicted_crop"] = out["predicted_id"].map(ID_TO_CLASS)
        out["split"] = split_count
        pooled[name].append(out.reset_index(drop=True))

if split_count == 0:
    raise RuntimeError(
        "No eligible common spatial splits found. No comparison files were saved."
    )

summary_lines = [
    "Pooled Spatial Validation - Crop Classification",
    "=" * 55,
    f"Eligible common spatial splits: {split_count}",
    f"Spatial block size: {BLOCK_SIZE} m",
    "Note: a point may appear in the test set in more than one split.",
    "Pooled metrics are repeated-holdout diagnostics, not independent accuracy estimates.",
]

for name, pieces in pooled.items():
    result = pd.concat(pieces, ignore_index=True)
    out_path = BASE / f"crop_{name.lower()}_pooled_spatial_predictions.csv"
    result.to_csv(out_path, index=False)

    cm = confusion_matrix(
        result["actual_id"], result["predicted_id"], labels=IDS
    )
    report = classification_report(
        result["actual_id"], result["predicted_id"],
        labels=IDS,
        target_names=[ID_TO_CLASS[i] for i in IDS],
        zero_division=0,
    )

    summary_lines.extend([
        "",
        f"{name.upper()} DATASET",
        "-" * 40,
        f"Pooled test predictions: {len(result)}",
        "Confusion matrix (rows=actual; columns=predicted)",
        "Class order: Coconut, Arecanut, Paddy",
        str(cm),
        "",
        report,
        f"Predictions saved: {out_path}",
    ])

report_path = BASE / "crop_pooled_spatial_class_report.txt"
report_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
print("\n".join(summary_lines))
print("\nSaved report:", report_path)
