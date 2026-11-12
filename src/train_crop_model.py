import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_CSV = os.path.join(
    PROJECT_ROOT,
    "data",
    "results",
    "crop_reference",
    "crop_training_samples_2022.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "results",
    "crop_reference"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "crop_random_forest_2022.joblib"
)

REPORT_PATH = os.path.join(
    RESULT_DIR,
    "crop_model_report_2022.txt"
)

PREDICTIONS_PATH = os.path.join(
    RESULT_DIR,
    "crop_test_predictions_2022.csv"
)


# ---------------------------------------------------------
# Read training data
# ---------------------------------------------------------

print("Reading crop training data...")

df = pd.read_csv(INPUT_CSV)

print("Total samples:", len(df))


features = [
    "blue",
    "green",
    "red",
    "nir",
    "swir",
    "ndvi",
    "ndwi",
    "ndbi"
]

target = "crop_class"


X = df[features].astype(np.float32)

class_mapping = {
    "Coconut": 1,
    "Arecanut": 2,
    "Paddy": 3
}

y = df[target].map(class_mapping)


if y.isna().any():
    raise RuntimeError(
        "Unknown crop class found in training data."
    )


y = y.astype(np.uint8)


print("\nClass distribution:")

for name, number in class_mapping.items():
    count = int((y == number).sum())
    print(f"{number} - {name}: {count}")


# ---------------------------------------------------------
# Stratified train/test split
# ---------------------------------------------------------

print("\nCreating stratified train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ---------------------------------------------------------
# Train Random Forest
# ---------------------------------------------------------

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

y_pred = model.predict(X_test)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    labels=[1, 2, 3],
    target_names=[
        "Coconut",
        "Arecanut",
        "Paddy"
    ],
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[1, 2, 3]
)


print("\n========================================")
print("CROP RANDOM FOREST RESULTS")
print("========================================")

print(f"\nHold-out accuracy: {accuracy:.4f}")
print(f"Hold-out accuracy: {accuracy * 100:.2f}%")

print("\nClassification report:")
print(report)

print("Confusion matrix:")
print(cm)


# ---------------------------------------------------------
# Feature importance
# ---------------------------------------------------------

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nFeature importance:")
print(importance.to_string(index=False))


# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("\nSaved model:")
print(MODEL_PATH)


# ---------------------------------------------------------
# Save test predictions
# ---------------------------------------------------------

test_results = X_test.copy()

test_results["actual_class"] = y_test.values
test_results["predicted_class"] = y_pred

reverse_mapping = {
    1: "Coconut",
    2: "Arecanut",
    3: "Paddy"
}

test_results["actual_crop"] = (
    test_results["actual_class"]
    .map(reverse_mapping)
)

test_results["predicted_crop"] = (
    test_results["predicted_class"]
    .map(reverse_mapping)
)

test_results.to_csv(
    PREDICTIONS_PATH,
    index=False
)

print("\nSaved test predictions:")
print(PREDICTIONS_PATH)


# ---------------------------------------------------------
# Save report
# ---------------------------------------------------------

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write("Crop Random Forest Model - 2022\n")
    f.write("=" * 45 + "\n\n")

    f.write(f"Total samples: {len(df)}\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Testing samples: {len(X_test)}\n\n")

    f.write(
        f"Hold-out accuracy: "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)\n\n"
    )

    f.write("Classification Report\n")
    f.write("-" * 45 + "\n")
    f.write(report)

    f.write("\nConfusion Matrix\n")
    f.write("-" * 45 + "\n")
    f.write(str(cm))

    f.write("\n\nFeature Importance\n")
    f.write("-" * 45 + "\n")
    f.write(importance.to_string(index=False))

print("\nSaved report:")
print(REPORT_PATH)

print("\nCrop model training completed successfully!")
