import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = r"D:\Intelligent_Land_Use_Change_Analysis"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "land_use_random_forest.joblib"
)

TRAINING_PATH = os.path.join(
    BASE_DIR,
    "data",
    "training",
    "training_all_years.npz"
)
OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "results",
    "model_evaluation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = {
    1: "Water",
    2: "Vegetation/Forest",
    3: "Agriculture",
    4: "Built-up",
    5: "Barren/Open"
}

print("=" * 80)
print("LULC RANDOM FOREST MODEL EVALUATION")
print("=" * 80)

# ---------------------------------------------------------
# Load training data
# ---------------------------------------------------------

data = np.load(TRAINING_PATH)

X = data["X"]
y = data["y"]

print(f"\nDataset shape: {X.shape}")
print(f"Labels shape:  {y.shape}")

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

model = joblib.load(MODEL_PATH)

print(f"\nModel: {type(model).__name__}")
print(f"Number of trees: {model.n_estimators}")

# ---------------------------------------------------------
# Stratified validation split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples:   {len(y_train):,}")
print(f"Validation samples: {len(y_test):,}")

# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 80)
print(f"VALIDATION ACCURACY: {accuracy * 100:.2f}%")
print("=" * 80)

# ---------------------------------------------------------
# Classification report
# ---------------------------------------------------------

target_names = [
    CLASS_NAMES[1],
    CLASS_NAMES[2],
    CLASS_NAMES[3],
    CLASS_NAMES[4],
    CLASS_NAMES[5]
]

report = classification_report(
    y_test,
    y_pred,
    labels=[1, 2, 3, 4, 5],
    target_names=target_names,
    digits=4
)

print("\nCLASSIFICATION REPORT")
print("-" * 80)
print(report)

# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[1, 2, 3, 4, 5]
)

cm_df = pd.DataFrame(
    cm,
    index=target_names,
    columns=target_names
)

print("\nCONFUSION MATRIX")
print("-" * 80)
print(cm_df)

# ---------------------------------------------------------
# Feature importance
# ---------------------------------------------------------

feature_names = [
    "Blue",
    "Green",
    "Red",
    "NIR",
    "SWIR",
    "NDVI",
    "NDWI",
    "NDBI"
]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print("-" * 80)
print(
    importance_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

report_path = os.path.join(
    OUTPUT_DIR,
    "lulc_model_evaluation.txt"
)

with open(report_path, "w", encoding="utf-8") as f:

    f.write("LULC RANDOM FOREST MODEL EVALUATION\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Dataset shape: {X.shape}\n")
    f.write(f"Training samples: {len(y_train):,}\n")
    f.write(f"Validation samples: {len(y_test):,}\n")
    f.write(f"Validation accuracy: {accuracy * 100:.2f}%\n\n")

    f.write("CLASSIFICATION REPORT\n")
    f.write("-" * 80 + "\n")
    f.write(report)

    f.write("\nCONFUSION MATRIX\n")
    f.write("-" * 80 + "\n")
    f.write(cm_df.to_string())

    f.write("\n\nFEATURE IMPORTANCE\n")
    f.write("-" * 80 + "\n")
    f.write(
        importance_df.to_string(
            index=False
        )
    )

cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.csv"
)

cm_df.to_csv(cm_path)

importance_path = os.path.join(
    OUTPUT_DIR,
    "feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)

print("\nSaved:")
print(report_path)
print(cm_path)
print(importance_path)

print("\n" + "=" * 80)
print("MODEL EVALUATION COMPLETED")
print("=" * 80)