import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import joblib


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = r"D:\Intelligent_Land_Use_Change_Analysis"

TRAINING_FILE = os.path.join(
    BASE_DIR,
    "data",
    "training",
    "training_all_years.npz"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "land_use_random_forest.joblib"
)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("Loading training data...")

data = np.load(
    TRAINING_FILE
)

X = data["X"]
y = data["y"]

print(
    f"Features shape: {X.shape}"
)

print(
    f"Labels shape: {y.shape}"
)


# ============================================================
# CHECK CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")

class_names = {
    1: "Water",
    2: "Vegetation/Forest",
    3: "Agriculture",
    4: "Built-up",
    5: "Barren/Open"
}

for class_id in range(1, 6):

    count = np.sum(
        y == class_id
    )

    print(
        f"Class {class_id} "
        f"({class_names[class_id]}): "
        f"{count}"
    )


# ============================================================
# SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print()
print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    n_jobs=-1,
    max_depth=20
)

model.fit(
    X_train,
    y_train
)

print(
    "Training completed!"
)


# ============================================================
# EVALUATE MODEL
# ============================================================

print("\nEvaluating model...")

y_pred = model.predict(
    X_test
)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    f"\nAccuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        y_pred,
        labels=[1, 2, 3, 4, 5],
        target_names=[
            "Water",
            "Vegetation/Forest",
            "Agriculture",
            "Built-up",
            "Barren/Open"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print(
    "Confusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=[1, 2, 3, 4, 5]
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)

print(
    "\nModel saved to:"
)

print(
    MODEL_FILE
)

print(
    "\nRandom Forest training completed successfully!"
)