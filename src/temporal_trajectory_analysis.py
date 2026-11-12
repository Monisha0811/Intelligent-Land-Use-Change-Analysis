import os
import numpy as np
import pandas as pd
import rasterio

BASE_DIR = r"D:\Intelligent_Land_Use_Change_Analysis"

CLASSIFIED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "classified",
    "mangaluru"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "results",
    "trajectories"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

YEARS = [2018, 2022, 2025]

CLASS_NAMES = {
    1: "Water",
    2: "Vegetation/Forest",
    3: "Agriculture",
    4: "Built-up",
    5: "Barren/Open"
}


def classify_trajectory(c1, c2, c3):

    # Same class in all three years
    if c1 == c2 == c3:
        return "Stable"

    # Changed from original class and returned
    # Example: Agriculture -> Vegetation -> Agriculture
    if c1 != c2 and c2 != c3 and c1 == c3:
        return "Temporary Change"

    # Changed in 2018-2022 and remained in the new class
    # Example: Agriculture -> Built-up -> Built-up
    if c1 != c2 and c2 == c3:
        return "Persistent Change"

    # Same class until 2022, then changed
    # Example: Agriculture -> Agriculture -> Built-up
    if c1 == c2 and c2 != c3:
        return "Recent Change"

    # Changed in both periods to different classes
    # Example: Agriculture -> Vegetation -> Water
    if c1 != c2 and c2 != c3:
        return "Progressive Change"

    return "Mixed Trajectory"
print("=" * 90)
print("TEMPORAL LULC TRAJECTORY ANALYSIS")
print("=" * 90)

rasters = {}

for year in YEARS:

    path = os.path.join(
        CLASSIFIED_DIR,
        f"land_use_{year}.tif"
    )

    print(f"\nLoading {year}...")

    with rasterio.open(path) as src:
        rasters[year] = src.read(1)

# ---------------------------------------------------------
# Find common valid pixels
# ---------------------------------------------------------

lulc_2018 = rasters[2018]
lulc_2022 = rasters[2022]
lulc_2025 = rasters[2025]

valid = (
    (lulc_2018 > 0) &
    (lulc_2022 > 0) &
    (lulc_2025 > 0)
)

c1 = lulc_2018[valid]
c2 = lulc_2022[valid]
c3 = lulc_2025[valid]

total_pixels = len(c1)

print(f"\nCommon valid pixels: {total_pixels:,}")

# ---------------------------------------------------------
# Create trajectory labels
# ---------------------------------------------------------

trajectory_types = []

for a, b, c in zip(c1, c2, c3):
    trajectory_types.append(
        classify_trajectory(
            int(a),
            int(b),
            int(c)
        )
    )

trajectory_types = np.array(
    trajectory_types,
    dtype=object
)

# ---------------------------------------------------------
# Create class sequence
# ---------------------------------------------------------

sequences = []

for a, b, c in zip(c1, c2, c3):

    sequence = (
        f"{CLASS_NAMES[int(a)]} -> "
        f"{CLASS_NAMES[int(b)]} -> "
        f"{CLASS_NAMES[int(c)]}"
    )

    sequences.append(sequence)

sequences = np.array(
    sequences,
    dtype=object
)

# ---------------------------------------------------------
# Summary by trajectory type
# ---------------------------------------------------------

trajectory_counts = (
    pd.Series(trajectory_types)
    .value_counts()
)

trajectory_summary = pd.DataFrame({
    "Trajectory Type": trajectory_counts.index,
    "Pixel Count": trajectory_counts.values
})

trajectory_summary["Percentage"] = (
    trajectory_summary["Pixel Count"] /
    total_pixels * 100
)

trajectory_summary["Area (km2)"] = (
    trajectory_summary["Pixel Count"] *
    0.0001
)

trajectory_summary = trajectory_summary[
    [
        "Trajectory Type",
        "Pixel Count",
        "Percentage",
        "Area (km2)"
    ]
]

# ---------------------------------------------------------
# Detailed class sequences
# ---------------------------------------------------------

sequence_counts = (
    pd.Series(sequences)
    .value_counts()
)

sequence_summary = pd.DataFrame({
    "Trajectory Sequence": sequence_counts.index,
    "Pixel Count": sequence_counts.values
})

sequence_summary["Percentage"] = (
    sequence_summary["Pixel Count"] /
    total_pixels * 100
)

sequence_summary["Area (km2)"] = (
    sequence_summary["Pixel Count"] *
    0.0001
)

sequence_summary = sequence_summary[
    [
        "Trajectory Sequence",
        "Pixel Count",
        "Percentage",
        "Area (km2)"
    ]
]

# ---------------------------------------------------------
# Stable class analysis
# ---------------------------------------------------------

stable_mask = (
    (c1 == c2) &
    (c2 == c3)
)

stable_classes = c1[stable_mask]

stable_counts = (
    pd.Series(stable_classes)
    .value_counts()
    .sort_index()
)

stable_summary = pd.DataFrame({
    "Class ID": stable_counts.index,
    "Class": [
        CLASS_NAMES[int(x)]
        for x in stable_counts.index
    ],
    "Stable Pixel Count": stable_counts.values
})

stable_summary["Stable Percentage"] = (
    stable_summary["Stable Pixel Count"] /
    total_pixels * 100
)

stable_summary["Stable Area (km2)"] = (
    stable_summary["Stable Pixel Count"] *
    0.0001
)

# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

trajectory_path = os.path.join(
    OUTPUT_DIR,
    "trajectory_type_summary.csv"
)

sequence_path = os.path.join(
    OUTPUT_DIR,
    "trajectory_sequence_summary.csv"
)

stable_path = os.path.join(
    OUTPUT_DIR,
    "stable_class_summary.csv"
)

trajectory_summary.to_csv(
    trajectory_path,
    index=False
)

sequence_summary.to_csv(
    sequence_path,
    index=False
)

stable_summary.to_csv(
    stable_path,
    index=False
)

# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

print("\n" + "=" * 90)
print("TRAJECTORY TYPE SUMMARY")
print("=" * 90)

print(
    trajectory_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

print("\n" + "=" * 90)
print("TOP 20 TRAJECTORY SEQUENCES")
print("=" * 90)

print(
    sequence_summary.head(20).to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

print("\n" + "=" * 90)
print("STABLE LAND-USE CLASSES")
print("=" * 90)

print(
    stable_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

print("\nSaved:")
print(trajectory_path)
print(sequence_path)
print(stable_path)

print("\n" + "=" * 90)
print("TEMPORAL TRAJECTORY ANALYSIS COMPLETED")
print("=" * 90)