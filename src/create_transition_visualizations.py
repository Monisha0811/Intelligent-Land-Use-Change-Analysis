import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "charts")

os.makedirs(OUTPUT_DIR, exist_ok=True)

classes = [
    "Water",
    "Vegetation/Forest",
    "Agriculture",
    "Built-up",
    "Barren/Open"
]

pairs = [
    ("2018", "2022"),
    ("2022", "2025"),
    ("2018", "2025")
]

print("\nCreating transition matrix heatmaps...")

for year1, year2 in pairs:

    filename = f"transition_matrix_{year1}_{year2}.csv"
    path = os.path.join(RESULTS_DIR, filename)

    if not os.path.exists(path):
        print(f"Missing file: {filename}")
        continue

    df = pd.read_csv(path)

    print(f"\nProcessing {year1} -> {year2}")
    print("Columns:", list(df.columns))

    if df.shape[1] >= 6:
        matrix = df.iloc[:, 1:6].copy()
        matrix.index = classes
        matrix.columns = classes
    else:
        print("Unexpected transition matrix format.")
        continue

    plt.figure(figsize=(10, 8))

    plt.imshow(matrix.values, aspect="auto")

    plt.xticks(range(5), classes, rotation=30, ha="right")
    plt.yticks(range(5), classes)

    plt.xlabel(f"Land-use class in {year2}")
    plt.ylabel(f"Land-use class in {year1}")
    plt.title(f"Land-use Transition Matrix: {year1} -> {year2}")

    for i in range(5):
        for j in range(5):
            value = matrix.iloc[i, j]
            plt.text(
                j,
                i,
                f"{value:,.0f}",
                ha="center",
                va="center",
                fontsize=9
            )

    plt.colorbar(label="Pixel Count")
    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_DIR,
        f"transition_matrix_{year1}_{year2}.png"
    )

    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved: {output_file}")

print("\n" + "=" * 65)
print("TRANSITION MATRIX VISUALIZATION COMPLETED!")
print("=" * 65)
