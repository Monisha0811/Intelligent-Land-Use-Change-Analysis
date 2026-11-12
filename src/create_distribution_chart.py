import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "charts")

os.makedirs(OUTPUT_DIR, exist_ok=True)

input_file = os.path.join(
    RESULTS_DIR,
    "land_use_area_statistics.csv"
)

df = pd.read_csv(input_file)

print("\nCreating land-use distribution chart...")

years = [2018, 2022, 2025]
class_order = [
    "Water",
    "Vegetation/Forest",
    "Agriculture",
    "Built-up",
    "Barren/Open"
]

# Prepare class percentages for each year
pivot_df = df.pivot(
    index="Class_Name",
    columns="Year",
    values="Percentage"
)

pivot_df = pivot_df.reindex(class_order)
pivot_df = pivot_df.reindex(columns=years)

ax = pivot_df.plot(
    kind="bar",
    figsize=(12, 7),
    width=0.8
)

ax.set_xlabel("Land-use Class")
ax.set_ylabel("Percentage of Mapped Area (%)")
ax.set_title("Land-use Class Distribution (2018, 2022 and 2025)")
ax.legend(title="Year")
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

plt.xticks(rotation=20, ha="right")
plt.tight_layout()

output_file = os.path.join(
    OUTPUT_DIR,
    "land_use_class_distribution.png"
)

plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close()

print("\nChart saved:")
print(output_file)

print("\nLAND-USE DISTRIBUTION CHART COMPLETED!")