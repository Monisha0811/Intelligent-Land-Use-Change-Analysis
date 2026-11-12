import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASSIFIED_DIR = os.path.join(BASE_DIR, "data", "classified")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "charts")

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = {
    2018: os.path.join(CLASSIFIED_DIR, "land_use_2018.tif"),
    2022: os.path.join(CLASSIFIED_DIR, "land_use_2022.tif"),
    2025: os.path.join(CLASSIFIED_DIR, "land_use_2025.tif")
}

print("\nCalculating common valid area for all three years...")

data = {}

for year, path in files.items():
    with rasterio.open(path) as src:
        data[year] = src.read(1)

common_mask = (
    (data[2018] != 0) &
    (data[2022] != 0) &
    (data[2025] != 0)
)

common_pixels = np.sum(common_mask)
pixel_area_km2 = 0.01 * 0.01

print(f"Common valid pixels: {common_pixels:,}")
print(f"Common valid area: {common_pixels * pixel_area_km2:.2f} km²")

classes = {
    1: "Water",
    2: "Vegetation/Forest",
    3: "Agriculture",
    4: "Built-up",
    5: "Barren/Open"
}

years = [2018, 2022, 2025]

trend_data = {}

for class_id, class_name in classes.items():
    values = []

    for year in years:
        count = np.sum(
            (data[year] == class_id) & common_mask
        )
        area = count * pixel_area_km2
        values.append(area)

    trend_data[class_name] = values

    print(
        f"{class_name:20s} "
        f"2018: {values[0]:8.2f} km² | "
        f"2022: {values[1]:8.2f} km² | "
        f"2025: {values[2]:8.2f} km²"
    )

print("\nCreating corrected land-use trend chart...")

plt.figure(figsize=(11, 7))

for class_name, values in trend_data.items():
    plt.plot(
        years,
        values,
        marker="o",
        linewidth=2,
        label=class_name
    )

plt.xlabel("Year")
plt.ylabel("Area (km²)")
plt.title("Land-use Area Trend: 2018–2025 (Common Valid Area)")
plt.xticks(years)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

output_file = os.path.join(
    OUTPUT_DIR,
    "land_use_trend_2018_2025.png"
)

plt.savefig(output_file, dpi=300)
plt.close()

print("\nCorrected trend chart saved:")
print(output_file)

print("\n" + "=" * 65)
print("CORRECTED TREND ANALYSIS COMPLETED!")
print("=" * 65)
