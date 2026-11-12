
import os
import numpy as np
import rasterio
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFIED_DIR = os.path.join(
    BASE_DIR, "data", "classified", "mangaluru"
)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "maps")

os.makedirs(OUTPUT_DIR, exist_ok=True)

YEARS = [2018, 2022, 2025]

COLORS = {
    1: [30, 144, 255, 180],  # Water
    2: [34, 139, 34, 180],   # Vegetation / Forest
    3: [255, 215, 0, 180],   # Agriculture
    4: [220, 60, 60, 180],   # Built-up
    5: [160, 120, 70, 180],  # Barren / Open
}

for year in YEARS:
    input_file = os.path.join(
        CLASSIFIED_DIR, f"land_use_{year}.tif"
    )
    output_file = os.path.join(
        OUTPUT_DIR, f"land_use_overlay_{year}.png"
    )
    temp_file = output_file + ".tmp.png"

    print(f"\nProcessing {year}...")

    with rasterio.open(input_file) as src:
        data = src.read(1)

        rgba = np.zeros(
            (src.height, src.width, 4), dtype=np.uint8
        )

        for class_id, color in COLORS.items():
            rgba[data == class_id] = color

        image = Image.fromarray(rgba)

    # Save to a temporary PNG, then replace the final file.
    image.save(temp_file, format="PNG")
    os.replace(temp_file, output_file)

    print("Dimensions:", image.size)
    print("Visible pixels:", int(np.count_nonzero(rgba[:, :, 3])))
    print("Saved:", output_file)

print("\nAll overlays generated successfully.")
