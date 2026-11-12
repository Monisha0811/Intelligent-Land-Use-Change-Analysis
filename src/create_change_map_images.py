import os
import numpy as np
import rasterio
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_DIR = os.path.join(BASE_DIR, "data", "change_maps")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "maps")

files = [
    ("change_2018_2022.tif", "change_map_2018_2022.png"),
    ("change_2022_2025.tif", "change_map_2022_2025.png"),
    ("change_2018_2025.tif", "change_map_2018_2025.png")
]

for input_file, output_file in files:

    with rasterio.open(os.path.join(INPUT_DIR, input_file)) as src:
        data = src.read(1)

    rgb = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)

    # No-data = white
    rgb[data == 0] = [255, 255, 255]

    # Unchanged = light gray
    rgb[data == 1] = [220, 220, 220]

    # Changed = red
    rgb[data == 2] = [220, 50, 50]

    Image.fromarray(rgb).save(
        os.path.join(OUTPUT_DIR, output_file)
    )

    print(f"Created: {output_file}")

print("Corrected change maps created successfully.")
