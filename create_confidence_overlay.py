
import os
import numpy as np
import rasterio
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIDENCE_DIR = os.path.join(
    BASE_DIR, "data", "results", "confidence"
)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "maps")

os.makedirs(OUTPUT_DIR, exist_ok=True)

YEARS = [2018, 2022, 2025]

for year in YEARS:
    input_file = os.path.join(
        CONFIDENCE_DIR, f"lulc_confidence_{year}.tif"
    )
    output_file = os.path.join(
        OUTPUT_DIR, f"confidence_overlay_{year}.png"
    )
    temp_file = output_file + ".tmp.png"

    print(f"\nProcessing {year}...")

    with rasterio.open(input_file) as src:
        confidence = src.read(1).astype(np.float32)

        # Confidence outputs use values between 0 and 1.
        valid = (
            np.isfinite(confidence)
            & (confidence > 0)
            & (confidence <= 1)
        )

        # Map lower confidence to red and higher confidence to green.
      
        rgba = np.zeros(
            (src.height, src.width, 4), dtype=np.uint8
        )

        low = valid & (confidence < 0.60)
        medium = valid & (confidence >= 0.60) & (confidence < 0.80)
        high = valid & (confidence >= 0.80)

        # Low confidence: red
        rgba[low] = [220, 55, 55, 190]

        # Medium confidence: yellow
        rgba[medium] = [255, 205, 40, 190]

        # High confidence: green
        rgba[high] = [35, 160, 75, 190]


       
        image = Image.fromarray(rgba)

    image.save(temp_file, format="PNG")
    os.replace(temp_file, output_file)

    print("Dimensions:", image.size)
    print("Visible pixels:", int(np.count_nonzero(rgba[:, :, 3])))
    print("Saved:", output_file)

print("\nAll confidence overlays generated successfully.")
