import os
import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.mask import mask
from matplotlib.colors import ListedColormap

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASSIFIED_DIR = os.path.join(
    BASE_DIR, "data", "classified", "mangaluru"
)

BOUNDARY_FILE = os.path.join(
    BASE_DIR, "data", "boundary", "mangaluru_taluk.gpkg"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "outputs", "maps"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# LULC classes
# ---------------------------------------------------------

class_names = {
    1: "Water",
    2: "Vegetation / Forest",
    3: "Agriculture",
    4: "Built-up",
    5: "Barren / Open"
}

# Transparent background + LULC colors
cmap = ListedColormap([
    "white",
    "#2196F3",   # Water
    "#228B22",   # Vegetation
    "#FFD700",   # Agriculture
    "#E53935",   # Built-up
    "#A67C52"    # Barren/Open
])

# ---------------------------------------------------------
# Load Mangaluru boundary
# ---------------------------------------------------------

boundary = gpd.read_file(BOUNDARY_FILE)

if boundary.empty:
    raise ValueError("Mangaluru boundary is empty.")

print("Boundary CRS:", boundary.crs)
print("Boundary:", boundary["KGISTalukN"].tolist())

# ---------------------------------------------------------
# Process each year
# ---------------------------------------------------------

for year in [2018, 2022, 2025]:

    input_file = os.path.join(
        CLASSIFIED_DIR,
        f"land_use_{year}.tif"
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        f"land_use_map_{year}.png"
    )

    print(f"\nProcessing {year}...")
    print("Input:", input_file)

    if not os.path.exists(input_file):
        print("ERROR: classified raster not found.")
        continue

    with rasterio.open(input_file) as src:

        # Make sure boundary uses same CRS as raster
        boundary_raster_crs = boundary.to_crs(src.crs)

        geometries = [
            feature.__geo_interface__
            for feature in boundary_raster_crs.geometry
        ]

        # -------------------------------------------------
        # Mask raster using actual Mangaluru polygon
        # -------------------------------------------------

        masked_data, masked_transform = mask(
            src,
            geometries,
            crop=True,
            nodata=0
        )

        classification = masked_data[0]

        # -------------------------------------------------
        # Convert outside-boundary pixels to transparent
        # -------------------------------------------------

        valid_mask = classification > 0

        # -------------------------------------------------
        # Create RGBA image
        # -------------------------------------------------

        rgba = np.zeros(
            (classification.shape[0],
             classification.shape[1],
             4),
            dtype=np.uint8
        )

        # LULC colors
        colors = {
            1: (33, 150, 243),     # Water
            2: (34, 139, 34),     # Vegetation
            3: (255, 215, 0),     # Agriculture
            4: (229, 57, 53),      # Built-up
            5: (166, 124, 82)      # Barren/Open
        }

        for class_id, color in colors.items():

            class_pixels = classification == class_id

            rgba[class_pixels, 0] = color[0]
            rgba[class_pixels, 1] = color[1]
            rgba[class_pixels, 2] = color[2]
            rgba[class_pixels, 3] = 255

        # Outside Mangaluru = fully transparent
        rgba[~valid_mask, 3] = 0

        # -------------------------------------------------
        # Save PNG
        # -------------------------------------------------

        plt.figure(figsize=(10, 10))

        plt.imshow(rgba, origin="upper")

        plt.axis("off")

        plt.tight_layout(pad=0)

        plt.savefig(
            output_file,
            dpi=200,
            bbox_inches="tight",
            pad_inches=0,
            transparent=True
        )

        plt.close()

        print("Saved:", output_file)

print("\nAll Mangaluru classification maps generated.")