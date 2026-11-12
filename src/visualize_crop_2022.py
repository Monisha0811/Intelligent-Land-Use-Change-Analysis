
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from pathlib import Path

PROJECT_ROOT = Path(r"D:\Intelligent_Land_Use_Change_Analysis")

raster_path = (
    PROJECT_ROOT / "data" / "classified" / "mangaluru"
    / "crop_2022_experimental.tif"
)

stats_path = (
    PROJECT_ROOT / "data" / "results" / "crop_reference"
    / "crop_area_statistics_2022_experimental.csv"
)

output_path = (
    PROJECT_ROOT / "outputs" / "charts"
    / "crop_2022_visual.png"
)

output_path.parent.mkdir(parents=True, exist_ok=True)

if not raster_path.exists():
    raise FileNotFoundError(f"Crop raster not found: {raster_path}")

with rasterio.open(raster_path) as src:
    crop = src.read(1)
    bounds = src.bounds
    print("Raster shape:", crop.shape)
    print("CRS:", src.crs)
    print("Bounds:", bounds)
    print("Unique values:", np.unique(crop).tolist())

crop_info = {
    1: ("Coconut", "#2ca02c"),
    2: ("Arecanut", "#1f77b4"),
    3: ("Paddy", "#f1c40f"),
}

# NoData/background is white; crop IDs are 1, 2, and 3.
cmap = ListedColormap([
    "white",
    crop_info[1][1],
    crop_info[2][1],
    crop_info[3][1],
])

fig, ax = plt.subplots(figsize=(12, 10))

ax.imshow(
    crop,
    cmap=cmap,
    interpolation="nearest",
    vmin=0,
    vmax=3,
    extent=[
        bounds.left, bounds.right,
        bounds.bottom, bounds.top
    ],
)

ax.set_title(
    "Experimental Crop Classification — Mangaluru, 2022",
    fontsize=15,
    pad=14
)
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")

legend_elements = [
    Patch(facecolor="white", edgecolor="gray", label="Background / NoData"),
    Patch(facecolor=crop_info[1][1], label="Coconut"),
    Patch(facecolor=crop_info[2][1], label="Arecanut"),
    Patch(facecolor=crop_info[3][1], label="Paddy"),
]

ax.legend(
    handles=legend_elements,
    loc="lower right",
    frameon=True
)

if stats_path.exists():
    stats = pd.read_csv(stats_path)

    summary_lines = ["Mapped crop area"]
    for _, row in stats.iterrows():
        summary_lines.append(
            f"{row['crop_class']}: {row['area_km2']:.2f} km² "
            f"({row['percentage_of_mapped_crop_area']:.2f}%)"
        )

    fig.text(
        0.02, 0.02,
        "\n".join(summary_lines),
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="gray",
            alpha=0.9
        ),
    )

fig.tight_layout(rect=[0, 0.13, 1, 1])

fig.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print("\nVisualization saved to:")
print(output_path)
