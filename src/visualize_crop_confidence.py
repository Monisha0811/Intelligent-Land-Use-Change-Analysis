from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent

CROP_PATH = ROOT / "data/classified/mangaluru/crop_2022_experimental.tif"
CONF_PATH = ROOT / "data/results/crop_reference/crop_confidence_2022_experimental.tif"
OUT_PATH = ROOT / "outputs/charts/crop_confidence_2022.png"

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with rasterio.open(CROP_PATH) as src:
    crop = src.read(1)
    bounds = src.bounds

with rasterio.open(CONF_PATH) as src:
    confidence = src.read(1)

if crop.shape != confidence.shape:
    raise ValueError("Crop and confidence rasters are not aligned.")

# 0 = background, 1 = high, 2 = medium, 3 = low confidence
diagnostic = np.zeros(crop.shape, dtype=np.uint8)
valid = np.isin(crop, [1, 2, 3]) & (confidence >= 0) & (confidence <= 1)

diagnostic[valid & (confidence >= 0.80)] = 1
diagnostic[valid & (confidence >= 0.60) & (confidence < 0.80)] = 2
diagnostic[valid & (confidence < 0.60)] = 3

cmap = ListedColormap([
    "#ffffff",
    "#2e8b57",
    "#e5a33d",
    "#c65b58",
])

fig, ax = plt.subplots(figsize=(11, 9))

ax.imshow(
    diagnostic,
    cmap=cmap,
    vmin=0,
    vmax=3,
    interpolation="nearest",
    extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
)

ax.set_title("Crop Classification Confidence — Mangaluru, 2022", fontsize=14)
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")

legend = [
    Patch(facecolor="#2e8b57", label="High (≥ 0.80)"),
    Patch(facecolor="#e5a33d", label="Medium (0.60–<0.80)"),
    Patch(facecolor="#c65b58", label="Low (< 0.60)"),
]

ax.legend(handles=legend, loc="lower right", frameon=True)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
plt.close(fig)

print("Saved diagnostic map:", OUT_PATH)
print("High confidence:", int(np.count_nonzero(diagnostic == 1)))
print("Medium confidence:", int(np.count_nonzero(diagnostic == 2)))
print("Low confidence:", int(np.count_nonzero(diagnostic == 3)))
print("Reminder: confidence is not verified accuracy.")
