import rasterio
import numpy as np

paths = {
    "Primary": "data/processed/mangaluru/sentinel2_2022_mangaluru.tif",
    "Backup": "data/processed/mangaluru/sentinel2_2022_backup_mangaluru.tif",
}

datasets = {}
for name, path in paths.items():
    with rasterio.open(path) as ds:
        data = ds.read()
        datasets[name] = {
            "data": data,
            "transform": ds.transform,
            "crs": ds.crs,
            "shape": data.shape,
        }
        print(f"\n{name}")
        print("Shape:", data.shape)
        print("CRS:", ds.crs)
        print("Transform:", ds.transform)
        for i, band in enumerate(data, 1):
            valid = band > 0
            print(
                f"Band {i} ({ds.descriptions[i-1]}): "
                f"valid={valid.sum()}, "
                f"min={band[valid].min() if valid.any() else 'N/A'}, "
                f"max={band[valid].max() if valid.any() else 'N/A'}"
            )

p, b = datasets["Primary"], datasets["Backup"]
print("\nGrid comparison")
print("Same shape:", p["shape"] == b["shape"])
print("Same CRS:", p["crs"] == b["crs"])
print("Same transform:", p["transform"] == b["transform"])
