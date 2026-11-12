import os
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REFERENCE_CSV = os.path.join(
    PROJECT_ROOT,
    "data",
    "results",
    "crop_reference",
    "mangaluru_crop_reference_points.csv"
)

IMAGE_2022 = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "mangaluru",
    "sentinel2_2022_mangaluru.tif"
)

LULC_2022 = os.path.join(
    PROJECT_ROOT,
    "data",
    "classified",
    "mangaluru",
    "land_use_2022.tif"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "results",
    "crop_reference"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "crop_training_samples_2022.csv"
)


# ---------------------------------------------------------
# Read crop reference points
# ---------------------------------------------------------

print("Reading crop reference points...")

df = pd.read_csv(REFERENCE_CSV)

print("Reference records:", len(df))
print("Columns:", list(df.columns))


# ---------------------------------------------------------
# Detect required columns
# ---------------------------------------------------------

def find_column(columns, candidates):
    lower_map = {str(c).lower(): c for c in columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    for c in columns:
        c_low = str(c).lower()
        for candidate in candidates:
            if candidate.lower() in c_low:
                return c

    return None


lat_col = find_column(
    df.columns,
    ["latitude", "lat"]
)

lon_col = find_column(
    df.columns,
    ["longitude", "longtitude", "lon", "long"]
)

crop_col = find_column(
    df.columns,
    ["crop_class", "cropname", "crop_name", "crop"]
)


if lat_col is None or lon_col is None or crop_col is None:
    raise RuntimeError(
        f"Could not identify required columns.\n"
        f"Latitude={lat_col}, Longitude={lon_col}, Crop={crop_col}\n"
        f"Available columns={list(df.columns)}"
    )


print("Latitude column :", lat_col)
print("Longitude column:", lon_col)
print("Crop column     :", crop_col)


# ---------------------------------------------------------
# Keep only the three crop classes for now
# ---------------------------------------------------------

target_classes = {
    "Coconut",
    "Arecanut",
    "Paddy"
}

df = df[df[crop_col].isin(target_classes)].copy()

print("\nCandidate crop records before raster filtering:")
print(df[crop_col].value_counts())


# ---------------------------------------------------------
# Open Sentinel-2 image and LULC classification
# ---------------------------------------------------------

print("\nOpening 2022 Sentinel-2 image...")

with rasterio.open(IMAGE_2022) as src_img, \
     rasterio.open(LULC_2022) as src_lulc:

    print("Image CRS:", src_img.crs)
    print("Image size:", src_img.width, "x", src_img.height)
    print("LULC CRS:", src_lulc.crs)

    if src_img.crs != src_lulc.crs:
        raise RuntimeError("Sentinel-2 and LULC CRS do not match.")

    transformer = Transformer.from_crs(
        "EPSG:4326",
        src_img.crs,
        always_xy=True
    )

    feature_rows = []

    skipped_outside = 0
    skipped_nodata = 0
    skipped_non_agriculture = 0

    for _, row in df.iterrows():

        try:
            lat = float(row[lat_col])
            lon = float(row[lon_col])
        except (ValueError, TypeError):
            continue

        x, y = transformer.transform(lon, lat)

        # Check raster bounds
        if not (
            src_img.bounds.left <= x <= src_img.bounds.right
            and src_img.bounds.bottom <= y <= src_img.bounds.top
        ):
            skipped_outside += 1
            continue

        # Sample Sentinel-2 bands
        values = next(src_img.sample([(x, y)]))

        # Expected order:
        # B02 Blue
        # B03 Green
        # B04 Red
        # B08 NIR
        # B11 SWIR
        if len(values) < 5:
            raise RuntimeError(
                f"Expected 5 bands, found {len(values)}"
            )

        blue, green, red, nir, swir = [
            float(v) for v in values[:5]
        ]

        # Skip invalid pixels
        vals = np.array(
            [blue, green, red, nir, swir],
            dtype=np.float32
        )

        if not np.all(np.isfinite(vals)):
            skipped_nodata += 1
            continue

        if np.all(vals == 0):
            skipped_nodata += 1
            continue

        # Sample existing 2022 LULC classification
        lulc_value = next(
            src_lulc.sample([(x, y)])
        )[0]

        if not np.isfinite(lulc_value) or lulc_value == 0:
            skipped_nodata += 1
            continue

        # Class 3 = Agriculture
        if int(lulc_value) != 3:
            skipped_non_agriculture += 1
            continue

        # Spectral indices
        ndvi = (nir - red) / (nir + red + 1e-9)
        ndwi = (green - nir) / (green + nir + 1e-9)
        ndbi = (swir - nir) / (swir + nir + 1e-9)

        feature_rows.append({
            "crop_class": row[crop_col],
            "latitude": lat,
            "longitude": lon,
            "blue": blue,
            "green": green,
            "red": red,
            "nir": nir,
            "swir": swir,
            "ndvi": ndvi,
            "ndwi": ndwi,
            "ndbi": ndbi
        })


# ---------------------------------------------------------
# Save result
# ---------------------------------------------------------

training_df = pd.DataFrame(feature_rows)

print("\n========================================")
print("Crop training sample extraction complete")
print("========================================")

print("Usable training samples:", len(training_df))
print("Skipped outside raster:", skipped_outside)
print("Skipped invalid pixels :", skipped_nodata)
print("Skipped non-agriculture:", skipped_non_agriculture)

if len(training_df) == 0:
    raise RuntimeError(
        "No usable crop training samples were found."
    )

print("\nSamples by crop:")
print(training_df["crop_class"].value_counts())

os.makedirs(OUTPUT_DIR, exist_ok=True)

training_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\nSaved:")
print(OUTPUT_CSV)
