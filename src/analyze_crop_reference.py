import os
import csv
import xml.etree.ElementTree as ET
from collections import Counter

import geopandas as gpd
from shapely.geometry import Point


BASE_DIR = r"D:\Intelligent_Land_Use_Change_Analysis"

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "crop_reference"
)

BOUNDARY_FILE = os.path.join(
    BASE_DIR,
    "data",
    "boundary",
    "mangaluru_taluk.gpkg"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "results",
    "crop_reference"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


FILES = [
    "mangaluru_A_2021_22.xml",
    "mangaluru_crop_survey_2021_22_B.xml"
]


def local_name(tag):
    """Remove XML namespace from a tag name."""
    return tag.split("}")[-1].strip()


def normalize_crop(crop):
    """Group related crop names into analysis classes."""

    crop = crop.strip()

    if crop in ["Paddy-H", "Paddy-L"]:
        return "Paddy"

    if crop == "Betel Nuts (Areca nuts)":
        return "Arecanut"

    if crop == "Coconut":
        return "Coconut"

    if crop == "Cashewnuts":
        return "Cashew"

    return crop


def get_child_text(element, wanted_name):
    """Find a child element regardless of XML namespace."""

    for child in element:

        if local_name(child.tag).lower() == wanted_name.lower():

            if child.text:
                return child.text.strip()

            return ""

    return ""


def parse_xml(path):

    records = []

    print("\nReading:", os.path.basename(path))

    tree = ET.parse(path)
    root = tree.getroot()

    # Every survey record is represented by an element
    # containing Cropname, Latitude and Longitude.
    for element in root.iter():

        crop = get_child_text(
            element,
            "Cropname"
        )

        latitude_text = get_child_text(
            element,
            "Latitude"
        )

        longitude_text = get_child_text(
            element,
            "Longtitude"
        )

        # Skip elements that aren't survey records.
        if not crop:
            continue

        if not latitude_text or not longitude_text:
            continue

        try:
            latitude = float(latitude_text)
            longitude = float(longitude_text)
        except ValueError:
            continue

        records.append(
            {
                "crop_raw": crop,
                "crop_class": normalize_crop(crop),
                "latitude": latitude,
                "Longtitude": longitude,
                "village": get_child_text(
                    element,
                    "Village_Name"
                ),
                "hobli": get_child_text(
                    element,
                    "Hobli_Name"
                ),
                "crop_extent": get_child_text(
                    element,
                    "Crop_Extent"
                ),
                "survey_date": get_child_text(
                    element,
                    "CropSurveyDate"
                ),
                "season": get_child_text(
                    element,
                    "Season"
                ),
                "year": get_child_text(
                    element,
                    "Years"
                )
            }
        )

    print(
        "Usable records:",
        len(records)
    )

    return records


# ---------------------------------------------------------
# Read both datasets
# ---------------------------------------------------------

all_records = []

for filename in FILES:

    path = os.path.join(
        DATA_DIR,
        filename
    )

    if not os.path.exists(path):

        print(
            "WARNING - file not found:",
            path
        )

        continue

    all_records.extend(
        parse_xml(path)
    )


print(
    "\nTotal records with coordinates:",
    len(all_records)
)


if not all_records:

    raise RuntimeError(
        "No usable crop records were found. "
        "Check the XML files."
    )


# ---------------------------------------------------------
# Create spatial points
# ---------------------------------------------------------

gdf = gpd.GeoDataFrame(
    all_records,
    geometry=[
        Point(
            record["Longtitude"],
            record["latitude"]
        )
        for record in all_records
    ],
    crs="EPSG:4326"
)


# ---------------------------------------------------------
# Load Mangaluru boundary
# ---------------------------------------------------------

boundary = gpd.read_file(
    BOUNDARY_FILE
).to_crs("EPSG:4326")


# Merge all boundary parts into one geometry.
boundary_geometry = boundary.geometry.union_all()


# ---------------------------------------------------------
# Check points inside Mangaluru
# ---------------------------------------------------------

inside_mask = gdf.geometry.within(
    boundary_geometry
)

inside_gdf = gdf.loc[
    inside_mask
].copy()

outside_count = (
    len(gdf) -
    len(inside_gdf)
)

print(
    "Inside Mangaluru:",
    len(inside_gdf)
)

print(
    "Outside Mangaluru:",
    outside_count
)


# ---------------------------------------------------------
# Crop class counts
# ---------------------------------------------------------

counts = Counter(
    inside_gdf["crop_class"]
)

print(
    "\nCrop classes inside Mangaluru:"
)

for crop, count in counts.most_common():

    print(
        f"{crop}: {count}"
    )


# ---------------------------------------------------------
# Candidate crop classes
# ---------------------------------------------------------

candidate_classes = [
    "Paddy",
    "Coconut",
    "Arecanut",
    "Cashew"
]

print(
    "\nCandidate classes:"
)

for crop in candidate_classes:

    print(
        f"{crop}: {counts.get(crop, 0)}"
    )


# ---------------------------------------------------------
# Save CSV
# ---------------------------------------------------------

csv_file = os.path.join(
    OUTPUT_DIR,
    "mangaluru_crop_reference_points.csv"
)

inside_gdf.drop(
    columns="geometry"
).to_csv(
    csv_file,
    index=False,
    encoding="utf-8"
)

print(
    "\nSaved CSV:",
    csv_file
)


# ---------------------------------------------------------
# Save GeoJSON
# ---------------------------------------------------------

geojson_file = os.path.join(
    OUTPUT_DIR,
    "mangaluru_crop_reference_points.geojson"
)

inside_gdf.to_file(
    geojson_file,
    driver="GeoJSON"
)

print(
    "Saved GeoJSON:",
    geojson_file
)


# ---------------------------------------------------------
# Save crop summary
# ---------------------------------------------------------

summary_file = os.path.join(
    OUTPUT_DIR,
    "crop_reference_summary.csv"
)

with open(
    summary_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Crop_Class",
        "Sample_Count"
    ])

    for crop, count in counts.most_common():

        writer.writerow([
            crop,
            count
        ])


print(
    "Saved summary:",
    summary_file
)

print(
    "\nCrop reference analysis completed successfully!"
)
