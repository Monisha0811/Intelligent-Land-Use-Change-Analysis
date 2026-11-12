import os
import xml.etree.ElementTree as ET
from collections import Counter


BASE_DIR = r"D:\Intelligent_Land_Use_Change_Analysis"

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "crop_reference"
)

FILES = [
    "mangaluru_A_2021_22",
    "mangaluru_crop_survey_2021_22_B"
]


def inspect_file(path):
    print("\n" + "=" * 70)
    print("FILE:", os.path.basename(path))
    print("=" * 70)

    if not os.path.exists(path):
        print("NOT FOUND:", path)
        return

    print("File size:", round(os.path.getsize(path) / 1024, 2), "KB")

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception as e:
        print("XML parsing failed:")
        print(e)
        return

    crops = []
    records = 0

    for output in root.iter():

        if not output.tag.endswith("output"):
            continue

        records += 1

        crop_name = None

        for child in output:

            if child.tag.endswith("Cropname"):
                crop_name = (
                    child.text.strip()
                    if child.text
                    else ""
                )

        if crop_name:
            crops.append(crop_name)

    counts = Counter(crops)

    print("Total records:", records)
    print("Records with crop name:", len(crops))

    print("\nCrop categories:")

    for crop, count in counts.most_common():
        print(f"{crop}: {count}")


for base_name in FILES:

    # Try .xml first, then .txt
    possible_files = [
        os.path.join(DATA_DIR, base_name + ".xml"),
        os.path.join(DATA_DIR, base_name + ".txt")
    ]

    found = None

    for path in possible_files:

        if os.path.exists(path):
            found = path
            break

    if found is None:

        print("\nFile not found for:", base_name)

        continue

    inspect_file(found)


print("\n" + "=" * 70)
print("Inspection completed.")
print("=" * 70)