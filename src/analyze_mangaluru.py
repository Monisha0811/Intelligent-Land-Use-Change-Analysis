import os
import csv
import numpy as np
import rasterio


BASE_DIR = r"D:\Intelligent_Land_Use_Change_Analysis"

CLASSIFIED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "classified",
    "mangaluru"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "data",
    "results",
    "mangaluru"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


YEARS = [2018, 2022, 2025]

CLASS_NAMES = {
    1: "Water",
    2: "Vegetation/Forest",
    3: "Agriculture",
    4: "Built-up",
    5: "Barren/Open"
}


# ---------------------------------------------------------
# Load rasters
# ---------------------------------------------------------

rasters = {}
arrays = {}

for year in YEARS:

    path = os.path.join(
        CLASSIFIED_DIR,
        f"land_use_{year}.tif"
    )

    print(f"Loading {year}: {path}")

    src = rasterio.open(path)

    rasters[year] = src
    arrays[year] = src.read(1)

    print(
        "Size:",
        src.width,
        "x",
        src.height
    )

    print("CRS:", src.crs)
    print("Resolution:", src.res)


# ---------------------------------------------------------
# Check spatial consistency
# ---------------------------------------------------------

reference = rasters[2018]

for year in YEARS[1:]:

    src = rasters[year]

    if src.width != reference.width:
        raise ValueError(f"Width mismatch: {year}")

    if src.height != reference.height:
        raise ValueError(f"Height mismatch: {year}")

    if src.crs != reference.crs:
        raise ValueError(f"CRS mismatch: {year}")

    if src.transform != reference.transform:
        raise ValueError(f"Transform mismatch: {year}")


# ---------------------------------------------------------
# Common valid area
# ---------------------------------------------------------

common_valid = np.ones(
    arrays[2018].shape,
    dtype=bool
)

for year in YEARS:

    common_valid &= (
        (arrays[year] >= 1) &
        (arrays[year] <= 5)
    )


common_pixels = int(
    np.sum(common_valid)
)

print("\nCommon valid pixels:", common_pixels)


# Pixel area in km²
pixel_width = abs(reference.transform.a)
pixel_height = abs(reference.transform.e)

pixel_area_km2 = (
    pixel_width * pixel_height
) / 1_000_000.0

common_area_km2 = (
    common_pixels * pixel_area_km2
)

print(
    "Common analysis area:",
    round(common_area_km2, 4),
    "km²"
)


# ---------------------------------------------------------
# Annual class statistics
# ---------------------------------------------------------

area_csv = os.path.join(
    RESULTS_DIR,
    "mangaluru_area_statistics.csv"
)

with open(
    area_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Year",
        "Class_ID",
        "Class_Name",
        "Pixels",
        "Area_km2",
        "Percentage"
    ])

    for year in YEARS:

        data = arrays[year]

        year_common = data[common_valid]

        print(f"\n{year}")

        for class_id in range(1, 6):

            pixels = int(
                np.sum(
                    year_common == class_id
                )
            )

            area = (
                pixels *
                pixel_area_km2
            )

            percentage = (
                area /
                common_area_km2
            ) * 100

            print(
                CLASS_NAMES[class_id],
                ":",
                pixels,
                "pixels |",
                round(area, 2),
                "km² |",
                round(percentage, 2),
                "%"
            )

            writer.writerow([
                year,
                class_id,
                CLASS_NAMES[class_id],
                pixels,
                round(area, 4),
                round(percentage, 4)
            ])


print(
    "\nSaved:",
    area_csv
)


# ---------------------------------------------------------
# Transition analysis
# ---------------------------------------------------------

def analyze_transition(year1, year2):

    a = arrays[year1][common_valid]
    b = arrays[year2][common_valid]

    matrix = np.zeros(
        (5, 5),
        dtype=np.int64
    )

    for from_class in range(1, 6):

        for to_class in range(1, 6):

            count = int(
                np.sum(
                    (a == from_class) &
                    (b == to_class)
                )
            )

            matrix[
                from_class - 1,
                to_class - 1
            ] = count

    # -----------------------------------------------------
    # Matrix CSV
    # -----------------------------------------------------

    matrix_file = os.path.join(
        RESULTS_DIR,
        f"transition_matrix_{year1}_{year2}.csv"
    )

    with open(
        matrix_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "From \\ To"
            ]
            +
            [
                CLASS_NAMES[i]
                for i in range(1, 6)
            ]
        )

        for i in range(5):

            writer.writerow(
                [
                    CLASS_NAMES[i + 1]
                ]
                +
                matrix[i].tolist()
            )

    # -----------------------------------------------------
    # Detailed transition CSV
    # -----------------------------------------------------

    details_file = os.path.join(
        RESULTS_DIR,
        f"transition_details_{year1}_{year2}.csv"
    )

    with open(
        details_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "From",
            "To",
            "Pixels",
            "Area_km2"
        ])

        for from_class in range(1, 6):

            for to_class in range(1, 6):

                pixels = matrix[
                    from_class - 1,
                    to_class - 1
                ]

                area = (
                    pixels *
                    pixel_area_km2
                )

                writer.writerow([
                    CLASS_NAMES[from_class],
                    CLASS_NAMES[to_class],
                    int(pixels),
                    round(area, 4)
                ])

    # -----------------------------------------------------
    # Change summary
    # -----------------------------------------------------

    changed = int(
        np.sum(a != b)
    )

    unchanged = int(
        np.sum(a == b)
    )

    changed_area = (
        changed *
        pixel_area_km2
    )

    unchanged_area = (
        unchanged *
        pixel_area_km2
    )

    print(
        f"\n{year1} → {year2}"
    )

    print(
        "Changed pixels:",
        changed
    )

    print(
        "Changed area:",
        round(changed_area, 2),
        "km²"
    )

    print(
        "Unchanged pixels:",
        unchanged
    )

    print(
        "Unchanged area:",
        round(unchanged_area, 2),
        "km²"
    )

    print(
        "Change percentage:",
        round(
            changed /
            common_pixels *
            100,
            2
        ),
        "%"
    )

    print("Top transitions:")

    transitions = []

    for i in range(5):

        for j in range(5):

            if i == j:
                continue

            transitions.append(
                (
                    int(matrix[i, j]),
                    i + 1,
                    j + 1
                )
            )

    transitions.sort(
        reverse=True
    )

    for pixels, from_id, to_id in transitions[:5]:

        print(
            CLASS_NAMES[from_id],
            "→",
            CLASS_NAMES[to_id],
            ":",
            pixels,
            "pixels =",
            round(
                pixels *
                pixel_area_km2,
                2
            ),
            "km²"
        )

    return matrix


# ---------------------------------------------------------
# Run transition comparisons
# ---------------------------------------------------------

transition_pairs = [
    (2018, 2022),
    (2022, 2025),
    (2018, 2025)
]

for year1, year2 in transition_pairs:

    analyze_transition(
        year1,
        year2
    )


# ---------------------------------------------------------
# Close rasters
# ---------------------------------------------------------

for src in rasters.values():
    src.close()


print(
    "\nMangaluru analysis completed successfully!"
)

print(
    "Results folder:",
    RESULTS_DIR
)