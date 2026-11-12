import os
import numpy as np
import pandas as pd
import rasterio

# ---------------------------------------------------------
# Agriculture Expansion / Contraction Analysis
#
# Classes:
# 1 = Water
# 2 = Vegetation/Forest
# 3 = Agriculture
# 4 = Built-up
# 5 = Barren/Open
# ---------------------------------------------------------

BASE_DIR = r"D:\Intelligent_Land_Use_Change_Analysis"

CLASSIFIED_DIR = os.path.join(
    BASE_DIR, "data", "classified", "mangaluru"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "data", "results", "agriculture_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

AGRICULTURE = 3

PIXEL_AREA_KM2 = 0.0001

CLASS_NAMES = {
    1: "Water",
    2: "Vegetation/Forest",
    3: "Agriculture",
    4: "Built-up",
    5: "Barren/Open"
}

PERIODS = [
    (2018, 2022),
    (2022, 2025),
    (2018, 2025)
]


def load_raster(year):
    path = os.path.join(
        CLASSIFIED_DIR,
        f"land_use_{year}.tif"
    )

    with rasterio.open(path) as src:
        return src.read(1)


def analyze_period(start_year, end_year):

    print("\n" + "=" * 60)
    print(f"Agriculture change: {start_year} -> {end_year}")
    print("=" * 60)

    start = load_raster(start_year)
    end = load_raster(end_year)

    # Common valid area
    valid = (start > 0) & (end > 0)

    start = start[valid]
    end = end[valid]

    # -----------------------------------------------------
    # Agriculture expansion:
    # Other class -> Agriculture
    # -----------------------------------------------------

    expansion_mask = (
        (end == AGRICULTURE) &
        (start != AGRICULTURE)
    )

    expansion_pixels = int(np.sum(expansion_mask))
    expansion_area = expansion_pixels * PIXEL_AREA_KM2

    # -----------------------------------------------------
    # Agriculture contraction:
    # Agriculture -> Other class
    # -----------------------------------------------------

    contraction_mask = (
        (start == AGRICULTURE) &
        (end != AGRICULTURE)
    )

    contraction_pixels = int(np.sum(contraction_mask))
    contraction_area = contraction_pixels * PIXEL_AREA_KM2

    # Net change
    start_agriculture = int(np.sum(start == AGRICULTURE))
    end_agriculture = int(np.sum(end == AGRICULTURE))

    start_area = start_agriculture * PIXEL_AREA_KM2
    end_area = end_agriculture * PIXEL_AREA_KM2

    net_change = end_area - start_area

    print(f"Start agriculture area: {start_area:.4f} km²")
    print(f"End agriculture area:   {end_area:.4f} km²")
    print(f"Expansion:              {expansion_area:.4f} km²")
    print(f"Contraction:            {contraction_area:.4f} km²")
    print(f"Net change:             {net_change:+.4f} km²")

    # -----------------------------------------------------
    # Detailed transition analysis
    # -----------------------------------------------------

    transition_results = []

    for source_class, source_name in CLASS_NAMES.items():

        if source_class == AGRICULTURE:
            continue

        mask = (
            (start == source_class) &
            (end == AGRICULTURE)
        )

        pixels = int(np.sum(mask))

        if pixels > 0:
            transition_results.append({
                "period": f"{start_year}-{end_year}",
                "transition": (
                    f"{source_name} -> Agriculture"
                ),
                "direction": "Expansion",
                "pixels": pixels,
                "area_km2": pixels * PIXEL_AREA_KM2
            })

    for destination_class, destination_name in CLASS_NAMES.items():

        if destination_class == AGRICULTURE:
            continue

        mask = (
            (start == AGRICULTURE) &
            (end == destination_class)
        )

        pixels = int(np.sum(mask))

        if pixels > 0:
            transition_results.append({
                "period": f"{start_year}-{end_year}",
                "transition": (
                    f"Agriculture -> {destination_name}"
                ),
                "direction": "Contraction",
                "pixels": pixels,
                "area_km2": pixels * PIXEL_AREA_KM2
            })

    return {
        "period": f"{start_year}-{end_year}",
        "start_area_km2": start_area,
        "end_area_km2": end_area,
        "expansion_area_km2": expansion_area,
        "contraction_area_km2": contraction_area,
        "net_change_km2": net_change
    }, transition_results


# ---------------------------------------------------------
# Run all periods
# ---------------------------------------------------------

summary_results = []
transition_results = []

for start_year, end_year in PERIODS:

    summary, transitions = analyze_period(
        start_year,
        end_year
    )

    summary_results.append(summary)
    transition_results.extend(transitions)


# Save summary
summary_df = pd.DataFrame(summary_results)

summary_file = os.path.join(
    OUTPUT_DIR,
    "agriculture_expansion_contraction_summary.csv"
)

summary_df.to_csv(summary_file, index=False)


# Save detailed transitions
transition_df = pd.DataFrame(
    transition_results
)

transition_file = os.path.join(
    OUTPUT_DIR,
    "agriculture_transition_details.csv"
)

transition_df.to_csv(
    transition_file,
    index=False
)


print("\n" + "=" * 60)
print("Agriculture expansion/contraction analysis completed.")
print("=" * 60)

print(f"\nSummary file:")
print(summary_file)

print(f"\nTransition file:")
print(transition_file)

print("\nSummary:")
print(summary_df.to_string(index=False))

print("\nDetailed transitions:")
print(
    transition_df.sort_values(
        ["period", "direction", "area_km2"],
        ascending=[True, True, False]
    ).to_string(index=False)
)