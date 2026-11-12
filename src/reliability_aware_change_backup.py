import os
import numpy as np
import pandas as pd
import rasterio

BASE_DIR = r"D:\Intelligent_Land_Use_Change_Analysis"

CLASSIFIED_DIR = os.path.join(
    BASE_DIR, "data", "classified", "mangaluru"
)

CONFIDENCE_DIR = os.path.join(
    BASE_DIR, "data", "results", "confidence"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "data", "results", "reliability"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

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


def get_reliability(source_conf, destination_conf):
    """
    Reliability based on confidence of both source and destination.
    """

    if source_conf >= 0.80 and destination_conf >= 0.80:
        return "High"

    elif source_conf >= 0.60 and destination_conf >= 0.60:
        return "Moderate"

    else:
        return "Low"


for start_year, end_year in PERIODS:

    print("\n" + "=" * 90)
    print(f"RELIABILITY-AWARE CHANGE ANALYSIS: {start_year} -> {end_year}")
    print("=" * 90)

    lulc_start_path = os.path.join(
        CLASSIFIED_DIR,
        f"land_use_{start_year}.tif"
    )

    lulc_end_path = os.path.join(
        CLASSIFIED_DIR,
        f"land_use_{end_year}.tif"
    )

    conf_start_path = os.path.join(
        CONFIDENCE_DIR,
        f"lulc_confidence_{start_year}.tif"
    )

    conf_end_path = os.path.join(
        CONFIDENCE_DIR,
        f"lulc_confidence_{end_year}.tif"
    )

    with rasterio.open(lulc_start_path) as src:
        lulc_start = src.read(1)

    with rasterio.open(lulc_end_path) as src:
        lulc_end = src.read(1)

    with rasterio.open(conf_start_path) as src:
        conf_start = src.read(1)

    with rasterio.open(conf_end_path) as src:
        conf_end = src.read(1)

    # Valid pixels in both years
    valid = (
        (lulc_start > 0) &
        (lulc_end > 0) &
        (conf_start > 0) &
        (conf_end > 0)
    )

    # Changed pixels only
    changed = valid & (lulc_start != lulc_end)

    start_classes = lulc_start[changed]
    end_classes = lulc_end[changed]

    start_conf = conf_start[changed]
    end_conf = conf_end[changed]

    print(f"Valid pixels: {np.sum(valid):,}")
    print(f"Changed pixels: {np.sum(changed):,}")

    if np.sum(changed) == 0:
        print("No changes detected.")
        continue

    records = []

    for i in range(len(start_classes)):

        source_id = int(start_classes[i])
        destination_id = int(end_classes[i])

        source_conf = float(start_conf[i])
        destination_conf = float(end_conf[i])

        reliability = get_reliability(
            source_conf,
            destination_conf
        )

        records.append({
            "Source Class ID": source_id,
            "Source Class": CLASS_NAMES[source_id],
            "Destination Class ID": destination_id,
            "Destination Class": CLASS_NAMES[destination_id],
            "Source Confidence": source_conf,
            "Destination Confidence": destination_conf,
            "Mean Confidence": (
                source_conf + destination_conf
            ) / 2,
            "Reliability": reliability
        })

    df = pd.DataFrame(records)

    # ---------------------------------------------------------
    # 1. Transition-level reliability summary
    # ---------------------------------------------------------

    summary = (
        df.groupby(
            [
                "Source Class",
                "Destination Class"
            ]
        )
        .agg(
            Changed_Pixels=("Reliability", "size"),
            Mean_Source_Confidence=(
                "Source Confidence",
                "mean"
            ),
            Mean_Destination_Confidence=(
                "Destination Confidence",
                "mean"
            ),
            Mean_Confidence=(
                "Mean Confidence",
                "mean"
            )
        )
        .reset_index()
    )

    # Reliability counts
    reliability_counts = (
        df.groupby(
            [
                "Source Class",
                "Destination Class",
                "Reliability"
            ]
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    summary = summary.merge(
        reliability_counts,
        on=["Source Class", "Destination Class"],
        how="left"
    )

    for column in ["High", "Moderate", "Low"]:
        if column not in summary.columns:
            summary[column] = 0

    summary["High (%)"] = (
        summary["High"] /
        summary["Changed_Pixels"] * 100
    )

    summary["Moderate (%)"] = (
        summary["Moderate"] /
        summary["Changed_Pixels"] * 100
    )

    summary["Low (%)"] = (
        summary["Low"] /
        summary["Changed_Pixels"] * 100
    )

    summary = summary.sort_values(
        "Changed_Pixels",
        ascending=False
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        f"reliability_summary_{start_year}_{end_year}.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # ---------------------------------------------------------
    # 2. Overall reliability
    # ---------------------------------------------------------

    overall = (
        df["Reliability"]
        .value_counts()
        .reindex(
            ["High", "Moderate", "Low"],
            fill_value=0
        )
    )

    overall_df = pd.DataFrame({
        "Period": [
            f"{start_year}-{end_year}"
        ] * 3,
        "Reliability": overall.index,
        "Changed Pixels": overall.values
    })

    total_changed = overall_df["Changed Pixels"].sum()

    overall_df["Percentage"] = (
        overall_df["Changed Pixels"] /
        total_changed * 100
    )

    overall_path = os.path.join(
        OUTPUT_DIR,
        f"overall_reliability_{start_year}_{end_year}.csv"
    )

    overall_df.to_csv(
        overall_path,
        index=False
    )

    # ---------------------------------------------------------
    # 3. Save detailed changed-pixel information
    # ---------------------------------------------------------

    detail_path = os.path.join(
        OUTPUT_DIR,
        f"change_details_{start_year}_{end_year}.csv"
    )

    df.to_csv(
        detail_path,
        index=False
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print("\nOverall reliability:")
    print(
        overall_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    print("\nTop transitions:")
    print(
        summary[
            [
                "Source Class",
                "Destination Class",
                "Changed_Pixels",
                "Mean_Confidence",
                "High (%)",
                "Moderate (%)",
                "Low (%)"
            ]
        ]
        .head(10)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    print("\nSaved:")
    print(summary_path)
    print(overall_path)
    print(detail_path)


print("\n" + "=" * 90)
print("RELIABILITY-AWARE CHANGE ANALYSIS COMPLETED")
print("=" * 90)