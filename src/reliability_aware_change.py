
import os
import csv
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
    5: "Barren/Open",
}
VALID_CLASSES = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
RELIABILITY_LEVELS = ["High", "Moderate", "Low"]
BLOCK_ROWS = 128


def get_reliability(source_conf, destination_conf):
    """Apply the project's chosen operational confidence thresholds."""
    minimum_confidence = min(source_conf, destination_conf)

    if minimum_confidence >= 0.80:
        return "High"
    if minimum_confidence >= 0.60:
        return "Moderate"
    return "Low"


def validate_rasters(paths):
    """Check dimensions, CRS, and transform across all four rasters."""
    with rasterio.open(paths[0]) as reference:
        ref_shape = reference.shape
        ref_crs = reference.crs
        ref_transform = reference.transform

    for path in paths[1:]:
        with rasterio.open(path) as raster:
            if raster.shape != ref_shape:
                raise ValueError(f"Raster dimensions do not match: {path}")
            if raster.crs != ref_crs:
                raise ValueError(f"Raster CRS does not match: {path}")
            if raster.transform != ref_transform:
                raise ValueError(f"Raster transform does not match: {path}")


def process_period(start_year, end_year):
    print("\n" + "=" * 80)
    print(f"RELIABILITY-AWARE CHANGE: {start_year} -> {end_year}")
    print("=" * 80)

    paths = [
        os.path.join(
            CLASSIFIED_DIR, f"land_use_{start_year}.tif"
        ),
        os.path.join(
            CLASSIFIED_DIR, f"land_use_{end_year}.tif"
        ),
        os.path.join(
            CONFIDENCE_DIR, f"lulc_confidence_{start_year}.tif"
        ),
        os.path.join(
            CONFIDENCE_DIR, f"lulc_confidence_{end_year}.tif"
        ),
    ]

    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required raster not found: {path}")

    validate_rasters(paths)

    # Keys: source class, destination class.
    stats = {}
    total_valid = 0
    total_changed = 0
    reliability_totals = {level: 0 for level in RELIABILITY_LEVELS}

    summary_path = os.path.join(
        OUTPUT_DIR,
        f"reliability_summary_{start_year}_{end_year}.csv",
    )
    overall_path = os.path.join(
        OUTPUT_DIR,
        f"overall_reliability_{start_year}_{end_year}.csv",
    )
    detail_path = os.path.join(
        OUTPUT_DIR,
        f"change_details_{start_year}_{end_year}.csv",
    )

    detail_fields = [
        "Source Class ID",
        "Source Class",
        "Destination Class ID",
        "Destination Class",
        "Source Confidence",
        "Destination Confidence",
        "Mean Confidence",
        "Reliability",
    ]

    with (
        rasterio.open(paths[0]) as src_start,
        rasterio.open(paths[1]) as src_end,
        rasterio.open(paths[2]) as conf_start,
        rasterio.open(paths[3]) as conf_end,
        open(detail_path, "w", newline="", encoding="utf-8-sig") as detail_file,
    ):
        writer = csv.DictWriter(detail_file, fieldnames=detail_fields)
        writer.writeheader()

        for row in range(0, src_start.height, BLOCK_ROWS):
            height = min(BLOCK_ROWS, src_start.height - row)
            window = rasterio.windows.Window(
                0, row, src_start.width, height
            )

            a = src_start.read(1, window=window)
            b = src_end.read(1, window=window)
            ca = conf_start.read(1, window=window).astype(np.float32)
            cb = conf_end.read(1, window=window).astype(np.float32)

            valid = (
                np.isin(a, VALID_CLASSES)
                & np.isin(b, VALID_CLASSES)
                & np.isfinite(ca)
                & np.isfinite(cb)
                & (ca > 0)
                & (cb > 0)
                & (ca <= 1)
                & (cb <= 1)
            )

            total_valid += int(valid.sum())
            changed = valid & (a != b)
            total_changed += int(changed.sum())

            if not changed.any():
                continue

            source_ids = a[changed].astype(np.uint8)
            destination_ids = b[changed].astype(np.uint8)
            source_conf = ca[changed]
            destination_conf = cb[changed]
            mean_conf = (source_conf + destination_conf) / 2

            levels = np.where(
                (source_conf >= 0.80) & (destination_conf >= 0.80),
                "High",
                np.where(
                    (source_conf >= 0.60) & (destination_conf >= 0.60),
                    "Moderate",
                    "Low",
                ),
            )

            # Aggregate per transition and reliability category.
            pair_codes = source_ids.astype(np.int16) * 10 + destination_ids
            for pair_code in np.unique(pair_codes):
                pair_mask = pair_codes == pair_code
                source_id = int(pair_code // 10)
                destination_id = int(pair_code % 10)
                key = (source_id, destination_id)

                if key not in stats:
                    stats[key] = {
                        "count": 0,
                        "source_conf_sum": 0.0,
                        "destination_conf_sum": 0.0,
                        "mean_conf_sum": 0.0,
                        "High": 0,
                        "Moderate": 0,
                        "Low": 0,
                    }

                item = stats[key]
                item["count"] += int(pair_mask.sum())
                item["source_conf_sum"] += float(source_conf[pair_mask].sum())
                item["destination_conf_sum"] += float(
                    destination_conf[pair_mask].sum()
                )
                item["mean_conf_sum"] += float(mean_conf[pair_mask].sum())

                for level in RELIABILITY_LEVELS:
                    count = int(np.count_nonzero(pair_mask & (levels == level)))
                    item[level] += count
                    reliability_totals[level] += count

            # Stream detailed records rather than storing them in memory.
            indices = np.flatnonzero(changed)
            for index in indices:
                source_id = int(a.flat[index])
                destination_id = int(b.flat[index])
                source_value = float(ca.flat[index])
                destination_value = float(cb.flat[index])
                writer.writerow({
                    "Source Class ID": source_id,
                    "Source Class": CLASS_NAMES[source_id],
                    "Destination Class ID": destination_id,
                    "Destination Class": CLASS_NAMES[destination_id],
                    "Source Confidence": source_value,
                    "Destination Confidence": destination_value,
                    "Mean Confidence": (source_value + destination_value) / 2,
                    "Reliability": get_reliability(
                        source_value, destination_value
                    ),
                })

    print(f"Valid pixels: {total_valid:,}")
    print(f"Changed pixels: {total_changed:,}")

    if total_changed == 0:
        print("No changes detected. Detailed CSV is empty.")
        return

    rows = []
    for (source_id, destination_id), item in stats.items():
        count = item["count"]
        record = {
            "Source Class": CLASS_NAMES[source_id],
            "Destination Class": CLASS_NAMES[destination_id],
            "Changed_Pixels": count,
            "Mean_Source_Confidence": item["source_conf_sum"] / count,
            "Mean_Destination_Confidence": item["destination_conf_sum"] / count,
            "Mean_Confidence": item["mean_conf_sum"] / count,
            "High": item["High"],
            "Moderate": item["Moderate"],
            "Low": item["Low"],
            "High (%)": item["High"] / count * 100,
            "Moderate (%)": item["Moderate"] / count * 100,
            "Low (%)": item["Low"] / count * 100,
        }
        rows.append(record)

    summary = pd.DataFrame(rows).sort_values(
        "Changed_Pixels", ascending=False
    )
    summary.to_csv(summary_path, index=False)

    overall_rows = []
    for level in RELIABILITY_LEVELS:
        count = reliability_totals[level]
        overall_rows.append({
            "Period": f"{start_year}-{end_year}",
            "Reliability": level,
            "Changed Pixels": count,
            "Percentage": count / total_changed * 100,
        })

    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(overall_path, index=False)

    print("\nOverall reliability:")
    print(overall_df.to_string(
        index=False, float_format=lambda value: f"{value:.2f}"
    ))

    print("\nTop transitions:")
    print(summary[[
        "Source Class",
        "Destination Class",
        "Changed_Pixels",
        "Mean_Confidence",
        "High (%)",
        "Moderate (%)",
        "Low (%)",
    ]].head(10).to_string(
        index=False, float_format=lambda value: f"{value:.2f}"
    ))

    print("\nSaved:")
    print(summary_path)
    print(overall_path)
    print(detail_path)


for start_year, end_year in [(2018, 2022), (2022, 2025), (2018, 2025)]:
    process_period(start_year, end_year)

print("\nReliability-aware change analysis completed.")
