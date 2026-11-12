import os
import numpy as np
import rasterio


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = r"D:\Intelligent_Land_Use_Change_Analysis"

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

TRAINING_DIR = os.path.join(
    BASE_DIR,
    "data",
    "training"
)

os.makedirs(TRAINING_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

YEARS = [2018, 2022, 2025]

# Maximum number of samples for each class from each year
SAMPLES_PER_CLASS = 5000


# ============================================================
# CALCULATE SPECTRAL INDICES
# ============================================================

def calculate_indices(bands):

    """
    Sentinel-2 bands:

    0 = Blue
    1 = Green
    2 = Red
    3 = NIR
    4 = SWIR

    Returns:
        NDVI
        NDWI
        NDBI
    """

    blue = bands[0]
    green = bands[1]
    red = bands[2]
    nir = bands[3]
    swir = bands[4]

    ndvi = (
        (nir - red) /
        (nir + red + 1e-10)
    )

    ndwi = (
        (green - nir) /
        (green + nir + 1e-10)
    )

    ndbi = (
        (swir - nir) /
        (swir + nir + 1e-10)
    )

    return ndvi, ndwi, ndbi


# ============================================================
# CREATE TRAINING SAMPLES FOR ONE YEAR
# ============================================================

def create_samples(year):

    input_file = os.path.join(
        PROCESSED_DIR,
        f"sentinel2_{year}.tif"
    )

    print()
    print("=" * 55)
    print(f"Preparing training samples for {year}...")
    print("=" * 55)

    all_features = []
    all_labels = []

    with rasterio.open(input_file) as src:

        # ----------------------------------------------------
        # Read image block by block
        # ----------------------------------------------------

        for _, window in src.block_windows(1):

            data = src.read(
                window=window
            ).astype(np.float32)

            if data.size == 0:
                continue

            blue = data[0]
            green = data[1]
            red = data[2]
            nir = data[3]
            swir = data[4]

            # Calculate spectral indices
            ndvi, ndwi, ndbi = calculate_indices(data)

            # Flatten
            blue = blue.ravel()
            green = green.ravel()
            red = red.ravel()
            nir = nir.ravel()
            swir = swir.ravel()

            ndvi = ndvi.ravel()
            ndwi = ndwi.ravel()
            ndbi = ndbi.ravel()

            # ------------------------------------------------
            # Remove invalid / NoData pixels
            # ------------------------------------------------

            valid = (
                np.isfinite(blue) &
                np.isfinite(green) &
                np.isfinite(red) &
                np.isfinite(nir) &
                np.isfinite(swir) &
                np.isfinite(ndvi) &
                np.isfinite(ndwi) &
                np.isfinite(ndbi) &
                (blue > 0) &
                (green > 0) &
                (red > 0) &
                (nir > 0) &
                (swir > 0)
            )

            if not np.any(valid):
                continue

            # ------------------------------------------------
            # Valid spectral values
            # ------------------------------------------------

            v_blue = blue[valid]
            v_green = green[valid]
            v_red = red[valid]
            v_nir = nir[valid]
            v_swir = swir[valid]

            v_ndvi = ndvi[valid]
            v_ndwi = ndwi[valid]
            v_ndbi = ndbi[valid]

            features = np.column_stack([
                v_blue,
                v_green,
                v_red,
                v_nir,
                v_swir,
                v_ndvi,
                v_ndwi,
                v_ndbi
            ])

            # ------------------------------------------------
            # Adaptive percentile thresholds
            #
            # Instead of using fixed thresholds for every year,
            # thresholds are calculated from the actual image.
            # ------------------------------------------------

            ndvi_low = np.percentile(v_ndvi, 30)
            ndvi_high = np.percentile(v_ndvi, 70)

            ndwi_high = np.percentile(v_ndwi, 85)

            ndbi_high = np.percentile(v_ndbi, 75)

            ndbi_low = np.percentile(v_ndbi, 25)

            # ------------------------------------------------
            # Initial labels
            #
            # 1 = Water
            # 2 = Vegetation / Forest
            # 3 = Agriculture
            # 4 = Built-up
            # 5 = Barren / Open
            # ------------------------------------------------

            labels = np.zeros(
                len(features),
                dtype=np.uint8
            )

            # ------------------------------------------------
            # WATER
            #
            # Strong positive NDWI
            # ------------------------------------------------

            water = (
                v_ndwi >= ndwi_high
            )

            labels[water] = 1

            # ------------------------------------------------
            # VEGETATION / FOREST
            #
            # High NDVI and not water
            # ------------------------------------------------

            vegetation = (
                (v_ndvi >= ndvi_high) &
                (~water)
            )

            labels[vegetation] = 2

            # ------------------------------------------------
            # BUILT-UP
            #
            # Higher NDBI + relatively low vegetation
            # ------------------------------------------------

            builtup = (
                (v_ndbi >= ndbi_high) &
                (v_ndvi <= ndvi_low) &
                (~water) &
                (~vegetation)
            )

            labels[builtup] = 4

            # ------------------------------------------------
            # AGRICULTURE
            #
            # Moderate vegetation signal
            # ------------------------------------------------

            agriculture = (
                (v_ndvi > ndvi_low) &
                (v_ndvi < ndvi_high) &
                (~water) &
                (~vegetation) &
                (~builtup)
            )

            labels[agriculture] = 3

            # ------------------------------------------------
            # BARREN / OPEN
            #
            # Low vegetation and relatively low NDBI
            # ------------------------------------------------

            barren = (
                (v_ndvi <= ndvi_low) &
                (v_ndbi <= ndbi_low) &
                (~water) &
                (~vegetation) &
                (~builtup) &
                (~agriculture)
            )

            labels[barren] = 5

            # ------------------------------------------------
            # Keep only labelled pixels
            # ------------------------------------------------

            labelled = labels > 0

            if np.any(labelled):

                all_features.append(
                    features[labelled]
                )

                all_labels.append(
                    labels[labelled]
                )

    # ========================================================
    # COMBINE ALL WINDOWS
    # ========================================================

    if not all_features:

        raise RuntimeError(
            f"No training samples could be created for {year}."
        )

    X = np.vstack(
        all_features
    )

    y = np.concatenate(
        all_labels
    )

    print(
        f"Total initial samples for {year}: "
        f"{len(y)}"
    )

    # ========================================================
    # SHOW CLASS DISTRIBUTION
    # ========================================================

    class_names = {
        1: "Water",
        2: "Vegetation/Forest",
        3: "Agriculture",
        4: "Built-up",
        5: "Barren/Open"
    }

    rng = np.random.default_rng(42 + year)

    selected_indices = []

    print()
    print("Class distribution:")

    for class_id in range(1, 6):

        indices = np.where(
            y == class_id
        )[0]

        print(
            f"Class {class_id} "
            f"({class_names[class_id]}): "
            f"{len(indices)} available samples"
        )

        if len(indices) == 0:

            print(
                f"WARNING: Class {class_id} "
                f"has no samples in {year}."
            )

            continue

        sample_count = min(
            SAMPLES_PER_CLASS,
            len(indices)
        )

        selected = rng.choice(
            indices,
            size=sample_count,
            replace=False
        )

        selected_indices.extend(
            selected.tolist()
        )

    # ========================================================
    # FINAL SAMPLES
    # ========================================================

    selected_indices = np.array(
        selected_indices,
        dtype=np.int64
    )

    X = X[selected_indices]
    y = y[selected_indices]

    # Shuffle
    shuffle = rng.permutation(
        len(y)
    )

    X = X[shuffle]
    y = y[shuffle]

    print(
        f"Final samples for {year}: "
        f"{len(y)}"
    )

    # ========================================================
    # SAVE YEAR-SPECIFIC TRAINING DATA
    # ========================================================

    output_file = os.path.join(
        TRAINING_DIR,
        f"training_{year}.npz"
    )

    np.savez_compressed(
        output_file,
        X=X,
        y=y
    )

    print(
        f"Saved: {output_file}"
    )

    return X, y


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    all_X = []
    all_y = []

    # --------------------------------------------------------
    # Generate samples for all years
    # --------------------------------------------------------

    for year in YEARS:

        X, y = create_samples(
            year
        )

        all_X.append(X)
        all_y.append(y)

    # --------------------------------------------------------
    # Combine all years
    # --------------------------------------------------------

    X_all = np.vstack(
        all_X
    )

    y_all = np.concatenate(
        all_y
    )

    # --------------------------------------------------------
    # Shuffle combined dataset
    # --------------------------------------------------------

    rng = np.random.default_rng(
        42
    )

    shuffle = rng.permutation(
        len(y_all)
    )

    X_all = X_all[shuffle]
    y_all = y_all[shuffle]

    # --------------------------------------------------------
    # Save combined training data
    # --------------------------------------------------------

    combined_file = os.path.join(
        TRAINING_DIR,
        "training_all_years.npz"
    )

    np.savez_compressed(
        combined_file,
        X=X_all,
        y=y_all
    )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print()
    print("-" * 55)
    print("Training-data preparation completed!")
    print("-" * 55)

    print(
        f"Total training samples: "
        f"{len(y_all)}"
    )

    print(
        f"Features: "
        f"{X_all.shape[1]}"
    )

    print(
        f"Saved: "
        f"{combined_file}"
    )

    print()
    print("Final combined class distribution:")

    class_names = {
        1: "Water",
        2: "Vegetation/Forest",
        3: "Agriculture",
        4: "Built-up",
        5: "Barren/Open"
    }

    for class_id in range(1, 6):

        count = np.sum(
            y_all == class_id
        )

        print(
            f"Class {class_id} "
            f"({class_names[class_id]}): "
            f"{count}"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()