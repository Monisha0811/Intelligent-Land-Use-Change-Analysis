"""
Dynamic Crop Classification and Analysis Service (Phase 5).

Classifies agricultural pixels into three distinct plantation/crop categories:
Coconut, Arecanut, and Paddy, using the verified Random Forest crop model.
Generates experimental crop distribution rasters, area summaries, and multi-temporal comparisons.
"""

import os
import logging
from pathlib import Path
from typing import Callable, Dict, Any, Optional, List, Tuple, Union

import numpy as np
import pandas as pd
import joblib
from PIL import Image

try:
    import rasterio
    from rasterio.windows import Window
    HAS_RASTERIO = True
except (ImportError, Exception):
    rasterio = None
    Window = None
    HAS_RASTERIO = False

from src.config import (
    BASE_DIR,
    PROCESSED_MANGALURU_DIR,
    CLASSIFIED_MANGALURU_DIR,
    CROP_MODEL_PATH,
    RESULTS_DIR,
    MAPS_DIR,
    CHARTS_DIR,
    CROP_CLASSES,
    CROP_COLORS,
    PIXEL_AREA_KM2,
    RF_FEATURE_NAMES,
)

logger = logging.getLogger("crop_service")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [CropService] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

CROP_RESULT_DIR = RESULTS_DIR / "crop_reference"
CROP_RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Exact visualization colors matching existing 2022 crop map:
# 1: Coconut #2ca02c (Green), 2: Arecanut #1f77b4 (Blue), 3: Paddy #f1c40f (Yellow)
CROP_RGB_COLORS = {
    0: (255, 255, 255),   # NoData / Non-Agricultural (White)
    1: (44, 160, 44),     # Coconut
    2: (31, 119, 180),    # Arecanut
    3: (241, 196, 15)     # Paddy
}


class CropService:
    """Service handling multi-temporal crop classification and area analysis."""

    def __init__(self):
        self._model = None

    def get_model(self):
        """Lazy load and cache the trained Random Forest crop classification model."""
        if self._model is None:
            if not CROP_MODEL_PATH.exists():
                raise FileNotFoundError(f"Trained crop model not found at {CROP_MODEL_PATH}")
            logger.info(f"Loading trained crop Random Forest model from {CROP_MODEL_PATH}...")
            loaded = joblib.load(str(CROP_MODEL_PATH))
            self._model = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
            logger.info("Crop Random Forest model loaded successfully.")
        return self._model

    def list_available_crop_years(self) -> List[str]:
        """
        List years that have either existing crop outputs or both
        processed imagery and LULC classification rasters ready for crop analysis.
        """
        crop_rasters = list(CLASSIFIED_MANGALURU_DIR.glob("crop_*.tif"))
        crop_years = set()
        for f in crop_rasters:
            # Match crop_2022_experimental.tif or crop_2022.tif
            parts = f.stem.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                crop_years.add(parts[1])

        # Also find any year ready for processing (has both sentinel2_{year}_mangaluru.tif and land_use_{year}.tif)
        lulc_rasters = list(CLASSIFIED_MANGALURU_DIR.glob("land_use_*.tif"))
        ready_years = set()
        for f in lulc_rasters:
            tag = f.stem.replace("land_use_", "")
            img_path = PROCESSED_MANGALURU_DIR / f"sentinel2_{tag}_mangaluru.tif"
            if img_path.exists():
                ready_years.add(tag)

        combined = sorted(list(crop_years | ready_years), key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
        return combined

    def calculate_spectral_indices(
        self,
        green: np.ndarray,
        red: np.ndarray,
        nir: np.ndarray,
        swir: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute NDVI, NDWI, and NDBI using 1e-9 epsilon matching classify_crops_2022.py."""
        eps = 1e-9
        ndvi = (nir - red) / (nir + red + eps)
        ndwi = (green - nir) / (green + nir + eps)
        ndbi = (swir - nir) / (swir + nir + eps)
        return ndvi, ndwi, ndbi

    def classify_crops(
        self,
        year: Union[str, int],
        progress_callback: Optional[Callable[[int, str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None
    ) -> Dict[str, Any]:
        """
        Perform crop classification on agricultural pixels for a designated year.
        Generates:
        1. Classified crop raster: data/classified/mangaluru/crop_{year}_experimental.tif
        2. Area statistics CSV: data/results/crop_reference/crop_area_statistics_{year}_experimental.csv
        3. Visual map image: outputs/maps/crop_map_{year}_experimental.png & outputs/charts/crop_{year}_visual.png
        """
        def report(pct: int, msg: str):
            if progress_callback:
                progress_callback(pct, msg)

        def check_abort():
            if is_cancelled and is_cancelled():
                raise RuntimeError("Crop classification cancelled by user.")

        year_str = str(year).strip()
        img_path = PROCESSED_MANGALURU_DIR / f"sentinel2_{year_str}_mangaluru.tif"
        lulc_path = CLASSIFIED_MANGALURU_DIR / f"land_use_{year_str}.tif"

        if not img_path.exists():
            raise FileNotFoundError(f"Processed 5-band image not found: {img_path}")
        if not lulc_path.exists():
            raise FileNotFoundError(f"Classified LULC raster not found: {lulc_path}")

        output_raster_path = CLASSIFIED_MANGALURU_DIR / f"crop_{year_str}_experimental.tif"
        output_stats_path = CROP_RESULT_DIR / f"crop_area_statistics_{year_str}_experimental.csv"
        output_map_path = MAPS_DIR / f"crop_map_{year_str}_experimental.png"
        output_chart_path = CHARTS_DIR / f"crop_{year_str}_visual.png"

        report(10, f"Loading trained crop Random Forest model for {year_str}...")
        check_abort()
        model = self.get_model()

        report(20, "Validating Sentinel-2 imagery and LULC raster alignment...")
        check_abort()

        profile = None
        if HAS_RASTERIO and rasterio is not None:
            with rasterio.open(str(img_path)) as src_img, rasterio.open(str(lulc_path)) as src_lulc:
                if (src_img.width != src_lulc.width or src_img.height != src_lulc.height):
                    raise RuntimeError(
                        f"Raster dimensions mismatch: Image ({src_img.width}x{src_img.height}) "
                        f"vs LULC ({src_lulc.width}x{src_lulc.height})"
                    )
                profile = src_img.profile.copy()
                height, width = src_img.height, src_img.width
                if src_img.count >= 5:
                    img_data = src_img.read([1, 2, 3, 4, 5])
                else:
                    from PIL import ImageSequence
                    pil_img = Image.open(str(img_path))
                    pages = [np.array(p, dtype=np.float32) for p in ImageSequence.Iterator(pil_img)]
                    if len(pages) >= 5:
                        img_data = np.stack(pages[:5], axis=0)
                    else:
                        raise ValueError(f"Expected at least 5 bands in image {img_path.name}, found {src_img.count}")
                lulc_data = src_lulc.read(1)
        else:
            # Fallback reading via PIL
            pil_lulc = Image.open(str(lulc_path))
            width, height = pil_lulc.size
            lulc_data = np.array(pil_lulc, dtype=np.uint8)

            from PIL import ImageSequence
            pil_img = Image.open(str(img_path))
            pages = [np.array(p, dtype=np.float32) for p in ImageSequence.Iterator(pil_img)]
            if len(pages) >= 5:
                img_data = np.stack(pages[:5], axis=0)
            elif len(pages) == 1 and pages[0].ndim == 3 and pages[0].shape[2] >= 5:
                img_data = np.transpose(pages[0][:, :, :5], (2, 0, 1))
            else:
                img_data = np.zeros((5, height, width), dtype=np.float32)

        blue = img_data[0].astype(np.float32)
        green = img_data[1].astype(np.float32)
        red = img_data[2].astype(np.float32)
        nir = img_data[3].astype(np.float32)
        swir = img_data[4].astype(np.float32)

        report(35, "Masking to agricultural pixels (LULC Class 3)...")
        check_abort()

        # Agriculture mask: Class 3 only
        agriculture_mask = (lulc_data == 3)

        # Valid spectral mask: finite and non-zero
        valid_spectral = (
            np.isfinite(blue) & np.isfinite(green) & np.isfinite(red) &
            np.isfinite(nir) & np.isfinite(swir) &
            ((blue != 0) | (green != 0) | (red != 0) | (nir != 0) | (swir != 0))
        )

        crop_mask = agriculture_mask & valid_spectral

        n_agriculture = int(np.count_nonzero(agriculture_mask))
        n_valid_crop = int(np.count_nonzero(crop_mask))
        logger.info(f"Year {year_str}: Agriculture pixels={n_agriculture:,}, Valid crop pixels={n_valid_crop:,}")

        crop_output = np.zeros((height, width), dtype=np.uint8)
        crop_counts = {1: 0, 2: 0, 3: 0}

        if n_valid_crop > 0:
            report(50, "Computing vegetation indices (NDVI, NDWI, NDBI) for crop pixels...")
            check_abort()

            ndvi, ndwi, ndbi = self.calculate_spectral_indices(green, red, nir, swir)

            report(65, f"Predicting crop classes on {n_valid_crop:,} agricultural pixels...")
            check_abort()

            X = np.column_stack([
                blue[crop_mask],
                green[crop_mask],
                red[crop_mask],
                nir[crop_mask],
                swir[crop_mask],
                ndvi[crop_mask],
                ndwi[crop_mask],
                ndbi[crop_mask]
            ]).astype(np.float32)

            # Block prediction to prevent memory spikes
            chunk_size = 200000
            predictions = np.zeros(n_valid_crop, dtype=np.uint8)
            for i in range(0, n_valid_crop, chunk_size):
                check_abort()
                end_i = min(i + chunk_size, n_valid_crop)
                pred_chunk = model.predict(X[i:end_i])
                predictions[i:end_i] = pred_chunk.astype(np.uint8)

            crop_output[crop_mask] = predictions

            unique_cls, counts = np.unique(predictions, return_counts=True)
            for u, c in zip(unique_cls, counts):
                u = int(u)
                if u in crop_counts:
                    crop_counts[u] = int(c)

        # -----------------------------------------------------------------
        # 1. Save Classified Crop Raster
        # -----------------------------------------------------------------
        report(80, f"Saving classified crop raster to {output_raster_path.name}...")
        check_abort()

        if HAS_RASTERIO and rasterio is not None and profile is not None:
            profile.update(
                count=1,
                dtype="uint8",
                nodata=0,
                compress="lzw"
            )
            with rasterio.open(str(output_raster_path), "w", **profile) as dst:
                dst.write(crop_output, 1)
                dst.set_band_description(1, "Crop: 1=Coconut, 2=Arecanut, 3=Paddy")
        else:
            pil_crop = Image.fromarray(crop_output)
            pil_crop.save(str(output_raster_path), format="TIFF", compression="tiff_lzw")

        # -----------------------------------------------------------------
        # 2. Compute and Save Crop Area Statistics CSV
        # -----------------------------------------------------------------
        report(88, "Calculating crop area statistics and proportions...")
        check_abort()

        stats_rows = []
        for crop_id, name in CROP_CLASSES.items():
            px = crop_counts[crop_id]
            area = px * PIXEL_AREA_KM2
            stats_rows.append({
                "crop_class_id": crop_id,
                "crop_class": name,
                "pixels": px,
                "area_km2": area
            })

        stats_df = pd.DataFrame(stats_rows)
        total_crop_area = stats_df["area_km2"].sum()
        if total_crop_area > 0:
            stats_df["percentage_of_mapped_crop_area"] = (stats_df["area_km2"] / total_crop_area) * 100.0
        else:
            stats_df["percentage_of_mapped_crop_area"] = 0.0

        stats_df.to_csv(str(output_stats_path), index=False)

        # -----------------------------------------------------------------
        # 3. Generate Crop Map Image
        # -----------------------------------------------------------------
        report(95, "Generating crop map PNG image...")
        check_abort()

        rgb_crop = np.zeros((height, width, 3), dtype=np.uint8)
        for crop_id, color in CROP_RGB_COLORS.items():
            rgb_crop[crop_output == crop_id] = color

        crop_pil_img = Image.fromarray(rgb_crop)
        crop_pil_img.save(str(output_map_path))
        crop_pil_img.save(str(output_chart_path))

        report(100, f"Crop classification for {year_str} completed successfully.")

        return {
            "success": True,
            "year": year_str,
            "raster_path": str(output_raster_path),
            "stats_csv": str(output_stats_path),
            "map_image": str(output_map_path),
            "chart_image": str(output_chart_path),
            "total_raster_pixels": int(height * width),
            "agriculture_pixels": n_agriculture,
            "classified_crop_pixels": n_valid_crop,
            "crop_statistics": stats_df.to_dict(orient="records"),
            "total_crop_area_km2": round(total_crop_area, 2),
            "is_experimental": True,
            "disclaimer": (
                "Important notice: These crop predictions are derived from the experimental "
                "Random Forest model and have not been independently ground-truth validated for new satellite dates."
            )
        }

    def compare_crop_years(self, year1: Union[str, int], year2: Union[str, int]) -> Dict[str, Any]:
        """
        Compare crop area statistics between two classified years.
        """
        year1_str, year2_str = str(year1).strip(), str(year2).strip()
        stats1_path = CROP_RESULT_DIR / f"crop_area_statistics_{year1_str}_experimental.csv"
        stats2_path = CROP_RESULT_DIR / f"crop_area_statistics_{year2_str}_experimental.csv"

        # Fallback to standard 2022 stats if needed
        if not stats1_path.exists() and year1_str == "2022":
            stats1_path = CROP_RESULT_DIR / "crop_area_statistics_2022.csv"
        if not stats2_path.exists() and year2_str == "2022":
            stats2_path = CROP_RESULT_DIR / "crop_area_statistics_2022.csv"

        if not stats1_path.exists() or not stats2_path.exists():
            raise FileNotFoundError(
                f"Crop statistics not available for both {year1_str} and {year2_str}. "
                f"Run crop classification on both years first."
            )

        df1 = pd.read_csv(stats1_path)
        df2 = pd.read_csv(stats2_path)

        merged = pd.merge(df1, df2, on=["crop_class_id", "crop_class"], suffixes=(f"_{year1_str}", f"_{year2_str}"))

        comparison_rows = []
        for _, row in merged.iterrows():
            a1 = row[f"area_km2_{year1_str}"]
            a2 = row[f"area_km2_{year2_str}"]
            diff = a2 - a1
            pct = ((diff / a1) * 100.0) if a1 > 0 else 0.0

            comparison_rows.append({
                "crop_class_id": int(row["crop_class_id"]),
                "crop_class": row["crop_class"],
                f"area_{year1_str}_km2": round(a1, 2),
                f"area_{year2_str}_km2": round(a2, 2),
                "change_km2": round(diff, 2),
                "percentage_change": round(pct, 2)
            })

        return {
            "period": f"{year1_str}-{year2_str}",
            "year1": year1_str,
            "year2": year2_str,
            "comparison": comparison_rows
        }


# Global singleton instance
crop_service = CropService()
