"""
Confidence, Reliability and Advanced Spatial Analytics Service (Phase 6).

Implements:
1. Pixel-level prediction confidence and reliability maps via Random Forest probabilities.
2. Class-wise confidence distribution statistics and metrics.
3. Multi-year land-use trajectory analysis (Stable, Temporary, Persistent, Recent, Progressive).
4. Relative spatial change-density hotspot detection via Gaussian spatial filtering.
5. Patch fragmentation metrics (patch counts, density, sizes) for agricultural and built-up land.
"""

import os
import csv
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

try:
    from scipy.ndimage import gaussian_filter, label
    HAS_SCIPY = True
except (ImportError, Exception):
    gaussian_filter = None
    label = None
    HAS_SCIPY = False

from src.config import (
    BASE_DIR,
    PROCESSED_MANGALURU_DIR,
    CLASSIFIED_MANGALURU_DIR,
    MAPS_DIR,
    RESULTS_DIR,
    CONFIDENCE_DIR,
    RELIABILITY_DIR,
    HOTSPOTS_DIR,
    TRAJECTORIES_DIR,
    AGRICULTURE_ANALYSIS_DIR,
    BUILTUP_ANALYSIS_DIR,
    LULC_MODEL_PATH,
    LULC_CLASSES,
    PIXEL_AREA_KM2,
    RF_FEATURE_NAMES,
)

logger = logging.getLogger("analytics_service")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [AnalyticsService] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class AnalyticsService:
    """Service providing confidence estimation, hotspot detection, trajectories, and fragmentation analytics."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path) if model_path else LULC_MODEL_PATH
        self._model = None

        # Ensure result directories exist
        CONFIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        RELIABILITY_DIR.mkdir(parents=True, exist_ok=True)
        HOTSPOTS_DIR.mkdir(parents=True, exist_ok=True)
        TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)
        AGRICULTURE_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        BUILTUP_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        MAPS_DIR.mkdir(parents=True, exist_ok=True)

    def get_model(self):
        """Lazy loader for the Random Forest LULC classification model."""
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"LULC model not found at {self.model_path}")
            logger.info(f"Loading LULC model from {self.model_path}")
            loaded = joblib.load(self.model_path)
            self._model = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
        return self._model

    @staticmethod
    def _save_raster(path: Union[str, Path], array: np.ndarray, profile: Optional[dict], dtype: str, nodata: Any):
        """Save a 2D numpy array as a single-band GeoTIFF."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if HAS_RASTERIO and rasterio is not None and profile is not None:
            out_prof = profile.copy()
            out_prof.update(
                dtype=dtype,
                count=1,
                nodata=nodata,
                compress="lzw"
            )
            with rasterio.open(str(path), "w", **out_prof) as dst:
                dst.write(array.astype(dtype), 1)
        else:
            img = Image.fromarray(array)
            img.save(str(path))

    # =========================================================================
    # 1. PREDICTION CONFIDENCE & RELIABILITY
    # =========================================================================

    def calculate_confidence(
        self,
        year: Union[int, str],
        processed_path: Optional[Union[str, Path]] = None,
        classified_path: Optional[Union[str, Path]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate pixel-level prediction confidence for a given year using Random Forest probabilities.

        Categorization:
        - Low: < 0.60
        - Medium / Moderate: 0.60 to < 0.80
        - High: >= 0.80
        """
        year_str = str(year)
        def report(pct: int, msg: str):
            if progress_callback:
                progress_callback(pct, msg)
            logger.info(f"[{pct}%] {msg}")

        def check_abort():
            if is_cancelled and is_cancelled():
                raise InterruptedError(f"Confidence calculation for {year_str} was cancelled by user.")

        report(5, f"Locating input rasters for {year_str} confidence analysis...")
        check_abort()

        if processed_path is None:
            # Check mangaluru folder first, then generic processed folder
            cand1 = PROCESSED_MANGALURU_DIR / f"sentinel2_{year_str}_mangaluru.tif"
            cand2 = BASE_DIR / "data" / "processed" / f"sentinel2_{year_str}.tif"
            if cand1.exists():
                processed_path = cand1
            elif cand2.exists():
                processed_path = cand2
            else:
                raise FileNotFoundError(f"Processed 5-band raster for {year_str} not found at {cand1} or {cand2}")
        else:
            processed_path = Path(processed_path)

        if classified_path is None:
            cand_cls = CLASSIFIED_MANGALURU_DIR / f"land_use_{year_str}.tif"
            if not cand_cls.exists():
                cand_cls = BASE_DIR / "data" / "classified" / f"land_use_{year_str}.tif"
            if not cand_cls.exists():
                raise FileNotFoundError(f"Classified LULC raster for {year_str} not found at {cand_cls}")
            classified_path = cand_cls
        else:
            classified_path = Path(classified_path)

        report(15, f"Reading processed 5-band raster: {processed_path.name}...")
        check_abort()

        profile = None
        if HAS_RASTERIO and rasterio is not None:
            with rasterio.open(str(processed_path)) as src:
                profile = src.profile.copy()
                height, width = src.height, src.width
                if src.count >= 5:
                    band_data = src.read([1, 2, 3, 4, 5])
                else:
                    from PIL import ImageSequence
                    pil_img = Image.open(str(processed_path))
                    pages = [np.array(p, dtype=np.float32) for p in ImageSequence.Iterator(pil_img)]
                    if len(pages) >= 5:
                        band_data = np.stack(pages[:5], axis=0)
                    else:
                        raise ValueError(f"Expected at least 5 bands in raster {processed_path.name}, found {src.count}")
            with rasterio.open(str(classified_path)) as cls_src:
                classification = cls_src.read(1)
        else:
            from PIL import ImageSequence
            pil_img = Image.open(str(processed_path))
            pages = [np.array(p, dtype=np.float32) for p in ImageSequence.Iterator(pil_img)]
            band_data = np.stack(pages[:5], axis=0)
            height, width = band_data.shape[1], band_data.shape[2]
            cls_img = Image.open(str(classified_path))
            classification = np.array(cls_img)

        blue = band_data[0].astype(np.float32)
        green = band_data[1].astype(np.float32)
        red = band_data[2].astype(np.float32)
        nir = band_data[3].astype(np.float32)
        swir = band_data[4].astype(np.float32)

        report(30, "Computing spectral indices (NDVI, NDWI, NDBI)...")
        check_abort()

        eps = 1e-10
        ndvi = np.divide(nir - red, nir + red + eps, out=np.zeros_like(nir, dtype=np.float32), where=(nir + red) != 0)
        ndwi = np.divide(green - nir, green + nir + eps, out=np.zeros_like(green, dtype=np.float32), where=(green + nir) != 0)
        ndbi = np.divide(swir - nir, swir + nir + eps, out=np.zeros_like(swir, dtype=np.float32), where=(swir + nir) != 0)

        # Mask valid pixels matching classification
        valid_mask = (
            (classification > 0) &
            np.isfinite(blue) & np.isfinite(green) & np.isfinite(red) &
            np.isfinite(nir) & np.isfinite(swir) &
            np.isfinite(ndvi) & np.isfinite(ndwi) & np.isfinite(ndbi) &
            ((blue != 0) | (green != 0) | (red != 0) | (nir != 0) | (swir != 0))
        )

        rows, cols = np.where(valid_mask)
        valid_count = len(rows)
        report(45, f"Valid study pixels to evaluate: {valid_count:,}...")
        check_abort()

        if valid_count == 0:
            raise ValueError(f"No valid pixels found for confidence evaluation in {year_str}")

        report(55, "Generating feature matrix and calculating Random Forest class probabilities...")
        check_abort()

        features_df = pd.DataFrame({
            "blue": blue[rows, cols],
            "green": green[rows, cols],
            "red": red[rows, cols],
            "nir": nir[rows, cols],
            "swir": swir[rows, cols],
            "ndvi": ndvi[rows, cols],
            "ndwi": ndwi[rows, cols],
            "ndbi": ndbi[rows, cols]
        })[RF_FEATURE_NAMES]

        model = self.get_model()
        probabilities = model.predict_proba(features_df)
        predicted_classes = classification[rows, cols]

        class_to_idx = {int(c): idx for idx, c in enumerate(model.classes_)}

        # Pixel confidence: probability assigned to the classified class
        confidence_vals = np.zeros(valid_count, dtype=np.float32)
        for i, c in enumerate(predicted_classes):
            c_int = int(c)
            if c_int in class_to_idx:
                confidence_vals[i] = probabilities[i, class_to_idx[c_int]]
            else:
                confidence_vals[i] = float(np.max(probabilities[i]))

        report(75, "Constructing confidence raster arrays and categories...")
        check_abort()

        conf_array = np.zeros((height, width), dtype=np.float32)
        conf_cat_array = np.zeros((height, width), dtype=np.uint8)

        conf_array[rows, cols] = confidence_vals

        # Categories: 1: Low (< 0.60), 2: Medium (0.60 to < 0.80), 3: High (>= 0.80)
        low_mask = confidence_vals < 0.60
        med_mask = (confidence_vals >= 0.60) & (confidence_vals < 0.80)
        high_mask = confidence_vals >= 0.80

        cat_vals = np.zeros(valid_count, dtype=np.uint8)
        cat_vals[low_mask] = 1
        cat_vals[med_mask] = 2
        cat_vals[high_mask] = 3
        conf_cat_array[rows, cols] = cat_vals

        report(85, "Saving confidence GeoTIFFs and RGBA Leaflet overlay...")
        check_abort()

        conf_tif = CONFIDENCE_DIR / f"lulc_confidence_{year_str}.tif"
        conf_class_tif = CONFIDENCE_DIR / f"lulc_confidence_class_{year_str}.tif"

        self._save_raster(conf_tif, conf_array, profile, "float32", 0.0)
        self._save_raster(conf_class_tif, conf_cat_array, profile, "uint8", 0)

        # Generate RGBA Leaflet Overlay
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[conf_cat_array == 1] = [220, 55, 55, 190]   # Low: Red
        rgba[conf_cat_array == 2] = [255, 205, 40, 190]  # Medium: Yellow
        rgba[conf_cat_array == 3] = [35, 160, 75, 190]   # High: Green

        overlay_png = MAPS_DIR / f"confidence_overlay_{year_str}.png"
        temp_overlay = MAPS_DIR / f"confidence_overlay_{year_str}.tmp.png"
        Image.fromarray(rgba).save(str(temp_overlay), format="PNG")
        if overlay_png.exists():
            overlay_png.unlink()
        temp_overlay.replace(overlay_png)

        report(92, "Computing class-wise confidence statistics and saving CSV...")
        check_abort()

        low_count = int(np.sum(low_mask))
        med_count = int(np.sum(med_mask))
        high_count = int(np.sum(high_mask))

        stats_summary = {
            "year": int(year_str) if year_str.isdigit() else year_str,
            "total_valid_pixels": valid_count,
            "low_pixels": low_count,
            "medium_pixels": med_count,
            "high_pixels": high_count,
            "low_percent": float(low_count / valid_count * 100),
            "medium_percent": float(med_count / valid_count * 100),
            "high_percent": float(high_count / valid_count * 100),
            "mean_confidence": float(np.mean(confidence_vals)),
            "median_confidence": float(np.median(confidence_vals)),
            "minimum_confidence": float(np.min(confidence_vals)),
            "maximum_confidence": float(np.max(confidence_vals))
        }

        stats_csv = CONFIDENCE_DIR / f"confidence_statistics_{year_str}.csv"
        pd.DataFrame([stats_summary]).to_csv(stats_csv, index=False)

        # Compute per-class breakdown
        class_breakdown = []
        for cid, cname in LULC_CLASSES.items():
            c_mask = predicted_classes == cid
            c_vals = confidence_vals[c_mask]
            c_count = len(c_vals)
            if c_count == 0:
                continue
            c_low = int(np.sum(c_vals < 0.60))
            c_med = int(np.sum((c_vals >= 0.60) & (c_vals < 0.80)))
            c_high = int(np.sum(c_vals >= 0.80))
            class_breakdown.append({
                "Year": int(year_str) if year_str.isdigit() else year_str,
                "Class ID": cid,
                "Class": cname,
                "Pixel Count": c_count,
                "Mean Confidence": float(np.mean(c_vals)),
                "Median Confidence": float(np.median(c_vals)),
                "Minimum Confidence": float(np.min(c_vals)),
                "Maximum Confidence": float(np.max(c_vals)),
                "Low < 0.60 (%)": float((c_low / c_count) * 100),
                "Medium 0.60-<0.80 (%)": float((c_med / c_count) * 100),
                "High >= 0.80 (%)": float((c_high / c_count) * 100)
            })

        # Update confidence_by_class.csv
        all_class_csv = CONFIDENCE_DIR / "confidence_by_class.csv"
        if all_class_csv.exists():
            try:
                existing_df = pd.read_csv(all_class_csv)
                # Filter out this year if already present
                existing_df = existing_df[existing_df["Year"] != stats_summary["year"]]
                merged_df = pd.concat([existing_df, pd.DataFrame(class_breakdown)], ignore_index=True)
                merged_df.sort_values(by=["Year", "Class ID"], inplace=True)
                merged_df.to_csv(all_class_csv, index=False)
            except Exception as e:
                logger.warning(f"Failed to merge confidence_by_class.csv: {e}")
                pd.DataFrame(class_breakdown).to_csv(all_class_csv, index=False)
        else:
            pd.DataFrame(class_breakdown).to_csv(all_class_csv, index=False)

        report(100, f"Confidence calculation for {year_str} completed successfully.")

        return {
            "year": stats_summary["year"],
            "summary": stats_summary,
            "class_breakdown": class_breakdown,
            "confidence_raster": str(conf_tif),
            "confidence_class_raster": str(conf_class_tif),
            "overlay_png": str(overlay_png),
            "statistics_csv": str(stats_csv)
        }

    # =========================================================================
    # 2. CHANGE HOTSPOT DETECTION
    # =========================================================================

    def detect_hotspots(
        self,
        year1: Union[int, str],
        year2: Union[int, str],
        sigma: float = 15.0,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Detect relative spatial change-density hotspots between two years using Gaussian spatial filtering.

        Hotspot Classes:
        - 1: Low (< 25 score)
        - 2: Moderate (25 to < 50 score)
        - 3: High (50 to < 75 score)
        - 4: Very High (>= 75 score)
        """
        y1_str, y2_str = str(year1), str(year2)

        def report(pct: int, msg: str):
            if progress_callback:
                progress_callback(pct, msg)
            logger.info(f"[{pct}%] {msg}")

        def check_abort():
            if is_cancelled and is_cancelled():
                raise InterruptedError(f"Hotspot detection between {y1_str} and {y2_str} was cancelled.")

        report(10, f"Loading classified rasters for {y1_str} and {y2_str}...")
        check_abort()

        p1 = CLASSIFIED_MANGALURU_DIR / f"land_use_{y1_str}.tif"
        p2 = CLASSIFIED_MANGALURU_DIR / f"land_use_{y2_str}.tif"

        if not p1.exists():
            raise FileNotFoundError(f"Classified raster for {y1_str} not found at {p1}")
        if not p2.exists():
            raise FileNotFoundError(f"Classified raster for {y2_str} not found at {p2}")

        profile = None
        if HAS_RASTERIO and rasterio is not None:
            with rasterio.open(str(p1)) as s1, rasterio.open(str(p2)) as s2:
                lulc1 = s1.read(1)
                lulc2 = s2.read(1)
                profile = s1.profile.copy()
        else:
            lulc1 = np.array(Image.open(str(p1)))
            lulc2 = np.array(Image.open(str(p2)))

        report(25, "Identifying valid pixels and detecting land use changes...")
        check_abort()

        valid = (np.isin(lulc1, [1, 2, 3, 4, 5]) & np.isin(lulc2, [1, 2, 3, 4, 5]))
        changed = valid & (lulc1 != lulc2)

        valid_count = int(np.sum(valid))
        changed_count = int(np.sum(changed))

        if valid_count == 0:
            raise ValueError(f"No common valid pixels found between {y1_str} and {y2_str}")

        report(40, f"Applying spatial smoothing filter (sigma={sigma})...")
        check_abort()

        score = np.full(lulc1.shape, 255.0, dtype=np.float32)
        classes = np.full(lulc1.shape, 255, dtype=np.uint8)

        if changed_count == 0:
            score[valid] = 0.0
            classes[valid] = 1
        else:
            if not HAS_SCIPY or gaussian_filter is None:
                raise RuntimeError("scipy.ndimage is required for hotspot detection.")

            change_float = changed.astype(np.float32)
            valid_float = valid.astype(np.float32)

            numerator = gaussian_filter(change_float, sigma=sigma)
            denominator = gaussian_filter(valid_float, sigma=sigma)

            density = np.zeros(lulc1.shape, dtype=np.float32)
            supported = valid & (denominator > 1e-6)
            density[supported] = numerator[supported] / denominator[supported]

            valid_density = density[valid]
            min_val = float(valid_density.min())
            max_val = float(valid_density.max())

            score_vals = np.zeros(lulc1.shape, dtype=np.float32)
            if max_val > min_val:
                score_vals[valid] = ((density[valid] - min_val) / (max_val - min_val)) * 100.0

            score[valid] = score_vals[valid]

            # Hotspot classes: 1: Low, 2: Moderate, 3: High, 4: Very High
            classes[valid & (score_vals < 25.0)] = 1
            classes[valid & (score_vals >= 25.0) & (score_vals < 50.0)] = 2
            classes[valid & (score_vals >= 50.0) & (score_vals < 75.0)] = 3
            classes[valid & (score_vals >= 75.0)] = 4

        report(70, "Saving hotspot GeoTIFF rasters and statistics...")
        check_abort()

        score_file = HOTSPOTS_DIR / f"change_hotspot_score_{y1_str}_{y2_str}.tif"
        class_file = HOTSPOTS_DIR / f"change_hotspot_class_{y1_str}_{y2_str}.tif"

        self._save_raster(score_file, score, profile, "float32", 255.0)
        self._save_raster(class_file, classes, profile, "uint8", 255)

        # Generate RGBA Leaflet overlay for hotspots
        h, w = lulc1.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[classes == 1] = [65, 182, 196, 170]   # Low: Light Cyan / Blue
        rgba[classes == 2] = [254, 204, 92, 180]   # Moderate: Yellow
        rgba[classes == 3] = [253, 141, 60, 190]   # High: Orange
        rgba[classes == 4] = [227, 26, 28, 200]    # Very High: Deep Red

        hotspot_png = MAPS_DIR / f"hotspot_overlay_{y1_str}_{y2_str}.png"
        temp_hotspot_png = MAPS_DIR / f"hotspot_overlay_{y1_str}_{y2_str}.tmp.png"
        Image.fromarray(rgba).save(str(temp_hotspot_png), format="PNG")
        if hotspot_png.exists():
            hotspot_png.unlink()
        temp_hotspot_png.replace(hotspot_png)

        report(85, "Compiling summary statistics...")
        check_abort()

        class_names = {1: "Low", 2: "Moderate", 3: "High", 4: "Very High"}
        stats_rows = []
        for cid, cname in class_names.items():
            cnt = int(np.count_nonzero(valid & (classes == cid)))
            pct = float(cnt / valid_count * 100.0) if valid_count > 0 else 0.0
            area = float(cnt * PIXEL_AREA_KM2)
            stats_rows.append({
                "Hotspot Class": cid,
                "Class Name": cname,
                "Pixels": cnt,
                "Percentage": pct,
                "Area (km2)": area
            })

        csv_file = HOTSPOTS_DIR / f"change_hotspot_statistics_{y1_str}_{y2_str}.csv"
        pd.DataFrame(stats_rows).to_csv(csv_file, index=False)

        report(100, f"Hotspot detection {y1_str}-{y2_str} completed.")

        return {
            "period": f"{y1_str}-{y2_str}",
            "valid_pixels": valid_count,
            "changed_pixels": changed_count,
            "statistics": stats_rows,
            "score_raster": str(score_file),
            "class_raster": str(class_file),
            "overlay_png": str(hotspot_png),
            "csv_file": str(csv_file)
        }

    # =========================================================================
    # 3. MULTI-YEAR LAND-USE TRAJECTORY ANALYSIS
    # =========================================================================

    def analyze_trajectories(
        self,
        years: Optional[List[Union[int, str]]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze multi-temporal pixel trajectories across three years (default: 2018, 2022, 2025).

        Trajectory Categories:
        - Stable: C1 == C2 == C3
        - Temporary Change: C1 != C2 and C2 != C3 and C1 == C3
        - Persistent Change: C1 != C2 and C2 == C3
        - Recent Change: C1 == C2 and C2 != C3
        - Progressive Change: C1 != C2 and C2 != C3 and C1 != C3
        """
        if years is None:
            years = [2018, 2022, 2025]
        if len(years) != 3:
            raise ValueError(f"Trajectory analysis requires exactly 3 years, got {years}")

        y1, y2, y3 = str(years[0]), str(years[1]), str(years[2])

        def report(pct: int, msg: str):
            if progress_callback:
                progress_callback(pct, msg)
            logger.info(f"[{pct}%] {msg}")

        def check_abort():
            if is_cancelled and is_cancelled():
                raise InterruptedError(f"Trajectory analysis across {years} cancelled.")

        report(15, f"Loading classified rasters for {y1}, {y2}, and {y3}...")
        check_abort()

        rasters = {}
        for y in [y1, y2, y3]:
            path = CLASSIFIED_MANGALURU_DIR / f"land_use_{y}.tif"
            if not path.exists():
                raise FileNotFoundError(f"Classified raster for {y} not found at {path}")
            if HAS_RASTERIO and rasterio is not None:
                with rasterio.open(str(path)) as src:
                    rasters[y] = src.read(1)
            else:
                rasters[y] = np.array(Image.open(str(path)))

        c1_all, c2_all, c3_all = rasters[y1], rasters[y2], rasters[y3]

        report(35, "Filtering common valid pixels across all three years...")
        check_abort()

        valid = (c1_all > 0) & (c2_all > 0) & (c3_all > 0)
        c1 = c1_all[valid]
        c2 = c2_all[valid]
        c3 = c3_all[valid]

        total_pixels = len(c1)
        if total_pixels == 0:
            raise ValueError("No common valid pixels found across the three years.")

        report(55, "Evaluating pixel trajectory types...")
        check_abort()

        # Vectorized classification of trajectory types
        traj_types = np.full(total_pixels, "Mixed Trajectory", dtype=object)

        is_stable = (c1 == c2) & (c2 == c3)
        is_temp = (c1 != c2) & (c2 != c3) & (c1 == c3)
        is_persist = (c1 != c2) & (c2 == c3)
        is_recent = (c1 == c2) & (c2 != c3)
        is_prog = (c1 != c2) & (c2 != c3) & (c1 != c3)

        traj_types[is_stable] = "Stable"
        traj_types[is_temp] = "Temporary Change"
        traj_types[is_persist] = "Persistent Change"
        traj_types[is_recent] = "Recent Change"
        traj_types[is_prog] = "Progressive Change"

        report(75, "Generating trajectory sequence combinations...")
        check_abort()

        # Class sequence labels
        sequences = [
            f"{LULC_CLASSES.get(int(a), f'Class {a}')} -> {LULC_CLASSES.get(int(b), f'Class {b}')} -> {LULC_CLASSES.get(int(c), f'Class {c}')}"
            for a, b, c in zip(c1, c2, c3)
        ]

        # Trajectory type summary
        counts_type = pd.Series(traj_types).value_counts()
        type_summary = pd.DataFrame({
            "Trajectory Type": counts_type.index,
            "Pixel Count": counts_type.values,
            "Percentage": (counts_type.values / total_pixels) * 100.0,
            "Area (km2)": counts_type.values * PIXEL_AREA_KM2
        })

        # Sequence summary
        counts_seq = pd.Series(sequences).value_counts()
        seq_summary = pd.DataFrame({
            "Trajectory Sequence": counts_seq.index,
            "Pixel Count": counts_seq.values,
            "Percentage": (counts_seq.values / total_pixels) * 100.0,
            "Area (km2)": counts_seq.values * PIXEL_AREA_KM2
        })

        # Stable class breakdown
        stable_c = c1[is_stable]
        counts_stable = pd.Series(stable_c).value_counts().sort_index()
        stable_summary = pd.DataFrame({
            "Class ID": counts_stable.index,
            "Class": [LULC_CLASSES.get(int(x), f"Class {x}") for x in counts_stable.index],
            "Stable Pixel Count": counts_stable.values,
            "Stable Percentage": (counts_stable.values / total_pixels) * 100.0,
            "Stable Area (km2)": counts_stable.values * PIXEL_AREA_KM2
        })

        report(90, "Writing trajectory analysis CSV reports...")
        check_abort()

        traj_type_csv = TRAJECTORIES_DIR / "trajectory_type_summary.csv"
        seq_csv = TRAJECTORIES_DIR / "trajectory_sequence_summary.csv"
        stable_csv = TRAJECTORIES_DIR / "stable_class_summary.csv"

        type_summary.to_csv(traj_type_csv, index=False)
        seq_summary.to_csv(seq_csv, index=False)
        stable_summary.to_csv(stable_csv, index=False)

        report(100, "Trajectory analysis completed successfully.")

        return {
            "years": [int(y) if str(y).isdigit() else str(y) for y in [y1, y2, y3]],
            "total_pixels": total_pixels,
            "trajectory_types": type_summary.to_dict(orient="records"),
            "top_sequences": seq_summary.head(20).to_dict(orient="records"),
            "stable_classes": stable_summary.to_dict(orient="records"),
            "trajectory_csv": str(traj_type_csv),
            "sequence_csv": str(seq_csv),
            "stable_csv": str(stable_csv)
        }

    # =========================================================================
    # 4. PATCH AND FRAGMENTATION METRICS
    # =========================================================================

    def analyze_fragmentation(
        self,
        year: Optional[Union[int, str]] = None,
        class_id: int = 3,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Calculate landscape patch fragmentation metrics for a specific class (3: Agriculture, 4: Built-up)
        using 8-connected component labeling.

        Metrics:
        - Number of patches
        - Largest patch area (km²) and percentage of class area
        - Mean patch area (km²)
        - Median patch area (km²)
        - Patch density per 100 km²
        """
        if year is not None:
            years_to_process = [str(year)]
        else:
            # Auto-detect all available classified years
            files = sorted(CLASSIFIED_MANGALURU_DIR.glob("land_use_*.tif"))
            years_to_process = [f.stem.replace("land_use_", "") for f in files if f.stem.replace("land_use_", "").isdigit()]
            if not years_to_process:
                years_to_process = ["2018", "2022", "2025"]

        def report(pct: int, msg: str):
            if progress_callback:
                progress_callback(pct, msg)
            logger.info(f"[{pct}%] {msg}")

        def check_abort():
            if is_cancelled and is_cancelled():
                raise InterruptedError("Fragmentation analysis cancelled.")

        report(10, f"Analyzing fragmentation for Class {class_id} across {len(years_to_process)} year(s)...")
        check_abort()

        results = []
        structure = np.ones((3, 3), dtype=np.uint8)  # 8-connectivity

        for idx, y in enumerate(years_to_process, start=1):
            check_abort()
            path = CLASSIFIED_MANGALURU_DIR / f"land_use_{y}.tif"
            if not path.exists():
                logger.warning(f"Classified raster not found for year {y}, skipping.")
                continue

            if HAS_RASTERIO and rasterio is not None:
                with rasterio.open(str(path)) as src:
                    data = src.read(1)
            else:
                data = np.array(Image.open(str(path)))

            valid = data > 0
            valid_pixels = int(np.sum(valid))
            valid_area_km2 = float(valid_pixels * PIXEL_AREA_KM2)

            class_mask = data == class_id
            class_pixels = int(np.sum(class_mask))
            class_area = float(class_pixels * PIXEL_AREA_KM2)

            if not HAS_SCIPY or label is None:
                raise RuntimeError("scipy.ndimage is required for patch labeling.")

            labeled, patch_count = label(class_mask, structure=structure)

            if patch_count > 0:
                patch_sizes = np.bincount(labeled.ravel())[1:]
                patch_areas = patch_sizes * PIXEL_AREA_KM2
                largest_patch = float(np.max(patch_areas))
                mean_patch = float(np.mean(patch_areas))
                median_patch = float(np.median(patch_areas))
            else:
                largest_patch = 0.0
                mean_patch = 0.0
                median_patch = 0.0

            patch_density = float(patch_count / valid_area_km2 * 100.0) if valid_area_km2 > 0 else 0.0
            largest_prop = float(largest_patch / class_area * 100.0) if class_area > 0 else 0.0

            res = {
                "year": int(y) if y.isdigit() else y,
                "class_id": class_id,
                "class_name": LULC_CLASSES.get(class_id, f"Class {class_id}"),
                "class_pixels": class_pixels,
                "class_area_km2": class_area,
                "percentage_of_valid": float(class_pixels / valid_pixels * 100.0) if valid_pixels > 0 else 0.0,
                "number_of_patches": patch_count,
                "largest_patch_km2": largest_patch,
                "mean_patch_km2": mean_patch,
                "median_patch_km2": median_patch,
                "patch_density_per_100km2": patch_density,
                "largest_patch_percentage_of_class": largest_prop
            }
            results.append(res)
            report(int(10 + (idx / len(years_to_process)) * 75), f"Finished year {y} ({patch_count:,} patches)")

        # Save to appropriate destination CSV
        if class_id == 3:
            out_csv = AGRICULTURE_ANALYSIS_DIR / "agriculture_fragmentation_statistics.csv"
            # Format columns to match existing template expectations
            df_out = pd.DataFrame([{
                "year": r["year"],
                "agriculture_pixels": r["class_pixels"],
                "agriculture_area_km2": r["class_area_km2"],
                "agriculture_percentage": r["percentage_of_valid"],
                "number_of_patches": r["number_of_patches"],
                "largest_patch_km2": r["largest_patch_km2"],
                "mean_patch_km2": r["mean_patch_km2"],
                "median_patch_km2": r["median_patch_km2"],
                "patch_density_per_100km2": r["patch_density_per_100km2"],
                "largest_patch_percentage_of_agriculture": r["largest_patch_percentage_of_class"]
            } for r in results])
            df_out.to_csv(out_csv, index=False)
        elif class_id == 4:
            out_csv = BUILTUP_ANALYSIS_DIR / "builtup_fragmentation_statistics.csv"
            df_out = pd.DataFrame([{
                "Year": r["year"],
                "Patch Count": r["number_of_patches"],
                "Largest Patch Area (km2)": r["largest_patch_km2"],
                "Mean Patch Area (km2)": r["mean_patch_km2"],
                "Median Patch Area (km2)": r["median_patch_km2"],
                "Patch Density per 100 km2": r["patch_density_per_100km2"]
            } for r in results])
            df_out.to_csv(out_csv, index=False)

        report(100, f"Fragmentation analysis for Class {class_id} completed.")
        return results

    # =========================================================================
    # 5. TRANSITION RELIABILITY ANALYSIS
    # =========================================================================

    def analyze_reliability(
        self,
        year1: Union[int, str],
        year2: Union[int, str],
        progress_callback: Optional[Callable[[int, str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate joint prediction reliability for detected transitions between two years.

        Reliability is defined by the minimum confidence: min(conf_year1, conf_year2).
        - High: min >= 0.80
        - Moderate: 0.60 <= min < 0.80
        - Low: min < 0.60
        """
        y1_str, y2_str = str(year1), str(year2)

        def report(pct: int, msg: str):
            if progress_callback:
                progress_callback(pct, msg)
            logger.info(f"[{pct}%] {msg}")

        def check_abort():
            if is_cancelled and is_cancelled():
                raise InterruptedError(f"Reliability analysis between {y1_str} and {y2_str} cancelled.")

        report(10, f"Checking confidence rasters for {y1_str} and {y2_str}...")
        check_abort()

        c1_tif = CONFIDENCE_DIR / f"lulc_confidence_{y1_str}.tif"
        c2_tif = CONFIDENCE_DIR / f"lulc_confidence_{y2_str}.tif"

        # If confidence rasters don't exist, calculate them automatically
        if not c1_tif.exists():
            report(15, f"Confidence raster for {y1_str} missing; generating...")
            self.calculate_confidence(y1_str)
        if not c2_tif.exists():
            report(30, f"Confidence raster for {y2_str} missing; generating...")
            self.calculate_confidence(y2_str)

        p1_tif = CLASSIFIED_MANGALURU_DIR / f"land_use_{y1_str}.tif"
        p2_tif = CLASSIFIED_MANGALURU_DIR / f"land_use_{y2_str}.tif"

        if not p1_tif.exists() or not p2_tif.exists():
            raise FileNotFoundError(f"Classified rasters missing for {y1_str} or {y2_str}")

        report(50, "Reading classification and confidence layers...")
        check_abort()

        if HAS_RASTERIO and rasterio is not None:
            with rasterio.open(str(p1_tif)) as s1, rasterio.open(str(p2_tif)) as s2, \
                 rasterio.open(str(c1_tif)) as sc1, rasterio.open(str(c2_tif)) as sc2:
                lulc1 = s1.read(1)
                lulc2 = s2.read(1)
                conf1 = sc1.read(1)
                conf2 = sc2.read(1)
        else:
            lulc1 = np.array(Image.open(str(p1_tif)))
            lulc2 = np.array(Image.open(str(p2_tif)))
            conf1 = np.array(Image.open(str(c1_tif)))
            conf2 = np.array(Image.open(str(c2_tif)))

        report(70, "Computing joint transition reliability metrics...")
        check_abort()

        valid = (np.isin(lulc1, [1, 2, 3, 4, 5]) & np.isin(lulc2, [1, 2, 3, 4, 5]) & (conf1 > 0) & (conf2 > 0))
        valid_count = int(np.sum(valid))
        if valid_count == 0:
            raise ValueError("No common valid pixels with confidence data found.")

        min_conf = np.minimum(conf1, conf2)
        changed = valid & (lulc1 != lulc2)
        total_changed = int(np.sum(changed))

        high_mask = min_conf >= 0.80
        med_mask = (min_conf >= 0.60) & (min_conf < 0.80)
        low_mask = min_conf < 0.60

        # Overall changed transitions reliability
        high_cnt = int(np.sum(changed & high_mask))
        med_cnt = int(np.sum(changed & med_mask))
        low_cnt = int(np.sum(changed & low_mask))

        overall_rows = [
            {"Reliability": "High", "Description": "Joint confidence >= 0.80", "Pixels": high_cnt,
             "Percentage": (high_cnt / total_changed * 100.0) if total_changed > 0 else 0.0,
             "Area (km2)": high_cnt * PIXEL_AREA_KM2},
            {"Reliability": "Moderate", "Description": "Joint confidence 0.60 - <0.80", "Pixels": med_cnt,
             "Percentage": (med_cnt / total_changed * 100.0) if total_changed > 0 else 0.0,
             "Area (km2)": med_cnt * PIXEL_AREA_KM2},
            {"Reliability": "Low", "Description": "Joint confidence < 0.60", "Pixels": low_cnt,
             "Percentage": (low_cnt / total_changed * 100.0) if total_changed > 0 else 0.0,
             "Area (km2)": low_cnt * PIXEL_AREA_KM2},
        ]

        report(88, "Saving reliability CSV tables...")
        check_abort()

        overall_csv = RELIABILITY_DIR / f"overall_reliability_{y1_str}_{y2_str}.csv"
        pd.DataFrame(overall_rows).to_csv(overall_csv, index=False)

        # Transition breakdown table
        transition_rows = []
        for src_id, src_name in LULC_CLASSES.items():
            for dst_id, dst_name in LULC_CLASSES.items():
                if src_id == dst_id:
                    continue
                pair_mask = changed & (lulc1 == src_id) & (lulc2 == dst_id)
                pair_count = int(np.sum(pair_mask))
                if pair_count == 0:
                    continue
                p_high = int(np.sum(pair_mask & high_mask))
                p_med = int(np.sum(pair_mask & med_mask))
                p_low = int(np.sum(pair_mask & low_mask))
                transition_rows.append({
                    "From": src_name,
                    "To": dst_name,
                    "Total Pixels": pair_count,
                    "High": p_high,
                    "Moderate": p_med,
                    "Low": p_low,
                    "High (%)": float(p_high / pair_count * 100.0),
                    "Moderate (%)": float(p_med / pair_count * 100.0),
                    "Low (%)": float(p_low / pair_count * 100.0)
                })

        summary_csv = RELIABILITY_DIR / f"reliability_summary_{y1_str}_{y2_str}.csv"
        pd.DataFrame(transition_rows).to_csv(summary_csv, index=False)

        report(100, f"Reliability analysis between {y1_str} and {y2_str} completed.")

        return {
            "period": f"{y1_str}-{y2_str}",
            "total_changed_pixels": total_changed,
            "overall_reliability": overall_rows,
            "transition_details": transition_rows,
            "overall_csv": str(overall_csv),
            "summary_csv": str(summary_csv)
        }

    # =========================================================================
    # 6. RETRIEVAL & QUERY HELPERS
    # =========================================================================

    def get_confidence_stats(self, year: Union[int, str]) -> Optional[Dict[str, Any]]:
        csv_path = CONFIDENCE_DIR / f"confidence_statistics_{year}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            return df.to_dict(orient="records")[0] if not df.empty else None
        return None

    def get_hotspot_stats(self, period: str) -> List[Dict[str, Any]]:
        p = period.replace("-", "_")
        csv_path = HOTSPOTS_DIR / f"change_hotspot_statistics_{p}.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path).to_dict(orient="records")
        return []

    def get_reliability_stats(self, period: str) -> List[Dict[str, Any]]:
        p = period.replace("-", "_")
        csv_path = RELIABILITY_DIR / f"overall_reliability_{p}.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path).to_dict(orient="records")
        return []


# Service Singleton
analytics_service = AnalyticsService()
