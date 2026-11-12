"""
Automated LULC Preprocessing and Prediction Service (Phase 3).

Integrates Sentinel-2 Level-2A imagery retrieval with band extraction, bilinear resampling,
study area clipping, spectral index calculation, and pixel-wise Random Forest classification.
"""

import os
import glob
import json
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
    from rasterio.enums import Resampling
    from rasterio.mask import mask
    HAS_RASTERIO = True
except (ImportError, Exception):
    rasterio = None
    Window = None
    Resampling = None
    mask = None
    HAS_RASTERIO = False

try:
    import scipy.ndimage as ndimage
    HAS_SCIPY = True
except (ImportError, Exception):
    ndimage = None
    HAS_SCIPY = False

from src.config import (
    BASE_DIR,
    RAW_DIR,
    PROCESSED_MANGALURU_DIR,
    CLASSIFIED_MANGALURU_DIR,
    MAPS_DIR,
    RESULTS_DIR,
    LULC_MODEL_PATH,
    BOUNDARY_GEOJSON,
    LULC_CLASSES,
    REFLECTANCE_SCALE,
    PIXEL_AREA_KM2,
    RF_FEATURE_NAMES,
)

logger = logging.getLogger("lulc_service")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [LULCService] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# Exact colormap for LULC maps matching existing 2018/2022/2025 maps
LULC_COLOR_MAP = {
    0: (255, 255, 255),  # NoData / Background (White)
    1: (33, 150, 243),   # Water: #2196F3
    2: (34, 139, 34),    # Vegetation/Forest: #228B22
    3: (255, 215, 0),    # Agriculture: #FFD700
    4: (229, 57, 53),    # Built-up: #E53935
    5: (166, 124, 82)    # Barren/Open: #A67C52
}


class LULCService:
    """Service handling Sentinel-2 preprocessing and automated LULC classification."""

    def __init__(self):
        self._model = None

    def get_model(self):
        """Lazy load and cache the trained LULC Random Forest model."""
        if self._model is None:
            if not LULC_MODEL_PATH.exists():
                raise FileNotFoundError(f"Trained LULC model not found at {LULC_MODEL_PATH}")
            logger.info(f"Loading trained LULC Random Forest model from {LULC_MODEL_PATH}...")
            loaded = joblib.load(str(LULC_MODEL_PATH))
            self._model = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
            logger.info("LULC Random Forest model loaded successfully.")
        return self._model

    def find_scene_bands(self, scene_path_or_year: Union[str, Path, int]) -> Dict[str, Path]:
        """
        Locate required Sentinel-2 bands inside a product directory or raw year folder.
        Required: B02_10m, B03_10m, B04_10m, B08_10m, B11_20m.
        Optional: SCL_20m.
        """
        search_root = None
        if isinstance(scene_path_or_year, int) or (isinstance(scene_path_or_year, str) and scene_path_or_year.isdigit()):
            search_root = RAW_DIR / str(scene_path_or_year)
        else:
            search_root = Path(scene_path_or_year)

        if not search_root.exists():
            raise FileNotFoundError(f"Scene search directory does not exist: {search_root}")

        band_patterns = {
            "B02": "*_B02_10m.*",
            "B03": "*_B03_10m.*",
            "B04": "*_B04_10m.*",
            "B08": "*_B08_10m.*",
            "B11": "*_B11_20m.*",
            "SCL": "*_SCL_20m.*"
        }

        found_bands: Dict[str, Path] = {}

        for band_key, pattern in band_patterns.items():
            matches = list(search_root.rglob(pattern))
            # Also try matching without resolution suffix if needed (e.g. *_B02.*)
            if not matches and band_key != "SCL":
                matches = list(search_root.rglob(f"*_{band_key}.*"))

            if matches:
                # Prefer .jp2 then .tif
                jp2_matches = [m for m in matches if m.suffix.lower() == ".jp2"]
                found_bands[band_key] = jp2_matches[0] if jp2_matches else matches[0]
            elif band_key != "SCL":
                raise FileNotFoundError(
                    f"Required band '{band_key}' not found under {search_root}. "
                    f"Ensure the product is extracted with 10m and 20m granules."
                )
            else:
                found_bands["SCL"] = None

        logger.info(f"Located Sentinel-2 bands in {search_root.name}: {list(found_bands.keys())}")
        return found_bands

    def calculate_indices(
        self,
        blue: np.ndarray,
        green: np.ndarray,
        red: np.ndarray,
        nir: np.ndarray,
        swir: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute NDVI, NDWI, and NDBI using exact formulas from training pipeline.
        Epsilon of 1e-10 prevents division by zero.
        """
        eps = 1e-10
        ndvi = (nir - red) / (nir + red + eps)
        ndwi = (green - nir) / (green + nir + eps)
        ndbi = (swir - nir) / (swir + nir + eps)
        return ndvi, ndwi, ndbi

    def prepare_feature_array(
        self,
        blue: np.ndarray,
        green: np.ndarray,
        red: np.ndarray,
        nir: np.ndarray,
        swir: np.ndarray,
        ndvi: np.ndarray,
        ndwi: np.ndarray,
        ndbi: np.ndarray,
        valid_mask: np.ndarray
    ) -> np.ndarray:
        """
        Assemble the 8 features in the verified order required by the Random Forest model:
        [blue, green, red, nir, swir, ndvi, ndwi, ndbi].
        """
        return np.column_stack([
            blue[valid_mask],
            green[valid_mask],
            red[valid_mask],
            nir[valid_mask],
            swir[valid_mask],
            ndvi[valid_mask],
            ndwi[valid_mask],
            ndbi[valid_mask]
        ]).astype(np.float32)

    def resample_band(
        self,
        source_array: np.ndarray,
        target_shape: Tuple[int, int],
        is_continuous: bool = True
    ) -> np.ndarray:
        """
        Resample 20m band to match 10m target shape.
        Continuous reflectance bands (e.g. B11) use bilinear resampling (order=1).
        Categorical bands (e.g. SCL) use nearest-neighbour (order=0).
        """
        if source_array.shape == target_shape:
            return source_array

        if HAS_SCIPY and ndimage is not None:
            zoom_y = target_shape[0] / source_array.shape[0]
            zoom_x = target_shape[1] / source_array.shape[1]
            order = 1 if is_continuous else 0
            resampled = ndimage.zoom(source_array, (zoom_y, zoom_x), order=order)
            # Ensure exact dimensions
            return resampled[:target_shape[0], :target_shape[1]]

        # Fallback using PIL
        resample_mode = Image.BILINEAR if is_continuous else Image.NEAREST
        pil_img = Image.fromarray(source_array)
        resized = pil_img.resize((target_shape[1], target_shape[0]), resample=resample_mode)
        return np.array(resized)

    def classify_scene(
        self,
        processed_image_path: Union[str, Path],
        output_raster_path: Optional[Union[str, Path]] = None,
        tag: str = "",
        progress_callback: Optional[Callable[[int, str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None
    ) -> Dict[str, Any]:
        """
        Run pixel-wise Random Forest classification on a processed 5-band raster.
        Generates classified GeoTIFF, class-wise area statistics, and map visualization PNG.
        """
        def report(pct: int, msg: str):
            if progress_callback:
                progress_callback(pct, msg)

        def check_abort():
            if is_cancelled and is_cancelled():
                raise RuntimeError("Classification cancelled by user.")

        processed_path = Path(processed_image_path)
        if not processed_path.exists():
            raise FileNotFoundError(f"Processed 5-band raster not found: {processed_path}")

        if not tag:
            # Extract tag from filename, e.g. sentinel2_2024_mangaluru.tif -> 2024
            name = processed_path.stem
            parts = name.split("_")
            tag = parts[1] if len(parts) > 1 else "new"

        if output_raster_path is None:
            output_raster_path = CLASSIFIED_MANGALURU_DIR / f"land_use_{tag}.tif"
        else:
            output_raster_path = Path(output_raster_path)

        map_png_path = MAPS_DIR / f"land_use_map_{tag}.png"

        report(55, f"Loading Random Forest LULC model for {tag}...")
        check_abort()
        model = self.get_model()

        report(60, f"Reading processed 5-band raster: {processed_path.name}...")
        check_abort()

        profile = None
        if HAS_RASTERIO and rasterio is not None:
            with rasterio.open(str(processed_path)) as src:
                profile = src.profile.copy()
                height, width = src.height, src.width
                if src.count >= 5:
                    band_data = src.read([1, 2, 3, 4, 5])  # Shape: (5, H, W)
                else:
                    # Fallback in case multi-frame TIFF was opened with count < 5
                    from PIL import ImageSequence
                    pil_img = Image.open(str(processed_path))
                    pages = [np.array(p, dtype=np.float32) for p in ImageSequence.Iterator(pil_img)]
                    if len(pages) >= 5:
                        band_data = np.stack(pages[:5], axis=0)
                    else:
                        raise ValueError(f"Expected at least 5 bands in raster {processed_path.name}, found {src.count}")
        else:
            # Fallback reading via PIL
            from PIL import ImageSequence
            pil_img = Image.open(str(processed_path))
            pages = [np.array(p, dtype=np.float32) for p in ImageSequence.Iterator(pil_img)]
            if len(pages) >= 5:
                band_data = np.stack(pages[:5], axis=0)
            elif len(pages) == 1 and pages[0].ndim == 3 and pages[0].shape[2] >= 5:
                band_data = np.transpose(pages[0][:, :, :5], (2, 0, 1))
            else:
                width, height = pil_img.size
                band_data = np.zeros((5, height, width), dtype=np.float32)
            height, width = band_data.shape[1], band_data.shape[2]

        blue = band_data[0].astype(np.float32)
        green = band_data[1].astype(np.float32)
        red = band_data[2].astype(np.float32)
        nir = band_data[3].astype(np.float32)
        swir = band_data[4].astype(np.float32)

        report(68, "Calculating spectral indices (NDVI, NDWI, NDBI)...")
        check_abort()
        ndvi, ndwi, ndbi = self.calculate_indices(blue, green, red, nir, swir)

        # Valid pixel criteria matching classify_2022_mangaluru.py:
        # Must be finite and all bands positive (unmasked / valid area)
        valid_mask = (
            np.isfinite(blue) & np.isfinite(green) & np.isfinite(red) &
            np.isfinite(nir) & np.isfinite(swir) &
            np.isfinite(ndvi) & np.isfinite(ndwi) & np.isfinite(ndbi) &
            (blue > 0) & (green > 0) & (red > 0) & (nir > 0) & (swir > 0)
        )

        total_pixels = int(height * width)
        valid_pixels = int(np.sum(valid_mask))
        logger.info(f"Total pixels: {total_pixels:,}, Valid pixels: {valid_pixels:,} ({(valid_pixels/total_pixels)*100:.1f}%)")

        classified_map = np.zeros((height, width), dtype=np.uint8)

        if valid_pixels > 0:
            report(75, f"Running Random Forest inference on {valid_pixels:,} pixels...")
            check_abort()

            # Flatten valid pixels into feature matrix
            X = self.prepare_feature_array(
                blue, green, red, nir, swir, ndvi, ndwi, ndbi, valid_mask
            )

            # Predict in blocks to avoid memory spikes
            chunk_size = 500000
            predictions = np.zeros(valid_pixels, dtype=np.uint8)
            for i in range(0, valid_pixels, chunk_size):
                check_abort()
                end_i = min(i + chunk_size, valid_pixels)
                pred_chunk = model.predict(X[i:end_i])
                predictions[i:end_i] = pred_chunk.astype(np.uint8)

            classified_map[valid_mask] = predictions

        report(88, f"Saving classified raster to {output_raster_path.name}...")
        check_abort()

        # Save classified GeoTIFF
        if HAS_RASTERIO and rasterio is not None and profile is not None:
            profile.update(
                dtype="uint8",
                count=1,
                compress="lzw",
                nodata=0
            )
            with rasterio.open(str(output_raster_path), "w", **profile) as dst:
                dst.write(classified_map, 1)
                dst.set_band_description(1, "LULC: 1=Water, 2=Vegetation, 3=Agriculture, 4=Built-up, 5=Barren")
        else:
            # Fallback saving via PIL TIFF
            pil_out = Image.fromarray(classified_map)
            pil_out.save(str(output_raster_path), format="TIFF", compression="tiff_lzw")

        report(92, "Generating classified map PNG visualization...")
        check_abort()

        # Generate classified PNG map using exact RGB colormap
        rgb_img = np.zeros((height, width, 3), dtype=np.uint8)
        for class_id, color in LULC_COLOR_MAP.items():
            rgb_img[classified_map == class_id] = color

        Image.fromarray(rgb_img).save(str(map_png_path))
        logger.info(f"Saved classified map visualization to {map_png_path}")

        # Compute class-wise area statistics
        report(96, "Calculating class-wise area statistics...")
        stats_rows = []
        for class_id, class_name in LULC_CLASSES.items():
            count = int(np.sum(classified_map == class_id))
            area_km2 = round(count * PIXEL_AREA_KM2, 4)
            pct = round((count / valid_pixels * 100.0), 4) if valid_pixels > 0 else 0.0
            stats_rows.append({
                "Year": tag,
                "Class_ID": class_id,
                "Class_Name": class_name,
                "Pixels": count,
                "Area_km2": area_km2,
                "Percentage": pct
            })

        # Update CSV statistics record safely
        area_stats_file = RESULTS_DIR / "land_use_area_statistics.csv"
        if area_stats_file.exists():
            try:
                existing_df = pd.read_csv(area_stats_file)
                # Filter out previous entries for this exact tag to avoid duplication
                existing_df = existing_df[existing_df["Year"].astype(str) != str(tag)]
                new_df = pd.concat([existing_df, pd.DataFrame(stats_rows)], ignore_index=True)
                new_df.to_csv(area_stats_file, index=False)
            except Exception as e:
                logger.warning(f"Could not append to {area_stats_file}: {e}")

        report(100, f"LULC classification for {tag} completed successfully.")

        return {
            "success": True,
            "tag": tag,
            "classified_raster": str(output_raster_path),
            "map_image": str(map_png_path),
            "statistics": stats_rows,
            "total_pixels": total_pixels,
            "valid_pixels": valid_pixels
        }

    def run_full_pipeline(
        self,
        scene_path_or_year: Union[str, Path, int],
        tag: str = "",
        progress_callback: Optional[Callable[[int, str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None
    ) -> Dict[str, Any]:
        """
        Execute full end-to-end Phase 3 pipeline:
        Locate bands -> Resample B11 -> Clip to Mangaluru -> Compute Features -> Predict -> Map & Stats.
        """
        def report(pct: int, msg: str):
            if progress_callback:
                progress_callback(pct, msg)

        def check_abort():
            if is_cancelled and is_cancelled():
                raise RuntimeError("Pipeline cancelled by user.")

        report(5, "Locating Sentinel-2 Level-2A bands...")
        check_abort()
        bands = self.find_scene_bands(scene_path_or_year)

        if not tag:
            if isinstance(scene_path_or_year, (int, str)) and str(scene_path_or_year).isdigit():
                tag = str(scene_path_or_year)
            else:
                tag = Path(scene_path_or_year).name.split("_")[2][:4] if "_" in str(scene_path_or_year) else "custom"

        # Check if pre-processed raster already exists for historical years to avoid redundant work
        processed_target = PROCESSED_MANGALURU_DIR / f"sentinel2_{tag}_mangaluru.tif"
        if not processed_target.exists():
            report(15, f"Preprocessing and aligning bands for {tag}...")
            check_abort()

            # Read reference 10m band (B02)
            b02_path = bands["B02"]
            if HAS_RASTERIO and rasterio is not None:
                with rasterio.open(str(b02_path)) as src:
                    ref_profile = src.profile.copy()
                    height, width = src.height, src.width
                    blue = src.read(1).astype(np.float32)

                with rasterio.open(str(bands["B03"])) as src:
                    green = src.read(1).astype(np.float32)
                with rasterio.open(str(bands["B04"])) as src:
                    red = src.read(1).astype(np.float32)
                with rasterio.open(str(bands["B08"])) as src:
                    nir = src.read(1).astype(np.float32)

                # Read and resample B11 (20m to 10m) using bilinear interpolation
                report(30, "Resampling continuous SWIR band (B11) using bilinear interpolation...")
                check_abort()
                with rasterio.open(str(bands["B11"])) as src:
                    swir = src.read(
                        1,
                        out_shape=(height, width),
                        resampling=Resampling.bilinear
                    ).astype(np.float32)

                # Clip using Mangaluru Taluk boundary
                report(40, "Clipping aligned bands to Mangaluru Taluk boundary...")
                check_abort()
                boundary_path = BOUNDARY_GEOJSON
                if boundary_path.exists():
                    import geopandas as gpd
                    boundary = gpd.read_file(str(boundary_path))
                    # Reproject boundary to raster CRS
                    boundary_crs = boundary.to_crs(ref_profile["crs"])
                    # Combine bands into memory dataset for clipping
                    ref_profile.update(count=5, dtype="float32", compress="lzw")
                    # Scale reflectance
                    blue /= REFLECTANCE_SCALE
                    green /= REFLECTANCE_SCALE
                    red /= REFLECTANCE_SCALE
                    nir /= REFLECTANCE_SCALE
                    swir /= REFLECTANCE_SCALE

                    with rasterio.open(str(processed_target), "w", **ref_profile) as dst:
                        dst.write(blue, 1)
                        dst.write(green, 2)
                        dst.write(red, 3)
                        dst.write(nir, 4)
                        dst.write(swir, 5)

                    # Now mask with boundary
                    with rasterio.open(str(processed_target)) as src:
                        clipped, transform = mask(src, boundary_crs.geometry, crop=True, nodata=0)
                        clip_profile = src.profile.copy()
                        clip_profile.update(
                            height=clipped.shape[1],
                            width=clipped.shape[2],
                            transform=transform,
                            nodata=0
                        )

                    with rasterio.open(str(processed_target), "w", **clip_profile) as dst:
                        dst.write(clipped)
                        dst.set_band_description(1, "Blue")
                        dst.set_band_description(2, "Green")
                        dst.set_band_description(3, "Red")
                        dst.set_band_description(4, "NIR")
                        dst.set_band_description(5, "SWIR")
            else:
                # Fallback: create scaled processed file
                logger.info("Using lightweight raster preprocessor fallback...")
                b02_img = Image.open(str(bands["B02"]))
                w, h = b02_img.size
                blue = np.array(b02_img, dtype=np.float32) / REFLECTANCE_SCALE
                green = np.array(Image.open(str(bands["B03"])), dtype=np.float32) / REFLECTANCE_SCALE
                red = np.array(Image.open(str(bands["B04"])), dtype=np.float32) / REFLECTANCE_SCALE
                nir = np.array(Image.open(str(bands["B08"])), dtype=np.float32) / REFLECTANCE_SCALE

                swir_raw = np.array(Image.open(str(bands["B11"])), dtype=np.float32)
                swir = self.resample_band(swir_raw, (h, w), is_continuous=True) / REFLECTANCE_SCALE

                pages = [Image.fromarray(b) for b in [blue, green, red, nir, swir]]
                pages[0].save(str(processed_target), save_all=True, append_images=pages[1:], format="TIFF")

        report(50, f"Starting LULC classification for {tag}...")
        check_abort()

        return self.classify_scene(
            processed_image_path=processed_target,
            tag=tag,
            progress_callback=progress_callback,
            is_cancelled=is_cancelled
        )


# Global singleton instance
lulc_service = LULCService()
