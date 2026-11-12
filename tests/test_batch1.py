"""
Comprehensive automated tests for Batch 1:
- Phase 3: Automated LULC Preprocessing and Prediction
- Phase 4: Dynamic Land Use Change Detection
- Phase 5: Dynamic Crop Analysis
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
from PIL import Image

from app import app
from src.config import (
    BASE_DIR,
    RAW_DIR,
    CLASSIFIED_MANGALURU_DIR,
    PROCESSED_MANGALURU_DIR,
    CHANGE_MAPS_DIR,
    MAPS_DIR,
    RESULTS_DIR,
    LULC_MODEL_PATH,
    CROP_MODEL_PATH,
    LULC_CLASSES,
    CROP_CLASSES,
    PIXEL_AREA_KM2,
)
from src.services.lulc_service import LULCService, lulc_service
from src.services.change_service import ChangeService, change_service
from src.services.crop_service import CropService, crop_service

try:
    import rasterio
    from rasterio.transform import from_origin
    HAS_RASTERIO = True
except (ImportError, Exception):
    rasterio = None
    from_origin = None
    HAS_RASTERIO = False


def create_test_5band_geotiff(path: Path, bands: list, crs="EPSG:32643") -> None:
    """Create a proper 5-band GeoTIFF fixture using Rasterio if available."""
    h, w = bands[0].shape
    if HAS_RASTERIO and rasterio is not None:
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "nodata": 0,
            "width": w,
            "height": h,
            "count": 5,
            "crs": crs,
            "transform": from_origin(0, 0, 10, 10),
            "compress": "lzw"
        }
        with rasterio.open(str(path), "w", **profile) as dst:
            for i, b in enumerate(bands, start=1):
                dst.write(b.astype(np.float32), i)
    else:
        pages = [Image.fromarray(b.astype(np.float32)) for b in bands]
        pages[0].save(str(path), save_all=True, append_images=pages[1:], format="TIFF")


def create_test_1band_geotiff(path: Path, data: np.ndarray, crs="EPSG:32643") -> None:
    """Create a proper 1-band GeoTIFF fixture using Rasterio if available."""
    h, w = data.shape
    if HAS_RASTERIO and rasterio is not None:
        profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "nodata": 0,
            "width": w,
            "height": h,
            "count": 1,
            "crs": crs,
            "transform": from_origin(0, 0, 10, 10),
            "compress": "lzw"
        }
        with rasterio.open(str(path), "w", **profile) as dst:
            dst.write(data.astype(np.uint8), 1)
    else:
        Image.fromarray(data.astype(np.uint8)).save(str(path), format="TIFF")


class TestPhase3LULCService(unittest.TestCase):
    """Unit and functional tests for Phase 3 LULC service."""

    def setUp(self):
        self.service = LULCService()

    def test_find_scene_bands_missing_band(self):
        """find_scene_bands raises FileNotFoundError if any required band is missing."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Create only B02, B03, B04, B08 (missing B11)
            for b in ["B02_10m", "B03_10m", "B04_10m", "B08_10m"]:
                (temp_dir / f"test_{b}.jp2").touch()

            with self.assertRaises(FileNotFoundError) as ctx:
                self.service.find_scene_bands(temp_dir)
            self.assertIn("B11", str(ctx.exception))
        finally:
            shutil.rmtree(temp_dir)

    def test_find_scene_bands_success(self):
        """find_scene_bands locates all 5 bands correctly."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            for b in ["B02_10m", "B03_10m", "B04_10m", "B08_10m", "B11_20m"]:
                (temp_dir / f"test_{b}.jp2").touch()

            bands = self.service.find_scene_bands(temp_dir)
            self.assertEqual(len(bands), 6)  # 5 required + SCL (None)
            self.assertIn("B02", bands)
            self.assertIn("B11", bands)
            self.assertIsNone(bands["SCL"])
        finally:
            shutil.rmtree(temp_dir)

    def test_calculate_indices(self):
        """calculate_indices computes NDVI, NDWI, and NDBI correctly without division by zero."""
        blue = np.array([0.1, 0.2], dtype=np.float32)
        green = np.array([0.15, 0.25], dtype=np.float32)
        red = np.array([0.05, 0.3], dtype=np.float32)
        nir = np.array([0.45, 0.1], dtype=np.float32)
        swir = np.array([0.2, 0.4], dtype=np.float32)

        ndvi, ndwi, ndbi = self.service.calculate_indices(blue, green, red, nir, swir)

        # Expected NDVI: (0.45 - 0.05) / (0.45 + 0.05) = 0.4 / 0.5 = 0.8
        self.assertAlmostEqual(ndvi[0], 0.8, places=4)
        # Expected NDWI: (0.15 - 0.45) / (0.15 + 0.45) = -0.3 / 0.6 = -0.5
        self.assertAlmostEqual(ndwi[0], -0.5, places=4)
        # Expected NDBI: (0.2 - 0.45) / (0.2 + 0.45) = -0.25 / 0.65 = -0.3846
        self.assertAlmostEqual(ndbi[0], -0.3846, places=3)

    def test_resample_band_continuous(self):
        """resample_band performs bilinear resampling for continuous data (e.g. B11)."""
        src = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        target_shape = (4, 4)
        resampled = self.service.resample_band(src, target_shape, is_continuous=True)
        self.assertEqual(resampled.shape, target_shape)
        # Values should be smoothly interpolated between 10 and 40
        self.assertGreaterEqual(resampled.min(), 9.9)
        self.assertLessEqual(resampled.max(), 40.1)

    def test_prepare_feature_array(self):
        """prepare_feature_array constructs exact 8-feature matrix in training order."""
        n = 5
        blue = np.ones(n, dtype=np.float32) * 1
        green = np.ones(n, dtype=np.float32) * 2
        red = np.ones(n, dtype=np.float32) * 3
        nir = np.ones(n, dtype=np.float32) * 4
        swir = np.ones(n, dtype=np.float32) * 5
        ndvi = np.ones(n, dtype=np.float32) * 6
        ndwi = np.ones(n, dtype=np.float32) * 7
        ndbi = np.ones(n, dtype=np.float32) * 8
        valid = np.array([True, True, False, True, False])

        features = self.service.prepare_feature_array(
            blue, green, red, nir, swir, ndvi, ndwi, ndbi, valid
        )

        self.assertEqual(features.shape, (3, 8))
        self.assertEqual(features.dtype, np.float32)
        # Verify order of features: 1, 2, 3, 4, 5, 6, 7, 8
        self.assertEqual(list(features[0]), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    def test_classify_scene_synthetic(self):
        """classify_scene runs Random Forest inference on synthetic raster and generates outputs."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Create a small synthetic 5-band TIFF (e.g. 20x20 pixels)
            h, w = 20, 20
            # Set values typical of vegetation / water
            blue = np.full((h, w), 0.05, dtype=np.float32)
            green = np.full((h, w), 0.15, dtype=np.float32)
            red = np.full((h, w), 0.08, dtype=np.float32)
            nir = np.full((h, w), 0.45, dtype=np.float32)
            swir = np.full((h, w), 0.12, dtype=np.float32)
            # Mask out a border as 0 (NoData)
            blue[0, :] = 0; green[0, :] = 0; red[0, :] = 0; nir[0, :] = 0; swir[0, :] = 0

            synthetic_tif = temp_dir / "synthetic_processed.tif"
            create_test_5band_geotiff(synthetic_tif, [blue, green, red, nir, swir])

            out_raster = temp_dir / "test_classified.tif"
            progress_mock = MagicMock()

            # Mock MAPS_DIR and RESULTS_DIR to temp_dir
            with patch("src.services.lulc_service.MAPS_DIR", temp_dir):
                with patch("src.services.lulc_service.RESULTS_DIR", temp_dir):
                    result = self.service.classify_scene(
                        processed_image_path=synthetic_tif,
                        output_raster_path=out_raster,
                        tag="test_synth",
                        progress_callback=progress_mock
                    )

            self.assertTrue(result["success"])
            self.assertEqual(result["tag"], "test_synth")
            self.assertTrue(out_raster.exists())
            self.assertGreater(result["valid_pixels"], 0)
            self.assertEqual(len(result["statistics"]), 5)  # 5 LULC classes
            # Verify progress reached 100
            progress_mock.assert_called_with(100, "LULC classification for test_synth completed successfully.")
        finally:
            shutil.rmtree(temp_dir)


class TestPhase4ChangeService(unittest.TestCase):
    """Unit and functional tests for Phase 4 Change Detection service."""

    def setUp(self):
        self.service = ChangeService()

    def test_list_available_years(self):
        """list_available_years finds existing historical classified rasters (2018, 2022, 2025)."""
        years = self.service.list_available_years()
        self.assertIn("2018", years)
        self.assertIn("2022", years)
        self.assertIn("2025", years)

    def test_get_available_periods(self):
        """get_available_periods includes standard historical periods."""
        periods = self.service.get_available_periods()
        self.assertIn("2018-2022", periods)
        self.assertIn("2022-2025", periods)
        self.assertIn("2018-2025", periods)

    def test_run_change_detection_identical_years_rejected(self):
        """run_change_detection rejects comparing identical years."""
        with self.assertRaises(ValueError) as ctx:
            self.service.run_change_detection("2022", "2022")
        self.assertIn("Cannot compare a year with itself", str(ctx.exception))

    def test_run_change_detection_synthetic(self):
        """run_change_detection computes transition matrix, spatial change map, and area change."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Create two synthetic classified rasters (10x10)
            h, w = 10, 10
            # Year 1: all Agriculture (3)
            map1 = np.full((h, w), 3, dtype=np.uint8)
            # Year 2: half Agriculture (3), half Built-up (4)
            map2 = np.full((h, w), 3, dtype=np.uint8)
            map2[:5, :] = 4  # 50 pixels transitioned from Agriculture to Built-up

            tif1 = temp_dir / "land_use_synth1.tif"
            tif2 = temp_dir / "land_use_synth2.tif"
            create_test_1band_geotiff(tif1, map1)
            create_test_1band_geotiff(tif2, map2)

            progress_mock = MagicMock()

            with patch("src.services.change_service.CLASSIFIED_MANGALURU_DIR", temp_dir):
                with patch("src.services.change_service.CHANGE_MAPS_DIR", temp_dir):
                    with patch("src.services.change_service.MAPS_DIR", temp_dir):
                        with patch("src.services.change_service.RESULTS_DIR", temp_dir):
                            res = self.service.run_change_detection(
                                "synth1", "synth2", progress_callback=progress_mock
                            )

            self.assertTrue(res["success"])
            self.assertEqual(res["total_pixels"], 100)
            self.assertEqual(res["valid_pixels"], 100)
            self.assertEqual(res["changed_pixels"], 50)
            self.assertEqual(res["unchanged_pixels"], 50)
            self.assertEqual(res["changed_percentage"], 50.0)

            # Check critical transition identification (Agriculture -> Built-up)
            crit = res["critical_transitions"]
            self.assertEqual(len(crit), 1)
            self.assertEqual(crit[0]["transition"], "Agriculture -> Built-up")
            self.assertEqual(crit[0]["pixels"], 50)

            # Verify files created
            self.assertTrue(Path(res["change_raster"]).exists())
            self.assertTrue(Path(res["change_map_png"]).exists())
            self.assertTrue((temp_dir / "area_change_synth1_synth2.csv").exists())
            self.assertTrue((temp_dir / "transition_matrix_synth1_synth2.csv").exists())
            self.assertTrue((temp_dir / "transition_details_synth1_synth2.csv").exists())
        finally:
            shutil.rmtree(temp_dir)


class TestPhase5CropService(unittest.TestCase):
    """Unit and functional tests for Phase 5 Crop service."""

    def setUp(self):
        self.service = CropService()

    def test_list_available_crop_years(self):
        """list_available_crop_years finds years ready for crop analysis."""
        years = self.service.list_available_crop_years()
        self.assertIn("2022", years)

    def test_classify_crops_synthetic(self):
        """classify_crops classifies agricultural pixels (Class 3) into Coconut/Arecanut/Paddy."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            h, w = 15, 15
            # LULC raster: center 5x5 is Agriculture (3), outside is Vegetation (2)
            lulc = np.full((h, w), 2, dtype=np.uint8)
            lulc[5:10, 5:10] = 3

            # 5-band image with reasonable reflectance
            blue = np.full((h, w), 0.05, dtype=np.float32)
            green = np.full((h, w), 0.12, dtype=np.float32)
            red = np.full((h, w), 0.07, dtype=np.float32)
            nir = np.full((h, w), 0.40, dtype=np.float32)
            swir = np.full((h, w), 0.18, dtype=np.float32)
            lulc_path = temp_dir / "land_use_testcrop.tif"
            img_path = temp_dir / "sentinel2_testcrop_mangaluru.tif"

            create_test_1band_geotiff(lulc_path, lulc)
            create_test_5band_geotiff(img_path, [blue, green, red, nir, swir])

            progress_mock = MagicMock()

            with patch("src.services.crop_service.CLASSIFIED_MANGALURU_DIR", temp_dir):
                with patch("src.services.crop_service.PROCESSED_MANGALURU_DIR", temp_dir):
                    with patch("src.services.crop_service.CROP_RESULT_DIR", temp_dir):
                        with patch("src.services.crop_service.MAPS_DIR", temp_dir):
                            with patch("src.services.crop_service.CHARTS_DIR", temp_dir):
                                res = self.service.classify_crops(
                                    "testcrop", progress_callback=progress_mock
                                )

            self.assertTrue(res["success"])
            self.assertEqual(res["agriculture_pixels"], 25)
            self.assertEqual(res["classified_crop_pixels"], 25)
            self.assertTrue(res["is_experimental"])
            self.assertIn("not been independently ground-truth validated", res["disclaimer"])

            # Verify crop raster only contains crop values (1, 2, 3) inside the 5x5 agriculture patch
            crop_img = Image.open(res["raster_path"])
            crop_arr = np.array(crop_img)
            self.assertTrue(np.all(crop_arr[:5, :] == 0))
            self.assertTrue(np.all(crop_arr[10:, :] == 0))
            self.assertTrue(np.all((crop_arr[5:10, 5:10] >= 1) & (crop_arr[5:10, 5:10] <= 3)))
        finally:
            shutil.rmtree(temp_dir)

    def test_compare_crop_years_synthetic(self):
        """compare_crop_years computes differences in crop areas between two years."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            df1 = pd.DataFrame([
                {"crop_class_id": 1, "crop_class": "Coconut", "pixels": 1000, "area_km2": 0.1, "percentage_of_mapped_crop_area": 50.0},
                {"crop_class_id": 2, "crop_class": "Arecanut", "pixels": 800, "area_km2": 0.08, "percentage_of_mapped_crop_area": 40.0},
                {"crop_class_id": 3, "crop_class": "Paddy", "pixels": 200, "area_km2": 0.02, "percentage_of_mapped_crop_area": 10.0}
            ])
            df2 = pd.DataFrame([
                {"crop_class_id": 1, "crop_class": "Coconut", "pixels": 1200, "area_km2": 0.12, "percentage_of_mapped_crop_area": 54.55},
                {"crop_class_id": 2, "crop_class": "Arecanut", "pixels": 800, "area_km2": 0.08, "percentage_of_mapped_crop_area": 36.36},
                {"crop_class_id": 3, "crop_class": "Paddy", "pixels": 200, "area_km2": 0.02, "percentage_of_mapped_crop_area": 9.09}
            ])

            df1.to_csv(temp_dir / "crop_area_statistics_y1_experimental.csv", index=False)
            df2.to_csv(temp_dir / "crop_area_statistics_y2_experimental.csv", index=False)

            with patch("src.services.crop_service.CROP_RESULT_DIR", temp_dir):
                comp = self.service.compare_crop_years("y1", "y2")

            self.assertEqual(comp["period"], "y1-y2")
            self.assertEqual(len(comp["comparison"]), 3)
            coconut = [c for c in comp["comparison"] if c["crop_class"] == "Coconut"][0]
            self.assertEqual(coconut["change_km2"], 0.02)
            self.assertEqual(coconut["percentage_change"], 20.0)
        finally:
            shutil.rmtree(temp_dir)


class TestBatch1FlaskIntegration(unittest.TestCase):
    """Integration tests for Batch 1 Flask routes and APIs."""

    def setUp(self):
        self.app = app.test_client()

    def test_api_lulc_years(self):
        """GET /api/lulc/years returns list of available classified years."""
        resp = self.app.get("/api/lulc/years")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("2022", data["years"])

    def test_api_lulc_classify_missing_params(self):
        """POST /api/lulc/classify returns 400 when scene_path/year missing."""
        resp = self.app.post("/api/lulc/classify", json={})
        self.assertEqual(resp.status_code, 400)

    def test_api_lulc_classify_submits_job(self):
        """POST /api/lulc/classify submits background job and returns 202."""
        with patch.object(lulc_service, "run_full_pipeline", return_value={"mock": "done"}):
            resp = self.app.post("/api/lulc/classify", json={"year": "2022", "tag": "test_tag"})
            self.assertEqual(resp.status_code, 202)
            data = resp.get_json()
            self.assertTrue(data["success"])
            self.assertIn("job_id", data)

    def test_api_change_periods(self):
        """GET /api/change/periods returns periods and years."""
        resp = self.app.get("/api/change/periods")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("2018-2022", data["periods"])

    def test_api_change_analyze_invalid(self):
        """POST /api/change/analyze rejects missing or identical years."""
        resp = self.app.post("/api/change/analyze", json={"year1": "2022", "year2": "2022"})
        self.assertEqual(resp.status_code, 400)

    def test_api_change_analyze_submits_job(self):
        """POST /api/change/analyze with async=True returns 202 with job_id."""
        with patch.object(change_service, "run_change_detection", return_value={"mock": "done"}):
            resp = self.app.post("/api/change/analyze", json={"year1": "2018", "year2": "2022", "async": True})
            self.assertEqual(resp.status_code, 202)
            data = resp.get_json()
            self.assertTrue(data["success"])
            self.assertIn("job_id", data)

    def test_api_crop_years(self):
        """GET /api/crop/years returns available crop years."""
        resp = self.app.get("/api/crop/years")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("2022", data["years"])

    def test_api_crop_classify_submits_job(self):
        """POST /api/crop/classify returns 202 with job_id."""
        with patch.object(crop_service, "classify_crops", return_value={"mock": "done"}):
            resp = self.app.post("/api/crop/classify", json={"year": "2022"})
            self.assertEqual(resp.status_code, 202)
            data = resp.get_json()
            self.assertTrue(data["success"])
            self.assertIn("job_id", data)

    def test_change_analysis_page(self):
        """GET /change-analysis renders page with 200."""
        resp = self.app.get("/change-analysis?period=2018-2022")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("Change Analysis", html)

    def test_crop_analysis_page(self):
        """GET /crop-analysis renders page with 200."""
        resp = self.app.get("/crop-analysis?year=2022")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("Experimental Crop Analysis", html)
        self.assertIn("Coconut", html)
        self.assertIn("Arecanut", html)


if __name__ == "__main__":
    unittest.main()
