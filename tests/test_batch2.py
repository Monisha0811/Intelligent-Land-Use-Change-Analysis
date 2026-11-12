"""
Comprehensive automated tests for Batch 2:
- Phase 6: Confidence, Reliability and Advanced Spatial Analytics
- Phase 7: Dashboard Integration, Visualizations, and Result Exports
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
    CLASSIFIED_MANGALURU_DIR,
    PROCESSED_MANGALURU_DIR,
    MAPS_DIR,
    RESULTS_DIR,
    CONFIDENCE_DIR,
    RELIABILITY_DIR,
    HOTSPOTS_DIR,
    TRAJECTORIES_DIR,
    AGRICULTURE_ANALYSIS_DIR,
    BUILTUP_ANALYSIS_DIR,
    LULC_CLASSES,
    PIXEL_AREA_KM2,
)
from src.services.analytics_service import AnalyticsService, analytics_service

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


def create_test_1band_geotiff(path: Path, array: np.ndarray, dtype="uint8", nodata=0, crs="EPSG:32643") -> None:
    """Create a proper 1-band GeoTIFF fixture using Rasterio if available."""
    h, w = array.shape
    if HAS_RASTERIO and rasterio is not None:
        profile = {
            "driver": "GTiff",
            "dtype": dtype,
            "nodata": nodata,
            "width": w,
            "height": h,
            "count": 1,
            "crs": crs,
            "transform": from_origin(0, 0, 10, 10),
            "compress": "lzw"
        }
        with rasterio.open(str(path), "w", **profile) as dst:
            dst.write(array.astype(dtype), 1)
    else:
        img = Image.fromarray(array.astype(dtype))
        img.save(str(path), format="TIFF")


class TestPhase6AnalyticsService(unittest.TestCase):
    """Tests for Phase 6 spatial analytics functions."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.service = AnalyticsService()

        # Create isolated subdirectories inside temp
        for p in [
            self.temp_path / "confidence",
            self.temp_path / "reliability",
            self.temp_path / "hotspots",
            self.temp_path / "trajectories",
            self.temp_path / "agriculture_analysis",
            self.temp_path / "builtup_analysis",
            self.temp_path / "maps"
        ]:
            p.mkdir(parents=True, exist_ok=True)

        # Isolate output directories inside temporary directory to protect real project data
        self.patchers = [
            patch("src.services.analytics_service.CONFIDENCE_DIR", self.temp_path / "confidence"),
            patch("src.services.analytics_service.RELIABILITY_DIR", self.temp_path / "reliability"),
            patch("src.services.analytics_service.HOTSPOTS_DIR", self.temp_path / "hotspots"),
            patch("src.services.analytics_service.TRAJECTORIES_DIR", self.temp_path / "trajectories"),
            patch("src.services.analytics_service.AGRICULTURE_ANALYSIS_DIR", self.temp_path / "agriculture_analysis"),
            patch("src.services.analytics_service.BUILTUP_ANALYSIS_DIR", self.temp_path / "builtup_analysis"),
            patch("src.services.analytics_service.MAPS_DIR", self.temp_path / "maps"),
        ]
        for p in self.patchers:
            p.start()

    def tearDown(self):
        for p in self.patchers:
            p.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_confidence_calculation_synthetic(self):
        """Test prediction confidence calculation with synthetic 5-band raster and mock RF model."""
        h, w = 30, 30
        blue = np.full((h, w), 0.05, dtype=np.float32)
        green = np.full((h, w), 0.10, dtype=np.float32)
        red = np.full((h, w), 0.08, dtype=np.float32)
        nir = np.full((h, w), 0.35, dtype=np.float32)
        swir = np.full((h, w), 0.15, dtype=np.float32)

        proc_tif = Path(self.temp_dir) / "sentinel2_test_mangaluru.tif"
        cls_tif = Path(self.temp_dir) / "land_use_test.tif"

        create_test_5band_geotiff(proc_tif, [blue, green, red, nir, swir])
        cls_data = np.full((h, w), 2, dtype=np.uint8)  # Vegetation
        create_test_1band_geotiff(cls_tif, cls_data, dtype="uint8", nodata=0)

        # Mock Random Forest model with 5 classes and high confidence for class 2
        mock_model = MagicMock()
        mock_model.classes_ = np.array([1, 2, 3, 4, 5])
        mock_model.n_features_in_ = 8

        def mock_predict_proba(X):
            n = len(X)
            # High probability for class 2: 0.85
            probs = np.full((n, 5), 0.0375, dtype=np.float32)
            probs[:, 1] = 0.85
            return probs

        mock_model.predict_proba.side_effect = mock_predict_proba

        with patch.object(self.service, "get_model", return_value=mock_model):
            res = self.service.calculate_confidence(
                year="test",
                processed_path=proc_tif,
                classified_path=cls_tif
            )

        self.assertIn("summary", res)
        self.assertEqual(res["summary"]["total_valid_pixels"], h * w)
        self.assertEqual(res["summary"]["high_pixels"], h * w)
        self.assertAlmostEqual(res["summary"]["mean_confidence"], 0.85, places=2)
        self.assertTrue(Path(res["confidence_raster"]).exists())
        self.assertTrue(Path(res["confidence_class_raster"]).exists())
        self.assertTrue(Path(res["overlay_png"]).exists())
        self.assertTrue(Path(res["statistics_csv"]).exists())

    def test_confidence_missing_files_error(self):
        """Verify FileNotFoundError if input rasters do not exist."""
        with self.assertRaises(FileNotFoundError):
            self.service.calculate_confidence(year="9999_nonexistent")

    def test_hotspot_detection_synthetic(self):
        """Test relative spatial change-density hotspot detection with synthetic changed pixels."""
        h, w = 40, 40
        y1_tif = Path(self.temp_dir) / "land_use_y1.tif"
        y2_tif = Path(self.temp_dir) / "land_use_y2.tif"

        lulc1 = np.full((h, w), 3, dtype=np.uint8)  # Agriculture
        lulc2 = np.full((h, w), 3, dtype=np.uint8)
        # Cluster of change in center: 4 (Built-up)
        lulc2[15:25, 15:25] = 4

        create_test_1band_geotiff(y1_tif, lulc1, dtype="uint8", nodata=0)
        create_test_1band_geotiff(y2_tif, lulc2, dtype="uint8", nodata=0)

        with patch("src.services.analytics_service.CLASSIFIED_MANGALURU_DIR", Path(self.temp_dir)):
            res = self.service.detect_hotspots(year1="y1", year2="y2", sigma=5.0)

        self.assertEqual(res["period"], "y1-y2")
        self.assertEqual(res["valid_pixels"], h * w)
        self.assertEqual(res["changed_pixels"], 100)
        self.assertTrue(Path(res["score_raster"]).exists())
        self.assertTrue(Path(res["class_raster"]).exists())
        self.assertTrue(Path(res["overlay_png"]).exists())
        self.assertTrue(len(res["statistics"]) > 0)

    def test_hotspot_detection_no_changes(self):
        """Test hotspot detection when two years are completely identical."""
        h, w = 20, 20
        y1_tif = Path(self.temp_dir) / "land_use_same1.tif"
        y2_tif = Path(self.temp_dir) / "land_use_same2.tif"

        lulc = np.full((h, w), 2, dtype=np.uint8)
        create_test_1band_geotiff(y1_tif, lulc, dtype="uint8", nodata=0)
        create_test_1band_geotiff(y2_tif, lulc, dtype="uint8", nodata=0)

        with patch("src.services.analytics_service.CLASSIFIED_MANGALURU_DIR", Path(self.temp_dir)):
            res = self.service.detect_hotspots(year1="same1", year2="same2")

        self.assertEqual(res["changed_pixels"], 0)
        # All valid pixels assigned Low hotspot score
        low_stat = [s for s in res["statistics"] if s["Hotspot Class"] == 1][0]
        self.assertEqual(low_stat["Pixels"], h * w)

    def test_trajectory_analysis_synthetic(self):
        """Test 3-year trajectory classification categories."""
        h, w = 5, 1
        # Pixel 0: Stable (1 -> 1 -> 1)
        # Pixel 1: Temporary (3 -> 2 -> 3)
        # Pixel 2: Persistent (3 -> 4 -> 4)
        # Pixel 3: Recent (3 -> 3 -> 4)
        # Pixel 4: Progressive (3 -> 2 -> 4)
        c1 = np.array([[1], [3], [3], [3], [3]], dtype=np.uint8)
        c2 = np.array([[1], [2], [4], [3], [2]], dtype=np.uint8)
        c3 = np.array([[1], [3], [4], [4], [4]], dtype=np.uint8)

        t1_tif = Path(self.temp_dir) / "land_use_t1.tif"
        t2_tif = Path(self.temp_dir) / "land_use_t2.tif"
        t3_tif = Path(self.temp_dir) / "land_use_t3.tif"

        create_test_1band_geotiff(t1_tif, c1, dtype="uint8", nodata=0)
        create_test_1band_geotiff(t2_tif, c2, dtype="uint8", nodata=0)
        create_test_1band_geotiff(t3_tif, c3, dtype="uint8", nodata=0)

        with patch("src.services.analytics_service.CLASSIFIED_MANGALURU_DIR", Path(self.temp_dir)):
            res = self.service.analyze_trajectories(years=["t1", "t2", "t3"])

        self.assertEqual(res["total_pixels"], 5)
        type_dict = {row["Trajectory Type"]: row["Pixel Count"] for row in res["trajectory_types"]}
        self.assertEqual(type_dict.get("Stable"), 1)
        self.assertEqual(type_dict.get("Temporary Change"), 1)
        self.assertEqual(type_dict.get("Persistent Change"), 1)
        self.assertEqual(type_dict.get("Recent Change"), 1)
        self.assertEqual(type_dict.get("Progressive Change"), 1)
        self.assertTrue(Path(res["trajectory_csv"]).exists())

    def test_fragmentation_analysis_synthetic(self):
        """Test patch fragmentation metric computation for a known binary pattern."""
        h, w = 20, 20
        grid = np.zeros((h, w), dtype=np.uint8)
        # Create 2 disjoint patches of Agriculture (class 3)
        grid[2:5, 2:5] = 3    # 3x3 = 9 pixels
        grid[10:14, 10:14] = 3  # 4x4 = 16 pixels
        # Set background to 1 (Water) so valid pixels = h*w
        grid[grid == 0] = 1

        f_tif = Path(self.temp_dir) / "land_use_frag.tif"
        create_test_1band_geotiff(f_tif, grid, dtype="uint8", nodata=0)

        with patch("src.services.analytics_service.CLASSIFIED_MANGALURU_DIR", Path(self.temp_dir)):
            results = self.service.analyze_fragmentation(year="frag", class_id=3)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["number_of_patches"], 2)
        self.assertEqual(r["class_pixels"], 25)
        # Largest patch: 16 pixels * 0.0001 km² = 0.0016 km²
        self.assertAlmostEqual(r["largest_patch_km2"], 0.0016, places=5)
        # Mean patch: 12.5 pixels * 0.0001 km² = 0.00125 km²
        self.assertAlmostEqual(r["mean_patch_km2"], 0.00125, places=5)
        # Largest patch % of agriculture: 16/25 * 100 = 64.0%
        self.assertAlmostEqual(r["largest_patch_percentage_of_class"], 64.0, places=1)

    def test_reliability_analysis_synthetic(self):
        """Test transition reliability evaluation based on minimum joint confidence."""
        h, w = 10, 10
        # Year 1: Class 3, Year 2: Class 4 (all changed)
        l1 = np.full((h, w), 3, dtype=np.uint8)
        l2 = np.full((h, w), 4, dtype=np.uint8)

        # Half high confidence (0.85 & 0.90 -> min 0.85 = High)
        # Half moderate confidence (0.65 & 0.70 -> min 0.65 = Moderate)
        conf1 = np.full((h, w), 0.65, dtype=np.float32)
        conf2 = np.full((h, w), 0.70, dtype=np.float32)
        conf1[:5, :] = 0.85
        conf2[:5, :] = 0.90

        p1_tif = Path(self.temp_dir) / "land_use_r1.tif"
        p2_tif = Path(self.temp_dir) / "land_use_r2.tif"
        c1_tif = Path(self.temp_dir) / "lulc_confidence_r1.tif"
        c2_tif = Path(self.temp_dir) / "lulc_confidence_r2.tif"

        create_test_1band_geotiff(p1_tif, l1, dtype="uint8", nodata=0)
        create_test_1band_geotiff(p2_tif, l2, dtype="uint8", nodata=0)
        create_test_1band_geotiff(c1_tif, conf1, dtype="float32", nodata=0)
        create_test_1band_geotiff(c2_tif, conf2, dtype="float32", nodata=0)

        with patch("src.services.analytics_service.CLASSIFIED_MANGALURU_DIR", Path(self.temp_dir)), \
             patch("src.services.analytics_service.CONFIDENCE_DIR", Path(self.temp_dir)):
            res = self.service.analyze_reliability(year1="r1", year2="r2")

        self.assertEqual(res["total_changed_pixels"], 100)
        rel_dict = {row["Reliability"]: row["Pixels"] for row in res["overall_reliability"]}
        self.assertEqual(rel_dict["High"], 50)
        self.assertEqual(rel_dict["Moderate"], 50)
        self.assertEqual(rel_dict["Low"], 0)

    def test_analytics_cancellation(self):
        """Test that user cancellation cleanly interrupts processing."""
        def cancelled():
            return True

        with self.assertRaises(InterruptedError):
            self.service.calculate_confidence(
                year="cancel_test",
                is_cancelled=cancelled
            )


class TestPhase7FlaskIntegration(unittest.TestCase):
    """Tests for Flask API routes, export endpoints, and template rendering for Batch 2."""

    def setUp(self):
        self.client = app.test_client()

    def test_api_analytics_confidence_submits_job(self):
        """POST /api/analytics/confidence submits background job."""
        with patch.object(analytics_service, "calculate_confidence", return_value={"status": "mock"}):
            response = self.client.post("/api/analytics/confidence", json={"year": 2022})
            self.assertEqual(response.status_code, 202)
            data = response.get_json()
            self.assertTrue(data["success"])
            self.assertIn("job_id", data)

    def test_api_analytics_confidence_missing_year(self):
        """POST /api/analytics/confidence without year returns 400."""
        response = self.client.post("/api/analytics/confidence", json={})
        self.assertEqual(response.status_code, 400)

    def test_api_analytics_hotspots_submits_job(self):
        """POST /api/analytics/hotspots submits background job."""
        with patch.object(analytics_service, "detect_hotspots", return_value={"status": "mock"}):
            response = self.client.post("/api/analytics/hotspots", json={"start_year": 2018, "end_year": 2022})
            self.assertEqual(response.status_code, 202)
            data = response.get_json()
            self.assertTrue(data["success"])
            self.assertIn("job_id", data)

    def test_api_analytics_trajectories_submits_job(self):
        """POST /api/analytics/trajectories submits background job."""
        with patch.object(analytics_service, "analyze_trajectories", return_value={"status": "mock"}):
            response = self.client.post("/api/analytics/trajectories", json={"years": [2018, 2022, 2025]})
            self.assertEqual(response.status_code, 202)
            data = response.get_json()
            self.assertTrue(data["success"])

    def test_api_analytics_fragmentation_submits_job(self):
        """POST /api/analytics/fragmentation submits background job."""
        with patch.object(analytics_service, "analyze_fragmentation", return_value=[{"status": "mock"}]):
            response = self.client.post("/api/analytics/fragmentation", json={"year": 2022, "class_id": 3})
            self.assertEqual(response.status_code, 202)
            data = response.get_json()
            self.assertTrue(data["success"])

    def test_api_analytics_reliability_submits_job(self):
        """POST /api/analytics/reliability submits background job."""
        with patch.object(analytics_service, "analyze_reliability", return_value={"status": "mock"}):
            response = self.client.post("/api/analytics/reliability", json={"period": "2018-2022"})
            self.assertEqual(response.status_code, 202)
            data = response.get_json()
            self.assertTrue(data["success"])

    def test_api_analytics_stats_confidence(self):
        """GET /api/analytics/stats?type=confidence&year=2022 returns JSON data."""
        response = self.client.get("/api/analytics/stats?type=confidence&year=2022")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["type"], "confidence")

    def test_export_classified_raster(self):
        """GET /export/raster/classified/2022 downloads GeoTIFF."""
        response = self.client.get("/export/raster/classified/2022")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("Content-Disposition", "")[:10], "attachment")

    def test_export_change_raster(self):
        """GET /export/raster/change/2018-2022 downloads GeoTIFF."""
        response = self.client.get("/export/raster/change/2018-2022")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("Content-Disposition", "")[:10], "attachment")

    def test_export_csv_whitelisted(self):
        """GET /export/csv/results/land_use_area_statistics.csv downloads CSV."""
        response = self.client.get("/export/csv/results/land_use_area_statistics.csv")
        self.assertEqual(response.status_code, 200)
        self.assertIn("attachment", response.headers.get("Content-Disposition", ""))

    def test_export_csv_invalid_category_rejected(self):
        """GET /export/csv/unauthorized_cat/file.csv returns 400."""
        response = self.client.get("/export/csv/unauthorized_cat/secret.txt")
        self.assertEqual(response.status_code, 400)

    def test_explorer_page_contains_hotspot_layers(self):
        """GET /explorer renders page with hotspot overlays in script."""
        response = self.client.get("/explorer")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Change Hotspots 2018-2022", html)
        self.assertIn("hotspot-overlay", html)

    def test_statistics_page_contains_download_links(self):
        """GET /statistics renders page with download buttons."""
        response = self.client.get("/statistics")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("/export/csv/results/land_use_area_statistics.csv", html)
        self.assertIn("/export/csv/confidence/confidence_by_class.csv", html)

    def test_change_analysis_page_contains_exports(self):
        """GET /change-analysis renders page with export buttons."""
        response = self.client.get("/change-analysis?period=2018-2022")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("/export/raster/change/2018-2022", html)
        self.assertIn("/export/csv/results/transition_details_2018_2022.csv", html)

    def test_crop_analysis_page_contains_exports(self):
        """GET /crop-analysis renders page with export buttons and experimental caveats."""
        response = self.client.get("/crop-analysis?year=2022")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("/export/raster/crop/2022", html)
        self.assertIn("Important limitation:", html)


if __name__ == "__main__":
    unittest.main()
