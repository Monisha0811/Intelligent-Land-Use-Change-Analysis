"""
Unit and integration tests for Phase 2: Copernicus Data Space Satellite Retrieval.
"""

import io
import json
import os
import shutil
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from app import app
from src.config import BASE_DIR, RAW_DIR
from src.job_manager import job_manager
from src.services.satellite_service import SatelliteService, satellite_service


class TestSatelliteService(unittest.TestCase):
    """Unit tests for SatelliteService logic, authentication, and spatial calculations."""

    def setUp(self):
        self.service = SatelliteService()

    def test_credentials_status_unconfigured(self):
        """Verify credentials status reports False when env vars are absent."""
        with patch.dict(os.environ, {}, clear=True):
            status = self.service.get_credentials_status()
            self.assertFalse(status["is_configured"])
            self.assertFalse(status["has_username"])

    def test_get_auth_token_missing_credentials(self):
        """get_auth_token should raise ValueError when no credentials provided."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                self.service.get_auth_token()
            self.assertIn("CDSE credentials are not configured", str(ctx.exception))

    @patch("requests.post")
    def test_get_auth_token_success_and_caching(self, mock_post):
        """Verify successful OAuth token retrieval and caching."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "access_token": "mock-test-jwt-token-xyz",
            "expires_in": 3600
        }
        mock_post.return_value = mock_resp

        # First call fetches token
        token = self.service.get_auth_token(username="user1", password="pw1")
        self.assertEqual(token, "mock-test-jwt-token-xyz")
        self.assertEqual(mock_post.call_count, 1)

        # Second call returns cached token without network request
        cached_token = self.service.get_auth_token(username="user1", password="pw1")
        self.assertEqual(cached_token, "mock-test-jwt-token-xyz")
        self.assertEqual(mock_post.call_count, 1)

    @patch("requests.post")
    def test_get_auth_token_unauthorized(self, mock_post):
        """Verify PermissionError is raised on HTTP 401."""
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Invalid credentials"
        mock_post.return_value = mock_resp

        with self.assertRaises(PermissionError):
            self.service.get_auth_token(username="baduser", password="badpassword")

    def test_get_study_area_geometry(self):
        """Verify study area geometry loads Mangaluru Taluk boundary correctly."""
        geom, wkt, bbox_wkt = self.service.get_study_area_geometry()
        self.assertIsNotNone(geom)
        self.assertFalse(geom.is_empty)
        self.assertTrue(wkt.startswith("POLYGON"))
        self.assertTrue(bbox_wkt.startswith("POLYGON"))
        # Mangaluru Taluk coordinates in EPSG:4326 (~74.8 to 75.1 E, 12.8 to 13.1 N)
        minx, miny, maxx, maxy = geom.bounds
        self.assertGreater(minx, 74.0)
        self.assertLess(maxx, 76.0)
        self.assertGreater(miny, 12.0)
        self.assertLess(maxy, 14.0)

    def test_search_scenes_validation(self):
        """Verify search_scenes validates input dates."""
        # Invalid date format
        with self.assertRaises(ValueError):
            self.service.search_scenes("2024/01/01", "2024-02-01")

        # Start date after end date
        with self.assertRaises(ValueError):
            self.service.search_scenes("2024-03-01", "2024-01-01")

    @patch("requests.get")
    def test_search_scenes_success(self, mock_get):
        """Verify search_scenes parses OData response, computes coverage, and filters."""
        # Mock catalogue response with two scenes: one 43PDQ and one 43PDR (different tile)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "value": [
                {
                    "Id": "prod-1",
                    "Name": "S2B_MSIL2A_20240115T045959_N0510_R119_T43PDQ_20240115T083000",
                    "ContentDate": {"Start": "2024-01-15T04:59:59.024Z"},
                    "ContentLength": 850000000,
                    "GeoFootprint": {
                        "type": "Polygon",
                        "coordinates": [[[74.5, 12.5], [75.5, 12.5], [75.5, 13.5], [74.5, 13.5], [74.5, 12.5]]]
                    },
                    "Attributes": [
                        {"Name": "cloudCover", "Value": 5.4}
                    ]
                },
                {
                    "Id": "prod-2",
                    "Name": "S2A_MSIL2A_20240110T045959_N0510_R119_T43PDR_20240110T083000",
                    "ContentDate": {"Start": "2024-01-10T04:59:59.024Z"},
                    "ContentLength": 800000000,
                    "GeoFootprint": None,
                    "Attributes": [
                        {"Name": "cloudCover", "Value": 1.2}
                    ]
                }
            ]
        }
        mock_get.return_value = mock_resp

        scenes = self.service.search_scenes("2024-01-01", "2024-01-31", max_cloud_cover=20.0, tile="43PDQ")

        # prod-2 should be filtered out by tile filter
        self.assertEqual(len(scenes), 1)
        scene = scenes[0]
        self.assertEqual(scene["id"], "prod-1")
        self.assertEqual(scene["cloud_cover"], 5.4)
        self.assertGreater(scene["coverage_percent"], 90.0)
        self.assertEqual(scene["formatted_date"], "2024-01-15")

    def test_download_scene_already_exists(self):
        """Verify early completion when scene is already downloaded and bands exist."""
        temp_raw = Path(tempfile.mkdtemp())
        safe_name = "S2B_MSIL2A_TEST_ALREADY_EXISTS.SAFE"
        year_dir = temp_raw / "2024"
        safe_dir = year_dir / safe_name
        granule_dir = safe_dir / "GRANULE" / "L2A_T43PDQ" / "IMG_DATA" / "R10m"
        granule_dir_20m = safe_dir / "GRANULE" / "L2A_T43PDQ" / "IMG_DATA" / "R20m"
        granule_dir.mkdir(parents=True, exist_ok=True)
        granule_dir_20m.mkdir(parents=True, exist_ok=True)

        try:
            # Create dummy band files
            for b in ["B02_10m", "B03_10m", "B04_10m", "B08_10m"]:
                (granule_dir / f"T43PDQ_{b}.jp2").touch()
            (granule_dir_20m / "T43PDQ_B11_20m.jp2").touch()

            # Temporarily point download_scene target to temp_raw
            progress_mock = MagicMock()
            with patch("src.services.satellite_service.RAW_DIR", temp_raw):
                result = self.service.download_scene(
                    product_id="test-id",
                    product_name="S2B_MSIL2A_TEST_ALREADY_EXISTS",
                    date="2024-01-15",
                    progress_callback=progress_mock
                )

            self.assertTrue(result["already_existed"])
            self.assertEqual(len(result["bands_found"]), 5)
            progress_mock.assert_called_with(100, f"Scene 'S2B_MSIL2A_TEST_ALREADY_EXISTS' already exists locally in {safe_name}.")

        finally:
            if temp_raw.exists():
                shutil.rmtree(temp_raw)

    @patch("src.services.satellite_service.SatelliteService.get_auth_token")
    @patch("requests.get")
    def test_download_scene_streaming_and_extraction(self, mock_get, mock_auth):
        """Verify streaming download creates part file, unzips, and removes temporary archive."""
        mock_auth.return_value = "dummy-token"

        # Create an in-memory zip archive containing mock Sentinel-2 SAFE structure
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            prefix = "S2B_MSIL2A_MOCK_2024.SAFE"
            zf.writestr(f"{prefix}/GRANULE/L2A_T43PDQ/IMG_DATA/R10m/T43PDQ_B02_10m.jp2", b"B02 data")
            zf.writestr(f"{prefix}/GRANULE/L2A_T43PDQ/IMG_DATA/R10m/T43PDQ_B03_10m.jp2", b"B03 data")
            zf.writestr(f"{prefix}/GRANULE/L2A_T43PDQ/IMG_DATA/R10m/T43PDQ_B04_10m.jp2", b"B04 data")
            zf.writestr(f"{prefix}/GRANULE/L2A_T43PDQ/IMG_DATA/R10m/T43PDQ_B08_10m.jp2", b"B08 data")
            zf.writestr(f"{prefix}/GRANULE/L2A_T43PDQ/IMG_DATA/R20m/T43PDQ_B11_20m.jp2", b"B11 data")
        zip_bytes = zip_buffer.getvalue()

        # Mock streaming response
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Length": str(len(zip_bytes))}
        # Yield in chunks
        chunk_size = 512
        chunks = [zip_bytes[i:i + chunk_size] for i in range(0, len(zip_bytes), chunk_size)]
        mock_resp.iter_content.return_value = iter(chunks)
        mock_get.return_value = mock_resp

        temp_raw_root = Path(tempfile.mkdtemp())
        try:
            progress_mock = MagicMock()
            with patch("src.services.satellite_service.RAW_DIR", temp_raw_root):
                result = self.service.download_scene(
                    product_id="mock-product-id",
                    product_name="S2B_MSIL2A_MOCK_2024",
                    date="2024-02-15",
                    progress_callback=progress_mock
                )

            self.assertFalse(result["already_existed"])
            self.assertEqual(result["year"], "2024")
            self.assertEqual(len(result["bands_found"]), 5)
            # Verify zip was extracted and removed
            safe_extracted = temp_raw_root / "2024" / "S2B_MSIL2A_MOCK_2024.SAFE"
            self.assertTrue(safe_extracted.is_dir())
            zip_remaining = temp_raw_root / "2024" / "S2B_MSIL2A_MOCK_2024.zip"
            self.assertFalse(zip_remaining.exists())

        finally:
            if temp_raw_root.exists():
                shutil.rmtree(temp_raw_root)

    @patch("src.services.satellite_service.SatelliteService.get_auth_token")
    @patch("requests.get")
    def test_download_scene_cancellation(self, mock_get, mock_auth):
        """Verify download terminates and cleans up when cancellation is requested."""
        mock_auth.return_value = "dummy-token"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Length": "10000000"}
        mock_resp.iter_content.return_value = iter([b"chunk1", b"chunk2", b"chunk3"])
        mock_get.return_value = mock_resp

        # Cancel condition triggers on second check
        cancel_calls = [False, True]
        def is_cancelled_mock():
            if cancel_calls:
                return cancel_calls.pop(0)
            return True

        temp_raw_root = Path(tempfile.mkdtemp())
        try:
            with patch("src.services.satellite_service.RAW_DIR", temp_raw_root):
                with self.assertRaises(RuntimeError) as ctx:
                    self.service.download_scene(
                        product_id="mock-cancel-id",
                        product_name="S2B_MSIL2A_MOCK_CANCEL",
                        date="2024-02-15",
                        is_cancelled=is_cancelled_mock
                    )

            self.assertIn("cancelled", str(ctx.exception).lower())
            # Verify partial file is removed
            part_file = temp_raw_root / "2024" / "S2B_MSIL2A_MOCK_CANCEL.zip.part"
            self.assertFalse(part_file.exists())

        finally:
            if temp_raw_root.exists():
                shutil.rmtree(temp_raw_root)


class TestSatelliteFlaskRoutes(unittest.TestCase):
    """Integration tests for Flask routes and background job submission."""

    def setUp(self):
        self.app = app.test_client()

    def test_satellite_retrieval_page(self):
        """GET /satellite-retrieval should return 200 and render the HTML interface."""
        resp = self.app.get("/satellite-retrieval")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("Automatic Satellite Data Retrieval", html)
        self.assertIn("Copernicus Sentinel-2 L2A Scene Search", html)
        self.assertIn("Mangaluru Taluk", html)

    def test_api_credentials_status(self):
        """GET /api/satellite/credentials-status should return JSON status."""
        resp = self.app.get("/api/satellite/credentials-status")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("credentials", data)
        self.assertIn("is_configured", data["credentials"])

    def test_api_search_missing_dates(self):
        """POST /api/satellite/search should return 400 when dates are missing."""
        resp = self.app.post("/api/satellite/search", json={})
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data["success"])
        self.assertIn("required", data["error"])

    def test_api_search_invalid_cloud(self):
        """POST /api/satellite/search should return 400 for out-of-range cloud cover."""
        resp = self.app.post("/api/satellite/search", json={
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "max_cloud": 150
        })
        self.assertEqual(resp.status_code, 400)

    @patch("src.services.satellite_service.SatelliteService.search_scenes")
    def test_api_search_success(self, mock_search):
        """POST /api/satellite/search should return 200 with scene list."""
        mock_search.return_value = [
            {
                "id": "prod-1",
                "name": "S2A_MSIL2A_TEST",
                "date": "2024-01-15T00:00:00Z",
                "formatted_date": "2024-01-15",
                "cloud_cover": 2.5,
                "coverage_percent": 100.0,
                "size_mb": 750.0,
                "already_downloaded": False
            }
        ]

        resp = self.app.post("/api/satellite/search", json={
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "max_cloud": 20.0,
            "tile": "43PDQ"
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["scenes"][0]["id"], "prod-1")

    def test_api_download_missing_params(self):
        """POST /api/satellite/download should return 400 for missing fields."""
        resp = self.app.post("/api/satellite/download", json={})
        self.assertEqual(resp.status_code, 400)

    def test_api_download_unconfigured_credentials(self):
        """POST /api/satellite/download should reject download if CDSE unconfigured."""
        with patch.object(satellite_service, "get_credentials_status", return_value={"is_configured": False}):
            resp = self.app.post("/api/satellite/download", json={
                "product_id": "test-id",
                "product_name": "S2_TEST_SCENE"
            })
            self.assertEqual(resp.status_code, 400)
            data = resp.get_json()
            self.assertIn("not configured", data["error"])

    def test_api_download_submits_job(self):
        """POST /api/satellite/download should submit job and return 202 with job_id."""
        with patch.object(satellite_service, "get_credentials_status", return_value={"is_configured": True}):
            with patch.object(satellite_service, "download_scene", return_value={"mock": "done"}):
                resp = self.app.post("/api/satellite/download", json={
                    "product_id": "mock-prod-123",
                    "product_name": "S2B_MSIL2A_20240215_T43PDQ",
                    "date": "2024-02-15"
                })
                self.assertEqual(resp.status_code, 202)
                data = resp.get_json()
                self.assertTrue(data["success"])
                self.assertIn("job_id", data)
                job_id = data["job_id"]

                # Verify job exists in job manager
                time.sleep(0.1)
                job_resp = self.app.get(f"/api/jobs/{job_id}")
                self.assertEqual(job_resp.status_code, 200)
                job_data = job_resp.get_json()
                self.assertTrue(job_data["success"])
                self.assertEqual(job_data["job"]["job_type"], "satellite_download")


if __name__ == "__main__":
    unittest.main()
