"""
Copernicus Data Space Ecosystem (CDSE) Satellite Retrieval Service.

Provides authentication, scene catalogue searching, spatial coverage calculation,
and authenticated streaming downloads for Sentinel-2 Level-2A imagery.
"""

import os
import re
import json
import time
import zipfile
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Any, Optional, List, Tuple

import requests
from shapely.geometry import shape

from src.config import (
    BASE_DIR,
    RAW_DIR,
    BOUNDARY_GEOJSON,
    CDSE_CATALOGUE_URL,
    CDSE_TOKEN_URL,
    get_cdse_credentials,
)

logger = logging.getLogger("satellite_service")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [SatelliteService] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class SatelliteService:
    """Service handling Copernicus Data Space Ecosystem (CDSE) interactions."""

    def __init__(self):
        self._token_cache: Dict[str, Any] = {
            "access_token": None,
            "expires_at": 0
        }
        self._aoi_geom = None
        self._aoi_bbox_wkt = None
        self._aoi_wkt = None

    def get_credentials_status(self) -> Dict[str, bool]:
        """Check whether CDSE credentials are configured without exposing values."""
        creds = get_cdse_credentials()
        return {
            "is_configured": creds["is_configured"],
            "has_username": bool(creds["username"] and creds["password"]),
            "has_client_id": bool(creds["client_id"] and creds["client_secret"])
        }

    def get_auth_token(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        force_refresh: bool = False
    ) -> str:
        """
        Authenticate with CDSE Keycloak OAuth2 service and return Bearer access token.
        Caches active token until 30 seconds before expiration.
        """
        # Return cached token if still valid
        now = time.time()
        if not force_refresh and self._token_cache["access_token"] and now < self._token_cache["expires_at"]:
            return self._token_cache["access_token"]

        # Resolve credentials
        env_creds = get_cdse_credentials()
        uname = (username or env_creds["username"]).strip()
        pwd = (password or env_creds["password"]).strip()
        cid = (client_id or env_creds["client_id"]).strip()
        csec = (client_secret or env_creds["client_secret"]).strip()

        if not ((uname and pwd) or (cid and csec)):
            raise ValueError(
                "CDSE credentials are not configured. Please add CDSE_USERNAME and "
                "CDSE_PASSWORD to your .env file."
            )

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        if uname and pwd:
            data = {
                "client_id": "cdse-public",
                "username": uname,
                "password": pwd,
                "grant_type": "password"
            }
        else:
            data = {
                "client_id": cid,
                "client_secret": csec,
                "grant_type": "client_credentials"
            }

        try:
            response = requests.post(
                CDSE_TOKEN_URL,
                headers=headers,
                data=data,
                timeout=30
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error connecting to Copernicus Identity service: {e}")

        if response.status_code == 200:
            token_data = response.json()
            access_token = token_data.get("access_token")
            expires_in = token_data.get("expires_in", 600)

            # Cache token with 30s buffer
            self._token_cache = {
                "access_token": access_token,
                "expires_at": now + expires_in - 30
            }
            logger.info("Successfully authenticated with Copernicus Data Space Ecosystem.")
            return access_token

        # Handle authentication failures gracefully
        try:
            err_json = response.json()
            desc = err_json.get("error_description") or err_json.get("error") or response.text
        except Exception:
            desc = response.text

        if response.status_code in [400, 401]:
            raise PermissionError(
                f"CDSE Authentication failed ({response.status_code}): {desc}. "
                "Please verify your CDSE_USERNAME and CDSE_PASSWORD."
            )
        else:
            raise RuntimeError(f"CDSE Identity server error (HTTP {response.status_code}): {desc}")

    def get_study_area_geometry(self) -> Tuple[Any, str, str]:
        """
        Load Mangaluru Taluk boundary and return geometry, full WKT, and bounding box WKT.
        """
        if self._aoi_geom is not None:
            return self._aoi_geom, self._aoi_wkt, self._aoi_bbox_wkt

        if not BOUNDARY_GEOJSON.exists():
            raise FileNotFoundError(f"Boundary file not found at: {BOUNDARY_GEOJSON}")

        with open(BOUNDARY_GEOJSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        features = data.get("features", [])
        if not features:
            raise ValueError(f"No features found in boundary file: {BOUNDARY_GEOJSON}")

        self._aoi_geom = shape(features[0]["geometry"])
        self._aoi_wkt = self._aoi_geom.wkt

        # Bounding box envelope polygon for compact OData queries
        minx, miny, maxx, maxy = self._aoi_geom.bounds
        self._aoi_bbox_wkt = f"POLYGON(({minx} {miny}, {maxx} {miny}, {maxx} {maxy}, {minx} {maxy}, {minx} {miny}))"

        return self._aoi_geom, self._aoi_wkt, self._aoi_bbox_wkt

    def search_scenes(
        self,
        start_date: str,
        end_date: str,
        max_cloud_cover: float = 30.0,
        tile: str = "43PDQ",
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search Sentinel-2 Level-2A surface reflectance products intersecting Mangaluru Taluk.
        
        Args:
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD
            max_cloud_cover: Maximum acceptable cloud percentage (0 to 100)
            tile: Sentinel-2 MGRS tile filter (e.g. '43PDQ')
            limit: Maximum scenes to retrieve from catalog
        """
        # Validate date formats
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        if not date_pattern.match(start_date) or not date_pattern.match(end_date):
            raise ValueError("Dates must be formatted as YYYY-MM-DD.")

        if start_date > end_date:
            raise ValueError(f"Start date ({start_date}) cannot be after end date ({end_date}).")

        aoi_geom, _, bbox_wkt = self.get_study_area_geometry()

        # Build OData filter
        spatial_filter = f"OData.CSC.Intersects(area=geography'SRID=4326;{bbox_wkt}')"
        collection_filter = "Collection/Name eq 'SENTINEL-2'"
        product_filter = (
            "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
            "and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A')"
        )
        cloud_filter = (
            "Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
            f"and att/OData.CSC.DoubleAttribute/Value le {float(max_cloud_cover)})"
        )
        date_filter = (
            f"ContentDate/Start gt {start_date}T00:00:00.000Z "
            f"and ContentDate/Start lt {end_date}T23:59:59.999Z"
        )

        filter_query = (
            f"{collection_filter} and {product_filter} and {cloud_filter} "
            f"and {spatial_filter} and {date_filter}"
        )

        params = {
            "$filter": filter_query,
            "$top": limit,
            "$expand": "Attributes"
        }

        logger.info(f"Querying CDSE Catalogue ({start_date} to {end_date}, cloud <= {max_cloud_cover}%)...")

        try:
            response = requests.get(
                CDSE_CATALOGUE_URL,
                params=params,
                timeout=60
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network failure connecting to CDSE catalogue: {e}")

        if response.status_code != 200:
            raise RuntimeError(f"CDSE catalogue returned HTTP {response.status_code}: {response.text[:500]}")

        catalog_data = response.json()
        raw_products = catalog_data.get("value", [])

        candidates = []

        for p in raw_products:
            name = p.get("Name", "")
            pid = p.get("Id", "")
            start_time_raw = p.get("ContentDate", {}).get("Start", "")
            content_length = p.get("ContentLength", 0)

            # Filter by tile if specified (Sentinel-2 L2A names contain tile like _T43PDQ_)
            if tile and f"T{tile}" not in name and tile not in name:
                continue

            # Extract cloudCover attribute
            cloud_cover = None
            for attr in p.get("Attributes", []):
                if attr.get("Name") == "cloudCover":
                    cloud_cover = attr.get("Value")
                    break

            # Calculate exact coverage percentage with Mangaluru Taluk
            coverage_pct = 0.0
            geo = p.get("GeoFootprint")
            if geo:
                try:
                    scene_geom = shape(geo)
                    inter = aoi_geom.intersection(scene_geom)
                    if not aoi_geom.is_empty:
                        coverage_pct = (inter.area / aoi_geom.area) * 100.0
                except Exception as e:
                    logger.warning(f"Could not compute coverage for {name}: {e}")

            # Check if product is already downloaded locally
            year_str = start_time_raw[:4] if len(start_time_raw) >= 4 else "unknown"
            local_safe_dir = RAW_DIR / year_str / f"{name}.SAFE"
            local_zip_file = RAW_DIR / year_str / f"{name}.zip"
            is_local = local_safe_dir.is_dir() or local_zip_file.is_file()

            candidates.append({
                "id": pid,
                "name": name,
                "date": start_time_raw,
                "formatted_date": start_time_raw[:10] if len(start_time_raw) >= 10 else "",
                "cloud_cover": round(float(cloud_cover), 2) if cloud_cover is not None else None,
                "coverage_percent": round(coverage_pct, 2),
                "size_mb": round(content_length / (1024 * 1024), 1) if content_length else None,
                "already_downloaded": is_local,
                "local_path": str(local_safe_dir) if local_safe_dir.is_dir() else (str(local_zip_file) if local_zip_file.is_file() else None)
            })

        # Sort: highest Mangaluru coverage first, then lowest cloud cover
        candidates.sort(
            key=lambda x: (
                -x["coverage_percent"],
                x["cloud_cover"] if x["cloud_cover"] is not None else 999.0
            )
        )

        logger.info(f"Found {len(candidates)} candidate scenes for Mangaluru Taluk.")
        return candidates

    def download_scene(
        self,
        product_id: str,
        product_name: str,
        date: str = "",
        progress_callback: Optional[Callable[[int, str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Download a selected Sentinel-2 Level-2A product from CDSE, stream to a temporary
        .part file, rename upon completion, and extract to the raw-data directory.
        """
        def report(pct: int, msg: str):
            if progress_callback:
                progress_callback(pct, msg)

        def check_abort():
            if is_cancelled and is_cancelled():
                raise RuntimeError("Download task cancelled by user.")

        report(5, f"Initiating retrieval for {product_name}...")
        check_abort()

        # Determine year subfolder
        year_str = date[:4] if len(date) >= 4 else (product_name.split("_")[2][:4] if "_" in product_name else "downloaded")
        target_dir = RAW_DIR / year_str
        target_dir.mkdir(parents=True, exist_ok=True)

        safe_dir = target_dir / (product_name if product_name.endswith(".SAFE") else f"{product_name}.SAFE")
        zip_path = target_dir / f"{product_name}.zip"
        part_path = target_dir / f"{product_name}.zip.part"

        # Check if already extracted locally
        if safe_dir.is_dir():
            # Check for required bands
            required_bands = ["B02_10m", "B03_10m", "B04_10m", "B08_10m", "B11_20m"]
            found = []
            for b in required_bands:
                matches = list(safe_dir.rglob(f"*_{b}.jp2"))
                if matches:
                    found.append(b)

            if len(found) >= len(required_bands):
                msg = f"Scene '{product_name}' already exists locally in {safe_dir.name}."
                logger.info(msg)
                report(100, msg)
                return {
                    "product_id": product_id,
                    "product_name": product_name,
                    "year": year_str,
                    "safe_path": str(safe_dir),
                    "already_existed": True,
                    "bands_found": found
                }

        # Authenticate with CDSE
        report(10, "Authenticating with Copernicus Data Space Ecosystem...")
        token = self.get_auth_token()
        check_abort()

        download_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
        headers = {"Authorization": f"Bearer {token}"}

        report(15, "Requesting product stream from CDSE...")
        logger.info(f"Requesting download from: {download_url}")

        try:
            # allow_redirects=True is critical as CDSE redirects to download storage node
            response = requests.get(
                download_url,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=90
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error initiating product download: {e}")

        if response.status_code == 401 or response.status_code == 403:
            raise PermissionError("CDSE authentication failed or session expired during download request.")
        elif response.status_code == 404:
            raise FileNotFoundError(f"Product '{product_id}' not found in CDSE catalog.")
        elif response.status_code != 200:
            raise RuntimeError(f"CDSE server returned HTTP {response.status_code} during download.")

        total_bytes = int(response.headers.get("content-length", 0))
        total_mb = total_bytes / (1024 * 1024) if total_bytes > 0 else 0
        downloaded_bytes = 0

        logger.info(f"Starting stream download to {part_path.name} (Total: {total_mb:.1f} MB)...")
        report(20, f"Downloading product archive ({total_mb:.1f} MB)...")

        chunk_size = 1024 * 1024  # 1 MB chunk

        try:
            with open(part_path, "wb") as f:
                last_report_time = time.time()
                for chunk in response.iter_content(chunk_size=chunk_size):
                    check_abort()
                    if chunk:
                        f.write(chunk)
                        downloaded_bytes += len(chunk)

                        # Update progress every ~1 second
                        current_time = time.time()
                        if current_time - last_report_time >= 1.0:
                            if total_bytes > 0:
                                # Scale progress from 20% to 80%
                                pct = 20 + int((downloaded_bytes / total_bytes) * 60)
                                curr_mb = downloaded_bytes / (1024 * 1024)
                                report(pct, f"Downloading: {curr_mb:.1f} MB / {total_mb:.1f} MB ({pct}%)")
                            else:
                                curr_mb = downloaded_bytes / (1024 * 1024)
                                report(50, f"Downloaded {curr_mb:.1f} MB...")
                            last_report_time = current_time

            check_abort()

            # Atomically rename .part to .zip
            if zip_path.exists():
                zip_path.unlink()
            part_path.rename(zip_path)
            report(82, "Download completed. Verifying archive...")

        except Exception:
            # Clean up partial download file upon error or cancellation
            if part_path.exists():
                try:
                    part_path.unlink()
                except Exception:
                    pass
            raise

        # Extract archive
        report(85, "Extracting Sentinel-2 .SAFE archive...")
        check_abort()

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(target_dir)

            report(95, "Extraction completed. Scanning bands...")

            # Clean up zip file to save disk space
            if zip_path.exists():
                try:
                    zip_path.unlink()
                    logger.info(f"Cleaned up temporary archive {zip_path.name}")
                except Exception as e:
                    logger.warning(f"Could not remove temporary zip: {e}")

            # Verify required bands
            required_bands = ["B02_10m", "B03_10m", "B04_10m", "B08_10m", "B11_20m"]
            found_bands = []
            if safe_dir.is_dir():
                for b in required_bands:
                    matches = list(safe_dir.rglob(f"*_{b}.jp2"))
                    if matches:
                        found_bands.append(b)

            msg = f"Scene {product_name} downloaded and extracted successfully."
            logger.info(msg)
            report(100, msg)

            return {
                "product_id": product_id,
                "product_name": product_name,
                "year": year_str,
                "safe_path": str(safe_dir),
                "already_existed": False,
                "bands_found": found_bands
            }

        except Exception as e:
            logger.error(f"Failed to extract product archive: {e}")
            raise RuntimeError(f"Product archive extraction failed: {e}")


# Global singleton instance
satellite_service = SatelliteService()
