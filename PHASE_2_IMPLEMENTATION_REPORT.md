# PHASE 2 IMPLEMENTATION REPORT: COPERNICUS DATA SPACE SATELLITE RETRIEVAL

**Project Title:** Intelligent Land Use Change Analysis Using Satellite Data and Machine Learning  
**Phase Completed:** Phase 2 – Copernicus Data Space Satellite Retrieval  
**Execution Date:** 2026-11-12  
**Status:** Successfully Implemented, Tested, and Verified (34/34 Tests Passing)  

---

## 1. Executive Summary

Phase 2 introduces automated retrieval of Sentinel-2 Level-2A surface reflectance imagery directly from the official Copernicus Data Space Ecosystem (CDSE). This module replaces manual satellite data collection workflows with an interactive web-based interface and resilient background processing engine.

All functionality has been integrated into the existing Flask application without modifying or replacing existing trained machine learning models, pre-computed raster datasets, or prior application routes.

---

## 2. Implemented Architecture & Components

```
Copernicus Data Space Ecosystem (CDSE)
   │
   ├─ OAuth2 Authentication (Keycloak: https://identity.dataspace.copernicus.eu)
   │     └─ Token caching with automated refresh (expires_in buffer: 30s)
   │
   ├─ OData v1 Catalogue API (https://catalogue.dataspace.copernicus.eu/odata/v1/Products)
   │     ├─ Spatial Filter: Intersects(area=geography'SRID=4326;POLYGON(...)')
   │     ├─ Product Filter: S2MSI2A (Bottom-of-Atmosphere Level-2A surface reflectance)
   │     ├─ Cloud Filter: cloudCover <= max_cloud_cover
   │     ├─ Temporal Filter: ContentDate/Start between start_date and end_date
   │     └─ MGRS Tile Filter: T43PDQ (Mangaluru Taluk)
   │
   └─ OData v1 Download Stream (https://catalogue.dataspace.copernicus.eu/odata/v1/Products({id})/$value)
         ├─ Follows redirects to CDSE S3 object storage nodes
         ├─ Streams into temporary .part files with real-time byte counters
         ├─ Cooperative cancellation checks on every chunk
         ├─ Atomic rename to .zip and extraction into data/raw/<year>/
         ├─ Band verification: B02 (10m), B03 (10m), B04 (10m), B08 (10m), B11 (20m)
         └─ Safe cleanup of temporary zip archives and partial downloads
```

---

## 3. Files Created and Modified

| File | Type | Purpose |
|------|------|---------|
| `src/services/__init__.py` | Created | Exposes `satellite_service` singleton and `SatelliteService` class. |
| `src/services/satellite_service.py` | Created | Encapsulates CDSE OAuth2 auth, study area geometry parsing, OData catalogue querying, exact Mangaluru coverage calculation, and streaming download/extraction. |
| `templates/satellite_retrieval.html` | Created | Interactive web UI for date/cloud filtering, scene search results display, "Select Best Scene" action, and real-time job progress monitoring. |
| `templates/home.html` | Modified | Added "Satellite Retrieval" navigation item. |
| `templates/explorer.html` | Modified | Added "Satellite Retrieval" and "Crop Analysis" navigation items. |
| `templates/change_analysis.html` | Modified | Added "Satellite Retrieval" and "Crop Analysis" navigation items. |
| `templates/statistics.html` | Modified | Added "Satellite Retrieval" and "Crop Analysis" navigation items. |
| `templates/ml_model.html` | Modified | Added "Satellite Retrieval" and "Crop Analysis" navigation items. |
| `app.py` | Modified | Added `GET /satellite-retrieval`, `GET /api/satellite/credentials-status`, `POST /api/satellite/search`, and `POST /api/satellite/download` endpoints. |
| `tests/test_phase2.py` | Created | Comprehensive unit and integration test suite (18 tests) with mocked network calls. |

---

## 4. API Endpoints Introduced

### 1. `GET /satellite-retrieval`
- **Description:** Renders the user interface for satellite data retrieval.
- **Parameters:** None.
- **Context:** Passes `credentials_status` so the frontend can display whether CDSE credentials are configured in `.env`.

### 2. `GET /api/satellite/credentials-status`
- **Description:** Returns the configuration status of Copernicus credentials without exposing secrets.
- **Response Format:**
  ```json
  {
    "success": true,
    "credentials": {
      "is_configured": true,
      "has_username": true,
      "has_client_id": false
    }
  }
  ```

### 3. `POST /api/satellite/search`
- **Description:** Queries the CDSE OData catalogue for Level-2A Sentinel-2 products intersecting Mangaluru Taluk.
- **Request Body:**
  ```json
  {
    "start_date": "2024-01-01",
    "end_date": "2024-03-31",
    "max_cloud": 20.0,
    "tile": "43PDQ"
  }
  ```
- **Response Format:**
  ```json
  {
    "success": true,
    "count": 3,
    "scenes": [
      {
        "id": "1a2b3c4d-...",
        "name": "S2B_MSIL2A_20240115T045959_N0510_R119_T43PDQ_20240115T083000",
        "date": "2024-01-15T04:59:59.024Z",
        "formatted_date": "2024-01-15",
        "cloud_cover": 4.12,
        "coverage_percent": 100.0,
        "size_mb": 842.5,
        "already_downloaded": false,
        "local_path": null
      }
    ]
  }
  ```

### 4. `POST /api/satellite/download`
- **Description:** Submits an asynchronous background job to stream and unpack a selected scene.
- **Request Body:**
  ```json
  {
    "product_id": "1a2b3c4d-...",
    "product_name": "S2B_MSIL2A_20240115T045959_N0510_R119_T43PDQ_20240115T083000",
    "date": "2024-01-15"
  }
  ```
- **Response Status:** HTTP `202 Accepted`
- **Response Body:**
  ```json
  {
    "success": true,
    "job_id": "98e6dded845e",
    "message": "Download job '98e6dded845e' submitted for scene 'S2B_MSIL2A_20240115T045959_N0510_R119_T43PDQ_20240115T083000'."
  }
  ```

---

## 5. Security & Credential Protection

1. **Environment Variables:** Credentials are read from `.env` using `get_cdse_credentials()` from `src/config.py`.
2. **Zero Secret Exposure:** Credentials are never sent to the client, rendered in HTML, written to log files, or included in JSON responses.
3. **Git Protection:** `.env` remains in `.gitignore`. Verified via `git status` that `.env` is uncommitted and untracked.
4. **Mocked Unit Tests:** All automated tests mock external network interactions using `unittest.mock.patch`, allowing the full test suite to execute securely offline without requiring real CDSE credentials.

---

## 6. Verification and Test Results

### Phase 2 Test Suite (`tests/test_phase2.py`)
- **Total Tests:** 18
- **Passed:** 18
- **Failed:** 0
- **Execution Time:** 0.167 seconds

**Verified Scenarios:**
1. `test_credentials_status_unconfigured`: Unset credentials report `is_configured=False`.
2. `test_get_auth_token_missing_credentials`: Raises `ValueError` when credentials are missing.
3. `test_get_auth_token_success_and_caching`: Obtains Bearer token and returns cached token on subsequent calls without redundant network requests.
4. `test_get_auth_token_unauthorized`: Raises `PermissionError` on HTTP 401.
5. `test_get_study_area_geometry`: Correctly loads GeoJSON boundary, validates bounding coordinates (74.8–75.1°E, 12.8–13.1°N), and generates WKT polygon.
6. `test_search_scenes_validation`: Rejects malformed dates and inverted date ranges.
7. `test_search_scenes_success`: Parses OData catalogue response, calculates spatial intersection percentage, and filters by tile `43PDQ`.
8. `test_download_scene_already_exists`: Fast-returns with 100% completion when scene and required bands already exist locally.
9. `test_download_scene_streaming_and_extraction`: Streams chunked response to `.part` file, renames to `.zip`, extracts `.SAFE` hierarchy, verifies required bands, and cleans up temporary archive.
10. `test_download_scene_cancellation`: Aborts stream upon user cancellation and deletes partial `.part` file.
11. `test_satellite_retrieval_page`: Confirms `GET /satellite-retrieval` returns HTTP 200 with complete UI markup.
12. `test_api_credentials_status`: Confirms `GET /api/satellite/credentials-status` returns JSON status.
13. `test_api_search_missing_dates`: Confirms `POST /api/satellite/search` returns 400 for missing dates.
14. `test_api_search_invalid_cloud`: Confirms `POST /api/satellite/search` returns 400 for invalid cloud threshold.
15. `test_api_search_success`: Confirms `POST /api/satellite/search` returns 200 with scene list.
16. `test_api_download_missing_params`: Confirms `POST /api/satellite/download` returns 400 when required fields are missing.
17. `test_api_download_unconfigured_credentials`: Confirms `POST /api/satellite/download` returns 400 when CDSE credentials are not set in `.env`.
18. `test_api_download_submits_job`: Confirms `POST /api/satellite/download` submits job to `job_manager` and returns HTTP 202 with valid `job_id`.

### Regression Verification (`tests/test_phase1.py`)
- **Total Tests:** 16
- **Passed:** 16
- **Failed:** 0
- **Execution Time:** 0.472 seconds

**Cumulative Suite Status:** 34 tests passing across Phase 1 and Phase 2.

---

## 7. Next Step: Phase 3 Scope & Preparation

As per instructions, execution is paused for user review and approval before proceeding to Phase 3:
- **Phase 3 Objective:** Automated LULC Preprocessing & Prediction Pipeline.
- **Key Deliverables in Phase 3:**
  - Band alignment and extraction service (`B02`, `B03`, `B04`, `B08`, `B11`).
  - Bilinear resampling for continuous reflectance band `B11` (20m to 10m).
  - Categorical SCL handling with nearest-neighbour interpolation.
  - Spectral index calculation: NDVI, NDWI, NDBI using existing formulas.
  - 8-feature array preparation matching exact trained Random Forest order.
  - Pixel-wise LULC inference producing GeoTIFF raster in `data/classified/mangaluru/`.
  - PNG map rendering for web application display.
