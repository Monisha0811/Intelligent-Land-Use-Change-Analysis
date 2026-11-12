"""
Centralized Configuration and Environment Management for
Intelligent Land Use Change Analysis Project.
"""

import os
from pathlib import Path


# ============================================================
# BASE & DIRECTORY PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_MANGALURU_DIR = PROCESSED_DIR / "mangaluru"
CLASSIFIED_DIR = DATA_DIR / "classified"
CLASSIFIED_MANGALURU_DIR = CLASSIFIED_DIR / "mangaluru"
CHANGE_MAPS_DIR = DATA_DIR / "change_maps"
RESULTS_DIR = DATA_DIR / "results"
TRAINING_DIR = DATA_DIR / "training"
CROP_REFERENCE_DIR = DATA_DIR / "crop_reference"
BOUNDARY_DIR = DATA_DIR / "boundary"

OUTPUTS_DIR = BASE_DIR / "outputs"
MAPS_DIR = OUTPUTS_DIR / "maps"
CHARTS_DIR = OUTPUTS_DIR / "charts"

# Models (verifying both singular 'model' and plural 'models')
MODEL_DIR = BASE_DIR / "model"
MODELS_DIR = BASE_DIR / "models"

LULC_MODEL_PATH = MODEL_DIR / "land_use_random_forest.joblib"
# The existing crop training/classification scripts reference models/crop_random_forest_2022.joblib
CROP_MODEL_PATH = MODELS_DIR / "crop_random_forest_2022.joblib"
# Fallback if in model/
if not CROP_MODEL_PATH.exists() and (MODEL_DIR / "crop_classifier_2022.joblib").exists():
    CROP_MODEL_PATH = MODEL_DIR / "crop_classifier_2022.joblib"

BOUNDARY_GPKG = BOUNDARY_DIR / "mangaluru_taluk.gpkg"
BOUNDARY_GEOJSON = BOUNDARY_DIR / "mangaluru_taluk.geojson"

CONFIDENCE_DIR = RESULTS_DIR / "confidence"
RELIABILITY_DIR = RESULTS_DIR / "reliability"
HOTSPOTS_DIR = RESULTS_DIR / "hotspots"
TRAJECTORIES_DIR = RESULTS_DIR / "trajectories"
AGRICULTURE_ANALYSIS_DIR = RESULTS_DIR / "agriculture_analysis"
BUILTUP_ANALYSIS_DIR = RESULTS_DIR / "builtup_analysis"


# ============================================================
# LOAD .env FILE (Standard library, zero-dependency)
# ============================================================

def load_dotenv(dotenv_path=None):
    """
    Parse a .env file and set environment variables.
    Does not overwrite existing environment variables.
    """
    if dotenv_path is None:
        dotenv_path = BASE_DIR / ".env"
    else:
        dotenv_path = Path(dotenv_path)

    if not dotenv_path.is_file():
        return False

    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    # Strip surrounding quotes if present
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    os.environ.setdefault(key, val)
        return True
    except Exception as e:
        print(f"Warning: Failed to load .env file: {e}")
        return False


# Execute dotenv loading on import
load_dotenv()


# ============================================================
# DOMAIN CONSTANTS
# ============================================================

# Five LULC classes (Strictly preserved)
LULC_CLASSES = {
    1: "Water",
    2: "Vegetation/Forest",
    3: "Agriculture",
    4: "Built-up",
    5: "Barren/Open"
}

# Color palette for maps and Leaflet overlays [R, G, B, Alpha]
LULC_COLORS = {
    1: [30, 144, 255, 180],   # Water: Dodger Blue
    2: [34, 139, 34, 180],    # Vegetation: Forest Green
    3: [255, 215, 0, 180],    # Agriculture: Gold
    4: [220, 60, 60, 180],    # Built-up: Crimson Red
    5: [160, 120, 70, 180],   # Barren/Open: Brown
}

# Crop classes (Strictly preserved from existing model)
CROP_CLASSES = {
    1: "Coconut",
    2: "Arecanut",
    3: "Paddy"
}

CROP_COLORS = {
    1: [39, 129, 74, 200],    # Coconut: Deep Green
    2: [196, 109, 36, 200],   # Arecanut: Amber/Orange
    3: [40, 120, 181, 200],   # Paddy: Blue/Cyan
}

# The eight Random Forest input features in verified training order
RF_FEATURE_NAMES = [
    "blue",
    "green",
    "red",
    "nir",
    "swir",
    "ndvi",
    "ndwi",
    "ndbi"
]

# Reflectance scaling factor for Sentinel-2 Level-2A DN values
REFLECTANCE_SCALE = 10000.0

# 10m Sentinel-2 pixel area in km² (10m x 10m = 100m² = 0.0001 km²)
PIXEL_AREA_KM2 = 0.0001

# Spatial projection for Mangaluru Taluk
TARGET_CRS = "EPSG:32643"
SENTINEL2_TILE = "43PDQ"

# Copernicus Data Space Ecosystem (CDSE) API Endpoints
CDSE_CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_DOWNLOAD_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_cdse_credentials():
    """
    Retrieve CDSE credentials from environment variables securely.
    """
    username = os.environ.get("CDSE_USERNAME", "").strip()
    password = os.environ.get("CDSE_PASSWORD", "").strip()
    client_id = os.environ.get("CDSE_CLIENT_ID", "").strip()
    client_secret = os.environ.get("CDSE_CLIENT_SECRET", "").strip()

    is_configured = bool((username and password) or (client_id and client_secret))

    return {
        "username": username,
        "password": password,
        "client_id": client_id,
        "client_secret": client_secret,
        "is_configured": is_configured
    }


def ensure_project_directories():
    """
    Ensure all operational output directories exist.
    """
    dirs = [
        PROCESSED_MANGALURU_DIR,
        CLASSIFIED_MANGALURU_DIR,
        CHANGE_MAPS_DIR,
        RESULTS_DIR,
        MAPS_DIR,
        CHARTS_DIR,
        RESULTS_DIR / "confidence",
        RESULTS_DIR / "reliability",
        RESULTS_DIR / "hotspots",
        RESULTS_DIR / "trajectories",
        RESULTS_DIR / "agriculture_analysis",
        RESULTS_DIR / "builtup_analysis",
        RESULTS_DIR / "crop_reference"
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# Automatically ensure required directories on startup
ensure_project_directories()
