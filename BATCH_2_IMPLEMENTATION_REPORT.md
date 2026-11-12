# Batch 2 Implementation Report: Phase 6 & Phase 7

**Project:** Intelligent Land Use Change Analysis Using Satellite Data and Machine Learning  
**Study Area:** Mangaluru Taluk, Karnataka, India (`EPSG:32643`, Sentinel-2 Tile `43PDQ`)  
**Scope:** Phase 6 (Confidence, Reliability & Advanced Spatial Analytics) and Phase 7 (Dashboard Integration, Visualizations & Result Exports)

---

## 1. Executive Summary

Batch 2 completes the analytical, spatial, and user-facing integration milestones of the project:
1. **Phase 6:** Engineered an automated, high-performance spatial analytics service (`src/services/analytics_service.py`) integrating:
   - Pixel-level Random Forest prediction probability and reliability index estimation.
   - Categorical confidence levels: **Low** ($< 0.60$), **Medium** ($0.60 \le P < 0.80$), and **High** ($\ge 0.80$).
   - Class-wise confidence distributions across all 5 LULC categories.
   - Relative spatial change-density hotspot detection via Gaussian spatial filtering ($\sigma = 15$ pixels) normalized to 0–100 and classified into 4 tiers.
   - 3-Year multi-temporal land use trajectory analysis categorized into *Stable*, *Temporary Change*, *Persistent Change*, *Recent Change*, and *Progressive Change*.
   - Landscape ecology patch fragmentation metrics (8-connectivity patch labeling, count, largest/mean/median patch area, patch density per $100\text{ km}^2$) for agriculture and built-up land.
   - Multi-temporal transition reliability assessment based on minimum joint confidence $\min(\text{conf}_{t1}, \text{conf}_{t2})$.
2. **Phase 7:** Unified all system modules across the Flask web interface:
   - Added asynchronous job triggers and API endpoints for analytics.
   - Integrated hotspot and confidence overlays directly into the Leaflet web map (`/explorer`).
   - Added complete result export and direct download endpoints (`/export/raster/...`, `/export/csv/...`) across the application.
   - Formatted and linked statistics cards with download capabilities on `templates/home.html`, `templates/change_analysis.html`, `templates/crop_analysis.html`, and `templates/statistics.html`.
   - Verified that all 80 automated unit and integration tests pass with 100% success.

---

## 2. Phase 6: Analytical Methods & Implementations

### 2.1 Prediction Confidence Estimation
- **Probability Inference:** For each valid study-area pixel, the Random Forest model calculates class probability vectors $P(c \mid \mathbf{x})$. Pixel confidence is assigned as the probability of the predicted class:
  $$\text{Confidence}(p) = P(C_{\text{pred}} \mid \mathbf{x})$$
- **Confidence Bins:**
  - **High:** $\ge 0.80$ (Color: `[35, 160, 75, 190]`, Green)
  - **Medium / Moderate:** $0.60 \le P < 0.80$ (Color: `[255, 205, 40, 190]`, Yellow)
  - **Low:** $< 0.60$ (Color: `[220, 55, 55, 190]`, Red)
- **Artifacts Saved:**
  - GeoTIFF continuous: `data/results/confidence/lulc_confidence_<year>.tif`
  - GeoTIFF categorical: `data/results/confidence/lulc_confidence_class_<year>.tif`
  - Leaflet RGBA overlay: `outputs/maps/confidence_overlay_<year>.png`
  - Summary CSVs: `data/results/confidence/confidence_statistics_<year>.csv` and `confidence_by_class.csv`

### 2.2 Change Hotspot Detection
- **Spatial Filtering:** To identify geographic concentrations of change while accounting for boundary support, a 2D Gaussian filter ($\sigma = 15$) is applied:
  $$\text{Density}(x, y) = \frac{\mathcal{G}_{\sigma} * \mathbb{I}(\text{Changed})}{\mathcal{G}_{\sigma} * \mathbb{I}(\text{Valid})}$$
- **Normalization & Tiers:**
  $$\text{Score} = \frac{\text{Density} - \text{Density}_{\min}}{\text{Density}_{\max} - \text{Density}_{\min}} \times 100$$
  - **Tier 1 (Low):** $\text{Score} < 25$ (`[65, 182, 196, 170]`)
  - **Tier 2 (Moderate):** $25 \le \text{Score} < 50$ (`[254, 204, 92, 180]`)
  - **Tier 3 (High):** $50 \le \text{Score} < 75$ (`[253, 141, 60, 190]`)
  - **Tier 4 (Very High):** $\text{Score} \ge 75$ (`[227, 26, 28, 200]`)
- **Artifacts Saved:**
  - GeoTIFF score: `data/results/hotspots/change_hotspot_score_<y1>_<y2>.tif`
  - GeoTIFF class: `data/results/hotspots/change_hotspot_class_<y1>_<y2>.tif`
  - Leaflet RGBA overlay: `outputs/maps/hotspot_overlay_<y1>_<y2>.png`
  - Statistics CSV: `data/results/hotspots/change_hotspot_statistics_<y1>_<y2>.csv`

### 2.3 Multi-Temporal Trajectory Analysis
- Evaluates 3-year classification sequences $(C_1, C_2, C_3)$ across common valid pixels:
  - **Stable:** $C_1 = C_2 = C_3$
  - **Temporary Change:** $C_1 \ne C_2 \land C_2 \ne C_3 \land C_1 = C_3$
  - **Persistent Change:** $C_1 \ne C_2 \land C_2 = C_3$
  - **Recent Change:** $C_1 = C_2 \land C_2 \ne C_3$
  - **Progressive Change:** $C_1 \ne C_2 \land C_2 \ne C_3 \land C_1 \ne C_3$
- Generates type summary, sequence summary, and stable class breakdown CSVs in `data/results/trajectories/`.

### 2.4 Landscape Patch Fragmentation
- Connective component labeling (`scipy.ndimage.label`) with 8-neighbor structuring element:
  - Number of discrete patches ($N_p$).
  - Largest patch area ($A_{\max}$ in $\text{km}^2$) and percentage of total class area.
  - Mean and median patch sizes ($\text{km}^2$).
  - Patch Density: $\text{PD} = \frac{N_p}{A_{\text{study}}} \times 100 \text{ patches / } 100\text{ km}^2$.
- Updates `data/results/agriculture_analysis/agriculture_fragmentation_statistics.csv` and `data/results/builtup_analysis/builtup_fragmentation_statistics.csv`.

### 2.5 Transition Reliability
- Minimum joint confidence $\min(\text{conf}_{y1}, \text{conf}_{y2})$ categorized into High ($\ge 0.80$), Moderate ($0.60 \le P < 0.80$), and Low ($< 0.60$).
- Output CSVs: `data/results/reliability/overall_reliability_<y1>_<y2>.csv` and `reliability_summary_<y1>_<y2>.csv`.

---

## 3. Phase 7: Web Application & Dashboard Integration

### 3.1 REST API & Asynchronous Job Endpoints (`app.py`)
| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/analytics/confidence` | `POST` | Asynchronous job submission for LULC confidence calculation. |
| `/api/analytics/hotspots` | `POST` | Asynchronous job submission for spatial hotspot detection. |
| `/api/analytics/trajectories` | `POST` | Asynchronous job submission for 3-year trajectory analysis. |
| `/api/analytics/fragmentation`| `POST` | Asynchronous job submission for patch fragmentation metrics. |
| `/api/analytics/reliability`  | `POST` | Asynchronous job submission for transition reliability assessment. |
| `/api/analytics/stats`        | `GET`  | Query precomputed analytics statistics. |

### 3.2 Result Exports and Downloads
| Endpoint | Description |
| :--- | :--- |
| `/export/raster/classified/<year>` | Download full-resolution classified LULC GeoTIFF. |
| `/export/raster/change/<period>` | Download 2-year change detection GeoTIFF. |
| `/export/raster/crop/<year>` | Download experimental crop classification GeoTIFF. |
| `/export/raster/confidence/<year>` | Download continuous prediction confidence GeoTIFF. |
| `/export/raster/hotspot/<period>` | Download change hotspot class GeoTIFF. |
| `/export/csv/<category>/<filename>`| Whitelisted download for results, confidence, hotspot, trajectory, and fragmentation CSVs. |

### 3.3 Interactive Leaflet Map Enhancements (`templates/explorer.html`)
- Added Hotspot overlay layers: *Change Hotspots 2018–2022*, *2022–2025*, and *2018–2025*.
- Extended dynamic legend to automatically switch between **Land Use Classes**, **Model Confidence**, and **Change Hotspots** depending on active map layers.
- Integrated opacity slider controlling all active raster overlays.

---

## 4. Verification and Automated Test Results

Executed complete regression and feature test suite:
```powershell
python -m pytest tests -v
```

### Summary:
- **`tests/test_phase1.py`:** 16 passed
- **`tests/test_phase2.py`:** 18 passed
- **`tests/test_batch1.py`:** 23 passed
- **`tests/test_batch2.py`:** 23 passed
- **Total Tests:** **80 passed** in 3.51s (100% passing, 0 failures, 0 skipped).
