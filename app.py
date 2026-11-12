
from flask import Flask, render_template, request, send_from_directory, send_file, jsonify

import os
import pandas as pd
from src.job_manager import job_manager
from src.services.satellite_service import satellite_service
from src.services.lulc_service import lulc_service
from src.services.change_service import change_service
from src.services.crop_service import crop_service

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
MAPS_DIR = os.path.join(BASE_DIR, "outputs", "maps")
CLASSIFIED_DIR = os.path.join(
    BASE_DIR, "data", "classified", "mangaluru"
)


# =========================================================
# Helper
# =========================================================

def load_csv(relative_path):
    path = os.path.join(BASE_DIR, relative_path)

    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


# =========================================================
# Home
# =========================================================

@app.route("/")
def index():
    selected_year = request.args.get("year", default=2022, type=int)

    if selected_year not in [2018, 2022, 2025]:
        selected_year = 2022

    area_df = load_csv("data/results/land_use_area_statistics.csv")

    statistics = []

    if not area_df.empty:
        area_df["Year"] = pd.to_numeric(
            area_df["Year"], errors="coerce"
        )

        year_df = area_df[area_df["Year"] == selected_year]

        statistics = year_df.to_dict(orient="records")

    return render_template(
        "home.html",
        selected_year=selected_year,
        statistics=statistics
    )


# =========================================================
# Land Use Explorer
# =========================================================

@app.route("/explorer")
def explorer():
    return render_template("explorer.html")


# =========================================================
# Change Analysis
# =========================================================

@app.route("/change-analysis")
def change_analysis():
    periods = {
        "2018-2022": {
            "area": "area_change_2018_2022.csv",
            "transition": "transition_details_2018_2022.csv",
            "intensity": "change_intensity/change_intensity_statistics_2018_2022.csv",
            "hotspot": "hotspots/change_hotspot_statistics_2018_2022.csv",
            "reliability": "reliability/overall_reliability_2018_2022.csv"
        },
        "2022-2025": {
            "area": "area_change_2022_2025.csv",
            "transition": "transition_details_2022_2025.csv",
            "intensity": "change_intensity/change_intensity_statistics_2022_2025.csv",
            "hotspot": "hotspots/change_hotspot_statistics_2022_2025.csv",
            "reliability": "reliability/overall_reliability_2022_2025.csv"
        },
        "2018-2025": {
            "area": "area_change_2018_2025.csv",
            "transition": "transition_details_2018_2025.csv",
            "intensity": "change_intensity/multi_temporal_change_intensity_summary.csv",
            "hotspot": "hotspots/change_hotspot_statistics_2018_2025.csv",
            "reliability": "reliability/overall_reliability_2018_2025.csv"
        }
    }

    # Discover dynamically available periods from classified rasters
    dynamic_periods = change_service.get_available_periods()
    all_period_keys = list(dict.fromkeys(list(periods.keys()) + dynamic_periods))

    selected = request.args.get("period", "2018-2025")
    if selected not in all_period_keys and all_period_keys:
        selected = all_period_keys[0]

    if selected in periods:
        selected_files = periods[selected]
        area = load_csv(os.path.join("data", "results", selected_files["area"]))
        transition = load_csv(os.path.join("data", "results", selected_files["transition"]))
        intensity = load_csv(os.path.join("data", "results", selected_files["intensity"]))
        hotspot = load_csv(os.path.join("data", "results", selected_files["hotspot"]))
        reliability = load_csv(os.path.join("data", "results", selected_files["reliability"]))
    else:
        # Dynamic comparison between any two classified years
        if "-" in selected:
            y1, y2 = selected.split("-", 1)
            area_csv = os.path.join("data", "results", f"area_change_{y1}_{y2}.csv")
            transition_csv = os.path.join("data", "results", f"transition_details_{y1}_{y2}.csv")
            if not os.path.exists(os.path.join(BASE_DIR, area_csv)):
                try:
                    change_service.run_change_detection(y1, y2)
                except Exception as e:
                    print(f"Error computing dynamic change detection: {e}")

            area = load_csv(area_csv)
            transition = load_csv(transition_csv)
        else:
            area = pd.DataFrame()
            transition = pd.DataFrame()

        intensity = pd.DataFrame()
        hotspot = pd.DataFrame()
        reliability = pd.DataFrame()

    return render_template(
        "change_analysis.html",
        periods=all_period_keys,
        selected_period=selected,
        area=area.to_dict("records"),
        transition=transition.to_dict("records"),
        intensity=intensity.to_dict("records"),
        hotspot=hotspot.to_dict("records"),
        reliability=reliability.to_dict("records")
    )


# =========================================================
# Statistics
# =========================================================

@app.route("/statistics")
def statistics():

    area = load_csv(
        os.path.join(
            "data",
            "results",
            "land_use_area_statistics.csv"
        )
    )

    confidence = load_csv(
        os.path.join(
            "data",
            "results",
            "confidence",
            "confidence_by_class.csv"
        )
    )

    trajectory = load_csv(
        os.path.join(
            "data",
            "results",
            "trajectories",
            "trajectory_type_summary.csv"
        )
    )
    trajectory_sequences = load_csv(
    os.path.join(
        "data",
        "results",
        "trajectories",
        "trajectory_sequence_summary.csv"
    )
)

    builtup = load_csv(
        os.path.join(
            "data",
            "results",
            "builtup_analysis",
            "builtup_area_statistics.csv"
        )
    )

    agriculture = load_csv(
        os.path.join(
            "data",
            "results",
            "agriculture_analysis",
            "agriculture_fragmentation_statistics.csv"
        )
    )

    return render_template(
        "statistics.html",
        area=area.to_dict("records"),
        confidence=confidence.to_dict("records"),
        trajectory=trajectory.to_dict("records"),
        trajectory_sequences=trajectory_sequences.to_dict("records"),
        builtup=builtup.to_dict("records"),
        agriculture=agriculture.to_dict("records")
    )


# =========================================================
# ML Model
# =========================================================

@app.route("/ml-model")
def ml_model():

    model_report_path = os.path.join(
        RESULTS_DIR,
        "model_evaluation",
        "lulc_model_evaluation.txt"
    )

    report = ""

    if os.path.exists(model_report_path):
        with open(
            model_report_path,
            "r",
            encoding="utf-8"
        ) as file:
            report = file.read()

    feature_importance = load_csv(
        os.path.join(
            "data",
            "results",
            "model_evaluation",
            "feature_importance.csv"
        )
    )

    confusion_matrix = load_csv(
        os.path.join(
            "data",
            "results",
            "model_evaluation",
            "confusion_matrix.csv"
        )
    )

    return render_template(
        "ml_model.html",
        report=report,
        feature_importance=feature_importance.to_dict("records"),
        confusion_matrix=confusion_matrix.to_dict("records")
    )


# =========================================================
# Land-use overlay
# =========================================================

@app.route("/overlay/<int:year>")
def land_use_overlay(year):

    filename = f"land_use_overlay_{year}.png"

    filepath = os.path.join(
        MAPS_DIR,
        filename
    )

    if not os.path.exists(filepath):
        return f"Overlay for {year} not found", 404

    return send_from_directory(
        MAPS_DIR,
        filename
    )

@app.route("/confidence-overlay/<int:year>")
def confidence_overlay(year):
    if year not in [2018, 2022, 2025]:
        return "Unsupported year", 404

    filename = f"confidence_overlay_{year}.png"
    filepath = os.path.join(MAPS_DIR, filename)

    if not os.path.exists(filepath):
        return f"Confidence overlay for {year} not found", 404

    return send_from_directory(MAPS_DIR, filename)


# =========================================================
# Classified raster
# =========================================================

@app.route("/classified/<int:year>")
def classified_raster(year):

    filename = f"land_use_{year}.tif"

    filepath = os.path.join(
        CLASSIFIED_DIR,
        filename
    )

    if not os.path.exists(filepath):
        return "Classification raster not found", 404

    return send_file(
        filepath,
        mimetype="image/tiff"
    )


# =========================================================
# Classified map PNG
# =========================================================

@app.route("/map/<int:year>")
def map_image(year):

    filename = f"land_use_map_{year}.png"

    filepath = os.path.join(
        MAPS_DIR,
        filename
    )

    if not os.path.exists(filepath):
        return f"Land-use map for {year} not found", 404

    return send_from_directory(
        MAPS_DIR,
        filename
    )


# =========================================================
# Change map
# =========================================================

@app.route("/change-map/<period>")
def change_map(period):
    allowed = {
        "2018-2022": "change_map_2018_2022.png",
        "2022-2025": "change_map_2022_2025.png",
        "2018-2025": "change_map_2018_2025.png"
    }

    filename = allowed.get(period, f"change_map_{period}.png")
    filepath = os.path.join(MAPS_DIR, filename)

    if not os.path.exists(filepath):
        # Attempt dynamic creation if period matches year1-year2
        if "-" in period:
            y1, y2 = period.split("-", 1)
            try:
                change_service.run_change_detection(y1, y2)
            except Exception:
                pass

    if not os.path.exists(filepath):
        return f"Change map for {period} not found", 404

    return send_from_directory(
        MAPS_DIR,
        filename
    )



# =========================================================
# Experimental Crop Analysis
# =========================================================

@app.route("/crop-analysis")
def crop_analysis():
    available_years = crop_service.list_available_crop_years()
    selected_year = request.args.get("year", default="2022")
    if selected_year not in available_years and available_years:
        selected_year = available_years[0]

    stats_file = os.path.join(
        "data", "results", "crop_reference",
        f"crop_area_statistics_{selected_year}_experimental.csv"
    )
    if not os.path.exists(os.path.join(BASE_DIR, stats_file)) and selected_year == "2022":
        stats_file = os.path.join(
            "data", "results", "crop_reference",
            "crop_area_statistics_2022_experimental.csv"
        )

    crop_stats = load_csv(stats_file)

    return render_template(
        "crop_analysis.html",
        crop_stats=crop_stats.to_dict("records"),
        available_years=available_years,
        selected_year=selected_year
    )


# =========================================================
# Satellite Retrieval (Phase 2)
# =========================================================

@app.route("/satellite-retrieval")
def satellite_retrieval():
    """Render the Copernicus Data Space satellite scene retrieval interface."""
    creds_status = satellite_service.get_credentials_status()
    return render_template(
        "satellite_retrieval.html",
        credentials_status=creds_status
    )



# =========================================================
# Experimental Crop Map Images
# =========================================================

@app.route("/crop-map-image")
def crop_map_image():
    year = request.args.get("year", default="2022")
    filename = f"crop_{year}_visual.png"
    filepath = os.path.join(BASE_DIR, "outputs", "charts", filename)
    if not os.path.exists(filepath):
        # Fallback to default 2022 visual
        filename = "crop_2022_visual.png"
    return send_from_directory(
        os.path.join(BASE_DIR, "outputs", "charts"),
        filename
    )


@app.route("/crop-confidence-image")
def crop_confidence_image():
    return send_from_directory(
        os.path.join("outputs", "maps"),
        "crop_confidence_2022_experimental.png"
    )


# =========================================================
# Start application
# =========================================================
@app.route("/chart/<path:filename>")
def chart_image(filename):
    allowed_charts = {
        "area_comparison_2018_2022.png",
        "area_comparison_2018_2025.png",
        "area_comparison_2022_2025.png",
        "crop_2022_visual.png",
        "crop_confidence_2022.png",
        "land_use_class_distribution.png",
        "land_use_trend_2018_2025.png",
        "net_change_2018_2025.png",
        "transition_matrix_2018_2022.png",
        "transition_matrix_2018_2025.png",
        "transition_matrix_2022_2025.png",
    }

    if filename not in allowed_charts:
        return "Chart not found", 404

    return send_from_directory(
        os.path.join(BASE_DIR, "outputs", "charts"),
        filename
    )


# =========================================================
# Job Management APIs (Asynchronous Tasks)
# =========================================================

@app.route("/api/health")
def api_health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Intelligent Land Use Change Analysis",
        "timestamp": pd.Timestamp.now().isoformat()
    })


@app.route("/api/jobs")
def api_list_jobs():
    """List recent background jobs with optional filtering."""
    job_type = request.args.get("type")
    status = request.args.get("status")
    limit = request.args.get("limit", default=50, type=int)

    jobs = job_manager.list_jobs(job_type=job_type, status=status, limit=limit)
    return jsonify({
        "success": True,
        "count": len(jobs),
        "jobs": jobs
    })


@app.route("/api/jobs/<job_id>")
def api_get_job(job_id):
    """Retrieve detailed progress and status for a specific job."""
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({
            "success": False,
            "error": f"Job '{job_id}' not found."
        }), 404

    return jsonify({
        "success": True,
        "job": job
    })


@app.route("/api/jobs/<job_id>/cancel", methods=["POST"])
def api_cancel_job(job_id):
    """Request cancellation of an active or pending background job."""
    job = job_manager.get_job_object(job_id)
    if not job:
        return jsonify({
            "success": False,
            "error": f"Job '{job_id}' not found."
        }), 404

    if job.status in ["completed", "failed"]:
        return jsonify({
            "success": False,
            "error": f"Cannot cancel job '{job_id}' because it has already {job.status}."
        }), 400

    success = job_manager.cancel_job(job_id)
    return jsonify({
        "success": success,
        "message": f"Cancellation requested for job '{job_id}'."
    })


# =========================================================
# Satellite APIs (Phase 2)
# =========================================================

@app.route("/api/satellite/credentials-status")
def api_satellite_credentials_status():
    """Return status of Copernicus Data Space credentials without exposing secrets."""
    return jsonify({
        "success": True,
        "credentials": satellite_service.get_credentials_status()
    })


@app.route("/api/satellite/search", methods=["POST"])
def api_satellite_search():
    """Search for Sentinel-2 Level-2A scenes intersecting Mangaluru Taluk."""
    data = request.get_json(silent=True) or {}

    start_date = data.get("start_date")
    end_date = data.get("end_date")
    max_cloud = data.get("max_cloud", 30.0)
    tile = data.get("tile", "43PDQ")

    if not start_date or not end_date:
        return jsonify({
            "success": False,
            "error": "Both 'start_date' and 'end_date' (YYYY-MM-DD) are required."
        }), 400

    try:
        max_cloud = float(max_cloud)
        if not (0.0 <= max_cloud <= 100.0):
            return jsonify({
                "success": False,
                "error": "max_cloud must be between 0 and 100."
            }), 400
    except (ValueError, TypeError):
        return jsonify({
            "success": False,
            "error": "max_cloud must be a valid number between 0 and 100."
        }), 400

    try:
        scenes = satellite_service.search_scenes(
            start_date=str(start_date).strip(),
            end_date=str(end_date).strip(),
            max_cloud_cover=max_cloud,
            tile=str(tile).strip() if tile else ""
        )
        return jsonify({
            "success": True,
            "count": len(scenes),
            "scenes": scenes
        })
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
    except RuntimeError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 502
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Unexpected error during scene search: {str(e)}"
        }), 500


@app.route("/api/satellite/download", methods=["POST"])
def api_satellite_download():
    """Submit a background job to download and extract a Sentinel-2 scene from CDSE."""
    data = request.get_json(silent=True) or {}

    product_id = data.get("product_id")
    product_name = data.get("product_name")
    date = data.get("date", "")

    if not product_id or not product_name:
        return jsonify({
            "success": False,
            "error": "Both 'product_id' and 'product_name' are required."
        }), 400

    # Check CDSE credentials
    creds_status = satellite_service.get_credentials_status()
    if not creds_status["is_configured"]:
        return jsonify({
            "success": False,
            "error": (
                "Copernicus Data Space Ecosystem (CDSE) credentials are not configured. "
                "Please configure CDSE_USERNAME and CDSE_PASSWORD in your .env file."
            )
        }), 400

    job_id = job_manager.submit_job(
        job_type="satellite_download",
        target_func=satellite_service.download_scene,
        product_id=str(product_id).strip(),
        product_name=str(product_name).strip(),
        date=str(date).strip() if date else "",
        description=f"Downloading Sentinel-2 scene: {product_name}"
    )

    return jsonify({
        "success": True,
        "job_id": job_id,
        "message": f"Download job '{job_id}' submitted for scene '{product_name}'."
    }), 202


# =========================================================
# Phase 3: LULC Classification APIs
# =========================================================

@app.route("/api/lulc/years")
def api_lulc_years():
    """List available classified LULC years."""
    years = change_service.list_available_years()
    return jsonify({
        "success": True,
        "count": len(years),
        "years": years
    })


@app.route("/api/lulc/classify", methods=["POST"])
def api_lulc_classify():
    """Submit background job for automated LULC preprocessing and classification."""
    data = request.get_json(silent=True) or {}
    scene = data.get("scene_path") or data.get("year")
    tag = data.get("tag", "")

    if not scene:
        return jsonify({
            "success": False,
            "error": "A 'scene_path' (e.g. .SAFE folder or raw directory) or 'year' is required."
        }), 400

    job_id = job_manager.submit_job(
        job_type="lulc_classification",
        target_func=lulc_service.run_full_pipeline,
        scene_path_or_year=scene,
        tag=tag,
        description=f"LULC Classification Pipeline for {tag or scene}"
    )

    return jsonify({
        "success": True,
        "job_id": job_id,
        "message": f"LULC classification job submitted for '{tag or scene}'."
    }), 202


# =========================================================
# Phase 4: Dynamic Change Detection APIs
# =========================================================

@app.route("/api/change/periods")
def api_change_periods():
    """List available change detection periods and classified years."""
    periods = change_service.get_available_periods()
    years = change_service.list_available_years()
    return jsonify({
        "success": True,
        "periods": periods,
        "years": years
    })


@app.route("/api/change/analyze", methods=["POST"])
def api_change_analyze():
    """Perform dynamic change detection between two years."""
    data = request.get_json(silent=True) or {}
    year1 = data.get("year1")
    year2 = data.get("year2")
    run_async = data.get("async", False)

    if not year1 or not year2:
        return jsonify({
            "success": False,
            "error": "Both 'year1' and 'year2' are required."
        }), 400

    if str(year1).strip() == str(year2).strip():
        return jsonify({
            "success": False,
            "error": "Cannot compare a year with itself."
        }), 400

    if run_async:
        job_id = job_manager.submit_job(
            job_type="change_detection",
            target_func=change_service.run_change_detection,
            year1=year1,
            year2=year2,
            description=f"Change Detection: {year1} -> {year2}"
        )
        return jsonify({
            "success": True,
            "job_id": job_id,
            "message": f"Change detection job submitted for {year1} -> {year2}."
        }), 202

    try:
        results = change_service.run_change_detection(year1=year1, year2=year2)
        return jsonify({
            "success": True,
            "results": results
        })
    except (ValueError, FileNotFoundError) as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error running change detection: {str(e)}"
        }), 500


# =========================================================
# Phase 5: Dynamic Crop Analysis APIs
# =========================================================

@app.route("/api/crop/years")
def api_crop_years():
    """List years available for crop analysis."""
    years = crop_service.list_available_crop_years()
    return jsonify({
        "success": True,
        "count": len(years),
        "years": years
    })


@app.route("/api/crop/classify", methods=["POST"])
def api_crop_classify():
    """Submit background job to classify crops for an agricultural year."""
    data = request.get_json(silent=True) or {}
    year = data.get("year")

    if not year:
        return jsonify({
            "success": False,
            "error": "'year' parameter is required."
        }), 400

    job_id = job_manager.submit_job(
        job_type="crop_classification",
        target_func=crop_service.classify_crops,
        year=year,
        description=f"Crop Classification for {year}"
    )

    return jsonify({
        "success": True,
        "job_id": job_id,
        "message": f"Crop classification job submitted for year {year}."
    }), 202


@app.route("/api/crop/compare")
def api_crop_compare():
    """Compare crop statistics between two years."""
    year1 = request.args.get("year1")
    year2 = request.args.get("year2")

    if not year1 or not year2:
        return jsonify({
            "success": False,
            "error": "Both 'year1' and 'year2' query parameters are required."
        }), 400

    try:
        comparison = crop_service.compare_crop_years(year1, year2)
        return jsonify({
            "success": True,
            "comparison": comparison
        })
    except FileNotFoundError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 404
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
