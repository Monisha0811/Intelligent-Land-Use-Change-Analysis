
from flask import Flask, render_template, request, send_from_directory, send_file, jsonify

import os
import pandas as pd
from src.job_manager import job_manager

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

    selected = request.args.get("period", "2018-2025")
    if selected not in periods:
        selected = "2018-2025"

    selected_files = periods[selected]

    area = load_csv(os.path.join("data", "results", selected_files["area"]))
    transition = load_csv(os.path.join("data", "results", selected_files["transition"]))
    intensity = load_csv(os.path.join("data", "results", selected_files["intensity"]))
    hotspot = load_csv(os.path.join("data", "results", selected_files["hotspot"]))
    reliability = load_csv(os.path.join("data", "results", selected_files["reliability"]))

    return render_template(
        "change_analysis.html",
        periods=periods.keys(),
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

    filename = allowed.get(period)

    if not filename:
        return "Invalid change-analysis period", 404

    filepath = os.path.join(
        MAPS_DIR,
        filename
    )

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
    crop_stats = load_csv(
        os.path.join(
            "data", "results", "crop_reference",
            "crop_area_statistics_2022_experimental.csv"
        )
    )

    return render_template(
        "crop_analysis.html",
        crop_stats=crop_stats.to_dict("records")
    )



# =========================================================
# Experimental Crop Map Images
# =========================================================

@app.route("/crop-map-image")
def crop_map_image():
    return send_from_directory(
        os.path.join("outputs", "charts"),
        "crop_2022_visual.png"
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


if __name__ == "__main__":
    app.run(debug=True)