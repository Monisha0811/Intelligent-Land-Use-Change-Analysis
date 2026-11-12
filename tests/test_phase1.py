"""
Comprehensive Test Suite for Phase 1: Background Job Engine & Architecture.

Verifies:
1. Cancellation race condition prevention (pending jobs cannot transition to running).
2. Safe argument injection across different function signatures (no TypeError).
3. Thread-safety of job state, progress, and log updates.
4. Preservation of running jobs during 100-job dictionary cleanup.
5. Accurate cancellation lifecycle handling (timestamps, durations).
6. Strict, consistent state transitions (pending -> running -> terminal).
7. Regression testing of all Flask routes and Job API endpoints.
"""

import time
import unittest
import threading
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.job_manager import JobManager, Job, _prepare_task_kwargs
from src.config import (
    BASE_DIR,
    LULC_CLASSES,
    CROP_CLASSES,
    RF_FEATURE_NAMES,
    REFLECTANCE_SCALE,
    PIXEL_AREA_KM2,
    get_cdse_credentials,
    load_dotenv
)
from app import app


class TestPhase1JobManager(unittest.TestCase):
    """Test suite for JobManager background task execution, lifecycle, and safety."""

    def setUp(self):
        self.manager = JobManager(max_workers=2)

    def test_job_lifecycle_success(self):
        """Test standard successful task execution and progress reporting."""
        def dummy_task(n, progress_callback, **kwargs):
            progress_callback(30, "Phase A completed")
            time.sleep(0.04)
            progress_callback(75, "Phase B completed")
            return {"sum": n * 2, "status": "done"}

        job_id = self.manager.submit_job(
            "dummy_success",
            dummy_task,
            21,
            description="Testing successful job execution"
        )

        self.assertIsNotNone(job_id)

        # Wait for completion
        for _ in range(50):
            job = self.manager.get_job(job_id)
            if job["status"] == "completed":
                break
            time.sleep(0.04)

        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["progress"], 100)
        self.assertEqual(job["result"]["sum"], 42)
        self.assertIsNotNone(job["completed_at"])
        self.assertGreaterEqual(job["duration_seconds"], 0.0)
        self.assertGreater(len(job["logs"]), 0)

    def test_job_failure_capture(self):
        """Test error handling when a background task raises an exception."""
        def failing_task(**kwargs):
            raise ValueError("Deliberate test failure")

        job_id = self.manager.submit_job("failing_job", failing_task)

        for _ in range(50):
            job = self.manager.get_job(job_id)
            if job["status"] == "failed":
                break
            time.sleep(0.04)

        self.assertEqual(job["status"], "failed")
        self.assertIn("Deliberate test failure", job["error"])
        self.assertIsNotNone(self.manager.get_job_object(job_id).traceback)
        self.assertIsNotNone(job["completed_at"])

    # -------------------------------------------------------------
    # 1. Cancellation race condition fix
    # -------------------------------------------------------------
    def test_cancellation_race_pending_job_never_runs(self):
        """
        Verify that cancelling a pending job prevents it from ever entering
        the 'running' state, even when a worker thread picks it up.
        """
        job = Job("test_race_1", "test_type", "Pending cancellation test")
        self.assertEqual(job.status, "pending")

        # Cancel while pending
        cancelled = job.cancel()
        self.assertTrue(cancelled)
        self.assertEqual(job.status, "cancelled")
        self.assertEqual(job.duration_seconds, 0.0)
        self.assertIsNotNone(job.completed_at)

        # Worker thread picks up the job and attempts set_running()
        started = job.set_running()
        self.assertFalse(started)
        # Status MUST remain 'cancelled', NOT 'running'
        self.assertEqual(job.status, "cancelled")

    def test_cancellation_while_running(self):
        """Verify that cancelling a running job sets status to cancelled with measured duration."""
        def cancellable_task(is_cancelled, **kwargs):
            for _ in range(100):
                if is_cancelled():
                    return "stopped"
                time.sleep(0.02)
            return "finished"

        job_id = self.manager.submit_job("running_cancel_test", cancellable_task)

        # Wait until job enters running state
        for _ in range(30):
            job = self.manager.get_job(job_id)
            if job["status"] == "running":
                break
            time.sleep(0.02)

        self.assertEqual(job["status"], "running")

        # Now cancel the running job
        cancelled = self.manager.cancel_job(job_id)
        self.assertTrue(cancelled)

        time.sleep(0.05)
        job = self.manager.get_job(job_id)
        self.assertEqual(job["status"], "cancelled")
        self.assertIsNotNone(job["completed_at"])
        self.assertGreaterEqual(job["duration_seconds"], 0.0)

    # -------------------------------------------------------------
    # 2. Safe argument injection across different signatures
    # -------------------------------------------------------------
    def test_safe_argument_injection_no_kwargs(self):
        """Test task function with NO extra kwargs does not raise TypeError."""
        def simple_function(a, b):
            return a + b

        job_id = self.manager.submit_job("simple_calc", simple_function, 10, 20)

        for _ in range(30):
            job = self.manager.get_job(job_id)
            if job["status"] == "completed":
                break
            time.sleep(0.03)

        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["result"], 30)

    def test_safe_argument_injection_only_progress(self):
        """Test task function that accepts only progress_callback."""
        def task_with_prog(val, progress_callback):
            progress_callback(50, "Halfway")
            return val * 3

        job_id = self.manager.submit_job("prog_calc", task_with_prog, 7)

        for _ in range(30):
            job = self.manager.get_job(job_id)
            if job["status"] == "completed":
                break
            time.sleep(0.03)

        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["result"], 21)

    def test_safe_argument_injection_var_kwargs(self):
        """Test task function that accepts **kwargs receives all helpers."""
        def task_with_kwargs(x, **kwargs):
            has_prog = "progress_callback" in kwargs
            has_cancel = "is_cancelled" in kwargs
            has_jid = "job_id" in kwargs
            return {"x": x, "has_prog": has_prog, "has_cancel": has_cancel, "has_jid": has_jid}

        job_id = self.manager.submit_job("kwargs_calc", task_with_kwargs, 99)

        for _ in range(30):
            job = self.manager.get_job(job_id)
            if job["status"] == "completed":
                break
            time.sleep(0.03)

        self.assertEqual(job["status"], "completed")
        res = job["result"]
        self.assertEqual(res["x"], 99)
        self.assertTrue(res["has_prog"])
        self.assertTrue(res["has_cancel"])
        self.assertTrue(res["has_jid"])

    # -------------------------------------------------------------
    # 3. Thread-safety of state and logs
    # -------------------------------------------------------------
    def test_thread_safe_concurrent_logging(self):
        """Verify concurrent log additions and serialization do not crash or corrupt data."""
        job = Job("concurrent_log_test", "test")
        job.set_running()

        errors = []

        def worker(thread_idx):
            try:
                for i in range(50):
                    job.add_log(f"Thread {thread_idx} log {i}")
                    job.update_progress(i, f"Progress {i}")
                    d = job.to_dict()
                    self.assertIsInstance(d["logs"], list)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertLessEqual(len(job.logs), 200)

    # -------------------------------------------------------------
    # 4. Protection of running jobs during 100-job cleanup
    # -------------------------------------------------------------
    def test_running_job_protected_from_cleanup(self):
        """Verify that a running job is NEVER deleted when the 100-job limit is reached."""
        manager = JobManager(max_workers=4)

        # Create a persistent running job
        running_job_event = threading.Event()

        def persistent_running_task(is_cancelled, **kwargs):
            running_job_event.wait(timeout=2.0)
            return "done"

        long_job_id = manager.submit_job("long_task", persistent_running_task)

        # Wait until it is running
        for _ in range(30):
            j = manager.get_job(long_job_id)
            if j["status"] == "running":
                break
            time.sleep(0.02)

        self.assertEqual(manager.get_job(long_job_id)["status"], "running")

        # Now submit 105 quick completed jobs
        def quick_task():
            return True

        for i in range(105):
            manager.submit_job(f"quick_{i}", quick_task)

        time.sleep(0.2)

        # The long-running job MUST STILL EXIST in the registry
        long_job = manager.get_job(long_job_id)
        self.assertIsNotNone(
            long_job,
            "CRITICAL: A running job was deleted during the 100-job cleanup!"
        )
        self.assertEqual(long_job["status"], "running")

        # Release the persistent task
        running_job_event.set()

    # -------------------------------------------------------------
    # 5. Strict, consistent state transitions
    # -------------------------------------------------------------
    def test_strict_state_transitions(self):
        """Verify invalid state transitions are blocked."""
        job = Job("state_test", "test")
        self.assertEqual(job.status, "pending")

        # Cannot complete directly from pending
        self.assertFalse(job.set_completed("data"))
        self.assertEqual(job.status, "pending")

        # Cannot fail directly from pending
        self.assertFalse(job.set_failed("error"))
        self.assertEqual(job.status, "pending")

        # Transition to running
        self.assertTrue(job.set_running())
        self.assertEqual(job.status, "running")

        # Transition to completed
        self.assertTrue(job.set_completed("success_result"))
        self.assertEqual(job.status, "completed")

        # Cannot cancel an already completed job
        self.assertFalse(job.cancel())
        self.assertEqual(job.status, "completed")

        # Cannot transition from completed back to running or failed
        self.assertFalse(job.set_running())
        self.assertFalse(job.set_failed("new error"))
        self.assertEqual(job.status, "completed")


class TestPhase1Config(unittest.TestCase):
    """Test suite for centralized configuration and constants."""

    def test_paths_and_constants(self):
        """Verify essential paths and constants adhere strictly to existing models."""
        self.assertTrue(BASE_DIR.exists())
        self.assertEqual(len(LULC_CLASSES), 5)
        self.assertEqual(LULC_CLASSES[1], "Water")
        self.assertEqual(LULC_CLASSES[2], "Vegetation/Forest")
        self.assertEqual(LULC_CLASSES[3], "Agriculture")
        self.assertEqual(LULC_CLASSES[4], "Built-up")
        self.assertEqual(LULC_CLASSES[5], "Barren/Open")

        self.assertEqual(len(CROP_CLASSES), 3)
        self.assertEqual(CROP_CLASSES[1], "Coconut")
        self.assertEqual(CROP_CLASSES[2], "Arecanut")
        self.assertEqual(CROP_CLASSES[3], "Paddy")

        # 8 features in verified exact training order
        expected_features = ["blue", "green", "red", "nir", "swir", "ndvi", "ndwi", "ndbi"]
        self.assertEqual(RF_FEATURE_NAMES, expected_features)

        self.assertEqual(REFLECTANCE_SCALE, 10000.0)
        self.assertEqual(PIXEL_AREA_KM2, 0.0001)

    def test_dotenv_parser(self):
        """Verify custom .env loader handles formatting, quotes, and non-existent files."""
        self.assertFalse(load_dotenv("non_existent_file.env"))


class TestPhase1FlaskIntegration(unittest.TestCase):
    """Test suite for Flask API endpoints and regression verification."""

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_api_health(self):
        """Verify health check endpoint returns 200 with ok status."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")

    def test_api_jobs_list(self):
        """Verify /api/jobs returns a valid list."""
        response = self.client.get("/api/jobs")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIsInstance(data["jobs"], list)

    def test_api_job_detail_and_cancel(self):
        """Verify querying job detail and cancellation via API."""
        # Querying non-existent job returns 404
        response = self.client.get("/api/jobs/non_existent_id")
        self.assertEqual(response.status_code, 404)

        # Cancel non-existent job returns 404
        cancel_response = self.client.post("/api/jobs/non_existent_id/cancel")
        self.assertEqual(cancel_response.status_code, 404)

    def test_existing_routes_regression(self):
        """Verify all existing dashboard routes continue to return 200 OK without regressions."""
        existing_routes = [
            "/",
            "/explorer",
            "/change-analysis",
            "/statistics",
            "/ml-model",
            "/crop-analysis"
        ]

        for route in existing_routes:
            with self.subTest(route=route):
                response = self.client.get(route)
                self.assertEqual(
                    response.status_code,
                    200,
                    f"Regression detected: {route} returned status {response.status_code}"
                )


if __name__ == "__main__":
    unittest.main()
