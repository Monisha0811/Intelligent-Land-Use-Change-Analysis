"""
Thread-safe Background Job Manager for Asynchronous Geospatial Tasks.

Provides non-blocking job execution, progress tracking, status reporting,
cancellation support, and error capture for Flask web application.
"""

import time
import uuid
import inspect
import logging
import traceback
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Any, Optional, List


# Set up logger
logger = logging.getLogger("job_manager")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [JobManager] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class Job:
    """
    Represents a single asynchronous background task with strict, thread-safe
    state transitions and bounded log tracking.
    """

    VALID_STATUSES = {"pending", "running", "completed", "failed", "cancelled"}
    TERMINAL_STATUSES = {"completed", "failed", "cancelled"}

    def __init__(self, job_id: str, job_type: str, description: str = ""):
        self.job_id = job_id
        self.job_type = job_type
        self.description = description
        self.status = "pending"
        self.progress = 0
        self.message = "Job queued..."
        self.created_at = datetime.now().isoformat()
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.duration_seconds: Optional[float] = None
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.traceback: Optional[str] = None
        self.logs: List[Dict[str, str]] = []

        self._start_time: Optional[float] = None
        self._cancel_event = threading.Event()
        self._lock = threading.RLock()

    def add_log(self, message: str):
        """Append a timestamped log entry under lock."""
        with self._lock:
            entry = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "message": message
            }
            self.logs.append(entry)
            if len(self.logs) > 200:
                self.logs.pop(0)

    def is_cancelled(self) -> bool:
        """Check if job cancellation has been requested."""
        return self._cancel_event.is_set()

    def set_running(self) -> bool:
        """
        Transition job from 'pending' to 'running'.
        Returns False if the job was already cancelled or is not pending.
        """
        with self._lock:
            # If cancellation was requested while pending, do not start
            if self._cancel_event.is_set() or self.status == "cancelled":
                if self.status != "cancelled":
                    self.status = "cancelled"
                    self.completed_at = datetime.now().isoformat()
                    self.duration_seconds = 0.0
                    self.message = "Job cancelled before execution started."
                return False

            if self.status != "pending":
                return False

            self.status = "running"
            self._start_time = time.time()
            self.started_at = datetime.now().isoformat()
            self.progress = 5
            self.message = "Job execution started..."
            self.add_log("Worker thread picked up task.")
            return True

    def set_completed(self, result: Any) -> bool:
        """
        Transition job from 'running' to 'completed'.
        Returns False if the job is not running (e.g. was cancelled).
        """
        with self._lock:
            if self.status != "running":
                return False

            self.status = "completed"
            self.progress = 100
            self.result = result
            self.message = "Job completed successfully."
            self.completed_at = datetime.now().isoformat()
            if self._start_time is not None:
                self.duration_seconds = round(time.time() - self._start_time, 2)
            self.add_log("Job completed successfully.")
            return True

    def set_failed(self, error: str, tb: Optional[str] = None) -> bool:
        """
        Transition job from 'running' to 'failed'.
        Returns False if the job is not running.
        """
        with self._lock:
            if self.status != "running":
                return False

            self.status = "failed"
            self.error = error
            self.traceback = tb
            self.message = f"Failed: {error}"
            self.completed_at = datetime.now().isoformat()
            if self._start_time is not None:
                self.duration_seconds = round(time.time() - self._start_time, 2)
            self.add_log(f"Error occurred: {error}")
            return True

    def cancel(self) -> bool:
        """
        Cancel a pending or running job.
        Cannot cancel completed or already failed/cancelled jobs.
        """
        with self._lock:
            if self.status in self.TERMINAL_STATUSES:
                return False

            self._cancel_event.set()
            previous_status = self.status
            self.status = "cancelled"
            self.completed_at = datetime.now().isoformat()

            if previous_status == "running" and self._start_time is not None:
                self.duration_seconds = round(time.time() - self._start_time, 2)
                self.message = "Job cancelled while running."
                self.add_log(f"Job was cancelled while running ({self.duration_seconds}s).")
            else:
                self.duration_seconds = 0.0
                self.message = "Job cancelled by user while pending."
                self.add_log("Job was cancelled while pending.")
            return True

    def update_progress(self, progress: int, message: Optional[str] = None):
        """Update job progress percentage and message under lock."""
        with self._lock:
            if self.status != "running":
                return
            self.progress = max(0, min(100, int(progress)))
            if message:
                self.message = message
                self.add_log(message)

    def to_dict(self) -> Dict[str, Any]:
        """Thread-safe serialization of job state for JSON API responses."""
        with self._lock:
            return {
                "job_id": self.job_id,
                "job_type": self.job_type,
                "description": self.description,
                "status": self.status,
                "progress": self.progress,
                "message": self.message,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "duration_seconds": self.duration_seconds,
                "result": self.result,
                "error": self.error,
                "traceback": self.traceback,
                "logs": list(self.logs[-20:]),  # Defensive copy of recent 20 logs
                "log_count": len(self.logs)
            }


def _prepare_task_kwargs(
    target_func: Callable,
    base_kwargs: Dict[str, Any],
    progress_callback: Callable,
    is_cancelled: Callable,
    job_id: str
) -> Dict[str, Any]:
    """
    Safely inject helpers only if target_func accepts them.
    Compatible with normal functions that do not expect kwargs.
    """
    task_kwargs = dict(base_kwargs)

    try:
        sig = inspect.signature(target_func)
        params = sig.parameters
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        param_names = set(params.keys())
    except (ValueError, TypeError):
        # Fallback for callables without inspectable signature (e.g. C-extensions or builtins)
        has_var_keyword = False
        param_names = set()

    if has_var_keyword or "progress_callback" in param_names:
        task_kwargs["progress_callback"] = progress_callback
    if has_var_keyword or "is_cancelled" in param_names:
        task_kwargs["is_cancelled"] = is_cancelled
    if has_var_keyword or "job_id" in param_names:
        task_kwargs["job_id"] = job_id

    return task_kwargs


class JobManager:
    """
    Singleton Manager for managing background worker threads, task lifecycle,
    and state registry.
    """

    def __init__(self, max_workers: int = 2):
        self._lock = threading.Lock()
        self._jobs: Dict[str, Job] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="geo_worker"
        )
        logger.info(f"Initialized JobManager with {max_workers} worker threads.")

    def submit_job(
        self,
        job_type: str,
        target_func: Callable,
        *args,
        description: str = "",
        **kwargs
    ) -> str:
        """
        Submit a long-running function to run asynchronously.

        Safely injects progress and cancellation helpers only if supported by target_func.
        """
        job_id = uuid.uuid4().hex[:12]
        job = Job(job_id=job_id, job_type=job_type, description=description)

        with self._lock:
            # Bounded registry: cleanup oldest TERMINAL jobs only.
            # Running or pending jobs are NEVER deleted.
            if len(self._jobs) >= 100:
                terminal_job_ids = [
                    jid for jid, j in self._jobs.items()
                    if j.status in Job.TERMINAL_STATUSES
                ]
                if terminal_job_ids:
                    terminal_job_ids.sort(key=lambda jid: self._jobs[jid].created_at)
                    excess = (len(self._jobs) - 100) + 1
                    for jid in terminal_job_ids[:excess]:
                        del self._jobs[jid]

            self._jobs[job_id] = job

        # Schedule execution in worker thread pool
        self._executor.submit(self._run_job_wrapper, job, target_func, args, kwargs)
        logger.info(f"Submitted job {job_id} ({job_type}): {description}")
        return job_id

    def _run_job_wrapper(self, job: Job, target_func: Callable, args, kwargs):
        """Worker wrapper handling job lifecycle, error handling, and completion."""
        # Attempt to transition from pending to running
        if not job.set_running():
            logger.info(f"Job {job.job_id} was not started (current status: {job.status}).")
            return

        logger.info(f"Job {job.job_id} started running.")

        def progress_callback(percent: int, message: Optional[str] = None):
            """Helper passed to worker tasks to report progress."""
            job.update_progress(percent, message)

        # Inspect target_func signature to inject only accepted helper parameters
        task_kwargs = _prepare_task_kwargs(
            target_func=target_func,
            base_kwargs=kwargs,
            progress_callback=progress_callback,
            is_cancelled=job.is_cancelled,
            job_id=job.job_id
        )

        try:
            if job.is_cancelled():
                return

            result = target_func(*args, **task_kwargs)

            if job.is_cancelled():
                return

            job.set_completed(result)

        except Exception as e:
            if job.is_cancelled():
                return
            tb = traceback.format_exc()
            job.set_failed(str(e), tb)
            logger.error(f"Job {job.job_id} failed: {e}\n{tb}")

        finally:
            # Final check: if cancelled midway, ensure completion timestamp and duration are recorded
            with job._lock:
                if job.status == "cancelled" and job.completed_at is None:
                    job.completed_at = datetime.now().isoformat()
                    if job._start_time is not None:
                        job.duration_seconds = round(time.time() - job._start_time, 2)
            logger.info(f"Job {job.job_id} ended with status: {job.status} ({job.duration_seconds}s)")

    def update_progress(self, job_id: str, progress: int, message: Optional[str] = None):
        """Update job progress percentage and message."""
        job = self.get_job_object(job_id)
        if job:
            job.update_progress(progress, message)

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running or pending job.
        Returns False if job does not exist or has already completed/failed.
        """
        job = self.get_job_object(job_id)
        if not job:
            return False
        return job.cancel()

    def get_job_object(self, job_id: str) -> Optional[Job]:
        """Retrieve raw Job object."""
        with self._lock:
            return self._jobs.get(job_id)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve serialized job info."""
        job = self.get_job_object(job_id)
        return job.to_dict() if job else None

    def list_jobs(self, job_type: Optional[str] = None, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """List serialized jobs with optional filtering."""
        with self._lock:
            jobs = list(self._jobs.values())

        if job_type:
            jobs = [j for j in jobs if j.job_type == job_type]
        if status:
            jobs = [j for j in jobs if j.status == status]

        # Sort newest first
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in jobs[:limit]]


# Global singleton instance
job_manager = JobManager(max_workers=2)
