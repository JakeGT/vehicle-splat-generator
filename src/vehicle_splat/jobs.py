"""Jobs, kept as JSON files beside their outputs (DATA_DIR/jobs/<id>/), and the one
worker that runs them: there is one GPU, so one splat trains at a time and the
rest wait their turn.

A job that was running when the service stopped is marked failed when it starts
again, rather than left looking busy forever; one still waiting is queued again."""
from __future__ import annotations

import datetime
import json
import pathlib
import queue
import threading
import time
import traceback
import uuid
from dataclasses import asdict
from typing import Callable

from .pipeline import STAGES, JobSpec

# Files a finished job may serve, by suffix after its model name.
FILE_SUFFIXES = (".splat", ".json", ".ply", "-turntable.mp4")


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class JobStore:
    def __init__(self, directory: pathlib.Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def path(self, job_id: str) -> pathlib.Path:
        return self.directory / job_id / "job.json"

    def outputs(self, job_id: str) -> pathlib.Path:
        return self.directory / job_id

    def create(self, spec: JobSpec) -> dict:
        job_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        job = {"id": job_id, "createdAt": now(), "status": "queued", "spec": asdict(spec),
               "stages": [{"name": name, "status": "waiting", "detail": None, "startedAt": None,
                           "seconds": None} for name in STAGES],
               "training": None, "report": None, "error": None, "files": []}
        self.save(job)
        return job

    def save(self, job: dict) -> None:
        path = self.path(job["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        partial.write_text(json.dumps(job, indent=2))
        partial.replace(path)

    def get(self, job_id: str) -> dict | None:
        path = self.path(job_id)
        if not path.is_file() or path.parent.parent != self.directory:
            return None
        return json.loads(path.read_text())

    def all(self) -> list[dict]:
        jobs = [json.loads(path.read_text()) for path in self.directory.glob("*/job.json")]
        return sorted(jobs, key=lambda job: job["createdAt"], reverse=True)

    def update(self, job_id: str, change: Callable[[dict], None]) -> dict:
        with self.lock:
            job = self.get(job_id)
            change(job)
            self.save(job)
            return job

    def recover(self) -> list[str]:
        """After a restart: running jobs failed, waiting ones queued again (their ids)."""
        waiting = []
        for job in self.all():
            if job["status"] == "running":
                self.update(job["id"], lambda j: j.update(status="failed", error="The service stopped while it ran."))
            elif job["status"] == "queued":
                waiting.append(job["id"])
        return list(reversed(waiting))


class JobReporter:
    """The pipeline's progress, written into the job as it goes."""

    def __init__(self, store: JobStore, job_id: str):
        self.store, self.job_id = store, job_id
        self.last_training_write = 0.0

    def stage(self, name: str, status: str, detail: str | None = None) -> None:
        def change(job):
            for stage in job["stages"]:
                if stage["name"] != name:
                    continue
                if status == "running" and stage["startedAt"] is None:
                    stage["startedAt"] = time.time()
                if status in ("done", "skipped", "failed") and stage["startedAt"] is not None:
                    stage["seconds"] = round(time.time() - stage["startedAt"], 1)
                # A stage that finishes without a word clears what it said it was
                # doing; one that fails is marked by the worker, which keeps it.
                stage["status"], stage["detail"] = status, detail
        self.store.update(self.job_id, change)

    def training(self, step: int, steps: int, loss: float, splats: int) -> None:
        # A file write per second is plenty for a page polling every second or two.
        if step != steps and time.monotonic() - self.last_training_write < 1.0:
            return
        self.last_training_write = time.monotonic()
        self.store.update(self.job_id, lambda job: job.update(
            training={"step": step, "steps": steps, "loss": round(loss, 4), "splats": splats, "at": time.time()}))


Runner = Callable[[JobSpec, pathlib.Path, JobReporter], dict]


class Worker:
    def __init__(self, store: JobStore, runner: Runner):
        self.store, self.runner = store, runner
        self.queue: queue.Queue[str] = queue.Queue()
        self.thread = threading.Thread(target=self._loop, name="splat-worker", daemon=True)

    def start(self) -> None:
        for job_id in self.store.recover():
            self.queue.put(job_id)
        self.thread.start()

    def submit(self, spec: JobSpec) -> dict:
        job = self.store.create(spec)
        self.queue.put(job["id"])
        return job

    def _loop(self) -> None:
        while True:
            self.run_one(self.queue.get())

    def run_one(self, job_id: str) -> None:
        job = self.store.update(job_id, lambda j: j.update(status="running", startedAt=now()))
        spec = JobSpec(**job["spec"])
        reporter = JobReporter(self.store, job_id)
        try:
            report = self.runner(spec, self.store.outputs(job_id), reporter)
            files = [spec.model_name + suffix for suffix in FILE_SUFFIXES
                     if (self.store.outputs(job_id) / (spec.model_name + suffix)).is_file()]
            self.store.update(job_id, lambda j: j.update(status="succeeded", finishedAt=now(), report=report,
                                                         files=files))
        except Exception as error:  # the job fails, the worker carries on with the next
            detail = "".join(traceback.format_exception_only(type(error), error)).strip()

            def failed(job):
                job.update(status="failed", finishedAt=now(), error=detail, trace=traceback.format_exc())
                for stage in job["stages"]:
                    if stage["status"] == "running":
                        stage["status"] = "failed"
            self.store.update(job_id, failed)
