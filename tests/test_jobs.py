from vehicle_splat.jobs import JobReporter, JobStore, Worker
from vehicle_splat.pipeline import STAGES, JobSpec


def finished(spec, out_dir, reporter):
    for stage in STAGES:
        reporter.stage(stage, "running")
        reporter.stage(stage, "done", f"{stage} went fine")
    for suffix in (".splat", ".json", "-turntable.mp4"):
        (out_dir / f"{spec.model_name}{suffix}").write_bytes(b"x")
    (out_dir / "scratch.bin").write_bytes(b"not served")
    return {"model": {"name": spec.model_name}}


def test_a_job_runs_and_lists_its_files(tmp_path):
    store = JobStore(tmp_path)
    worker = Worker(store, finished)
    job = worker.submit(JobSpec(cap_id=58438, paint_id="pwhite"))
    assert job["status"] == "queued"
    worker.run_one(worker.queue.get_nowait())
    done = store.get(job["id"])
    assert done["status"] == "succeeded"
    assert done["files"] == ["58438-pwhite.splat", "58438-pwhite.json", "58438-pwhite-turntable.mp4"]
    assert done["report"] == {"model": {"name": "58438-pwhite"}}
    assert [stage["status"] for stage in done["stages"]] == ["done"] * len(STAGES)
    assert all(stage["seconds"] is not None for stage in done["stages"])


def test_a_failed_job_says_why_and_the_worker_carries_on(tmp_path):
    def failing(spec, out_dir, reporter):
        reporter.stage("frames", "running", "fetching the spin")
        raise RuntimeError("No CUDA GPU is visible.")

    store = JobStore(tmp_path)
    worker = Worker(store, failing)
    first = worker.submit(JobSpec(cap_id=1))
    worker.run_one(first["id"])
    failed = store.get(first["id"])
    assert failed["status"] == "failed"
    assert failed["error"] == "RuntimeError: No CUDA GPU is visible."
    assert "Traceback" in failed["trace"]
    assert (failed["stages"][0]["status"], failed["stages"][0]["detail"]) == ("failed", "fetching the spin")

    worker.runner = finished
    second = worker.submit(JobSpec(cap_id=2))
    worker.run_one(second["id"])
    assert store.get(second["id"])["status"] == "succeeded"


def test_after_a_restart_running_jobs_fail_and_waiting_ones_queue_again(tmp_path):
    store = JobStore(tmp_path)
    running = store.create(JobSpec(cap_id=1))
    store.update(running["id"], lambda job: job.update(status="running"))
    waiting = [store.create(JobSpec(cap_id=cap_id)) for cap_id in (2, 3)]
    store.update(waiting[0]["id"], lambda job: job.update(createdAt="2026-01-01T00:00:00+00:00"))
    assert store.recover() == [waiting[0]["id"], waiting[1]["id"]]
    assert store.get(running["id"])["status"] == "failed"
    assert store.get(running["id"])["error"] == "The service stopped while it ran."


def test_a_job_id_cannot_reach_outside_the_store(tmp_path):
    store = JobStore(tmp_path / "jobs")
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "job.json").write_text("{}")
    assert store.get("../elsewhere") is None
    assert store.get("missing") is None


def test_training_progress_is_written_at_most_once_a_second_and_always_at_the_end(tmp_path):
    store = JobStore(tmp_path)
    job = store.create(JobSpec(cap_id=1))
    reporter = JobReporter(store, job["id"])
    reporter.training(1, 100, 0.5, 1000)
    reporter.training(51, 100, 0.25, 2000)
    assert store.get(job["id"])["training"]["step"] == 1
    reporter.training(100, 100, 0.125, 3000)
    training = store.get(job["id"])["training"]
    assert (training["step"], training["loss"], training["splats"]) == (100, 0.125, 3000)


def test_a_stage_that_finishes_without_a_word_drops_what_it_was_doing(tmp_path):
    store = JobStore(tmp_path)
    job = store.create(JobSpec(cap_id=1))
    reporter = JobReporter(store, job["id"])
    reporter.stage("preview", "running", "rendering the turntable")
    reporter.stage("preview", "done")
    preview = next(stage for stage in store.get(job["id"])["stages"] if stage["name"] == "preview")
    assert (preview["status"], preview["detail"]) == ("done", None)
