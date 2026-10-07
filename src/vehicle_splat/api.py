"""The service: a page to give a CAP id, see the van's renders and make its splat,
and the API behind it. Run locally, for one person; there is no sign-in, so keep
it on localhost (compose.yaml binds 127.0.0.1)."""
from __future__ import annotations

import pathlib
import re
import shutil

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings, load_settings
from .imagin import FROM_ABOVE, SPIN_ANGLES, Imagin, Vehicle
from .jobs import JobReporter, JobStore, Runner, Worker
from .pipeline import JobSpec

WEB = pathlib.Path(__file__).parent / "web"
# The page's thumbnails: plenty for a grid, a fraction of the training renders.
THUMBNAIL_WIDTH = 400
PAINT_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


class JobRequest(BaseModel):
    capId: int = Field(gt=0)
    paintId: str | None = Field(default=None, pattern=PAINT_ID.pattern)
    extraViews: str = Field(default="above", pattern="^(above|all|none)$")
    steps: int = Field(default=15_000, ge=500, le=100_000)
    maxSplats: int = Field(default=150_000, ge=10_000, le=1_000_000)


def paint(paint_id: str | None) -> str | None:
    if paint_id in (None, ""):
        return None
    if not PAINT_ID.match(paint_id):
        raise HTTPException(422, "A paint id is letters, digits, - and _.")
    return paint_id


def create_app(settings: Settings | None = None, runner: Runner | None = None, start_worker: bool = True,
               imagin: Imagin | None = None) -> FastAPI:
    """The app. The runner and Imagin are only passed in by the tests, to run
    without a GPU or the network."""
    settings = settings or load_settings()
    imagin = imagin or Imagin(settings.imagin_customer, settings.imagin_cache_dir)
    store = JobStore(settings.jobs_dir)
    if runner is None:
        from .pipeline import run

        def runner(spec: JobSpec, out_dir: pathlib.Path, reporter: JobReporter) -> dict:
            return run(spec, out_dir, imagin, reporter)
    worker = Worker(store, runner)
    if start_worker:
        worker.start()

    app = FastAPI(title="Vehicle splat")
    app.state.worker = worker

    @app.get("/api/health")
    def health():
        import torch

        return {"ok": True, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                "torch": torch.__version__, "portalPublishing": settings.portal_models_dir is not None}

    @app.get("/api/vehicles/{cap_id}")
    def vehicle(cap_id: int, paint_id: str | None = Query(default=None, alias="paintId")):
        """The van's renders, fetched (and cached) at thumbnail size: the spin, and
        each view code, marked with the code it only repeats."""
        if cap_id <= 0:
            raise HTTPException(422, "A CAP id is a positive number.")
        van = Vehicle(cap_id, paint(paint_id))
        try:
            imagin.spin(van, THUMBNAIL_WIDTH)
            views = imagin.views(van, THUMBNAIL_WIDTH)
        except (OSError, ValueError) as error:
            raise HTTPException(502, f"Imagin had no renders for CAP id {cap_id}: {error}") from error
        image = lambda angle: f"/api/images/{cap_id}/{angle}.png?width={THUMBNAIL_WIDTH}" + (  # noqa: E731
            f"&paintId={van.paint_id}" if van.paint_id else "")
        return {
            "capId": cap_id, "paintId": van.paint_id,
            "spin": [{"angle": angle, "url": image(angle)} for angle in SPIN_ANGLES],
            "views": [{"code": view.code, "url": image(view.code), "duplicateOf": view.duplicate_of,
                       "expectedFromAbove": view.code in FROM_ABOVE} for view in views],
        }

    @app.get("/api/images/{cap_id}/{angle}.png")
    def image(cap_id: int, angle: int, width: int = Query(default=THUMBNAIL_WIDTH, ge=100, le=2600),
              paint_id: str | None = Query(default=None, alias="paintId")):
        if angle not in SPIN_ANGLES and not 1 <= angle <= 33:
            raise HTTPException(404, "No such Imagin angle.")
        try:
            path = imagin.image(Vehicle(cap_id, paint(paint_id)), angle, width)
        except (OSError, ValueError) as error:
            raise HTTPException(502, str(error)) from error
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "max-age=86400"})

    @app.post("/api/jobs", status_code=202)
    def create_job(request: JobRequest):
        spec = JobSpec(cap_id=request.capId, paint_id=paint(request.paintId), extra_views=request.extraViews,
                       steps=request.steps, max_splats=request.maxSplats)
        return worker.submit(spec)

    @app.get("/api/jobs")
    def jobs(limit: int = Query(default=20, ge=1, le=200)):
        return [{key: job[key] for key in ("id", "createdAt", "status", "spec")} for job in store.all()[:limit]]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        found = store.get(job_id)
        if not found:
            raise HTTPException(404, "No such job.")
        found.pop("trace", None)
        return found

    @app.get("/api/jobs/{job_id}/files/{name}")
    def job_file(job_id: str, name: str):
        found = store.get(job_id)
        if not found or name not in found["files"]:
            raise HTTPException(404, "No such file.")
        return FileResponse(store.outputs(job_id) / name, filename=name)

    @app.post("/api/jobs/{job_id}/publish")
    def publish(job_id: str):
        """Copies the model into the portal's apps/portal/public/models, when it is mounted."""
        if settings.portal_models_dir is None:
            raise HTTPException(409, "No portal models folder is mounted (PORTAL_MODELS_DIR).")
        found = store.get(job_id)
        if not found or found["status"] != "succeeded":
            raise HTTPException(404, "No finished job.")
        name = JobSpec(**found["spec"]).model_name
        settings.portal_models_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for suffix in (".json", ".splat"):
            shutil.copyfile(store.outputs(job_id) / f"{name}{suffix}", settings.portal_models_dir / f"{name}{suffix}")
            copied.append(f"{name}{suffix}")
        return {"published": copied}

    app.mount("/", StaticFiles(directory=WEB, html=True), name="page")
    return app

