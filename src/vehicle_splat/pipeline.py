"""One van, Imagin's renders in, a splat out. The stages, in order:

1. **frames**: the 32-frame spin, and (unless asked not to) Imagin's other views,
   less the codes that only repeat its default view.
2. **camera**: the spin's camera, fitted to the van's outlines (hull.py).
3. **views**: each extra view placed round the van (views.py). An outline cannot
   tell front from back, so a first model, trained on the spin alone, renders both
   candidates and the one nearer the view's own picture is kept. Then the views to
   use: by default only those from clearly above the spin, which see what it cannot.
4. **training**: the spin and the chosen views (train.py).
5. **preview**: a turntable video.
6. **export**: the .splat, .ply and the manifest the portal reads (export.py)."""
from __future__ import annotations

import pathlib
import time
from dataclasses import asdict, dataclass
from typing import Literal, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .camera import intrinsics, turntable_viewmats
from .export import write_model
from .hull import SpinCamera, fit_spin_camera, hull_points, initial_splats
from .imagin import SPIN_ANGLES, Imagin, Vehicle
from .preview import write_turntable
from .views import ViewCamera, is_from_above, place_view, view_masks

ExtraViews = Literal["above", "all", "none"]
STAGES = ("frames", "camera", "views", "training", "preview", "export")
# An extra view whose outline the hull fills worse than this is not trusted.
MIN_OUTLINE_IOU = 0.8


@dataclass(frozen=True)
class JobSpec:
    cap_id: int
    paint_id: str | None = None
    extra_views: ExtraViews = "above"
    steps: int = 15_000
    max_splats: int = 150_000
    sh_degree: int = 3
    spin_width: int = 1200
    # The extra views frame the van looser than the spin, so they are fetched larger.
    view_width: int = 1600
    first_model_steps: int = 3000

    @property
    def vehicle(self) -> Vehicle:
        return Vehicle(self.cap_id, self.paint_id)

    @property
    def model_name(self) -> str:
        return str(self.cap_id) if not self.paint_id else f"{self.cap_id}-{self.paint_id}"


class Reporter(Protocol):
    def stage(self, name: str, status: str, detail: str | None = None) -> None: ...
    def training(self, step: int, steps: int, loss: float, splats: int) -> None: ...


def load_rgba(path: pathlib.Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32) / 255


def gpu() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is visible. In Docker on WSL the container needs the GPU "
                           "passed through: see the README's \"The GPU in WSL\".")
    return torch.device("cuda")


def run(spec: JobSpec, out_dir: pathlib.Path, imagin: Imagin, reporter: Reporter,
        device: torch.device | None = None) -> dict:
    device = device or gpu()
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"spec": asdict(spec)}

    # 1. Frames.
    reporter.stage("frames", "running", "fetching the spin")
    spin = np.stack([load_rgba(path) for path in imagin.spin(spec.vehicle, spec.spin_width)])
    extras: dict[int, np.ndarray] = {}
    if spec.extra_views != "none":
        reporter.stage("frames", "running", "fetching the other views")
        extras = {view.code: load_rgba(view.path) for view in imagin.views(spec.vehicle, spec.view_width)
                  if view.duplicate_of is None}
    count, height, width = spin.shape[:3]
    reporter.stage("frames", "done", f"{count} spin frames ({width} x {height}), {len(extras)} other views")

    # 2. The spin's camera.
    reporter.stage("camera", "running", "fitting the spin's camera")
    camera = fit_spin_camera(spin, device, lambda message: reporter.stage("camera", "running", message))
    report["camera"] = {**camera.as_manifest(count), "score": round(camera.score, 3)}
    reporter.stage("camera", "done", f"field of view {camera.fov_y:.1f}°, elevation {camera.elevation:.1f}°, "
                                     f"turning {'+' if camera.direction > 0 else '−'}, hull fit {camera.score:.3f}")

    # 3. The extra views.
    chosen: list[ViewCamera] = []
    if extras:
        chosen, report["views"] = _place_views(spec, spin, extras, camera, device, reporter)
    else:
        reporter.stage("views", "skipped", "no extra views asked for")

    # 4. Training.
    started = time.monotonic()
    reporter.stage("training", "running", "starting from the hull")
    from .train import TrainingView, initial_params, train

    means, colours, spacing = initial_splats(spin, camera, device, spec.max_splats // 2)
    views = _spin_views(spin, camera, device)
    for view in chosen:
        image = torch.from_numpy(extras[view.code]).to(device)
        views.append(TrainingView(image, view.viewmat().to(device),
                                  view.K(image.shape[1], image.shape[0]).to(device), refine_pose=True))
    params, _ = train(views, initial_params(means, colours, spacing, spec.sh_degree), spec.steps,
                      spec.max_splats, spec.sh_degree, reporter.training)
    report["training"] = {"steps": spec.steps, "views": len(views), "seconds": round(time.monotonic() - started),
                          "splats": len(params["means"])}
    reporter.stage("training", "done", f"{len(params['means']):,} splats from {len(views)} pictures")

    # 5. Preview.
    reporter.stage("preview", "running", "rendering the turntable")
    write_turntable(out_dir / f"{spec.model_name}-turntable.mp4", params, camera, count, spec.sh_degree)
    reporter.stage("preview", "done")

    # 6. Export.
    reporter.stage("export", "running")
    source = {"imagin": {"capId": spec.cap_id, "paintId": spec.paint_id, "spinAngles": list(SPIN_ANGLES),
                         "spinWidth": width, "views": [view.code for view in chosen]}}
    model = write_model(out_dir, spec.model_name, params, camera.as_manifest(count), source)
    report["model"] = {"name": spec.model_name, "splatCount": model["splatCount"]}
    reporter.stage("export", "done", f"{model['splatCount']:,} splats written")
    return report


def _spin_views(spin: np.ndarray, camera: SpinCamera, device) -> list:
    from .train import TrainingView

    count, height, width = spin.shape[:3]
    K = intrinsics(camera.fov_y, width, height).to(device)
    viewmats = turntable_viewmats(range(count), count, camera.elevation, camera.direction).to(device)
    return [TrainingView(torch.from_numpy(spin[index]).to(device), viewmats[index], K) for index in range(count)]


def _place_views(spec: JobSpec, spin: np.ndarray, extras: dict[int, np.ndarray], camera: SpinCamera,
                 device, reporter: Reporter) -> tuple[list[ViewCamera], list[dict]]:
    from .train import initial_params, train

    points = hull_points(spin, camera, device)
    points = points[torch.randperm(len(points), device=device)[:250_000]]
    masks = view_masks(extras, 200, device)
    placed: list[ViewCamera] = []
    for number, (code, mask) in enumerate(masks.items(), start=1):
        reporter.stage("views", "running", f"placing view {code} ({number} of {len(masks)})")
        placed.append(place_view(points, mask, camera.fov_y, code))

    reporter.stage("views", "running", f"a first model, from the spin alone ({spec.first_model_steps:,} steps)")
    means, colours, spacing = initial_splats(spin, camera, device, spec.max_splats // 2)
    first, _ = train(_spin_views(spin, camera, device), initial_params(means, colours, spacing, spec.sh_degree),
                     spec.first_model_steps, spec.max_splats, spec.sh_degree)

    reporter.stage("views", "running", "telling front from back")
    chosen, rows = [], []
    for view in placed:
        candidates = [candidate for candidate in (view, view.alternative) if candidate is not None]
        for candidate in candidates:
            candidate.picture_error = _picture_error(first, candidate, extras[view.code], spec.sh_degree)
        best = min(candidates, key=lambda candidate: candidate.picture_error)
        trusted = best.outline_iou >= MIN_OUTLINE_IOU
        from_above = is_from_above(best, camera.elevation)
        used = trusted and (spec.extra_views == "all" or from_above)
        reason = ("used" if used else "outline fits too poorly" if not trusted else "at the spin's own height")
        if used:
            chosen.append(best)
        rows.append({**best.report(), "fromAbove": from_above, "used": used, "reason": reason,
                     "alternative": next((c.report() for c in candidates if c is not best), None)})
    reporter.stage("views", "done", f"{len(chosen)} of {len(placed)} views used")
    return chosen, rows


def _picture_error(params, camera: ViewCamera, image: np.ndarray, sh_degree: int, width: int = 400) -> float:
    """How far the first model's picture from `camera` is from the view's own, both
    over grey, where either shows the van."""
    from .train import render

    device = params["means"].device
    height = round(width * image.shape[0] / image.shape[1])
    grey = torch.full((3,), 0.5, device=device)
    with torch.no_grad():
        rendered, alpha, _ = render(params, camera.viewmat().to(device), camera.K(width, height).to(device),
                                    width, height, grey, sh_degree)
    target = torch.from_numpy(image).to(device).permute(2, 0, 1)[None]
    target = F.interpolate(target, size=(height, width), mode="area")[0].permute(1, 2, 0)
    target_alpha = target[..., 3:]
    target = target[..., :3] * target_alpha + grey * (1 - target_alpha)
    van = ((alpha[..., 0] > 0.1) | (target_alpha[..., 0] > 0.1))
    return (rendered - target).abs().mean(-1)[van].mean().item()

