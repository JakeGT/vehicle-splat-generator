"""Placing Imagin's other views round the van.

Each view's camera is unknown, and framed tighter or looser than the spin. It is
found against the hull carved from the spin (hull.py), with the spin's lens: for
a direction (yaw, elevation) the hull is projected, and the camera's distance and
aim are solved so the hull's box lands on the view's outline. The best outline
fit wins.

An outline alone cannot tell a boxy van's front-left from its rear-right -- they
are mirror images -- nor which way round a view from straight above is. So each
view keeps two candidates, the best fit and the best one facing the other way,
and the pipeline picks between them by what they show (pipeline.py)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .camera import DISTANCE, focal_length, intrinsics, orbit_viewmat, project
from .hull import iou, mask_box, outline_of

# Views from at least this high count as "from above": they see what the spin cannot.
VIEWS_FROM_ABOVE = 75.0


@dataclass
class ViewCamera:
    code: int
    yaw: float
    elevation: float
    distance: float
    # Where the camera looks, in the world: the hull's box, aimed at the view's.
    target: torch.Tensor
    fov_y: float
    outline_iou: float
    # How far its picture is from the first model's, when the pipeline compared them.
    picture_error: float | None = None
    alternative: "ViewCamera | None" = field(default=None, repr=False)

    def viewmat(self) -> torch.Tensor:
        return orbit_viewmat(self.yaw, self.elevation, self.distance, self.target).float()

    def K(self, width: int, height: int) -> torch.Tensor:
        return intrinsics(self.fov_y, width, height)

    def report(self) -> dict:
        return {"code": self.code, "yaw": round(self.yaw, 1), "elevation": round(self.elevation, 1),
                "distance": round(self.distance, 3), "outlineIoU": round(self.outline_iou, 3),
                "pictureError": None if self.picture_error is None else round(self.picture_error, 4)}


def fit_camera(points: torch.Tensor, mask: torch.Tensor, yaw: float, elevation: float, fov_y: float,
               code: int = 0, passes: int = 2) -> ViewCamera:
    """The camera at (yaw, elevation) whose picture of the hull fills the mask's box."""
    height, width = mask.shape
    K = intrinsics(fov_y, width, height).to(points.device)
    focal = focal_length(fov_y, height)
    box = mask_box(mask)
    box_centre = torch.tensor([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=torch.float64)
    distance, target = DISTANCE, torch.zeros(3, dtype=torch.float64)
    for _ in range(passes):
        viewmat = orbit_viewmat(yaw, elevation, distance, target)
        uv, _ = project(points, viewmat, K)
        low, high = uv.min(0).values.double().cpu(), uv.max(0).values.double().cpu()
        # One scale for both sides: a picture's proportions are the camera's, not the fit's.
        scale = min((box[2] - box[0]) / (high[0] - low[0]).item(), (box[3] - box[1]) / (high[1] - low[1]).item())
        # Closer by the scale; the picture grows about its principal point.
        distance = distance / scale
        principal = torch.tensor([width / 2, height / 2], dtype=torch.float64)
        centre = ((low + high) / 2 - principal) * scale + principal
        shift = box_centre - centre
        # Moving the camera across its picture moves the van the other way.
        rotation = viewmat[:3, :3]
        target = target - (shift[0] * rotation[0] + shift[1] * rotation[1]) * distance / focal
    camera = ViewCamera(code, yaw, elevation, distance, target, fov_y, 0.0)
    camera.outline_iou = iou(outline_of(points, camera.viewmat().to(points.device), K, height, width), mask)
    return camera


def _best(points, mask, fov_y, yaws, elevations, code) -> ViewCamera:
    best = None
    for yaw in yaws:
        for elevation in elevations:
            camera = fit_camera(points, mask, float(yaw), float(elevation), fov_y, code, passes=1)
            if best is None or camera.outline_iou > best.outline_iou:
                best = camera
    return best


def _refined(points, mask, fov_y, rough: ViewCamera, code) -> ViewCamera:
    yaws = np.arange(rough.yaw - 8, rough.yaw + 8.1, 2)
    elevations = np.arange(max(rough.elevation - 4, -5), min(rough.elevation + 4, 90) + 0.1, 1)
    best = _best(points, mask, fov_y, yaws, elevations, code)
    return fit_camera(points, mask, best.yaw, best.elevation, fov_y, code)


def place_view(points: torch.Tensor, mask: torch.Tensor, fov_y: float, code: int,
               yaw_step: float = 10, elevation_step: float = 5) -> ViewCamera:
    """The view's best camera, with the best facing the other way as its alternative."""
    yaws = np.arange(0, 360, yaw_step)
    elevations = np.arange(-5, 90.1, elevation_step)
    rough = _best(points, mask, fov_y, yaws, elevations, code)
    best = _refined(points, mask, fov_y, rough, code)
    if best.elevation >= VIEWS_FROM_ABOVE:
        # From above, the outline is the same turned half round.
        alternative = fit_camera(points, mask, (best.yaw + 180) % 360, best.elevation, fov_y, code)
    else:
        opposite = [yaw for yaw in yaws if abs(((yaw - best.yaw + 180) % 360) - 180) >= 150]
        alternative = _refined(points, mask, fov_y, _best(points, mask, fov_y, opposite, elevations, code), code)
    best.alternative = alternative
    return best


def view_masks(images: dict[int, np.ndarray], width: int, device) -> dict[int, torch.Tensor]:
    from .hull import alpha_masks
    return {code: alpha_masks(image[None], width, device)[0] for code, image in images.items()}


def is_from_above(camera: ViewCamera, spin_elevation: float, margin: float = 8.0) -> bool:
    """Whether a view looks down on the van from clearly above the spin."""
    return camera.elevation >= spin_elevation + margin

