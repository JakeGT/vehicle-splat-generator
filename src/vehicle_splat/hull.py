"""The van's visual hull: what is inside its outline in every frame of the spin.

Imagin does not publish the spin's camera, so it is fitted here. For a guess at
(field of view, elevation, direction), a grid of points is carved down to those
inside the van's outline in all 32 frames, and the hull is projected back into
each. With the right guess the hull fills every outline; with a wrong one the
frames disagree, the hull shrinks, and it fills them badly. The hull also gives
training its first splats, and the extra views something to be placed against
(views.py)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .camera import DISTANCE, intrinsics, project, turntable_viewmats

# The ground shadow is faint; 0.6 keeps it out of the outline.
OUTLINE_ALPHA = 0.6


@dataclass(frozen=True)
class SpinCamera:
    fov_y: float
    elevation: float
    direction: int
    # How well the hull fills the outlines, averaged with neighbouring guesses.
    score: float = float("nan")

    def as_manifest(self, frame_count: int) -> dict:
        """The camera as the portal's manifest reads it (CRM repo, splatModel.ts)."""
        return {"convention": "opencv-turntable", "fovYDegrees": round(self.fov_y, 3),
                "elevationDegrees": round(self.elevation, 3), "direction": int(self.direction),
                "distance": DISTANCE, "frameCount": frame_count}


def alpha_masks(images: np.ndarray, width: int, device) -> torch.Tensor:
    """Outlines [V, h, w] of RGBA images [V, H, W, 4] in 0..1, `width` across."""
    alpha = torch.from_numpy(np.ascontiguousarray(images[..., 3])).unsqueeze(1).float()
    height = round(width * images.shape[1] / images.shape[2])
    small = F.interpolate(alpha, size=(height, width), mode="area").squeeze(1)
    return (small > OUTLINE_ALPHA).to(device)


def mask_box(mask: torch.Tensor) -> tuple[int, int, int, int]:
    ys, xs = torch.nonzero(mask, as_tuple=True)
    return xs.min().item(), ys.min().item(), xs.max().item() + 1, ys.max().item() + 1


def hull_grid(fov_y: float, resolution: int, aspect: float, device) -> tuple[torch.Tensor, float]:
    # A cube as wide as the frame is at the axis: the van fits in it whatever the guess.
    half = DISTANCE * math.tan(math.radians(fov_y) / 2) * aspect * 1.05
    axis = torch.linspace(-half, half, resolution, device=device)
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
    return grid, 2 * half / (resolution - 1)


def carve(camera: SpinCamera, masks: torch.Tensor, resolution: int):
    """The grid, which of its points are inside the hull, and the grid's spacing."""
    count, mask_height, mask_width = masks.shape
    device = masks.device
    K = intrinsics(camera.fov_y, mask_width, mask_height).to(device)
    viewmats = turntable_viewmats(range(count), count, camera.elevation, camera.direction).to(device)
    grid, spacing = hull_grid(camera.fov_y, resolution, mask_width / mask_height, device)
    inside = torch.ones(len(grid), dtype=torch.bool, device=device)
    for view in range(count):
        uv, z = project(grid, viewmats[view], K)
        pixel = uv.floor().long()
        seen = (z > 0) & (pixel[:, 0] >= 0) & (pixel[:, 0] < mask_width) & (pixel[:, 1] >= 0) & (pixel[:, 1] < mask_height)
        # Every Imagin frame shows the whole van, so a point outside a frame is not
        # on it, any more than one outside the outline.
        keep = torch.zeros_like(inside)
        keep[seen] = masks[view, pixel[seen, 1], pixel[seen, 0]]
        inside &= keep
    return grid, inside, spacing


def outline_of(points: torch.Tensor, viewmat: torch.Tensor, K: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """The outline [h, w] that points draw from a camera. The points are a little
    coarser than the pixels, so the gaps between them are closed."""
    uv, z = project(points, viewmat, K)
    pixel = uv.round().long()
    on = (z > 0) & (pixel[:, 0] >= 0) & (pixel[:, 0] < width) & (pixel[:, 1] >= 0) & (pixel[:, 1] < height)
    image = torch.zeros(height * width, device=points.device)
    image[pixel[on, 1] * width + pixel[on, 0]] = 1
    return F.max_pool2d(image.view(1, 1, height, width), 3, 1, 1)[0, 0] > 0


def iou(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a & b).sum() / (a | b).sum().clamp(min=1)).item()


def hull_score(camera: SpinCamera, masks: torch.Tensor, resolution: int) -> float:
    grid, inside, _ = carve(camera, masks, resolution)
    points = grid[inside]
    if len(points) == 0:
        return 0.0
    count, height, width = masks.shape
    K = intrinsics(camera.fov_y, width, height).to(masks.device)
    viewmats = turntable_viewmats(range(count), count, camera.elevation, camera.direction).to(masks.device)
    return float(np.mean([iou(outline_of(points, viewmats[view], K, height, width), masks[view])
                          for view in range(count)]))


def search(masks, resolution, fovs, elevations, directions, progress=None) -> SpinCamera:
    """Scores every guess, then takes the best after averaging each score with its
    neighbours'. The hull is carved from a grid, so one score moves by a point or
    two with how the grid happens to fall, and a lone high score is usually that
    rather than the camera."""
    best = None
    total, done = len(directions) * len(fovs) * len(elevations), 0
    for direction in directions:
        scores = np.zeros((len(fovs), len(elevations)))
        for i, fov_y in enumerate(fovs):
            for j, elevation in enumerate(elevations):
                scores[i, j] = hull_score(SpinCamera(float(fov_y), float(elevation), direction), masks, resolution)
                done += 1
                if progress:
                    progress(done, total)
        padded = np.pad(scores, 1, mode="edge")
        smoothed = sum(padded[a:a + len(fovs), b:b + len(elevations)] for a in range(3) for b in range(3)) / 9
        i, j = np.unravel_index(np.argmax(smoothed), smoothed.shape)
        if best is None or smoothed[i, j] > best.score:
            best = SpinCamera(float(fovs[i]), float(elevations[j]), direction, float(smoothed[i, j]))
    return best


def fit_spin_camera(spin: np.ndarray, device, progress: Callable[[str], None] | None = None,
                    fovs: Sequence[float] = tuple(range(4, 42, 2)),
                    elevations: Sequence[float] = tuple(range(-4, 26, 2)),
                    coarse=(80, 88), fine=(160, 176), rounds: int = 6) -> SpinCamera:
    """The spin's camera: a coarse search over every guess, then a fine one round
    the best, moved along for as long as the best is on its edge."""
    say = progress or (lambda message: None)
    width, resolution = coarse
    best = search(alpha_masks(spin, width, device), resolution, list(fovs), list(elevations), (1, -1),
                  lambda done, total: say(f"coarse search {done}/{total}"))
    say(f"coarse: fov {best.fov_y:.1f}°, elevation {best.elevation:.1f}°, direction {best.direction:+d}")
    width, resolution = fine
    masks = alpha_masks(spin, width, device)
    for round_ in range(rounds):
        fov_range = np.arange(max(best.fov_y - 2, 1), best.fov_y + 2.01, 0.5)
        elevation_range = np.arange(best.elevation - 2, best.elevation + 2.01, 0.5)
        best = search(masks, resolution, fov_range, elevation_range, (best.direction,),
                      lambda done, total: say(f"fine search {round_ + 1}: {done}/{total}"))
        on_edge = (best.fov_y in (fov_range[0], fov_range[-1]) and best.fov_y > 1) or \
            best.elevation in (elevation_range[0], elevation_range[-1])
        if not on_edge:
            break
    return best


def hull_points(spin: np.ndarray, camera: SpinCamera, device, mask_width=320, resolution=200) -> torch.Tensor:
    masks = alpha_masks(spin, mask_width, device)
    grid, inside, _ = carve(camera, masks, resolution)
    return grid[inside]


def initial_splats(spin: np.ndarray, camera: SpinCamera, device, max_points: int,
                   mask_width=400, resolution=200) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Training's first splats: the hull's surface, a voxel apart, each coloured by
    averaging what the frames show there."""
    masks = alpha_masks(spin, mask_width, device)
    grid, inside, spacing = carve(camera, masks, resolution)
    occupied = inside.view(resolution, resolution, resolution)
    padded = F.pad(occupied[None, None].float(), (1, 1, 1, 1, 1, 1))[0, 0]
    neighbours = sum(padded[1 + dx:resolution + 1 + dx, 1 + dy:resolution + 1 + dy, 1 + dz:resolution + 1 + dz]
                     for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)))
    points = grid[(occupied & (neighbours < 6)).reshape(-1)]
    if len(points) > max_points:
        points = points[torch.randperm(len(points), device=device)[:max_points]]
    count, height, width = spin.shape[:3]
    images = torch.from_numpy(spin).to(device)
    K = intrinsics(camera.fov_y, width, height).to(device)
    viewmats = turntable_viewmats(range(count), count, camera.elevation, camera.direction).to(device)
    total = torch.zeros(len(points), 3, device=device)
    weight = torch.zeros(len(points), 1, device=device)
    for view in range(count):
        uv, z = project(points, viewmats[view], K)
        pixel = uv.floor().long()
        on = (z > 0) & (pixel[:, 0] >= 0) & (pixel[:, 0] < width) & (pixel[:, 1] >= 0) & (pixel[:, 1] < height)
        sample = images[view, pixel[on, 1], pixel[on, 0]]
        alpha = sample[:, 3:]
        total[on] += sample[:, :3] * alpha
        weight[on] += alpha
    colours = (total / weight.clamp(min=1e-6)).clamp(0, 1)
    return points, colours, spacing
