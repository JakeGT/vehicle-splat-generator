"""The cameras Imagin renders a van from.

The world is OpenCV's, as gsplat trains in: x right, **y down**, z forward. The
van turns on world y. A camera orbits that axis: at a yaw round it, raised by an
elevation, a distance from what it looks at, with square pixels and the
principal point in the middle of the picture.

The 360° spin is one such camera at a fixed elevation and distance: frame k is
at yaw direction * k * 360° / frames. The customer portal rebuilds exactly that
camera from a model's manifest (CRM repo,
packages/ui/src/vehicleSpin/splat/turntableCamera.ts); keep the two the same.
tests/test_camera.py pins the matrix both sides are tested against."""
from __future__ import annotations

import math

import torch

# The spin camera's distance from the axis. Only the scene's scale follows from it.
DISTANCE = 4.0


def orbit_viewmat(yaw_deg: float, elevation_deg: float, distance: float = DISTANCE,
                  target: torch.Tensor | None = None) -> torch.Tensor:
    """The world-to-camera matrix [4, 4] (float64) of a camera `distance` from
    `target`, at `yaw_deg` round the axis and `elevation_deg` above it, looking at it."""
    yaw, phi = math.radians(yaw_deg), math.radians(elevation_deg)
    forward = torch.tensor([math.cos(phi) * math.sin(yaw), math.sin(phi), math.cos(phi) * math.cos(yaw)],
                           dtype=torch.float64)
    target = torch.zeros(3, dtype=torch.float64) if target is None else target.to(torch.float64)
    centre = target - distance * forward
    down = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    if abs(forward[1].item()) > 1 - 1e-9:
        # Straight down there is no "down" to keep; the picture's down is where
        # it tends from just short of straight down, so the yaw still turns it.
        down = torch.tensor([-math.sin(yaw), 0.0, -math.cos(yaw)], dtype=torch.float64)
    y_axis = down - (down @ forward) * forward
    y_axis = y_axis / y_axis.norm()
    x_axis = torch.linalg.cross(y_axis, forward)
    rotation = torch.stack([x_axis, y_axis, forward])
    viewmat = torch.eye(4, dtype=torch.float64)
    viewmat[:3, :3] = rotation
    viewmat[:3, 3] = -(rotation @ centre)
    return viewmat


def spin_yaw(position: float, count: int, direction: int) -> float:
    """The yaw, in degrees, of the spin's frame `position` (fractional between frames)."""
    return direction * position * 360.0 / count


def turntable_viewmats(positions, count: int, elevation_deg: float, direction: int) -> torch.Tensor:
    """The spin camera at each (fractional) frame position: [P, 4, 4] float32."""
    return torch.stack([orbit_viewmat(spin_yaw(float(position), count, direction), elevation_deg)
                        for position in positions]).float()


def focal_length(fov_y_deg: float, height: float) -> float:
    return (height / 2) / math.tan(math.radians(fov_y_deg) / 2)


def intrinsics(fov_y_deg: float, width: float, height: float) -> torch.Tensor:
    focal = focal_length(fov_y_deg, height)
    return torch.tensor([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=torch.float32)


def project(points: torch.Tensor, viewmat: torch.Tensor, K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixel positions [N, 2] and depths [N] of world points [N, 3]."""
    viewmat = viewmat.to(points)
    camera = points @ viewmat[:3, :3].T + viewmat[:3, 3]
    z = camera[:, 2]
    uv = camera[:, :2] / z.clamp(min=1e-6).unsqueeze(-1) * K[0, 0].to(points) + K[:2, 2].to(points)
    return uv, z
