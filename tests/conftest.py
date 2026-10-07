"""A made-up van, rendered exactly by ray casting, so the camera fitting and view
placing can be checked against cameras that are known. It is two boxes: a body
and a cab set to one side, so no turn or mirror of it looks the same as another."""
from __future__ import annotations

import io
import urllib.parse

import numpy as np
import pytest
import torch
from PIL import Image

from vehicle_splat.camera import focal_length, orbit_viewmat, turntable_viewmats
from vehicle_splat.hull import SpinCamera
from vehicle_splat.imagin import SPIN_ANGLES

# (low corner, high corner, colour). World y is down, so the cab's y is above the body's.
BOXES = (
    ((-0.22, -0.25, -0.55), (0.22, 0.15, 0.55), (0.88, 0.88, 0.92)),
    ((-0.22, -0.38, 0.15), (0.05, -0.25, 0.55), (0.15, 0.3, 0.8)),
)
SPIN_CAMERA = SpinCamera(fov_y=12.0, elevation=6.0, direction=-1)
SPIN_SIZE = (160, 100)


def render_van(viewmat: torch.Tensor, fov_y: float, width: int, height: int) -> np.ndarray:
    """The van from a camera, as straight RGBA [H, W, 4] in 0..1 on a transparent background."""
    viewmat = viewmat.double()
    rotation, translation = viewmat[:3, :3], viewmat[:3, 3]
    centre = -(rotation.T @ translation)
    focal = focal_length(fov_y, height)
    v, u = torch.meshgrid(torch.arange(height, dtype=torch.float64) + 0.5,
                          torch.arange(width, dtype=torch.float64) + 0.5, indexing="ij")
    direction = torch.stack([(u - width / 2) / focal, (v - height / 2) / focal, torch.ones_like(u)], -1) @ rotation
    image = torch.zeros(height, width, 4, dtype=torch.float64)
    nearest = torch.full((height, width), float("inf"), dtype=torch.float64)
    for low, high, colour in BOXES:
        near = (torch.tensor(low, dtype=torch.float64) - centre) / direction
        far = (torch.tensor(high, dtype=torch.float64) - centre) / direction
        entry = torch.minimum(near, far).amax(-1)
        leave = torch.maximum(near, far).amin(-1)
        closer = (leave >= entry) & (leave > 0) & (entry < nearest)
        nearest = torch.where(closer, entry, nearest)
        image[closer] = torch.tensor((*colour, 1.0), dtype=torch.float64)
    return image.float().numpy()


def render_spin(camera: SpinCamera = SPIN_CAMERA, size=SPIN_SIZE) -> np.ndarray:
    width, height = size
    viewmats = turntable_viewmats(range(32), 32, camera.elevation, camera.direction)
    return np.stack([render_van(viewmat, camera.fov_y, width, height) for viewmat in viewmats])


@pytest.fixture(scope="session")
def spin() -> np.ndarray:
    return render_spin()


def png(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray((image * 255).round().astype(np.uint8), "RGBA").save(buffer, "PNG")
    return buffer.getvalue()


# The fake Imagin's other views: (yaw, elevation, distance, target). They are
# framed differently from the spin and aimed a little off the van's middle, as
# Imagin's are, and show all of it. Every other code repeats code 1, as Imagin
# does for a code a van lacks.
VIEWS = {
    1: (35.0, 4.0, 4.2, (0.02, 0.0, 0.0)),
    5: (-120.0, 5.0, 4.4, (0.0, -0.02, 0.03)),
    27: (60.0, 31.0, 5.6, (0.0, 0.0, 0.02)),
    33: (30.0, 90.0, 6.8, (0.03, 0.0, 0.0)),
}


def render_view(code: int, width: int) -> np.ndarray:
    yaw, elevation, distance, target = VIEWS[code]
    image = render_van(orbit_viewmat(yaw, elevation, distance, torch.tensor(target)), SPIN_CAMERA.fov_y,
                       width, round(width * 5 / 8))
    alpha = image[..., 3]
    assert not (alpha[0].any() or alpha[-1].any() or alpha[:, 0].any() or alpha[:, -1].any()), \
        f"view {code} runs off its picture"
    return image


class FakeImagin:
    """Imagin's CDN, as a fetch function: renders the made-up van for each URL."""

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
        angle, width = int(query["angle"]), int(query["width"])
        height = round(width * 5 / 8)
        if angle in SPIN_ANGLES:
            viewmat = turntable_viewmats([angle - SPIN_ANGLES[0]], 32, SPIN_CAMERA.elevation, SPIN_CAMERA.direction)[0]
            return png(render_van(viewmat, SPIN_CAMERA.fov_y, width, height))
        return png(render_view(angle if angle in VIEWS else 1, width))


@pytest.fixture
def fake_imagin() -> FakeImagin:
    return FakeImagin()
