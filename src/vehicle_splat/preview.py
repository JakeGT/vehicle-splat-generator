"""A turntable video of a trained splat, rendered by gsplat itself: once round at
the spin's own height, as the portal shows it, then once from above, where the
extra views are what the roof was learnt from."""
from __future__ import annotations

import pathlib

import torch

from .camera import DISTANCE, intrinsics, orbit_viewmat, spin_yaw
from .hull import SpinCamera
from .train import render

# The portal's page background (#e8eef5), so the preview looks as the card will.
PAGE = (0xE8 / 255, 0xEE / 255, 0xF5 / 255)
RAISED_ELEVATION = 25.0


def write_turntable(path: pathlib.Path, params, camera: SpinCamera, frame_count: int, sh_degree: int,
                    width: int = 800, height: int = 500) -> None:
    import imageio.v2 as imageio

    device = params["means"].device
    K = intrinsics(camera.fov_y, width, height).to(device)
    background = torch.tensor(PAGE, device=device)
    # Four angles between every pair of the spin's frames; half as many from above.
    rings = [(camera.elevation, [p / 4 for p in range(frame_count * 4)]),
             (RAISED_ELEVATION, [p / 2 for p in range(frame_count * 2)])]
    writer = imageio.get_writer(path, fps=30, quality=8, macro_block_size=2)
    try:
        with torch.no_grad():
            for elevation, positions in rings:
                for position in positions:
                    yaw = spin_yaw(position, frame_count, camera.direction)
                    viewmat = orbit_viewmat(yaw, elevation, DISTANCE).float().to(device)
                    image, _, _ = render(params, viewmat, K, width, height, background, sh_degree)
                    writer.append_data((image.clamp(0, 1) * 255).byte().cpu().numpy())
    finally:
        writer.close()
