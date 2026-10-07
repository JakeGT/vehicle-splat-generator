import math

import numpy as np
import torch

from conftest import BOXES, SPIN_CAMERA
from vehicle_splat.hull import SpinCamera, alpha_masks, carve, hull_score, initial_splats, search

CPU = torch.device("cpu")


def inside_a_box(points: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    inside = torch.zeros(len(points), dtype=torch.bool)
    for low, high, _ in BOXES:
        inside |= ((points >= torch.tensor(low) - margin) & (points <= torch.tensor(high) + margin)).all(-1)
    return inside


def test_the_right_camera_is_found_among_the_guesses(spin):
    masks = alpha_masks(spin, 80, CPU)
    best = search(masks, 48, [8, 10, 12, 14, 16], [2, 4, 6, 8, 10], (1, -1))
    assert best.direction == SPIN_CAMERA.direction
    assert abs(best.fov_y - SPIN_CAMERA.fov_y) <= 2
    assert abs(best.elevation - SPIN_CAMERA.elevation) <= 2
    assert best.score > 0.85


def test_the_wrong_direction_fills_the_outlines_worse(spin):
    masks = alpha_masks(spin, 80, CPU)
    right = hull_score(SPIN_CAMERA, masks, 48)
    wrong = hull_score(SpinCamera(SPIN_CAMERA.fov_y, SPIN_CAMERA.elevation, -SPIN_CAMERA.direction), masks, 48)
    assert right > wrong + 0.05


def test_the_hull_holds_the_van_and_little_else(spin):
    grid, inside, spacing = carve(SPIN_CAMERA, alpha_masks(spin, 160, CPU), 96)
    in_van = inside_a_box(grid)
    # All of the van survives; what else does is near it (a hull is never tighter
    # than the van, and carving from one ring of cameras leaves some to spare).
    assert inside[in_van].float().mean() > 0.99
    assert inside_a_box(grid[inside], margin=0.12).all()
    assert math.isclose(spacing, (grid[1, 2] - grid[0, 2]).item(), rel_tol=1e-5)


def test_first_splats_are_on_the_surface_and_take_its_colours(spin):
    points, colours, spacing = initial_splats(spin, SPIN_CAMERA, CPU, 5000, 160, 80)
    assert len(points) == len(colours) <= 5000
    assert inside_a_box(points, margin=0.12).all()
    # The body is near white and the cab blue: most splats are the body's.
    body = torch.tensor(BOXES[0][2])
    assert ((colours - body).abs().max(-1).values < 0.1).float().mean() > 0.5
    assert spacing > 0


def test_outlines_ignore_a_faint_shadow():
    images = np.zeros((1, 20, 40, 4), dtype=np.float32)
    images[0, 5:15, 10:30, 3] = 1.0
    images[0, 15:18, 10:30, 3] = 0.3
    mask = alpha_masks(images, 40, CPU)[0]
    assert mask[5:15, 10:30].all()
    assert not mask[15:].any()
