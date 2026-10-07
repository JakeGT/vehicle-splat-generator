import pytest
import torch

from conftest import SPIN_CAMERA, VIEWS, render_view
from vehicle_splat.hull import hull_points
from vehicle_splat.views import ViewCamera, fit_camera, is_from_above, place_view, view_masks

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def points(spin):
    points = hull_points(spin, SPIN_CAMERA, CPU, 160, 96)
    generator = torch.Generator().manual_seed(0)
    return points[torch.randperm(len(points), generator=generator)[:30_000]]


def view_mask(code: int, width: int = 200):
    return view_masks({code: render_view(code, width)}, width, CPU)[code]


def yaw_apart(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def test_fitting_at_the_right_angle_finds_the_framing(points):
    yaw, elevation, distance, target = VIEWS[27]
    camera = fit_camera(points, view_mask(27), yaw, elevation, SPIN_CAMERA.fov_y)
    assert camera.outline_iou > 0.85
    # The hull is a little bigger than the van, so the fit stands a little back.
    assert distance * 0.95 < camera.distance < distance * 1.15


@pytest.mark.parametrize("code", [5, 27])
def test_a_view_is_placed_where_it_was_taken(points, code):
    yaw, elevation, _, _ = VIEWS[code]
    placed = place_view(points, view_mask(code), SPIN_CAMERA.fov_y, code)
    # The outline may prefer the mirror image; the pipeline tells them apart by
    # their pictures, so either candidate may be the right one.
    candidates = [placed, placed.alternative]
    assert any(yaw_apart(c.yaw, yaw) <= 6 and abs(c.elevation - elevation) <= 4 for c in candidates)
    assert yaw_apart(placed.yaw, placed.alternative.yaw) >= 120
    assert placed.outline_iou > 0.8


def test_from_straight_above_the_alternative_is_turned_half_round(points):
    placed = place_view(points, view_mask(33), SPIN_CAMERA.fov_y, 33)
    assert placed.elevation >= 75
    assert yaw_apart(placed.yaw + 180, placed.alternative.yaw) < 1e-6
    assert any(yaw_apart(c.yaw, VIEWS[33][0]) <= 8 for c in (placed, placed.alternative))


def test_from_above_means_clearly_above_the_spin():
    camera = lambda elevation: ViewCamera(1, 0, elevation, 4, torch.zeros(3), 12, 1)  # noqa: E731
    assert not is_from_above(camera(10), 6)
    assert is_from_above(camera(14), 6)
    assert is_from_above(camera(90), 6)
