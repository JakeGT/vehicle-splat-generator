import math

import torch

from vehicle_splat.camera import (DISTANCE, focal_length, intrinsics, orbit_viewmat, project, spin_yaw,
                                  turntable_viewmats)

# The spin camera half way between frames 3 and 4 of 32, 8° up, turning the
# negative way: column-major, as WebGL takes it. The CRM repo's portal test
# (apps/portal/src/components/deals/vehicleSplat.test.ts) pins the same numbers for its
# turntableCamera.ts, so a change to either side's convention breaks one of them.
PORTAL_MATRIX = [0.77301, 0.08829, -0.628219, 0, -0, 0.990268, 0.139173, 0,
                 0.634393, -0.107582, 0.765488, 0, -0, -0, 4, 1]


def test_matches_the_portals_turntable_camera():
    viewmat = turntable_viewmats([3.5], 32, 8, -1)[0]
    assert torch.allclose(viewmat.T.flatten(), torch.tensor(PORTAL_MATRIX), atol=1e-5)


def test_a_view_matrix_is_a_rotation_and_looks_at_its_target():
    target = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    viewmat = orbit_viewmat(-63.0, 21.0, 3.5, target)
    rotation = viewmat[:3, :3]
    assert torch.allclose(rotation @ rotation.T, torch.eye(3, dtype=torch.float64), atol=1e-12)
    assert math.isclose(torch.linalg.det(rotation).item(), 1.0, abs_tol=1e-12)
    in_camera = rotation @ target + viewmat[:3, 3]
    assert torch.allclose(in_camera, torch.tensor([0, 0, 3.5], dtype=torch.float64), atol=1e-12)


def test_elevation_raises_the_camera_and_the_picture_stays_upright():
    # y is down: a camera above the van has a negative y, and the van's top
    # (more negative y) is still higher in the picture.
    viewmat = orbit_viewmat(0.0, 30.0)
    centre = -(viewmat[:3, :3].T @ viewmat[:3, 3])
    assert centre[1] < 0
    K = intrinsics(20, 200, 100).double()
    uv, _ = project(torch.tensor([[0.0, -0.3, 0.0], [0.0, 0.3, 0.0]], dtype=torch.float64), viewmat, K)
    assert uv[0, 1] < uv[1, 1]


def test_straight_down_turns_with_the_yaw():
    # Straight down there is no up; the picture still turns as the yaw does,
    # continuing from just short of straight down.
    for yaw in (0.0, 40.0, 200.0):
        top = orbit_viewmat(yaw, 90.0)
        near = orbit_viewmat(yaw, 89.9)
        assert torch.allclose(top[:3, :3], near[:3, :3], atol=5e-3)


def test_spin_frames_go_round_once():
    assert spin_yaw(0, 32, -1) == 0
    assert spin_yaw(8, 32, 1) == 90
    assert spin_yaw(8, 32, -1) == -90
    first, last = turntable_viewmats([0, 32], 32, 5, 1)
    assert torch.allclose(first, last, atol=1e-6)


def test_the_lens_has_square_pixels_and_a_centred_principal_point():
    assert math.isclose(focal_length(90, 2), 1.0)
    K = intrinsics(90, 300, 2)
    assert K[0, 0] == K[1, 1] == 1.0
    assert (K[0, 2], K[1, 2]) == (150, 1)


def test_the_axis_is_in_the_middle_of_every_spin_frame():
    K = intrinsics(12, 160, 100)
    for viewmat in turntable_viewmats(range(0, 32, 5), 32, 6, -1):
        uv, z = project(torch.zeros(1, 3), viewmat, K)
        assert torch.allclose(uv[0], torch.tensor([80.0, 50.0]), atol=1e-4)
        assert math.isclose(z.item(), DISTANCE, rel_tol=1e-6)
