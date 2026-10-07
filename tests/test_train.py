"""The parts of training that need no gsplat: it is imported only where it renders."""
import math

import torch

from vehicle_splat.camera import orbit_viewmat
from vehicle_splat.train import SH_C0, _window, initial_params, refined_viewmat, rotation_from, ssim


def test_no_turn_is_no_rotation_and_a_quarter_turn_is_one():
    assert torch.allclose(rotation_from(torch.zeros(3)), torch.eye(3), atol=1e-6)
    quarter = rotation_from(torch.tensor([0.0, 0.0, math.pi / 2]))
    assert torch.allclose(quarter @ torch.tensor([1.0, 0, 0]), torch.tensor([0.0, 1, 0]), atol=1e-6)


def test_a_pose_nudge_can_be_learnt_from_nothing():
    viewmat = orbit_viewmat(30, 10).float()
    delta = torch.zeros(6, requires_grad=True)
    nudged = refined_viewmat(viewmat, delta)
    assert torch.allclose(nudged, viewmat, atol=1e-6)
    nudged[0, 3].backward()
    # A shift along the camera's own x moves its x translation one for one.
    assert math.isclose(delta.grad[3].item(), 1.0, rel_tol=1e-5)
    assert torch.isfinite(delta.grad).all()


def test_a_shift_moves_the_camera_in_its_own_frame():
    viewmat = orbit_viewmat(70, 20).float()
    nudged = refined_viewmat(viewmat, torch.tensor([0.0, 0, 0, 0.1, -0.2, 0.3]))
    assert torch.allclose(nudged[:3, :3], viewmat[:3, :3], atol=1e-6)
    assert torch.allclose(nudged[:3, 3] - viewmat[:3, 3], torch.tensor([0.1, -0.2, 0.3]), atol=1e-6)


def test_ssim_is_one_for_the_same_picture_and_less_for_another():
    generator = torch.Generator().manual_seed(1)
    image = torch.rand(32, 48, 3, generator=generator)
    window = _window(torch.device("cpu"))
    assert math.isclose(ssim(image, image, window).item(), 1.0, rel_tol=1e-5)
    assert ssim(image, torch.rand(32, 48, 3, generator=generator), window).item() < 0.2


def test_first_params_hold_the_hulls_colours():
    means = torch.rand(10, 3)
    colours = torch.rand(10, 3)
    params = initial_params(means, colours, 0.01, 3)
    assert params["shN"].shape == (10, 15, 3)
    assert torch.allclose(0.5 + SH_C0 * params["sh0"][:, 0], colours, atol=1e-6)
    assert torch.allclose(torch.sigmoid(params["opacities"]), torch.full((10,), 0.5))
    assert torch.allclose(torch.exp(params["scales"]), torch.full((10, 3), 0.007))
    # Its own copy: training moves the splats, not the hull it started from.
    params["means"].data += 1
    assert not torch.equal(params["means"], means)
