"""Training the splat with gsplat (https://github.com/nerfstudio-project/gsplat).

Each step renders one view's camera over a random background colour and compares
it with that view laid over the same colour, so the space round the van has to
stay transparent. The loss is L1 and SSIM, plus the outline (alpha) and MCMC's
small opacity and scale penalties. MCMC densification keeps the count under
`max_splats`, which is what keeps the web file small.

The spin's cameras are fitted and stay put. The extra views' cameras were only
placed by their outline (views.py), so each gets a small rotation and shift that
training refines, once the van has taken shape."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from .camera import DISTANCE

SH_C0 = 0.28209479177387814
# Pose refinement waits for the shape to settle first.
POSE_REFINE_FROM = 500


@dataclass
class TrainingView:
    # Straight (not premultiplied) RGBA in 0..1, [H, W, 4], on the training device.
    image: torch.Tensor
    viewmat: torch.Tensor
    K: torch.Tensor
    refine_pose: bool = False

    @property
    def size(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]


Progress = Callable[[int, int, float, int], None]


def initial_params(means: torch.Tensor, colours: torch.Tensor, spacing: float, sh_degree: int) -> torch.nn.ParameterDict:
    count, device = len(means), means.device
    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means.clone()),
        "scales": torch.nn.Parameter(torch.full((count, 3), math.log(spacing * 0.7), device=device)),
        "quats": torch.nn.Parameter(torch.tensor([1.0, 0, 0, 0], device=device).repeat(count, 1)),
        # 0.5, as gsplat's MCMC preset starts.
        "opacities": torch.nn.Parameter(torch.zeros(count, device=device)),
        "sh0": torch.nn.Parameter(((colours - 0.5) / SH_C0)[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros(count, (sh_degree + 1) ** 2 - 1, 3, device=device)),
    })


def render(params, viewmat: torch.Tensor, K: torch.Tensor, width: int, height: int,
           background: torch.Tensor, sh_degree: int):
    """The view's picture [H, W, 3] over `background`, its alpha [H, W, 1], and
    gsplat's working (for densification)."""
    from gsplat import rasterization

    colours = torch.cat([params["sh0"], params["shN"]], dim=1)
    image, alpha, info = rasterization(
        params["means"], params["quats"], torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
        colours, viewmat[None], K[None], width, height, sh_degree=sh_degree,
        backgrounds=background[None], packed=False)
    return image[0], alpha[0], info


def _window(device, size=11, sigma=1.5):
    x = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).expand(3, 1, size, size).contiguous()


def ssim(image: torch.Tensor, target: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    x, y = image.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]
    blur = lambda t: F.conv2d(t, window, padding=window.shape[-1] // 2, groups=3)  # noqa: E731
    mu_x, mu_y = blur(x), blur(y)
    sigma_x, sigma_y = blur(x * x) - mu_x ** 2, blur(y * y) - mu_y ** 2
    sigma_xy = blur(x * y) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) /
            ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))).mean()


def rotation_from(vector: torch.Tensor) -> torch.Tensor:
    """The rotation [3, 3] by `vector`'s length about its direction (Rodrigues)."""
    theta = torch.sqrt((vector * vector).sum() + 1e-12)
    x, y, z = vector
    zero = torch.zeros((), device=vector.device)
    skew = torch.stack([torch.stack([zero, -z, y]), torch.stack([z, zero, -x]), torch.stack([-y, x, zero])])
    return (torch.eye(3, device=vector.device) + torch.sin(theta) / theta * skew
            + (1 - torch.cos(theta)) / theta ** 2 * skew @ skew)


def refined_viewmat(viewmat: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """A camera nudged by `delta` (a rotation vector, then a shift), in its own frame."""
    nudge = torch.eye(4, device=viewmat.device)
    nudge = torch.cat([torch.cat([rotation_from(delta[:3]), delta[3:, None]], 1), nudge[3:]], 0)
    return nudge @ viewmat


def train(views: list[TrainingView], params: torch.nn.ParameterDict, steps: int, max_splats: int,
          sh_degree: int, progress: Progress | None = None) -> tuple[torch.nn.ParameterDict, list[torch.Tensor]]:
    """Trains `params` on `views`; returns them and each view's camera as refined."""
    from gsplat.strategy import MCMCStrategy

    device = params["means"].device
    window = _window(device)
    # gsplat's default: the cameras' spread, a little over.
    scene_scale = 1.1 * DISTANCE
    rates = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
             "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {name: torch.optim.Adam([params[name]], lr=rate, eps=1e-15) for name, rate in rates.items()}
    means_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1 / steps))
    strategy = MCMCStrategy(cap_max=max_splats, refine_stop_iter=int(steps * 0.85))
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()

    refined = [index for index, view in enumerate(views) if view.refine_pose]
    deltas = torch.nn.Parameter(torch.zeros(len(refined), 6, device=device))
    # About half a pixel of turn, and a pixel of shift, a step at most.
    pose_optimizer = torch.optim.Adam([{"params": [deltas], "lr": 1e-4}], eps=1e-15)
    pose_schedule = torch.optim.lr_scheduler.ExponentialLR(pose_optimizer, gamma=0.1 ** (1 / steps))
    shift_scale = torch.tensor([1.0, 1, 1, 10, 10, 10], device=device)
    slot = {index: position for position, index in enumerate(refined)}

    for step in range(steps):
        index = torch.randint(len(views), (1,)).item()
        view = views[index]
        width, height = view.size
        viewmat = view.viewmat
        if view.refine_pose:
            viewmat = refined_viewmat(viewmat, deltas[slot[index]] * shift_scale)
        background = torch.rand(3, device=device)
        image, alpha, info = render(params, viewmat, view.K, width, height, background,
                                    min(step // 1000, sh_degree))
        target_alpha = view.image[..., 3:]
        target = view.image[..., :3] * target_alpha + background * (1 - target_alpha)
        loss = (0.8 * (image - target).abs().mean() + 0.2 * (1 - ssim(image, target, window))
                + 0.1 * (alpha - target_alpha).abs().mean()
                + 0.01 * torch.sigmoid(params["opacities"]).mean() + 0.01 * torch.exp(params["scales"]).mean())
        strategy.step_pre_backward(params, optimizers, state, step, info)
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if refined and step >= POSE_REFINE_FROM:
            pose_optimizer.step()
        pose_optimizer.zero_grad(set_to_none=True)
        means_schedule.step()
        pose_schedule.step()
        strategy.step_post_backward(params, optimizers, state, step, info, lr=means_schedule.get_last_lr()[0])
        if progress and (step % 50 == 0 or step == steps - 1):
            progress(step + 1, steps, loss.item(), len(params["means"]))

    with torch.no_grad():
        cameras = [refined_viewmat(view.viewmat, deltas[slot[index]] * shift_scale) if view.refine_pose
                   else view.viewmat for index, view in enumerate(views)]
    return params, cameras
