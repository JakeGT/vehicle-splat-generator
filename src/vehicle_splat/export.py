"""Writing a trained splat out: the files the customer portal and splat viewers read."""
from __future__ import annotations

import datetime
import json
import pathlib

import numpy as np
import torch
import torch.nn.functional as F

from .train import SH_C0

# The .splat layout: per splat, position (3 float32), scale (3 float32), colour
# RGBA (4 uint8: base colour, then opacity) and rotation (4 uint8: w x y z, each
# q * 128 + 128). 32 bytes; the portal reads it in packages/ui/src/vehicleSpin/splat/splatModel.ts.
SPLAT_RECORD = np.dtype([("position", "<f4", 3), ("scale", "<f4", 3), ("colour", "u1", 4), ("rotation", "u1", 4)])


def splat_arrays(params) -> dict[str, np.ndarray]:
    """The visible splats, biggest and most opaque first -- as browser viewers
    expect, so a part-loaded file already looks like the van."""
    means = params["means"].detach().float().cpu().numpy()
    scales = torch.exp(params["scales"]).detach().float().cpu().numpy()
    quats = F.normalize(params["quats"].detach().float(), dim=-1).cpu().numpy()
    opacities = torch.sigmoid(params["opacities"]).detach().float().cpu().numpy()
    sh0 = params["sh0"].detach().float().cpu().numpy()[:, 0, :]
    shN = params["shN"].detach().float().cpu().numpy()
    visible = np.flatnonzero(opacities > 1 / 255)
    order = visible[np.argsort(-(scales.prod(axis=1) * opacities)[visible])]
    return {"means": means[order], "scales": scales[order], "quats": quats[order],
            "opacities": opacities[order], "sh0": sh0[order], "shN": shN[order]}


def write_splat(path: pathlib.Path, splats: dict[str, np.ndarray]) -> None:
    data = np.empty(len(splats["means"]), SPLAT_RECORD)
    data["position"] = splats["means"]
    data["scale"] = splats["scales"]
    colour = np.concatenate([0.5 + SH_C0 * splats["sh0"], splats["opacities"][:, None]], axis=1)
    data["colour"] = (colour * 255).round().clip(0, 255).astype(np.uint8)
    data["rotation"] = (splats["quats"] * 128 + 128).round().clip(0, 255).astype(np.uint8)
    path.write_bytes(data.tobytes())


def write_ply(path: pathlib.Path, splats: dict[str, np.ndarray]) -> None:
    """The standard 3D Gaussian Splatting PLY: activations undone (log scales,
    logit opacity), the higher spherical-harmonic bands channel by channel."""
    count = len(splats["means"])
    rest = splats["shN"].transpose(0, 2, 1).reshape(count, -1)
    names = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
             + [f"f_rest_{i}" for i in range(rest.shape[1])]
             + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])
    opacity = splats["opacities"].clip(1e-6, 1 - 1e-6)
    values = np.concatenate([splats["means"], np.zeros_like(splats["means"]), splats["sh0"], rest,
                             np.log(opacity / (1 - opacity))[:, None], np.log(splats["scales"]),
                             splats["quats"]], axis=1)
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {count}\n"
    header += "".join(f"property float {name}\n" for name in names) + "end_header\n"
    path.write_bytes(header.encode() + values.astype("<f4").tobytes())


def manifest(name: str, splat_count: int, camera: dict, source: dict) -> dict:
    """What the portal reads first (splatModel.ts's parseManifest): the .splat to
    load, and the spin's camera, so the model is framed as Imagin's renders are."""
    return {
        "version": 1,
        "format": "splat",
        "splat": f"{name}.splat",
        "splatCount": int(splat_count),
        "camera": camera,
        "source": source,
        "createdAt": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }


def write_model(directory: pathlib.Path, name: str, params, camera: dict, source: dict) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    splats = splat_arrays(params)
    write_splat(directory / f"{name}.splat", splats)
    write_ply(directory / f"{name}.ply", splats)
    model = manifest(name, len(splats["means"]), camera, source)
    (directory / f"{name}.json").write_text(json.dumps(model, indent=2) + "\n")
    return model
