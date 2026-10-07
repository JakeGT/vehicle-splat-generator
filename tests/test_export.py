import json
import math

import numpy as np
import torch

from vehicle_splat.export import SPLAT_RECORD, write_model
from vehicle_splat.hull import SpinCamera
from vehicle_splat.train import SH_C0


def params(sh_degree: int = 3):
    count = 4
    opacities = torch.tensor([0.9, 0.5, 0.002, 0.8])
    return {
        "means": torch.arange(count * 3, dtype=torch.float32).view(count, 3),
        "scales": torch.log(torch.tensor([[0.01] * 3, [0.02] * 3, [0.05] * 3, [0.001] * 3])),
        "quats": torch.tensor([[2.0, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]]),
        "opacities": torch.logit(opacities),
        "sh0": ((torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.5]]) - 0.5) / SH_C0)[:, None],
        "shN": torch.zeros(count, (sh_degree + 1) ** 2 - 1, 3),
    }


def test_the_splat_file_is_what_the_portal_and_viewers_read(tmp_path):
    camera = SpinCamera(9.0, 1.0, -1).as_manifest(32)
    model = write_model(tmp_path, "58438", params(), camera, {"imagin": {"capId": 58438}})
    records = np.frombuffer((tmp_path / "58438.splat").read_bytes(), SPLAT_RECORD)
    assert SPLAT_RECORD.itemsize == 32
    # The all-but-transparent splat is dropped; the rest go biggest and most opaque first.
    assert len(records) == model["splatCount"] == 3
    assert records["position"].tolist() == [[3, 4, 5], [0, 1, 2], [9, 10, 11]]
    assert np.allclose(records["scale"][0], 0.02)
    assert records["colour"].tolist() == [[0, 255, 0, 128], [255, 0, 0, 230], [128, 128, 128, 204]]
    # Rotations normalised, then w x y z each as q * 128 + 128.
    assert records["rotation"].tolist() == [[128, 255, 128, 128], [255, 128, 128, 128], [255, 128, 128, 128]]


def test_the_manifest_points_at_the_splat_and_carries_the_camera(tmp_path):
    camera = SpinCamera(9.0, 1.0, -1).as_manifest(32)
    write_model(tmp_path, "58438-pwhite", params(), camera, {"imagin": {"capId": 58438}})
    manifest = json.loads((tmp_path / "58438-pwhite.json").read_text())
    assert manifest["version"] == 1 and manifest["format"] == "splat"
    assert manifest["splat"] == "58438-pwhite.splat"
    assert manifest["splatCount"] == 3
    assert manifest["camera"] == {"convention": "opencv-turntable", "fovYDegrees": 9.0, "elevationDegrees": 1.0,
                                  "direction": -1, "distance": 4.0, "frameCount": 32}
    assert manifest["source"] == {"imagin": {"capId": 58438}}


def test_the_ply_is_standard_3dgs_with_activations_undone(tmp_path):
    write_model(tmp_path, "van", params(), {}, {})
    data = (tmp_path / "van.ply").read_bytes()
    header, body = data.split(b"end_header\n", 1)
    lines = header.decode().splitlines()
    assert "element vertex 3" in lines
    names = [line.split()[-1] for line in lines if line.startswith("property float")]
    assert len(names) == 3 + 3 + 3 + 45 + 1 + 3 + 4
    values = np.frombuffer(body, "<f4").reshape(3, len(names))
    first = dict(zip(names, values[0]))
    assert math.isclose(first["opacity"], math.log(0.5 / 0.5), abs_tol=1e-5)
    assert math.isclose(first["scale_0"], math.log(0.02), rel_tol=1e-5)
    assert math.isclose(first["f_dc_1"], 0.5 / SH_C0, rel_tol=1e-5)
