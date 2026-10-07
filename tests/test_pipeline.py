"""The pipeline's stages end to end, without a GPU: the made-up van comes from a
fake Imagin, and gsplat's training and rendering are replaced by stand-ins, so
what is checked is everything round them -- the camera fit carried into the
manifest, which views are used and why, and the files written."""
import json
import urllib.parse

import torch

import vehicle_splat.train
from conftest import SPIN_CAMERA
from vehicle_splat import hull, pipeline
from vehicle_splat.imagin import SPIN_ANGLES, Imagin
from vehicle_splat.pipeline import JobSpec, run

CPU = torch.device("cpu")


class Recorder:
    def __init__(self):
        self.stages, self.steps = [], []

    def stage(self, name, status, detail=None):
        self.stages.append((name, status))

    def training(self, step, steps, loss, splats):
        self.steps.append(step)


def stand_ins(monkeypatch):
    # The camera fit is tested on its own (test_hull.py); a full search is minutes on a CPU.
    monkeypatch.setattr(pipeline, "fit_spin_camera", lambda spin, device, progress: SPIN_CAMERA)
    monkeypatch.setattr(pipeline, "hull_points", lambda spin, camera, device: hull.hull_points(spin, camera, device, 160, 64))
    monkeypatch.setattr(pipeline, "initial_splats", lambda spin, camera, device, most: hull.initial_splats(
        spin, camera, device, min(most, 3000), 160, 64))
    trained = []

    def train(views, params, steps, max_splats, sh_degree, progress=None):
        trained.append(views)
        if progress:
            progress(steps, steps, 0.05, len(params["means"]))
        return params, [view.viewmat for view in views]

    def render(params, viewmat, K, width, height, background, sh_degree):
        return background.expand(height, width, 3), torch.zeros(height, width, 1), {}

    monkeypatch.setattr(vehicle_splat.train, "train", train)
    monkeypatch.setattr(vehicle_splat.train, "render", render)
    monkeypatch.setattr(pipeline, "write_turntable", lambda path, *rest: path.write_bytes(b"video"))
    return trained


def test_a_splat_from_the_spin_and_the_views_from_above(tmp_path, fake_imagin, monkeypatch):
    trained = stand_ins(monkeypatch)
    reporter = Recorder()
    spec = JobSpec(cap_id=58438, spin_width=160, view_width=200, steps=1000, first_model_steps=100)
    report = run(spec, tmp_path / "out", Imagin("test", tmp_path / "imagin", fake_imagin), reporter, CPU)

    assert [stage for stage, status in reporter.stages if status == "done"] == list(pipeline.STAGES)
    assert report["camera"]["fovYDegrees"] == SPIN_CAMERA.fov_y and report["camera"]["direction"] == -1

    rows = {row["code"]: row for row in report["views"]}
    assert set(rows) == {1, 5, 27, 33}
    assert [code for code, row in rows.items() if row["used"]] == [27, 33]
    assert rows[1]["reason"] == rows[5]["reason"] == "at the spin's own height"
    assert all(row["alternative"] is not None for row in rows.values())

    # The first model from the spin alone, then the real one with the two views,
    # whose poses are refined.
    assert [len(views) for views in trained] == [32, 34]
    assert [view.refine_pose for view in trained[1][32:]] == [True, True]
    assert report["training"]["views"] == 34

    manifest = json.loads((tmp_path / "out" / "58438.json").read_text())
    assert manifest["camera"] == SPIN_CAMERA.as_manifest(32)
    assert manifest["source"]["imagin"]["views"] == [27, 33]
    assert manifest["source"]["imagin"]["spinWidth"] == 160
    assert manifest["splatCount"] == report["model"]["splatCount"] > 0
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == [
        "58438-turntable.mp4", "58438.json", "58438.ply", "58438.splat"]


def test_the_spin_alone_skips_the_views(tmp_path, fake_imagin, monkeypatch):
    trained = stand_ins(monkeypatch)
    reporter = Recorder()
    spec = JobSpec(cap_id=58438, extra_views="none", spin_width=160, steps=1000)
    report = run(spec, tmp_path, Imagin("test", tmp_path / "imagin", fake_imagin), reporter, CPU)
    assert ("views", "skipped") in reporter.stages
    assert "views" not in report
    assert [len(views) for views in trained] == [32]
    fetched = {int(urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["angle"][0]) for url in fake_imagin.calls}
    assert fetched == set(SPIN_ANGLES)


def test_all_views_uses_those_at_the_spins_height_too(tmp_path, fake_imagin, monkeypatch):
    stand_ins(monkeypatch)
    spec = JobSpec(cap_id=58438, extra_views="all", spin_width=160, view_width=200, steps=1000,
                   first_model_steps=100)
    report = run(spec, tmp_path, Imagin("test", tmp_path / "imagin", fake_imagin), Recorder(), CPU)
    assert sorted(row["code"] for row in report["views"] if row["used"]) == [1, 5, 27, 33]
