import re

import pytest
from fastapi.testclient import TestClient

from test_jobs import finished
from vehicle_splat.api import WEB, create_app
from vehicle_splat.config import Settings
from vehicle_splat.imagin import Imagin


@pytest.fixture
def make_client(tmp_path, fake_imagin):
    def make(portal: bool = False):
        settings = Settings(tmp_path / "data", "test", tmp_path / "portal-models" if portal else None)
        app = create_app(settings, runner=finished, start_worker=False,
                         imagin=Imagin("test", settings.imagin_cache_dir, fake_imagin))
        return TestClient(app), app.state.worker
    return make


def test_the_page_is_served(make_client):
    client, _ = make_client()
    page = client.get("/")
    assert page.status_code == 200 and "Vehicle splat" in page.text
    assert client.get("/app.js").status_code == 200


def test_health_names_the_gpu_or_none(make_client):
    client, _ = make_client()
    health = client.get("/api/health").json()
    assert health["ok"] is True and "gpu" in health and health["portalPublishing"] is False


def test_a_van_shows_its_spin_and_its_views(make_client):
    client, _ = make_client()
    van = client.get("/api/vehicles/58438").json()
    assert van["capId"] == 58438 and van["paintId"] is None
    assert len(van["spin"]) == 32 and van["spin"][0]["url"] == "/api/images/58438/200.png?width=400"
    assert len(van["views"]) == 33
    own = {view["code"] for view in van["views"] if view["duplicateOf"] is None}
    assert own == {1, 5, 27, 33}
    assert [view["code"] for view in van["views"] if view["expectedFromAbove"]] == [17, 21, 27, 33]
    image = client.get(van["views"][4]["url"])
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"


def test_a_paint_is_carried_into_the_image_urls(make_client):
    client, _ = make_client()
    van = client.get("/api/vehicles/58438?paintId=pwhite").json()
    assert van["paintId"] == "pwhite"
    assert van["spin"][3]["url"].endswith("&paintId=pwhite")
    assert client.get(van["spin"][3]["url"]).status_code == 200


@pytest.mark.parametrize("path, status", [
    ("/api/vehicles/58438?paintId=../../etc", 422),
    ("/api/images/58438/150.png", 404),
    ("/api/images/58438/34.png", 404),
    ("/api/images/58438/200.png?width=5000", 422),
])
def test_bad_requests_are_refused(make_client, path, status):
    client, _ = make_client()
    assert client.get(path).status_code == status


def test_imagin_failing_is_a_bad_gateway(tmp_path):
    settings = Settings(tmp_path, "test", None)
    app = create_app(settings, runner=finished, start_worker=False,
                     imagin=Imagin("test", tmp_path, lambda url: b"nope"))
    response = TestClient(app).get("/api/vehicles/1")
    assert response.status_code == 502 and "CAP id 1" in response.json()["detail"]


@pytest.mark.parametrize("body", [{"capId": 0}, {"capId": 1, "paintId": "a b"}, {"capId": 1, "extraViews": "some"},
                                  {"capId": 1, "steps": 10}, {"capId": 1, "maxSplats": 5_000_000}])
def test_a_job_request_is_checked(make_client, body):
    client, _ = make_client()
    assert client.post("/api/jobs", json=body).status_code == 422


def test_a_job_runs_and_its_files_are_served(make_client):
    client, worker = make_client()
    job = client.post("/api/jobs", json={"capId": 58438, "extraViews": "none", "steps": 2000}).json()
    assert job["status"] == "queued" and job["spec"]["extra_views"] == "none" and job["spec"]["steps"] == 2000
    worker.run_one(worker.queue.get_nowait())
    done = client.get(f"/api/jobs/{job['id']}").json()
    assert done["status"] == "succeeded" and "trace" not in done
    assert client.get(f"/api/jobs/{job['id']}/files/58438.splat").content == b"x"
    assert client.get(f"/api/jobs/{job['id']}/files/scratch.bin").status_code == 404
    assert client.get("/api/jobs/nothing").status_code == 404
    listed = client.get("/api/jobs").json()
    assert [entry["id"] for entry in listed] == [job["id"]] and set(listed[0]) == {"id", "createdAt", "status", "spec"}


def test_publishing_needs_the_portal_folder_and_a_finished_job(make_client, tmp_path):
    client, worker = make_client()
    job = client.post("/api/jobs", json={"capId": 58438}).json()
    assert client.post(f"/api/jobs/{job['id']}/publish").status_code == 409

    client, worker = make_client(portal=True)
    assert client.get("/api/health").json()["portalPublishing"] is True
    job = client.post("/api/jobs", json={"capId": 58438, "paintId": "pwhite"}).json()
    assert client.post(f"/api/jobs/{job['id']}/publish").status_code == 404
    worker.run_one(job["id"])
    published = client.post(f"/api/jobs/{job['id']}/publish").json()
    assert published == {"published": ["58438-pwhite.json", "58438-pwhite.splat"]}
    assert sorted(path.name for path in (tmp_path / "portal-models").iterdir()) == published["published"]


def test_every_element_the_page_script_uses_is_on_the_page():
    script = (WEB / "app.js").read_text()
    page = (WEB / "index.html").read_text()
    used = set(re.findall(r"\$\('([A-Za-z]+)'\)", script))
    present = set(re.findall(r'id="([A-Za-z]+)"', page))
    assert used and used <= present, used - present
