import urllib.parse

import pytest

from vehicle_splat.imagin import SPIN_ANGLES, VIEW_CODES, Imagin, Vehicle


def test_the_url_names_the_van_and_only_a_paint_when_given():
    imagin = Imagin("uk-vanaways", None)
    plain = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(imagin.url(Vehicle(58438), 207, 1200)).query))
    assert plain == {"customer": "uk-vanaways", "capVehicleType": "lcv", "capId": "58438", "width": "1200",
                     "fileType": "png", "angle": "207"}
    painted = imagin.url(Vehicle(58438, "pwhite"), 1, 400)
    assert "paintId=pwhite" in painted


def test_renders_are_cached_by_van_width_and_angle(tmp_path, fake_imagin):
    imagin = Imagin("test", tmp_path, fake_imagin)
    first = imagin.image(Vehicle(1), 200, 160)
    again = imagin.image(Vehicle(1), 200, 160)
    assert first == again == tmp_path / "1-default" / "160" / "200.png"
    assert len(fake_imagin.calls) == 1
    imagin.image(Vehicle(1, "red"), 200, 160)
    imagin.image(Vehicle(1), 200, 320)
    assert len(fake_imagin.calls) == 3
    assert not list(tmp_path.rglob("*.part"))


def test_the_spin_is_32_frames_in_order(tmp_path, fake_imagin):
    paths = Imagin("test", tmp_path, fake_imagin).spin(Vehicle(1), 80)
    assert [path.stem for path in paths] == [str(angle) for angle in SPIN_ANGLES]


def test_views_that_only_repeat_the_default_are_marked(tmp_path, fake_imagin):
    views = Imagin("test", tmp_path, fake_imagin).views(Vehicle(1), 80)
    assert [view.code for view in views] == list(VIEW_CODES)
    own = [view.code for view in views if view.duplicate_of is None]
    assert own == [1, 5, 27, 33]
    assert all(view.duplicate_of == 1 for view in views if view.code not in own)


def test_an_answer_that_is_not_a_picture_is_refused_and_not_cached(tmp_path):
    imagin = Imagin("test", tmp_path, lambda url: b"<html>no such vehicle</html>")
    with pytest.raises(ValueError, match="no picture"):
        imagin.image(Vehicle(99), 200, 160)
    assert not list(tmp_path.rglob("*.png"))
