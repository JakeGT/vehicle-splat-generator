"""A van's renders from Imagin's CDN: the 360° spin, and its other views.

Every render is cached on disk by vehicle, width and angle, so a van is only
downloaded once whatever is asked of it afterwards."""
from __future__ import annotations

import hashlib
import pathlib
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

# The spin: 32 renders a full turn apart, in even steps, from about the van's
# own height.
SPIN_ANGLES = tuple(range(200, 232))
# Imagin's other view codes. A van has 13 of them; for a code it lacks, Imagin
# answers with its default view (code 1), byte for byte.
VIEW_CODES = tuple(range(1, 34))
# The views that look down on the van, among Imagin's standard set: what the
# spin cannot see. A hint for the page; the pipeline decides from where each
# view is found to be (pipeline.py).
FROM_ABOVE = frozenset({17, 21, 27, 33})

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

Fetch = Callable[[str], bytes]


@dataclass(frozen=True)
class Vehicle:
    cap_id: int
    paint_id: str | None = None

    @property
    def key(self) -> str:
        return f"{self.cap_id}-{self.paint_id or 'default'}"


@dataclass(frozen=True)
class View:
    code: int
    path: pathlib.Path
    # The code this one repeats, byte for byte, or None for a view of its own.
    duplicate_of: int | None


def download(url: str, attempts: int = 3) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "vehicle-splat"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(1 + attempt)
    raise AssertionError("unreachable")


class Imagin:
    def __init__(self, customer: str, cache_dir: pathlib.Path, fetch: Fetch = download):
        self.customer = customer
        self.cache_dir = cache_dir
        self.fetch = fetch

    def url(self, vehicle: Vehicle, angle: int, width: int) -> str:
        query = {"customer": self.customer, "capVehicleType": "lcv", "capId": vehicle.cap_id,
                 "width": width, "fileType": "png", "angle": angle}
        if vehicle.paint_id:
            query["paintId"] = vehicle.paint_id
        return "https://cdn.imagin.studio/getImage?" + urllib.parse.urlencode(query)

    def image(self, vehicle: Vehicle, angle: int, width: int) -> pathlib.Path:
        """The render, from the cache or Imagin. PNG, with its transparent background."""
        path = self.cache_dir / vehicle.key / str(width) / f"{angle}.png"
        if not path.exists():
            data = self.fetch(self.url(vehicle, angle, width))
            if not data.startswith(PNG_SIGNATURE):
                raise ValueError(f"Imagin sent no picture for CAP id {vehicle.cap_id}, angle {angle}")
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".part")
            partial.write_bytes(data)
            partial.replace(path)
        return path

    def images(self, vehicle: Vehicle, angles, width: int) -> list[pathlib.Path]:
        with ThreadPoolExecutor(8) as pool:
            return list(pool.map(lambda angle: self.image(vehicle, angle, width), angles))

    def spin(self, vehicle: Vehicle, width: int) -> list[pathlib.Path]:
        return self.images(vehicle, SPIN_ANGLES, width)

    def views(self, vehicle: Vehicle, width: int) -> list[View]:
        """Every view code, each marked with the code it merely repeats, if any."""
        paths = self.images(vehicle, VIEW_CODES, width)
        first: dict[str, int] = {}
        views = []
        for code, path in zip(VIEW_CODES, paths):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            views.append(View(code, path, first.get(digest)))
            first.setdefault(digest, code)
        return views
