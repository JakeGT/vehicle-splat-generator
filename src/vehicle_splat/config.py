"""The service's settings, from the environment."""
from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Jobs, their files, and Imagin's cached renders.
    data_dir: pathlib.Path
    # Imagin's customer key, as the portal's fixtures use it.
    imagin_customer: str
    # Where "Publish to portal" copies a model, when the portal's
    # apps/portal/public/models is mounted into the container. None hides it.
    portal_models_dir: pathlib.Path | None

    @property
    def jobs_dir(self) -> pathlib.Path:
        return self.data_dir / "jobs"

    @property
    def imagin_cache_dir(self) -> pathlib.Path:
        return self.data_dir / "imagin"


def load_settings() -> Settings:
    portal = os.environ.get("PORTAL_MODELS_DIR")
    return Settings(
        data_dir=pathlib.Path(os.environ.get("DATA_DIR", "data")).resolve(),
        imagin_customer=os.environ.get("IMAGIN_CUSTOMER", "uk-vanaways"),
        portal_models_dir=pathlib.Path(portal) if portal else None,
    )
