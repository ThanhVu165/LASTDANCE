"""Path configuration for offline jobs.

No artifact stores an absolute path. Runtime paths are derived from ``AIC_DATA``
or from the repository-local ``data/`` default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def get_data_root(environment: Mapping[str, str] | None = None) -> Path:
    values = os.environ if environment is None else environment
    configured = values.get("AIC_DATA", "data")
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = _REPOSITORY_ROOT / path
    return path.resolve(strict=False)


@dataclass(frozen=True, slots=True)
class DataLayout:
    root: Path

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> "DataLayout":
        return cls(root=get_data_root(environment))

    @property
    def videos(self) -> Path:
        return self.root / "videos"

    @property
    def keyframes(self) -> Path:
        return self.root / "keyframes"

    @property
    def shots(self) -> Path:
        return self.root / "shots"

    @property
    def index(self) -> Path:
        return self.root / "index"
