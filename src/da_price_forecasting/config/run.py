from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from ..paths import resolve_path
from .base import RepoConfigModel


class RunKind(str, Enum):
    LEAR_OPERATIONAL = "lear_operational"
    LEAR_ANC = "lear_anc"
    SQRA = "sqra"
    TABPFN_TS = "tabpfn_ts"
    TABPFN_LOCAL = "tabpfn_local"
    ERA5_DOWNLOAD = "era5_download"
    ERA5_AGGREGATE = "era5_aggregate"
    ICON_AGGREGATE = "icon_aggregate"
    ENERGY_ARENA_SUBMIT = "energy_arena_submit"
    VISUALIZATION_REPORT = "visualization_report"
    EVALUATION = "evaluation"


class RunConfig(RepoConfigModel):
    """Top-level runner config used by the single repo entry point."""

    kind: RunKind
    config_path: Path | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    plot: bool = False
    submit: bool = False
    save_outputs: bool = True

    @model_validator(mode="after")
    def _resolve_paths(self) -> "RunConfig":
        if self.config_path is None and not self.config:
            raise ValueError("Run config requires either config_path or an embedded config mapping.")
        if self.config_path is not None:
            self.config_path = resolve_path(self.config_path, self.repo_root)
        return self
