from __future__ import annotations

from datetime import date
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

from ..paths import resolve_path
from .base import RepoConfigModel


class EnergyArenaObjective(str, Enum):
    POINT = "point"
    QUANTILE = "quantile"


class LearOperationalEnergyArenaSource(BaseModel):
    kind: Literal["lear_operational"] = "lear_operational"
    config_path: Path | None = None
    config: dict[str, Any] | None = None
    run_before_submit: bool = True

    @model_validator(mode="after")
    def _require_config(self) -> "LearOperationalEnergyArenaSource":
        if self.config_path is None and self.config is None:
            raise ValueError("lear_operational source requires config_path or embedded config.")
        return self


class SqraEnergyArenaSource(BaseModel):
    kind: Literal["sqra"] = "sqra"
    config_path: Path | None = None
    config: dict[str, Any] | None = None
    run_before_submit: bool = True
    value_column: str | None = None
    quantile_columns: list[str] | None = None

    @model_validator(mode="after")
    def _require_config(self) -> "SqraEnergyArenaSource":
        if self.config_path is None and self.config is None:
            raise ValueError("sqra source requires config_path or embedded config.")
        return self


class ForecastFileEnergyArenaSource(BaseModel):
    kind: Literal["forecast_file"] = "forecast_file"
    path: Path
    name: str | None = None
    value_column: str | None = None
    quantile_columns: list[str] | None = None


class TabpfnTsEnergyArenaSource(BaseModel):
    kind: Literal["tabpfn_ts"] = "tabpfn_ts"
    config_path: Path | None = None
    config: dict[str, Any] | None = None
    run_before_submit: bool = True
    value_column: str | None = None
    quantile_columns: list[str] | None = None

    @model_validator(mode="after")
    def _require_config(self) -> "TabpfnTsEnergyArenaSource":
        if self.config_path is None and self.config is None:
            raise ValueError("tabpfn_ts source requires config_path or embedded config.")
        return self


class TabpfnLocalEnergyArenaSource(BaseModel):
    kind: Literal["tabpfn_local"] = "tabpfn_local"
    config_path: Path | None = None
    config: dict[str, Any] | None = None
    run_before_submit: bool = True
    value_column: str | None = None
    quantile_columns: list[str] | None = None

    @model_validator(mode="after")
    def _require_config(self) -> "TabpfnLocalEnergyArenaSource":
        if self.config_path is None and self.config is None:
            raise ValueError("tabpfn_local source requires config_path or embedded config.")
        return self


EnergyArenaForecastSourceConfig = Annotated[
    LearOperationalEnergyArenaSource
    | SqraEnergyArenaSource
    | ForecastFileEnergyArenaSource
    | TabpfnTsEnergyArenaSource
    | TabpfnLocalEnergyArenaSource,
    Field(discriminator="kind"),
]


class EnergyArenaSubmissionConfig(RepoConfigModel):
    source: EnergyArenaForecastSourceConfig | None = None
    lear_config_path: Path | None = None
    forecast_date: date
    challenge_id: int = 2
    target_tz: str = "Europe/Berlin"
    objective: EnergyArenaObjective = EnergyArenaObjective.POINT
    value_column: str | None = None
    quantile_columns: list[str] | None = None
    api_base_url: str = "https://api.energy-arena.org"
    submit_url: str | None = None
    api_key_env: str = "ENERGY_ARENA_API_KEY"
    api_key_header_name: str = "X-API-Key"
    api_key_prefix: str = ""
    forecast_visibility: str = "closed"
    leaderboard_visibility: str = "public"
    approach_name: str = "forecast_model"
    approach_description: str | None = None
    repo_url: str | None = None
    payload_template_path: Path | None = None
    artifacts_dir: Path = Path("results/energy_arena_submissions")
    request_timeout_seconds: int = 30

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_source(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("source") is None and data.get("lear_config_path") is not None:
            data = dict(data)
            data["source"] = {
                "kind": "lear_operational",
                "config_path": data["lear_config_path"],
                "run_before_submit": True,
            }
        return data

    @model_validator(mode="after")
    def _resolve_paths(self) -> "EnergyArenaSubmissionConfig":
        if self.source is None:
            raise ValueError("Energy Arena submission config requires a forecast source.")

        if self.lear_config_path is not None:
            self.lear_config_path = resolve_path(self.lear_config_path, self.repo_root)
        if isinstance(self.source, LearOperationalEnergyArenaSource) and self.source.config_path is not None:
            self.source.config_path = resolve_path(self.source.config_path, self.repo_root)
        elif isinstance(self.source, SqraEnergyArenaSource) and self.source.config_path is not None:
            self.source.config_path = resolve_path(self.source.config_path, self.repo_root)
        elif isinstance(self.source, ForecastFileEnergyArenaSource):
            self.source.path = resolve_path(self.source.path, self.repo_root)
        elif isinstance(self.source, TabpfnTsEnergyArenaSource) and self.source.config_path is not None:
            self.source.config_path = resolve_path(self.source.config_path, self.repo_root)
        elif isinstance(self.source, TabpfnLocalEnergyArenaSource) and self.source.config_path is not None:
            self.source.config_path = resolve_path(self.source.config_path, self.repo_root)

        if self.payload_template_path is not None:
            self.payload_template_path = resolve_path(self.payload_template_path, self.repo_root)
        self.artifacts_dir = resolve_path(self.artifacts_dir, self.repo_root)
        return self


EnergyArenaPointSubmissionConfig = EnergyArenaSubmissionConfig
