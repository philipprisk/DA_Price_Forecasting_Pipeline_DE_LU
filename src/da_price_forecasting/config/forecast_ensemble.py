from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from ..paths import resolve_path
from .base import RepoConfigModel


class ForecastEnsembleCalendarSwitchConfig(RepoConfigModel):
    condition: Literal[
        "holiday",
        "weekend",
        "nonworkday",
        "weekday_0",
        "weekday_1",
        "weekday_2",
        "weekday_3",
        "weekday_4",
        "weekday_5",
        "weekday_6",
    ]
    source: str


class ForecastEnsembleSourceConfig(RepoConfigModel):
    name: str
    path: Path
    weight: float = 1.0

    @model_validator(mode="after")
    def _resolve_paths(self) -> "ForecastEnsembleSourceConfig":
        self.path = resolve_path(self.path, self.repo_root)
        return self


class ForecastEnsembleConfig(RepoConfigModel):
    sources: list[ForecastEnsembleSourceConfig] = Field(default_factory=list)
    export_dir: Path
    join: str = "inner"
    method: Literal["fixed", "rolling_inverse_mae", "rolling_regime_inverse_mae"] = "fixed"
    train_days: int = 28
    min_train_days: int = 7
    weight_power: float = 1.0
    epsilon: float = 1e-6
    bias_correction: Literal["none", "rolling_mean_error", "rolling_hour_mean_error"] = "none"
    bias_train_days: int = 28
    bias_min_train_days: int = 7
    calendar_switches: list[ForecastEnsembleCalendarSwitchConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _resolve_paths(self) -> "ForecastEnsembleConfig":
        if len(self.sources) < 2:
            raise ValueError("Forecast ensemble requires at least two sources.")
        if self.join != "inner":
            raise ValueError("Only inner joins are currently supported for forecast ensembles.")
        if self.train_days < 1:
            raise ValueError("train_days must be positive.")
        if self.min_train_days < 1:
            raise ValueError("min_train_days must be positive.")
        if self.weight_power <= 0:
            raise ValueError("weight_power must be positive.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if self.bias_train_days < 1:
            raise ValueError("bias_train_days must be positive.")
        if self.bias_min_train_days < 1:
            raise ValueError("bias_min_train_days must be positive.")
        source_names = {source.name for source in self.sources}
        unknown_sources = [switch.source for switch in self.calendar_switches if switch.source not in source_names]
        if unknown_sources:
            raise ValueError(f"Calendar switches reference unknown source names: {sorted(set(unknown_sources))}")
        self.export_dir = resolve_path(self.export_dir, self.repo_root)
        return self
