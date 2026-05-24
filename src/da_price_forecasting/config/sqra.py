from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator

from ..paths import resolve_path
from .base import RepoConfigModel, _default_datetime, _parse_timestamp


class SqraConfig(RepoConfigModel):
    target_tz: str = "Europe/Berlin"
    import_paths: list[Path] = Field(
        default_factory=lambda: [
            Path("results/lear_op_results/dwd/d56/c1/exaa/forecast.csv"),
            Path("results/lear_op_results/dwd/d56/c5/fundamental/forecast.csv"),
            Path("results/lear_op_results/era5/d364/c5/fundamental/forecast.csv"),
        ]
    )
    quantiles: list[float] = Field(default_factory=lambda: [0.10, 0.25, 0.50, 0.75, 0.90])
    test_start: datetime = Field(
        default_factory=lambda: _default_datetime("2025-12-01T00:00:00+01:00")
    )
    test_end: datetime = Field(
        default_factory=lambda: _default_datetime("2026-02-28T23:45:00+01:00")
    )
    train_days_rolling: int = 60
    target_availability_lag_days: int = 0
    experiment_name: str = "sqra_dwd_exaa_enriched_d60"
    export_dir: Path = Path("results/sqra_results/dwd_exaa_enriched")

    @field_validator("test_start", "test_end", mode="before")
    @classmethod
    def _coerce_datetimes(cls, value: Any, info) -> datetime:
        target_tz = info.data.get("target_tz", "Europe/Berlin")
        return _parse_timestamp(value, target_tz)

    @model_validator(mode="after")
    def _resolve_paths(self) -> "SqraConfig":
        if self.train_days_rolling < 1:
            raise ValueError("train_days_rolling must be positive.")
        if self.target_availability_lag_days < 0:
            raise ValueError("target_availability_lag_days must be non-negative.")
        self.import_paths = [resolve_path(path, self.repo_root) for path in self.import_paths]
        self.export_dir = resolve_path(self.export_dir, self.repo_root)
        return self

    @property
    def quantile_columns(self) -> list[str]:
        return [f"q{tau:.3f}" for tau in self.quantiles]
