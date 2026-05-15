from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator

from ..paths import resolve_path
from .base import RepoConfigModel, _default_datetime, _parse_timestamp


class ExaaNaiveConfig(RepoConfigModel):
    target_tz: str = "Europe/Berlin"
    country_code_entsoe: str = "DE_LU"
    entsoe_api_key_env: str = "ENTSOE_API_KEY"
    entsoe_start_date: datetime = Field(
        default_factory=lambda: _default_datetime("2025-01-01T00:00:00+01:00")
    )
    entsoe_end_date: datetime = Field(
        default_factory=lambda: _default_datetime("2026-02-28T23:45:00+01:00")
    )
    test_start: datetime = Field(
        default_factory=lambda: _default_datetime("2025-12-01T00:00:00+01:00")
    )
    test_end: datetime = Field(
        default_factory=lambda: _default_datetime("2026-02-28T23:45:00+01:00")
    )
    experiment_name: str = "exaa_naive"
    export_dir: Path | None = None

    @field_validator("entsoe_start_date", "entsoe_end_date", "test_start", "test_end", mode="before")
    @classmethod
    def _coerce_datetimes(cls, value: Any, info) -> datetime:
        target_tz = info.data.get("target_tz", "Europe/Berlin")
        return _parse_timestamp(value, target_tz)

    @model_validator(mode="after")
    def _resolve_paths(self) -> "ExaaNaiveConfig":
        if self.export_dir is not None:
            self.export_dir = resolve_path(self.export_dir, self.repo_root)
        return self

    @property
    def resolved_export_dir(self) -> Path:
        if self.export_dir is not None:
            return self.export_dir
        return self.repo_root / "results" / "lear_op_results" / self.experiment_name
