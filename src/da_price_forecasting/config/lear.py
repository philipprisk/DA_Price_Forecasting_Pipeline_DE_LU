from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator

from ..paths import resolve_path
from .base import ForecastVariant, RepoConfigModel, WeatherSource, _default_datetime, _parse_timestamp


class LearOperationalConfig(RepoConfigModel):
    target_tz: str = "Europe/Berlin"
    post_regime_start: datetime = Field(
        default_factory=lambda: _default_datetime("2025-10-01T00:00:00+02:00")
    )
    country_code_entsoe: str = "DE_LU"
    entsoe_api_key_env: str = "ENTSOE_API_KEY"
    entsoe_start_date: datetime = Field(
        default_factory=lambda: _default_datetime("2024-01-01T00:00:00+01:00")
    )
    entsoe_end_date: datetime = Field(
        default_factory=lambda: _default_datetime("2026-04-23T00:00:00+02:00")
    )
    era5_dirs: list[Path] = Field(
        default_factory=lambda: [
            Path("data/processed/era5_aggregated"),
        ]
    )
    icon_dir: Path = Path("data/processed/icon_aggregated")
    dwd_folder_offset_date: date = date(2025, 10, 26)
    start_folder_date: date = date(2025, 8, 1)
    required_run: str = "09"
    skip_dates: list[date] = Field(
        default_factory=lambda: [
            date(2025, 10, 26),
            date(2025, 10, 27),
            date(2025, 10, 28),
        ]
    )
    use_vst: bool = True
    weather_source: WeatherSource = WeatherSource.DWD
    variant: ForecastVariant = ForecastVariant.EXAA_ONLY
    lars_start_date: datetime = Field(
        default_factory=lambda: _default_datetime("2025-12-01T00:00:00+01:00")
    )
    test_start: datetime = Field(
        default_factory=lambda: _default_datetime("2026-04-16T00:00:00+02:00")
    )
    test_end: datetime = Field(
        default_factory=lambda: _default_datetime("2026-04-23T23:45:00+02:00")
    )
    train_days_rolling: int = 56
    n_clusters: int = 5
    export_dir: Path | None = None

    @field_validator(
        "post_regime_start",
        "entsoe_start_date",
        "entsoe_end_date",
        "lars_start_date",
        "test_start",
        "test_end",
        mode="before",
    )
    @classmethod
    def _coerce_datetimes(cls, value: Any, info) -> datetime:
        target_tz = info.data.get("target_tz", "Europe/Berlin")
        return _parse_timestamp(value, target_tz)

    @model_validator(mode="after")
    def _resolve_paths(self) -> "LearOperationalConfig":
        self.era5_dirs = [resolve_path(path, self.repo_root) for path in self.era5_dirs]
        self.icon_dir = resolve_path(self.icon_dir, self.repo_root)
        if self.export_dir is not None:
            self.export_dir = resolve_path(self.export_dir, self.repo_root)
        return self

    @property
    def use_exaa(self) -> bool:
        return self.variant == ForecastVariant.EXAA

    @property
    def use_exaa_only(self) -> bool:
        return self.variant == ForecastVariant.EXAA_ONLY

    @property
    def experiment_name(self) -> str:
        weather_tag = self.weather_source.value.lower() if not self.use_exaa_only else "exaa_only"
        variant_tag = ""
        if self.variant == ForecastVariant.EXAA and not self.use_exaa_only:
            variant_tag = "_exaa"
        elif self.variant == ForecastVariant.FUNDAMENTAL and not self.use_exaa_only:
            variant_tag = "_fundamental"

        cluster_tag = f"_c{self.n_clusters}" if not self.use_exaa_only else ""
        return f"lear_{weather_tag}{variant_tag}{cluster_tag}_d{self.train_days_rolling}"

    @property
    def resolved_export_dir(self) -> Path:
        if self.export_dir is not None:
            return self.export_dir

        base = self.repo_root / "results" / "lear_op_results"
        if self.use_exaa_only:
            return base / "exaa_only" / f"d{self.train_days_rolling}"
        if self.use_exaa:
            return base / self.weather_source.value.lower() / f"d{self.train_days_rolling}" / f"c{self.n_clusters}" / "exaa"
        return base / self.weather_source.value.lower() / f"d{self.train_days_rolling}" / f"c{self.n_clusters}" / "fundamental"


class LearAncConfig(RepoConfigModel):
    target_tz: str = "Europe/Berlin"
    country_code_entsoe: str = "DE_LU"
    entsoe_api_key_env: str = "ENTSOE_API_KEY"
    entsoe_start_date: datetime = Field(
        default_factory=lambda: _default_datetime("2024-01-01T00:00:00+01:00")
    )
    entsoe_end_date: datetime = Field(
        default_factory=lambda: _default_datetime("2026-04-23T00:00:00+02:00")
    )
    era5_dirs: list[Path] = Field(
        default_factory=lambda: [
            Path("data/processed/era5_aggregated"),
        ]
    )
    icon_dir: Path = Path("data/processed/icon_aggregated")
    dwd_folder_offset_date: date = date(2025, 10, 26)
    start_folder_date: date = date(2025, 8, 1)
    required_run: str = "09"
    skip_dates: list[date] = Field(
        default_factory=lambda: [
            date(2025, 10, 26),
            date(2025, 10, 27),
            date(2025, 10, 28),
        ]
    )
    weather_source: WeatherSource = WeatherSource.ERA5
    variant: ForecastVariant = ForecastVariant.EXAA
    lars_start_date: datetime = Field(
        default_factory=lambda: _default_datetime("2025-12-01T00:00:00+01:00")
    )
    test_start: datetime = Field(
        default_factory=lambda: _default_datetime("2025-12-01T00:00:00+01:00")
    )
    test_end: datetime = Field(
        default_factory=lambda: _default_datetime("2026-02-28T23:45:00+01:00")
    )
    train_days_rolling: int = 112
    n_clusters: int = 5
    mtu_window_wind: list[int] = Field(default_factory=lambda: list(range(96)))
    mtu_window_solar: list[int] = Field(default_factory=lambda: list(range(96)))
    export_dir: Path | None = None

    @field_validator("entsoe_start_date", "entsoe_end_date", "lars_start_date", "test_start", "test_end", mode="before")
    @classmethod
    def _coerce_datetimes(cls, value: Any, info) -> datetime:
        target_tz = info.data.get("target_tz", "Europe/Berlin")
        return _parse_timestamp(value, target_tz)

    @model_validator(mode="after")
    def _resolve_paths(self) -> "LearAncConfig":
        self.era5_dirs = [resolve_path(path, self.repo_root) for path in self.era5_dirs]
        self.icon_dir = resolve_path(self.icon_dir, self.repo_root)
        if self.export_dir is not None:
            self.export_dir = resolve_path(self.export_dir, self.repo_root)
        return self

    @property
    def use_exaa(self) -> bool:
        return self.variant == ForecastVariant.EXAA

    @property
    def use_exaa_only(self) -> bool:
        return self.variant == ForecastVariant.EXAA_ONLY

    @property
    def experiment_name(self) -> str:
        weather_tag = self.weather_source.value.lower() if not self.use_exaa_only else "exaa_only"
        variant_tag = ""
        if self.variant == ForecastVariant.EXAA and not self.use_exaa_only:
            variant_tag = "_exaa"
        elif self.variant == ForecastVariant.FUNDAMENTAL and not self.use_exaa_only:
            variant_tag = "_fundamental"

        cluster_tag = f"_c{self.n_clusters}" if not self.use_exaa_only else ""
        return f"anc_{weather_tag}{variant_tag}{cluster_tag}_d{self.train_days_rolling}"

    @property
    def resolved_export_dir(self) -> Path:
        if self.export_dir is not None:
            return self.export_dir

        base = self.repo_root / "results" / "lear_anc_results"
        if self.use_exaa_only:
            return base / "exaa_only" / f"d{self.train_days_rolling}"
        if self.use_exaa:
            return base / self.weather_source.value.lower() / f"c{self.n_clusters}" / f"d{self.train_days_rolling}" / "exaa"
        return base / self.weather_source.value.lower() / f"c{self.n_clusters}" / f"d{self.train_days_rolling}" / "fundamental"
