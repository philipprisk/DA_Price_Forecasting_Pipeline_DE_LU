from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from ..paths import resolve_path
from .base import RepoConfigModel, _default_datetime, _parse_timestamp


class TabpfnTsConfig(RepoConfigModel):
    target_tz: str = "Europe/Berlin"
    country_code_entsoe: str = "DE_LU"
    entsoe_api_key_env: str = "ENTSOE_API_KEY"
    entsoe_start_date: datetime = Field(
        default_factory=lambda: _default_datetime("2025-01-01T00:00:00+01:00")
    )
    entsoe_end_date: datetime = Field(
        default_factory=lambda: _default_datetime("2026-04-23T23:45:00+02:00")
    )
    test_start: datetime = Field(
        default_factory=lambda: _default_datetime("2026-04-16T00:00:00+02:00")
    )
    test_end: datetime = Field(
        default_factory=lambda: _default_datetime("2026-04-23T23:45:00+02:00")
    )
    forecast_horizon: int = 96
    frequency: str = "15min"
    quantiles: list[float] = Field(default_factory=lambda: [0.10, 0.25, 0.50, 0.75, 0.90])
    use_exaa: bool = False
    use_load_forecast: bool = False
    max_context_length: int = 4096
    tabpfn_mode: Literal["client", "local"] = "client"
    tabpfn_output_selection: Literal["mean", "median", "mode"] = "median"
    tabpfn_model_config: dict[str, Any] = Field(
        default_factory=lambda: {
            "model_path": ".cache/models/tabpfn/tabpfn-v2-regressor-2noar4o2.ckpt",
        }
    )
    disable_telemetry: bool = True
    experiment_name: str | None = None
    export_dir: Path | None = None

    @field_validator("entsoe_start_date", "entsoe_end_date", "test_start", "test_end", mode="before")
    @classmethod
    def _coerce_datetimes(cls, value: Any, info) -> datetime:
        target_tz = info.data.get("target_tz", "Europe/Berlin")
        return _parse_timestamp(value, target_tz)

    @model_validator(mode="after")
    def _resolve_paths(self) -> "TabpfnTsConfig":
        if self.export_dir is not None:
            self.export_dir = resolve_path(self.export_dir, self.repo_root)
        return self

    @property
    def quantile_columns(self) -> list[str]:
        return [f"q{tau:.3f}" for tau in self.quantiles]

    @property
    def resolved_experiment_name(self) -> str:
        if self.experiment_name:
            return self.experiment_name

        covariates = []
        if self.use_exaa:
            covariates.append("exaa")
        if self.use_load_forecast:
            covariates.append("load")
        covariate_tag = "_".join(covariates) if covariates else "univariate"
        return f"tabpfn_ts_{covariate_tag}_{self.tabpfn_mode}"

    @property
    def resolved_export_dir(self) -> Path:
        if self.export_dir is not None:
            return self.export_dir
        return self.repo_root / "results" / "tabpfn_ts_results" / self.resolved_experiment_name


class TabpfnLocalConfig(RepoConfigModel):
    target_tz: str = "Europe/Berlin"
    country_code_entsoe: str = "DE_LU"
    entsoe_api_key_env: str = "ENTSOE_API_KEY"
    entsoe_start_date: datetime = Field(
        default_factory=lambda: _default_datetime("2025-01-01T00:00:00+01:00")
    )
    entsoe_end_date: datetime = Field(
        default_factory=lambda: _default_datetime("2026-04-23T23:45:00+02:00")
    )
    test_start: datetime = Field(
        default_factory=lambda: _default_datetime("2026-04-16T00:00:00+02:00")
    )
    test_end: datetime = Field(
        default_factory=lambda: _default_datetime("2026-04-23T23:45:00+02:00")
    )
    post_regime_start: datetime = Field(
        default_factory=lambda: _default_datetime("2025-10-01T00:00:00+02:00")
    )
    forecast_horizon: int = 96
    frequency: str = "15min"
    quantiles: list[float] = Field(default_factory=lambda: [0.10, 0.25, 0.50, 0.75, 0.90])
    use_exaa: bool = True
    use_load_forecast: bool = True
    price_lag_days: list[int] = Field(default_factory=lambda: [1, 2, 7])
    add_calendar_features: bool = True
    train_window_days: int | None = 365
    max_train_rows: int | None = 8192
    n_estimators: int = 8
    device: str = "auto"
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache", "batched"] = "fit_preprocessors"
    n_preprocessing_jobs: int = 1
    inference_precision: str = "auto"
    ignore_pretraining_limits: bool = False
    point_output_type: Literal["mean", "median", "mode"] = "median"
    predict_batch_size: int | None = 16
    random_state: int | None = 0
    experiment_name: str | None = None
    export_dir: Path | None = None

    @field_validator("entsoe_start_date", "entsoe_end_date", "test_start", "test_end", "post_regime_start", mode="before")
    @classmethod
    def _coerce_datetimes(cls, value: Any, info) -> datetime:
        target_tz = info.data.get("target_tz", "Europe/Berlin")
        return _parse_timestamp(value, target_tz)

    @model_validator(mode="after")
    def _resolve_paths(self) -> "TabpfnLocalConfig":
        if self.export_dir is not None:
            self.export_dir = resolve_path(self.export_dir, self.repo_root)
        return self

    @property
    def quantile_columns(self) -> list[str]:
        return [f"q{tau:.3f}" for tau in self.quantiles]

    @property
    def resolved_experiment_name(self) -> str:
        if self.experiment_name:
            return self.experiment_name

        covariates = []
        if self.use_exaa:
            covariates.append("exaa")
        if self.use_load_forecast:
            covariates.append("load")
        covariate_tag = "_".join(covariates) if covariates else "univariate"
        lag_tag = "_".join(f"d{lag}" for lag in self.price_lag_days) if self.price_lag_days else "nolag"
        rows_tag = f"n{self.max_train_rows}" if self.max_train_rows is not None else "all"
        return f"tabpfn_local_{covariate_tag}_{lag_tag}_{rows_tag}"

    @property
    def resolved_export_dir(self) -> Path:
        if self.export_dir is not None:
            return self.export_dir
        return self.repo_root / "results" / "tabpfn_local_results" / self.resolved_experiment_name
