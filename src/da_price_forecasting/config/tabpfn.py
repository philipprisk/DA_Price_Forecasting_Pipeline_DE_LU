from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ..paths import resolve_path
from .base import RepoConfigModel, _default_datetime, _parse_timestamp
from .features import (
    CommodityConfig as TabpfnCommodityConfig,
    CommodityInstrumentConfig as TabpfnCommodityInstrumentConfig,
    CovariateConfig as TabpfnFeatureConfig,
    CovariateName as TabpfnCovariate,
)


class TabpfnPriceRollingStatsConfig(BaseModel):
    enabled: bool = False
    windows_days: list[int] = Field(default_factory=lambda: [7])

    @field_validator("windows_days", mode="before")
    @classmethod
    def _coerce_windows_days(cls, value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, int):
            return [value]
        return list(value)


class TabpfnCovariateRampConfig(BaseModel):
    sources: list[str] = Field(default_factory=list)
    horizons_mtu: list[int] = Field(default_factory=lambda: [4, 12])

    @field_validator("sources", mode="before")
    @classmethod
    def _coerce_sources(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

    @field_validator("horizons_mtu", mode="before")
    @classmethod
    def _coerce_horizons_mtu(cls, value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, int):
            return [value]
        return list(value)


class TabpfnInteractionFeatureConfig(BaseModel):
    seq2_minus_price_lag_d1: bool = False


class TabpfnLocalEngineeringConfig(BaseModel):
    price_daily_stats: bool = False
    holiday: bool = False
    price_rolling_stats: TabpfnPriceRollingStatsConfig = Field(default_factory=TabpfnPriceRollingStatsConfig)
    covariate_ramps: TabpfnCovariateRampConfig = Field(default_factory=TabpfnCovariateRampConfig)
    interactions: TabpfnInteractionFeatureConfig = Field(default_factory=TabpfnInteractionFeatureConfig)


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
    features: TabpfnFeatureConfig | None = None
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
    def _normalise(self) -> "TabpfnTsConfig":
        self.features = _normalise_tabpfn_features(
            self.features,
            use_exaa=self.use_exaa,
            use_load_forecast=self.use_load_forecast,
        )
        self.use_exaa = "exaa" in self.features.covariates
        self.use_load_forecast = "load_forecast" in self.features.covariates
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
        for covariate in self.features.covariates if self.features is not None else []:
            covariates.append(_covariate_tag(covariate))
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
    features: TabpfnFeatureConfig | None = None
    engineered_features: TabpfnLocalEngineeringConfig = Field(default_factory=TabpfnLocalEngineeringConfig)
    price_lag_days: list[int] = Field(default_factory=lambda: [1, 2, 7])
    add_calendar_features: bool = True
    context_selection: Literal["tail", "recent_weekday_even"] = "tail"
    context_recent_days: int = 7
    context_same_weekday_fraction: float = 0.35
    context_even_history_fraction: float = 0.25
    add_curve_features: bool = False
    curve_feature_sources: list[str] = Field(default_factory=lambda: ["price_exaa", "load_fc"])
    curve_neighbor_offsets: list[int] = Field(default_factory=lambda: [-4, -1, 1, 4])
    add_curve_summary_features: bool = False
    use_categorical_features: bool = False
    categorical_feature_names: list[str] = Field(
        default_factory=lambda: [
            "mtu",
            "hour",
            "dayofweek",
            "month",
            "is_weekend",
            "is_15min_market",
        ]
    )
    target_transform: Literal["none", "asinh_robust"] = "none"
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
    def _normalise(self) -> "TabpfnLocalConfig":
        self.features = _normalise_tabpfn_features(
            self.features,
            use_exaa=self.use_exaa,
            use_load_forecast=self.use_load_forecast,
        )
        self.use_exaa = "exaa" in self.features.covariates
        self.use_load_forecast = "load_forecast" in self.features.covariates
        if self.context_recent_days < 0:
            raise ValueError("context_recent_days must be non-negative.")
        if self.context_same_weekday_fraction < 0 or self.context_even_history_fraction < 0:
            raise ValueError("context selection fractions must be non-negative.")
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
        for covariate in self.features.covariates if self.features is not None else []:
            covariates.append(_covariate_tag(covariate))
        covariate_tag = "_".join(covariates) if covariates else "univariate"
        lag_tag = "_".join(f"d{lag}" for lag in self.price_lag_days) if self.price_lag_days else "nolag"
        rows_tag = f"n{self.max_train_rows}" if self.max_train_rows is not None else "all"
        return f"tabpfn_local_{covariate_tag}_{lag_tag}_{rows_tag}"

    @property
    def resolved_export_dir(self) -> Path:
        if self.export_dir is not None:
            return self.export_dir
        return self.repo_root / "results" / "tabpfn_local_results" / self.resolved_experiment_name


def _normalise_tabpfn_features(
    features: TabpfnFeatureConfig | None,
    use_exaa: bool,
    use_load_forecast: bool,
) -> TabpfnFeatureConfig:
    if features is None:
        covariates: list[TabpfnCovariate] = []
        if use_exaa:
            covariates.append("exaa")
        if use_load_forecast:
            covariates.append("load_forecast")
        return TabpfnFeatureConfig(covariates=covariates)
    return features


def _covariate_tag(covariate: TabpfnCovariate) -> str:
    tags = {
        "exaa": "exaa",
        "load_forecast": "load",
        "ntc": "ntc",
        "generation_unavailability": "unavail",
        "renewable_generation_proxy": "renewproxy",
        "commodities": "commodities",
    }
    return tags[covariate]
