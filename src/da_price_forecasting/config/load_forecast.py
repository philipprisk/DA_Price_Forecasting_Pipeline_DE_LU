from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from ..paths import resolve_path
from .base import RepoConfigModel


class LoadForecastModelConfig(RepoConfigModel):
    """Train/evaluate a direct actual-load forecasting model."""

    target_tz: str = "Europe/Berlin"
    country_code_entsoe: str = "DE_LU"
    entsoe_api_key_env: str = "ENTSOE_API_KEY"
    entsoe_start_date: date = date(2025, 8, 1)
    entsoe_end_date: date = date(2026, 4, 23)
    chunk_days: int = 90

    actual_load_file: Path = Path("data/processed/load_forecast/actual_load.csv")
    icon_dir: Path = Path("data/processed/icon_aggregated_c5")
    weather_source: Literal["dwd_icon", "open_meteo"] = "dwd_icon"
    open_meteo_weather_file: Path = Path("data/processed/open_meteo/open_meteo_icon_d2_c25_load_weather.csv")
    open_meteo_cluster_file: Path = Path("data/clustering/icon_d2_clustering_c25.parquet")
    open_meteo_base_url: str = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    open_meteo_model: str | None = "icon_d2"
    open_meteo_start_date: date = date(2025, 8, 1)
    open_meteo_end_date: date = date(2026, 4, 23)
    open_meteo_hourly_variables: list[str] = Field(
        default_factory=lambda: [
            "temperature_2m",
            "dew_point_2m",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "cloud_cover",
            "precipitation",
            "snow_depth",
        ]
    )
    open_meteo_batch_size: int = 10
    open_meteo_cell_selection: str = "nearest"
    open_meteo_timeout_seconds: int = 60
    open_meteo_force_download: bool = False
    open_meteo_api_mode: Literal["historical_forecast", "single_run"] = "historical_forecast"
    open_meteo_single_run_hour_utc: str = "06:00"
    open_meteo_single_run_forecast_days: int = 2
    open_meteo_request_pause_seconds: float = 0.0
    open_meteo_retry_attempts: int = 5
    open_meteo_retry_backoff_seconds: float = 30.0
    open_meteo_point_selection: Literal["centroid", "grid_mean"] = "centroid"
    open_meteo_max_points_per_cluster: int | None = None
    export_dir: Path = Path("results/load_forecast_results/direct_dwd_icon_c5_hgb")

    include_calendar_features: bool = True
    include_holiday_features: bool = True
    include_bridge_day_features: bool = True
    include_actual_load_lag_features: bool = True
    include_weather_features: bool = True
    include_weighted_weather_features: bool = False
    include_weighted_weather_daily_features: bool = False
    include_weighted_weather_inertia_features: bool = False
    include_weather_cluster_spread_features: bool = False
    include_weather_time_interactions: bool = False
    include_regional_holiday_features: bool = False
    include_partial_load_features: bool = False
    include_partial_load_shape_features: bool = False
    include_entsoe_forecast_features: bool = False
    include_entsoe_error_lag_features: bool = False
    include_entsoe_error_rolling_features: bool = False
    include_rich_temperature_features: bool = False
    fetch_weather_only: bool = False
    calendar_harmonics: int = 1

    entsoe_load_forecast_file: Path = Path("data/processed/load_forecast/entsoe_load_forecast.csv")
    weather_cluster_weight_file: Path | None = None
    weather_cluster_id_column: str = "cluster_id"
    weather_cluster_weight_column: str = "weight"
    weather_weighted_feature_prefix: str = "weather_weighted"
    weather_weighted_feature_bases: list[str] = Field(default_factory=list)
    weighted_weather_daily_feature_bases: list[str] = Field(default_factory=list)
    weighted_weather_daily_stats: list[str] = Field(default_factory=lambda: ["mean", "min", "max"])
    weighted_weather_daily_lag_days: list[int] = Field(default_factory=lambda: [1, 7])
    weighted_weather_inertia_feature_bases: list[str] = Field(default_factory=list)
    weighted_weather_inertia_windows_hours: list[int] = Field(default_factory=lambda: [24, 48, 72])
    weighted_weather_inertia_stats: list[str] = Field(default_factory=lambda: ["mean", "delta_mean"])
    weather_spread_feature_bases: list[str] = Field(default_factory=list)
    weather_spread_stats: list[str] = Field(default_factory=lambda: ["min", "max", "range", "std"])
    weather_hdd_thresholds: list[float] = Field(default_factory=lambda: [18.0])
    weather_cdd_thresholds: list[float] = Field(default_factory=lambda: [22.0])
    keep_weather_cluster_features: bool = True

    regional_holiday_weight_file: Path | None = None
    regional_holiday_region_column: str = "region"
    regional_holiday_weight_column: str = "weight"
    regional_holiday_feature_prefix: str = "regional_holiday"

    partial_load_reference_day: int = 1
    partial_load_comparison_lag_days: int = 7
    partial_load_morning_end_hour: int = 11
    partial_load_morning_end_minute: int = 45
    partial_load_point_times: list[str] = Field(default_factory=list)

    actual_load_lag_days: list[int] = Field(default_factory=lambda: [2, 3, 7, 14])
    entsoe_error_lag_days: list[int] = Field(default_factory=lambda: [2, 3, 7, 14, 21])
    entsoe_error_rolling_windows_days: list[int] = Field(default_factory=lambda: [7, 28])
    entsoe_error_rolling_groups: list[Literal["global", "hour", "mtu"]] = Field(default_factory=lambda: ["global", "mtu"])
    entsoe_error_rolling_min_observations: int = 24
    target_availability_lag_days: int = 1

    start_folder_date: date = date(2025, 8, 1)
    required_run: str = "09"
    dwd_folder_offset_date: date = date(2025, 10, 26)
    skip_dates: list[date] = Field(
        default_factory=lambda: [
            date(2025, 10, 26),
            date(2025, 10, 27),
            date(2025, 10, 28),
        ]
    )
    dwd_icon_auto_update: bool = False
    dwd_icon_raw_dir: Path = Path("data/raw/dwd_icon_daily")
    dwd_icon_base_url: str = "https://opendata.dwd.de/weather/nwp/icon-d2/grib/"
    dwd_icon_download_variables: list[str] = Field(
        default_factory=lambda: [
            "t_2m",
            "td_2m",
            "p",
            "u_10m",
            "v_10m",
            "vmax_10m",
            "aswdir_s",
            "aswdifd_s",
            "tot_prec",
            "h_snow",
            "snow_gsp",
        ]
    )
    dwd_icon_download_timeout_seconds: int = 60
    dwd_icon_request_pause_seconds: float = 0.0
    dwd_icon_force_update: bool = False
    dwd_icon_catch_up_missing_days: bool = True
    dwd_icon_aggregation_shapefile_path: Path = Path("data/shapefile/ne_10m_admin_0_countries.shp")
    dwd_icon_aggregation_n_clusters: int | None = None
    dwd_icon_aggregation_buffer_km: int = 50

    model_type: Literal["hist_gradient_boosting", "lightgbm", "ridge"] = "hist_gradient_boosting"
    model_granularity: Literal["global", "hour_block"] = "global"
    hour_block_boundaries: list[int] = Field(default_factory=lambda: [0, 6, 11, 16, 21, 24])
    train_days_rolling: int = 112
    min_train_days: int = 14
    test_start: date = date(2025, 12, 1)
    test_end: date = date(2026, 2, 28)

    hgb_max_iter: int = 300
    hgb_learning_rate: float = 0.04
    hgb_max_leaf_nodes: int = 31
    hgb_l2_regularization: float = 0.1
    lgbm_n_estimators: int = 600
    lgbm_learning_rate: float = 0.02
    lgbm_num_leaves: int = 31
    lgbm_min_child_samples: int = 40
    lgbm_subsample: float = 1.0
    lgbm_colsample_bytree: float = 1.0
    lgbm_reg_lambda: float = 0.3
    ridge_alpha: float = 100.0
    random_state: int = 42
    max_features: int | None = None
    load_target_mode: Literal["actual_load", "entsoe_residual"] = "actual_load"
    clip_predictions_to_training_target_range: bool = False
    prediction_upper_quantile: float = Field(default=1.0, ge=0.0, le=1.0)
    apply_rolling_bias_correction: bool = False
    bias_correction_window_days: int = 28
    bias_correction_min_observations: int = 24
    bias_correction_group: Literal["global", "hour", "mtu"] = "hour"

    @field_validator("actual_load_lag_days", mode="before")
    @classmethod
    def _coerce_lag_days(cls, value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, int):
            return [value]
        return [int(item) for item in value]

    @field_validator("entsoe_error_lag_days", mode="before")
    @classmethod
    def _coerce_entsoe_error_lag_days(cls, value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, int):
            return [value]
        return [int(item) for item in value]

    @field_validator("entsoe_error_rolling_windows_days", mode="before")
    @classmethod
    def _coerce_entsoe_error_rolling_windows_days(cls, value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, int):
            return [value]
        return [int(item) for item in value]

    @field_validator("entsoe_error_rolling_groups", mode="before")
    @classmethod
    def _coerce_entsoe_error_rolling_groups(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    @field_validator("weather_weighted_feature_bases", mode="before")
    @classmethod
    def _coerce_weighted_feature_bases(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    @field_validator("weighted_weather_daily_feature_bases", mode="before")
    @classmethod
    def _coerce_weighted_daily_feature_bases(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    @field_validator("weighted_weather_inertia_feature_bases", mode="before")
    @classmethod
    def _coerce_weighted_inertia_feature_bases(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    @field_validator("weather_spread_feature_bases", mode="before")
    @classmethod
    def _coerce_weather_spread_feature_bases(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    @field_validator("weighted_weather_daily_stats", mode="before")
    @classmethod
    def _coerce_weighted_daily_stats(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    @field_validator("weighted_weather_inertia_stats", "weather_spread_stats", mode="before")
    @classmethod
    def _coerce_weather_stats(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    @field_validator("weighted_weather_daily_lag_days", mode="before")
    @classmethod
    def _coerce_weighted_daily_lag_days(cls, value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, int):
            return [value]
        return [int(item) for item in value]

    @field_validator("weighted_weather_inertia_windows_hours", mode="before")
    @classmethod
    def _coerce_weighted_inertia_windows_hours(cls, value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, int):
            return [value]
        return [int(item) for item in value]

    @field_validator("partial_load_point_times", mode="before")
    @classmethod
    def _coerce_partial_load_point_times(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    @field_validator("weather_hdd_thresholds", "weather_cdd_thresholds", mode="before")
    @classmethod
    def _coerce_weather_thresholds(cls, value: Any) -> list[float]:
        if value is None:
            return []
        if isinstance(value, (int, float)):
            return [float(value)]
        return [float(item) for item in value]

    @field_validator("hour_block_boundaries", mode="before")
    @classmethod
    def _coerce_hour_block_boundaries(cls, value: Any) -> list[int]:
        if value is None:
            return []
        return [int(item) for item in value]

    @field_validator("dwd_icon_download_variables", mode="before")
    @classmethod
    def _coerce_dwd_icon_download_variables(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    @model_validator(mode="after")
    def _resolve_paths(self) -> "LoadForecastModelConfig":
        if self.chunk_days < 1:
            raise ValueError("chunk_days must be positive.")
        if self.train_days_rolling < 1:
            raise ValueError("train_days_rolling must be positive.")
        if self.min_train_days < 1:
            raise ValueError("min_train_days must be positive.")
        if self.target_availability_lag_days < 0:
            raise ValueError("target_availability_lag_days must be non-negative.")
        if self.dwd_icon_download_timeout_seconds < 1:
            raise ValueError("dwd_icon_download_timeout_seconds must be positive.")
        if self.dwd_icon_request_pause_seconds < 0:
            raise ValueError("dwd_icon_request_pause_seconds must be non-negative.")
        if self.dwd_icon_aggregation_n_clusters is not None and self.dwd_icon_aggregation_n_clusters < 1:
            raise ValueError("dwd_icon_aggregation_n_clusters must be positive when provided.")
        if self.dwd_icon_aggregation_buffer_km < 0:
            raise ValueError("dwd_icon_aggregation_buffer_km must be non-negative.")
        if any(lag_day <= 0 for lag_day in self.actual_load_lag_days):
            raise ValueError("actual_load_lag_days must contain positive integers.")
        if any(lag_day <= 0 for lag_day in self.entsoe_error_lag_days):
            raise ValueError("entsoe_error_lag_days must contain positive integers.")
        if any(lag_day <= self.target_availability_lag_days for lag_day in self.entsoe_error_lag_days):
            raise ValueError("entsoe_error_lag_days must be greater than target_availability_lag_days.")
        if any(window_day <= 0 for window_day in self.entsoe_error_rolling_windows_days):
            raise ValueError("entsoe_error_rolling_windows_days must contain positive integers.")
        unknown_error_groups = set(self.entsoe_error_rolling_groups) - {"global", "hour", "mtu"}
        if unknown_error_groups:
            raise ValueError(f"entsoe_error_rolling_groups contains unsupported groups: {sorted(unknown_error_groups)}")
        if self.entsoe_error_rolling_min_observations < 1:
            raise ValueError("entsoe_error_rolling_min_observations must be positive.")
        if self.calendar_harmonics < 1:
            raise ValueError("calendar_harmonics must be at least 1.")
        if self.include_weighted_weather_features and not self.include_weather_features:
            raise ValueError("include_weighted_weather_features requires include_weather_features.")
        if self.weather_source == "open_meteo":
            if self.open_meteo_batch_size < 1:
                raise ValueError("open_meteo_batch_size must be positive.")
            if self.open_meteo_timeout_seconds < 1:
                raise ValueError("open_meteo_timeout_seconds must be positive.")
            if self.open_meteo_single_run_forecast_days < 1:
                raise ValueError("open_meteo_single_run_forecast_days must be positive.")
            if self.open_meteo_request_pause_seconds < 0:
                raise ValueError("open_meteo_request_pause_seconds must be non-negative.")
            if self.open_meteo_retry_attempts < 0:
                raise ValueError("open_meteo_retry_attempts must be non-negative.")
            if self.open_meteo_retry_backoff_seconds < 0:
                raise ValueError("open_meteo_retry_backoff_seconds must be non-negative.")
            if self.open_meteo_max_points_per_cluster is not None and self.open_meteo_max_points_per_cluster < 1:
                raise ValueError("open_meteo_max_points_per_cluster must be positive when provided.")
        if self.include_weighted_weather_features and self.weather_cluster_weight_file is None:
            raise ValueError("weather_cluster_weight_file is required when include_weighted_weather_features is true.")
        if self.include_weighted_weather_daily_features and not self.include_weighted_weather_features:
            raise ValueError("include_weighted_weather_daily_features requires include_weighted_weather_features.")
        if self.include_weighted_weather_inertia_features and not self.include_weighted_weather_features:
            raise ValueError("include_weighted_weather_inertia_features requires include_weighted_weather_features.")
        if self.include_weather_cluster_spread_features and not self.include_weather_features:
            raise ValueError("include_weather_cluster_spread_features requires include_weather_features.")
        if self.include_regional_holiday_features and self.regional_holiday_weight_file is None:
            raise ValueError("regional_holiday_weight_file is required when include_regional_holiday_features is true.")
        if not self.weather_cluster_id_column:
            raise ValueError("weather_cluster_id_column must not be empty.")
        if not self.weather_cluster_weight_column:
            raise ValueError("weather_cluster_weight_column must not be empty.")
        if not self.weather_weighted_feature_prefix:
            raise ValueError("weather_weighted_feature_prefix must not be empty.")
        if not self.regional_holiday_region_column:
            raise ValueError("regional_holiday_region_column must not be empty.")
        if not self.regional_holiday_weight_column:
            raise ValueError("regional_holiday_weight_column must not be empty.")
        if not self.regional_holiday_feature_prefix:
            raise ValueError("regional_holiday_feature_prefix must not be empty.")
        if self.partial_load_reference_day < 1:
            raise ValueError("partial_load_reference_day must be positive.")
        if self.partial_load_comparison_lag_days < 1:
            raise ValueError("partial_load_comparison_lag_days must be positive.")
        if not (0 <= self.partial_load_morning_end_hour <= 23):
            raise ValueError("partial_load_morning_end_hour must be between 0 and 23.")
        if self.partial_load_morning_end_minute not in {0, 15, 30, 45}:
            raise ValueError("partial_load_morning_end_minute must be one of 0, 15, 30, or 45.")
        valid_daily_stats = {"mean", "min", "max", "range"}
        unknown_daily_stats = set(self.weighted_weather_daily_stats) - valid_daily_stats
        if unknown_daily_stats:
            raise ValueError(f"weighted_weather_daily_stats contains unsupported stats: {sorted(unknown_daily_stats)}")
        if any(lag_day <= 0 for lag_day in self.weighted_weather_daily_lag_days):
            raise ValueError("weighted_weather_daily_lag_days must contain positive integers.")
        valid_inertia_stats = {"mean", "min", "max", "range", "delta_mean"}
        unknown_inertia_stats = set(self.weighted_weather_inertia_stats) - valid_inertia_stats
        if unknown_inertia_stats:
            raise ValueError(f"weighted_weather_inertia_stats contains unsupported stats: {sorted(unknown_inertia_stats)}")
        if any(window <= 0 for window in self.weighted_weather_inertia_windows_hours):
            raise ValueError("weighted_weather_inertia_windows_hours must contain positive integers.")
        valid_spread_stats = {"mean", "min", "max", "range", "std", "p10", "p90"}
        unknown_spread_stats = set(self.weather_spread_stats) - valid_spread_stats
        if unknown_spread_stats:
            raise ValueError(f"weather_spread_stats contains unsupported stats: {sorted(unknown_spread_stats)}")
        for point_time in self.partial_load_point_times:
            parts = point_time.split(":")
            if len(parts) != 2:
                raise ValueError("partial_load_point_times entries must use HH:MM format.")
            hour, minute = int(parts[0]), int(parts[1])
            if not (0 <= hour <= 23) or minute not in {0, 15, 30, 45}:
                raise ValueError("partial_load_point_times must be quarter-hour times between 00:00 and 23:45.")
        if self.include_rich_temperature_features:
            self.weather_hdd_thresholds = sorted({float(threshold) for threshold in self.weather_hdd_thresholds} | {18.0})
            self.weather_cdd_thresholds = sorted({float(threshold) for threshold in self.weather_cdd_thresholds} | {22.0})
        if any(threshold < -50 or threshold > 60 for threshold in self.weather_hdd_thresholds + self.weather_cdd_thresholds):
            raise ValueError("Weather degree thresholds must be plausible Celsius values.")
        if self.model_granularity == "hour_block":
            if len(self.hour_block_boundaries) < 2:
                raise ValueError("hour_block_boundaries must contain at least two boundaries.")
            if self.hour_block_boundaries[0] != 0 or self.hour_block_boundaries[-1] != 24:
                raise ValueError("hour_block_boundaries must start at 0 and end at 24.")
            if sorted(set(self.hour_block_boundaries)) != self.hour_block_boundaries:
                raise ValueError("hour_block_boundaries must be strictly increasing.")
            if any(boundary < 0 or boundary > 24 for boundary in self.hour_block_boundaries):
                raise ValueError("hour_block_boundaries must be between 0 and 24.")
        if self.lgbm_n_estimators < 1:
            raise ValueError("lgbm_n_estimators must be positive.")
        if self.lgbm_learning_rate <= 0:
            raise ValueError("lgbm_learning_rate must be positive.")
        if self.lgbm_num_leaves < 2:
            raise ValueError("lgbm_num_leaves must be at least 2.")
        if self.lgbm_min_child_samples < 1:
            raise ValueError("lgbm_min_child_samples must be positive.")
        if not (0 < self.lgbm_subsample <= 1.0):
            raise ValueError("lgbm_subsample must be in (0, 1].")
        if not (0 < self.lgbm_colsample_bytree <= 1.0):
            raise ValueError("lgbm_colsample_bytree must be in (0, 1].")
        if self.lgbm_reg_lambda < 0:
            raise ValueError("lgbm_reg_lambda must be non-negative.")
        if self.bias_correction_window_days < 1:
            raise ValueError("bias_correction_window_days must be positive.")
        if self.bias_correction_min_observations < 1:
            raise ValueError("bias_correction_min_observations must be positive.")

        self.actual_load_file = resolve_path(self.actual_load_file, self.repo_root)
        self.entsoe_load_forecast_file = resolve_path(self.entsoe_load_forecast_file, self.repo_root)
        self.icon_dir = resolve_path(self.icon_dir, self.repo_root)
        self.dwd_icon_raw_dir = resolve_path(self.dwd_icon_raw_dir, self.repo_root)
        self.dwd_icon_aggregation_shapefile_path = resolve_path(self.dwd_icon_aggregation_shapefile_path, self.repo_root)
        self.open_meteo_weather_file = resolve_path(self.open_meteo_weather_file, self.repo_root)
        self.open_meteo_cluster_file = resolve_path(self.open_meteo_cluster_file, self.repo_root)
        self.export_dir = resolve_path(self.export_dir, self.repo_root)
        if self.weather_cluster_weight_file is not None:
            self.weather_cluster_weight_file = resolve_path(self.weather_cluster_weight_file, self.repo_root)
        if self.regional_holiday_weight_file is not None:
            self.regional_holiday_weight_file = resolve_path(self.regional_holiday_weight_file, self.repo_root)
        return self


class EntsoeLoadForecastBenchmarkConfig(RepoConfigModel):
    """Benchmark ENTSO-E total load forecasts against actual load."""

    target_tz: str = "Europe/Berlin"
    country_code_entsoe: str = "DE_LU"
    entsoe_api_key_env: str = "ENTSOE_API_KEY"
    entsoe_start_date: date = date(2025, 8, 1)
    entsoe_end_date: date = date(2026, 4, 23)
    test_start: date | None = date(2025, 12, 1)
    test_end: date | None = date(2026, 2, 28)
    chunk_days: int = 90

    actual_load_file: Path = Path("data/processed/load_forecast/actual_load.csv")
    forecast_file: Path = Path("data/processed/load_forecast/entsoe_load_forecast.csv")
    export_dir: Path = Path("results/load_forecast_results/entsoe_load_forecast_benchmark")

    @model_validator(mode="after")
    def _resolve_paths(self) -> "EntsoeLoadForecastBenchmarkConfig":
        if self.chunk_days < 1:
            raise ValueError("chunk_days must be positive.")
        self.actual_load_file = resolve_path(self.actual_load_file, self.repo_root)
        self.forecast_file = resolve_path(self.forecast_file, self.repo_root)
        self.export_dir = resolve_path(self.export_dir, self.repo_root)
        return self
