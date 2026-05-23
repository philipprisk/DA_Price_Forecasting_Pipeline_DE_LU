from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from da_price_forecasting.config import EntsoeLoadForecastBenchmarkConfig, LoadForecastModelConfig
from da_price_forecasting.pipelines import load_forecast as lf


def _config(tmp_path: Path, **overrides) -> LoadForecastModelConfig:
    data = {
        "repo_root": tmp_path,
        "actual_load_file": tmp_path / "actual_load.csv",
        "icon_dir": tmp_path / "icon",
        "export_dir": tmp_path / "out",
        "include_holiday_features": False,
        "actual_load_lag_days": [2, 7],
        "train_days_rolling": 8,
        "min_train_days": 2,
        "model_type": "ridge",
    }
    data.update(overrides)
    return LoadForecastModelConfig(**data)


def test_actual_load_fetch_window_includes_partial_current_day(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        lf,
        "_current_operational_day",
        lambda target_tz: pd.Timestamp("2026-05-22", tz=target_tz),
    )

    config = _config(
        tmp_path,
        entsoe_start_date=date(2026, 5, 1),
        entsoe_end_date=date(2026, 5, 23),
        test_start=date(2026, 5, 23),
        test_end=date(2026, 5, 23),
        include_partial_load_features=True,
        partial_load_reference_day=1,
    )

    _, end = lf._actual_load_fetch_window(config)

    assert end == pd.Timestamp("2026-05-22", tz="Europe/Berlin")


def test_actual_load_fetch_window_without_partial_stops_at_latest_complete_day(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        lf,
        "_current_operational_day",
        lambda target_tz: pd.Timestamp("2026-05-22", tz=target_tz),
    )

    config = _config(
        tmp_path,
        entsoe_start_date=date(2026, 5, 1),
        entsoe_end_date=date(2026, 5, 23),
        test_start=date(2026, 5, 23),
        test_end=date(2026, 5, 23),
        include_partial_load_features=False,
    )

    _, end = lf._actual_load_fetch_window(config)

    assert end == pd.Timestamp("2026-05-21", tz="Europe/Berlin")


def test_actual_load_fetch_refreshes_current_partial_day(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        lf,
        "_current_operational_day",
        lambda target_tz: pd.Timestamp("2026-05-22", tz=target_tz),
    )

    actual_file = tmp_path / "actual_load.csv"
    cached_index = pd.to_datetime(
        [
            "2026-05-20T00:00:00+02:00",
            "2026-05-21T00:00:00+02:00",
            "2026-05-22T00:00:00+02:00",
        ],
        utc=True,
    ).tz_convert("Europe/Berlin")
    pd.DataFrame({"load_actual": [1.0, 2.0, 3.0]}, index=cached_index).to_csv(actual_file)

    fetched_index = pd.to_datetime(["2026-05-22T10:15:00+02:00"], utc=True).tz_convert("Europe/Berlin")
    fetched = pd.DataFrame({"load_actual": [99.0]}, index=fetched_index)
    calls = []

    def fake_fetch_actual_load(**kwargs):
        calls.append(kwargs)
        return fetched

    monkeypatch.setattr(lf, "fetch_actual_load", fake_fetch_actual_load)

    config = _config(
        tmp_path,
        actual_load_file=actual_file,
        entsoe_start_date=date(2026, 5, 20),
        entsoe_end_date=date(2026, 5, 23),
        test_start=date(2026, 5, 23),
        test_end=date(2026, 5, 23),
        include_partial_load_features=True,
        partial_load_reference_day=1,
    )

    result = lf._load_or_fetch_actual_load(config)

    assert len(calls) == 1
    assert calls[0]["start_day"] == pd.Timestamp("2026-05-22", tz="Europe/Berlin")
    assert result.loc[fetched_index[0], "load_actual"] == 99.0


def test_build_load_forecast_dataset_adds_calendar_lags_and_weather(monkeypatch, tmp_path: Path) -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=10 * 96, freq="15min")
    actual = pd.DataFrame({"load_actual": 50_000.0 + np.arange(len(index), dtype=float)}, index=index)
    weather = pd.DataFrame({"weather_t2m_C_cluster_0": np.linspace(0.0, 5.0, len(index))}, index=index)

    monkeypatch.setattr(lf, "_load_or_fetch_actual_load", lambda config: actual)
    monkeypatch.setattr(lf, "_build_load_weather_features", lambda config: weather)

    dataset = lf.build_load_forecast_dataset(_config(tmp_path))
    timestamp = index[2 * 96 + 5]

    assert "Load_Actual_MW" in dataset.columns
    assert "Load_Actual_MW_lag_d2" in dataset.columns
    assert "Load_Actual_MW_lag_d7" in dataset.columns
    assert "mtu_sin" in dataset.columns
    assert "weather_t2m_C_cluster_0" in dataset.columns
    assert dataset.loc[timestamp, "Load_Actual_MW_lag_d2"] == actual.loc[timestamp - pd.Timedelta(days=2), "load_actual"]


def test_build_load_forecast_dataset_keeps_future_feature_rows_without_actuals(monkeypatch, tmp_path: Path) -> None:
    actual_index = pd.date_range("2026-01-01T00:00:00+01:00", periods=3 * 96, freq="15min")
    weather_index = pd.date_range("2026-01-01T00:00:00+01:00", periods=4 * 96, freq="15min")
    actual = pd.DataFrame({"load_actual": 50_000.0 + np.arange(len(actual_index), dtype=float)}, index=actual_index)
    weather = pd.DataFrame({"weather_t2m_C_cluster_0": np.linspace(0.0, 5.0, len(weather_index))}, index=weather_index)

    monkeypatch.setattr(lf, "_load_or_fetch_actual_load", lambda config: actual)
    monkeypatch.setattr(lf, "_build_load_weather_features", lambda config: weather)

    dataset = lf.build_load_forecast_dataset(_config(tmp_path))
    future_timestamp = pd.Timestamp("2026-01-04T12:00:00+01:00")

    assert future_timestamp in dataset.index
    assert np.isnan(dataset.loc[future_timestamp, "Load_Actual_MW"])
    assert np.isfinite(dataset.loc[future_timestamp, "weather_t2m_C_cluster_0"])


def test_calendar_features_include_bridge_days_and_extra_harmonics() -> None:
    index = pd.date_range("2026-05-15T00:00:00+02:00", periods=1, freq="15min")

    features = lf._build_calendar_features(
        index,
        target_tz="Europe/Berlin",
        include_holidays=True,
        include_bridge_days=True,
        calendar_harmonics=3,
    )

    assert features.loc[index[0], "is_bridge_day"] == 1.0
    assert "doy_sin_3" in features.columns
    assert "week_cos_3" in features.columns


def test_regional_holiday_features_are_population_weighted(tmp_path: Path) -> None:
    index = pd.date_range("2026-01-06T00:00:00+01:00", periods=1, freq="15min")
    weight_file = tmp_path / "region_weights.csv"
    pd.DataFrame({"region": ["DE1", "LU0"], "weight": [2.0, 1.0]}).to_csv(weight_file, index=False)

    features = lf._build_regional_holiday_features(
        index,
        target_tz="Europe/Berlin",
        weight_file=weight_file,
        region_column="region",
        weight_column="weight",
        feature_prefix="regional_holiday",
    )

    assert np.isclose(features.loc[index[0], "regional_holiday_public_holiday_share"], 2.0 / 3.0)
    assert np.isclose(features.loc[index[0], "regional_holiday_nonworkday_share"], 2.0 / 3.0)


def test_weather_time_interactions_add_temperature_interactions() -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=2, freq="15min")
    features = pd.DataFrame(
        {
            "weather_t2m_C_cluster_0": [5.0, 6.0],
            "weather_hdd18_cluster_0": [13.0, 12.0],
            "mtu_sin": [0.0, 0.1],
            "mtu_cos": [1.0, 0.9],
            "is_weekend": [0.0, 0.0],
        },
        index=index,
    )

    result = lf._add_weather_time_interactions(features)

    assert result.loc[index[0], "weather_t2m_C_cluster_0_x_mtu_cos"] == 5.0
    assert "weather_hdd18_cluster_0_x_mtu_sin" in result.columns


def test_rich_temperature_weather_features_add_configured_thresholds(monkeypatch, tmp_path: Path) -> None:
    hourly_index = pd.date_range("2026-01-01T00:00:00", periods=2, freq="h", tz="Europe/Berlin")
    df_hourly = pd.DataFrame({"t2m_cluster_0": [283.15, 298.15]}, index=hourly_index)
    df_qh = pd.DataFrame(index=pd.date_range("2026-01-01T00:00:00", periods=8, freq="15min", tz="Europe/Berlin"))
    monkeypatch.setattr(lf, "load_dwd", lambda **kwargs: (df_hourly, df_qh))

    features = lf._build_load_weather_features(
        _config(
            tmp_path,
            include_rich_temperature_features=True,
            weather_hdd_thresholds=[12, 18],
            weather_cdd_thresholds=[20, 24],
        )
    )

    assert "weather_hdd12_cluster_0" in features.columns
    assert "weather_hdd18_cluster_0" in features.columns
    assert "weather_cdd20_cluster_0" in features.columns
    assert "weather_cdd24_cluster_0" in features.columns
    assert features.loc[hourly_index[0], "weather_hdd12_cluster_0"] == 2.0
    assert features.loc[hourly_index[1], "weather_cdd24_cluster_0"] == 1.0


def test_dwd_weather_auto_update_runs_before_loading_archive(monkeypatch, tmp_path: Path) -> None:
    from da_price_forecasting.preprocessing import dwd_icon_operational

    hourly_index = pd.date_range("2026-05-16T00:00:00", periods=2, freq="h", tz="Europe/Berlin")
    qh_index = pd.date_range("2026-05-16T00:00:00", periods=8, freq="15min", tz="Europe/Berlin")
    df_hourly = pd.DataFrame({"t2m_cluster_0": [283.15, 284.15]}, index=hourly_index)
    df_qh = pd.DataFrame({"ASWDIR_cluster_0": [1.0] * 8, "ASWDIFD_cluster_0": [2.0] * 8}, index=qh_index)
    calls = {}

    def fake_ensure_dwd_icon_weather(**kwargs):
        calls.update(kwargs)
        assert not calls.get("load_dwd_called", False)
        return [date(2026, 5, 15)]

    def fake_load_dwd(**kwargs):
        calls["load_dwd_called"] = True
        return df_hourly, df_qh

    monkeypatch.setattr(dwd_icon_operational, "ensure_dwd_icon_weather", fake_ensure_dwd_icon_weather)
    monkeypatch.setattr(lf, "load_dwd", fake_load_dwd)

    config = _config(
        tmp_path,
        required_run="06",
        test_start=date(2026, 5, 16),
        test_end=date(2026, 5, 16),
        dwd_folder_offset_date=date(2025, 10, 26),
        dwd_icon_auto_update=True,
        dwd_icon_raw_dir=tmp_path / "raw_dwd",
        dwd_icon_aggregation_shapefile_path=tmp_path / "countries.shp",
        dwd_icon_aggregation_n_clusters=25,
    )

    result = lf._build_load_weather_features(config)

    assert calls["forecast_start"] == date(2026, 5, 16)
    assert calls["forecast_end"] == date(2026, 5, 16)
    assert calls["run_hour"] == "06"
    assert calls["raw_base_dir"] == tmp_path / "raw_dwd"
    assert calls["n_clusters"] == 25
    assert calls["load_dwd_called"] is True
    assert "weather_t2m_C_cluster_0" in result.columns


def test_open_meteo_weather_features_use_existing_loader(monkeypatch, tmp_path: Path) -> None:
    weather_index = pd.date_range("2026-01-01T00:00:00", periods=2, freq="h", tz="Europe/Berlin")
    weather = pd.DataFrame(
        {
            "t2m_cluster_0": [283.15, 298.15],
            "td2m_cluster_0": [280.15, 290.15],
            "u10_cluster_0": [3.0, 4.0],
            "v10_cluster_0": [4.0, 3.0],
            "ssrd_cluster_0": [0.0, 100.0],
            "fdir_cluster_0": [0.0, 60.0],
        },
        index=weather_index,
    )
    monkeypatch.setattr(lf, "load_open_meteo", lambda **kwargs: weather)

    features = lf._build_load_weather_features(
        _config(
            tmp_path,
            weather_source="open_meteo",
            open_meteo_cluster_file=tmp_path / "clusters.parquet",
            open_meteo_weather_file=tmp_path / "open_meteo.csv",
            include_rich_temperature_features=True,
            weather_hdd_thresholds=[12, 18],
            weather_cdd_thresholds=[20, 24],
        )
    )

    assert "weather_hdd12_cluster_0" in features.columns
    assert "weather_cdd24_cluster_0" in features.columns
    assert "weather_wind_speed_10m_cluster_0" in features.columns
    assert "weather_solar_global_cluster_0" in features.columns
    assert "weather_rel_humidity_pct_cluster_0" in features.columns
    assert "weather_vpd_hPa_cluster_0" in features.columns
    assert np.isclose(features.loc[weather_index[0], "weather_wind_speed_10m_cluster_0"], 5.0)
    assert features.loc[weather_index[1], "weather_solar_diffuse_cluster_0"] == 40.0
    assert 0.0 <= features.loc[weather_index[0], "weather_rel_humidity_pct_cluster_0"] <= 100.0
    assert features.loc[weather_index[1], "weather_vpd_hPa_cluster_0"] > 0.0


def test_open_meteo_weather_features_extend_final_hour_to_full_quarter_day(
    monkeypatch,
    tmp_path: Path,
) -> None:
    weather_index = pd.date_range("2026-05-24", periods=24, freq="h", tz="Europe/Berlin")
    weather = pd.DataFrame(
        {
            "t2m_cluster_0": np.full(len(weather_index), 293.15),
            "ssrd_cluster_0": np.arange(len(weather_index), dtype=float),
        },
        index=weather_index,
    )
    monkeypatch.setattr(lf, "load_open_meteo", lambda **kwargs: weather)

    features = lf._build_load_weather_features(
        _config(
            tmp_path,
            weather_source="open_meteo",
            open_meteo_cluster_file=tmp_path / "clusters.parquet",
            open_meteo_weather_file=tmp_path / "open_meteo.csv",
        )
    )

    day = pd.Timestamp("2026-05-24", tz="Europe/Berlin")
    day_features = features.loc[day: day + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)]

    assert len(day_features) == 96
    assert day_features.index[-1] == pd.Timestamp("2026-05-24T23:45:00+02:00")
    assert day_features["weather_t2m_C_cluster_0"].notna().all()


def test_weighted_weather_features_use_cluster_weights() -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=2, freq="15min")
    features = pd.DataFrame(
        {
            "weather_t2m_C_cluster_0": [10.0, 20.0],
            "weather_t2m_C_cluster_1": [20.0, 40.0],
            "weather_solar_global_cluster_0": [1.0, 3.0],
            "weather_solar_global_cluster_1": [5.0, 7.0],
        },
        index=index,
    )
    weights = pd.Series({0: 1.0, 1: 3.0})

    result = lf._add_weighted_weather_features(features, weights)

    assert np.isclose(result.loc[index[0], "weather_weighted_t2m_C"], 17.5)
    assert np.isclose(result.loc[index[1], "weather_weighted_solar_global"], 6.0)


def test_build_load_forecast_dataset_adds_weighted_weather_interactions(monkeypatch, tmp_path: Path) -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=10 * 96, freq="15min")
    actual = pd.DataFrame({"load_actual": 50_000.0 + np.arange(len(index), dtype=float)}, index=index)
    weather = pd.DataFrame(
        {
            "weather_t2m_C_cluster_0": np.full(len(index), 10.0),
            "weather_t2m_C_cluster_1": np.full(len(index), 20.0),
            "weather_hdd18_cluster_0": np.full(len(index), 8.0),
            "weather_hdd18_cluster_1": np.full(len(index), 0.0),
        },
        index=index,
    )
    weight_file = tmp_path / "weather_cluster_weights.csv"
    pd.DataFrame({"cluster_id": [0, 1], "weight": [1.0, 3.0]}).to_csv(weight_file, index=False)

    monkeypatch.setattr(lf, "_load_or_fetch_actual_load", lambda config: actual)
    monkeypatch.setattr(lf, "_build_load_weather_features", lambda config: weather)

    dataset = lf.build_load_forecast_dataset(
        _config(
            tmp_path,
            include_weighted_weather_features=True,
            include_weather_time_interactions=True,
            weather_cluster_weight_file=weight_file,
        )
    )

    assert np.isclose(dataset.loc[index[0], "weather_weighted_t2m_C"], 17.5)
    assert "weather_weighted_t2m_C_x_mtu_cos" in dataset.columns
    assert "weather_weighted_hdd18_x_mtu_sin" in dataset.columns


def test_build_load_forecast_dataset_can_use_only_selected_weighted_weather(monkeypatch, tmp_path: Path) -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=10 * 96, freq="15min")
    actual = pd.DataFrame({"load_actual": 50_000.0 + np.arange(len(index), dtype=float)}, index=index)
    weather = pd.DataFrame(
        {
            "weather_t2m_C_cluster_0": np.full(len(index), 10.0),
            "weather_t2m_C_cluster_1": np.full(len(index), 20.0),
            "weather_solar_global_cluster_0": np.full(len(index), 100.0),
            "weather_solar_global_cluster_1": np.full(len(index), 200.0),
        },
        index=index,
    )
    weight_file = tmp_path / "weather_cluster_weights.csv"
    pd.DataFrame({"cluster_id": [0, 1], "weight": [1.0, 3.0]}).to_csv(weight_file, index=False)

    monkeypatch.setattr(lf, "_load_or_fetch_actual_load", lambda config: actual)
    monkeypatch.setattr(lf, "_build_load_weather_features", lambda config: weather)

    dataset = lf.build_load_forecast_dataset(
        _config(
            tmp_path,
            include_weighted_weather_features=True,
            include_weather_time_interactions=True,
            weather_cluster_weight_file=weight_file,
            weather_weighted_feature_bases=["t2m_C"],
            keep_weather_cluster_features=False,
        )
    )

    assert "weather_weighted_t2m_C" in dataset.columns
    assert "weather_weighted_t2m_C_x_mtu_cos" in dataset.columns
    assert "weather_weighted_solar_global" not in dataset.columns
    assert "weather_t2m_C_cluster_0" not in dataset.columns
    assert "weather_solar_global_cluster_0" not in dataset.columns


def test_partial_load_features_use_previous_morning_and_week_comparison() -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=12 * 96, freq="15min")
    actual = pd.DataFrame({"load_actual": np.arange(len(index), dtype=float)}, index=index)
    features = lf._build_partial_load_features(
        actual,
        pd.date_range("2026-01-10T00:00:00+01:00", periods=96, freq="15min"),
        target_tz="Europe/Berlin",
        reference_day=1,
        comparison_lag_days=7,
        morning_end_hour=11,
    )

    d1 = actual.loc["2026-01-09T00:00:00+01:00":"2026-01-09T11:45:00+01:00", "load_actual"]
    d8 = actual.loc["2026-01-02T00:00:00+01:00":"2026-01-02T11:45:00+01:00", "load_actual"]
    timestamp = pd.Timestamp("2026-01-10T12:00:00+01:00")

    assert np.isclose(features.loc[timestamp, "partial_load_d1_00_11_mean"], d1.mean())
    assert np.isclose(features.loc[timestamp, "partial_load_d1_00_11_mean_diff_d7"], d1.mean() - d8.mean())


def test_partial_load_features_support_quarter_hour_cutoff() -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=12 * 96, freq="15min")
    actual = pd.DataFrame({"load_actual": np.arange(len(index), dtype=float)}, index=index)
    features = lf._build_partial_load_features(
        actual,
        pd.date_range("2026-01-10T00:00:00+01:00", periods=96, freq="15min"),
        target_tz="Europe/Berlin",
        reference_day=1,
        comparison_lag_days=7,
        morning_end_hour=10,
        morning_end_minute=15,
    )

    d1 = actual.loc["2026-01-09T00:00:00+01:00":"2026-01-09T10:15:00+01:00", "load_actual"]
    timestamp = pd.Timestamp("2026-01-10T12:00:00+01:00")

    assert np.isclose(features.loc[timestamp, "partial_load_d1_00_1015_mean"], d1.mean())
    assert features.loc[timestamp, "partial_load_d1_00_1015_last"] == actual.loc[
        pd.Timestamp("2026-01-09T10:15:00+01:00"),
        "load_actual",
    ]


def test_partial_load_features_can_add_shape_and_point_values() -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=12 * 96, freq="15min")
    actual = pd.DataFrame({"load_actual": np.arange(len(index), dtype=float)}, index=index)
    features = lf._build_partial_load_features(
        actual,
        pd.date_range("2026-01-10T00:00:00+01:00", periods=96, freq="15min"),
        target_tz="Europe/Berlin",
        reference_day=1,
        comparison_lag_days=7,
        morning_end_hour=10,
        morning_end_minute=15,
        include_shape_features=True,
        point_times=["06:00", "10:15"],
    )

    timestamp = pd.Timestamp("2026-01-10T12:00:00+01:00")
    d1 = actual.loc["2026-01-09T00:00:00+01:00":"2026-01-09T10:15:00+01:00", "load_actual"]
    d8 = actual.loc["2026-01-02T00:00:00+01:00":"2026-01-02T10:15:00+01:00", "load_actual"]

    assert np.isclose(features.loc[timestamp, "partial_load_d1_00_1015_range"], d1.max() - d1.min())
    assert np.isclose(
        features.loc[timestamp, "partial_load_d1_00_1015_ramp_diff_d7"],
        (d1.iloc[-1] - d1.iloc[0]) - (d8.iloc[-1] - d8.iloc[0]),
    )
    assert features.loc[timestamp, "partial_load_d1_point_0600"] == actual.loc[
        pd.Timestamp("2026-01-09T06:00:00+01:00"),
        "load_actual",
    ]
    assert features.loc[timestamp, "partial_load_d1_point_1015"] == actual.loc[
        pd.Timestamp("2026-01-09T10:15:00+01:00"),
        "load_actual",
    ]


def test_entsoe_error_features_use_safe_lags_and_rolling_history(monkeypatch, tmp_path: Path) -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=12 * 96, freq="15min")
    mtu = index.hour * 4 + index.minute // 15
    day_number = (index.normalize() - index.normalize()[0]).days
    forecast = pd.DataFrame({"load_fc": np.full(len(index), 50_000.0)}, index=index)
    actual = pd.DataFrame({"load_actual": 50_000.0 + day_number * 100.0 + mtu}, index=index)
    monkeypatch.setattr(lf, "_load_or_fetch_model_entsoe_load_forecast", lambda config: forecast)

    config = _config(
        tmp_path,
        include_entsoe_error_lag_features=True,
        include_entsoe_error_rolling_features=True,
        entsoe_error_lag_days=[2, 7],
        entsoe_error_rolling_windows_days=[7],
        entsoe_error_rolling_groups=["global", "mtu"],
        entsoe_error_rolling_min_observations=1,
        target_availability_lag_days=1,
    )
    target_index = pd.date_range("2026-01-10T00:00:00+01:00", periods=96, freq="15min")
    features = lf._build_entsoe_error_features(config, actual, target_index)

    timestamp = pd.Timestamp("2026-01-10T12:00:00+01:00")
    source_timestamp = timestamp - pd.Timedelta(days=2)
    assert features.loc[timestamp, "Entsoe_Load_Error_MW_lag_d2"] == (
        actual.loc[source_timestamp, "load_actual"] - forecast.loc[source_timestamp, "load_fc"]
    )

    history = actual.join(forecast)
    history["error"] = history["load_actual"] - history["load_fc"]
    safe_history = history.loc["2026-01-02T00:00:00+01:00":"2026-01-08T23:45:00+01:00"]
    assert np.isclose(features.loc[timestamp, "Entsoe_Load_Error_MW_global_mean_7d"], safe_history["error"].mean())
    assert np.isclose(
        features.loc[timestamp, "Entsoe_Load_Error_MW_mtu_mean_7d"],
        safe_history.loc[(safe_history.index.hour == 12) & (safe_history.index.minute == 0), "error"].mean(),
    )


def test_weighted_weather_daily_features_add_daily_stats_and_diffs() -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=3 * 96, freq="15min")
    values = np.concatenate([np.full(96, 10.0), np.full(96, 12.0), np.linspace(15.0, 18.0, 96)])
    features = pd.DataFrame({"weather_weighted_t2m_C": values}, index=index)

    result = lf._add_weighted_weather_daily_features(
        features,
        prefix="weather_weighted",
        base_names=["t2m_C"],
        stats=["mean", "min", "max", "range"],
        lag_days=[1],
    )

    timestamp = pd.Timestamp("2026-01-03T12:00:00+01:00")
    assert np.isclose(result.loc[timestamp, "weather_weighted_t2m_C_daily_mean"], 16.5)
    assert np.isclose(result.loc[timestamp, "weather_weighted_t2m_C_daily_mean_diff_d1"], 4.5)
    assert np.isclose(result.loc[timestamp, "weather_weighted_t2m_C_daily_range"], 3.0)


def test_weather_cluster_spread_features_add_cross_cluster_stats() -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=2, freq="15min")
    features = pd.DataFrame(
        {
            "weather_t2m_C_cluster_0": [10.0, 20.0],
            "weather_t2m_C_cluster_1": [20.0, 40.0],
            "weather_hdd18_cluster_0": [8.0, 0.0],
            "weather_hdd18_cluster_1": [0.0, 0.0],
        },
        index=index,
    )

    result = lf._add_weather_cluster_spread_features(
        features,
        base_names=["t2m_C"],
        stats=["min", "max", "range", "std", "p10", "p90"],
    )

    assert "weather_spread_t2m_C_range" in result.columns
    assert "weather_spread_hdd18_range" not in result.columns
    assert np.isclose(result.loc[index[0], "weather_spread_t2m_C_range"], 10.0)
    assert np.isclose(result.loc[index[1], "weather_spread_t2m_C_p90"], 38.0)


def test_weighted_weather_inertia_features_add_rolling_history() -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=12, freq="15min")
    features = pd.DataFrame({"weather_weighted_t2m_C": np.arange(len(index), dtype=float)}, index=index)

    result = lf._add_weighted_weather_inertia_features(
        features,
        prefix="weather_weighted",
        base_names=["t2m_C"],
        windows_hours=[1],
        stats=["mean", "range", "delta_mean"],
    )

    timestamp = index[4]
    assert np.isclose(result.loc[timestamp, "weather_weighted_t2m_C_roll1h_mean"], 2.5)
    assert np.isclose(result.loc[timestamp, "weather_weighted_t2m_C_roll1h_range"], 3.0)
    assert np.isclose(result.loc[timestamp, "weather_weighted_t2m_C_minus_roll1h_mean"], 1.5)


def test_rolling_load_forecast_can_model_entsoe_residual(tmp_path: Path) -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=20 * 96, freq="15min")
    mtu = index.hour * 4 + index.minute // 15
    benchmark = 45_000.0 + 900.0 * np.sin(2 * np.pi * mtu / 96)
    residual = 100.0 + np.arange(len(index), dtype=float) * 0.05
    dataset = pd.DataFrame(index=index)
    dataset["Load_Benchmark_MW"] = benchmark
    dataset["feature_residual_trend"] = np.arange(len(index), dtype=float)
    dataset["Load_Actual_MW"] = benchmark + residual

    config = _config(
        tmp_path,
        load_target_mode="entsoe_residual",
        test_start=pd.Timestamp("2026-01-12").date(),
        test_end=pd.Timestamp("2026-01-12").date(),
    )
    forecast, _ = lf.rolling_load_forecast(dataset, config)

    assert "Load_Benchmark_MW" in forecast.columns
    assert forecast["Load_Model_MW"].notna().all()
    assert forecast["Load_Model_MW"].mean() > forecast["Load_Benchmark_MW"].mean()


def test_rolling_bias_correction_uses_only_past_available_errors(tmp_path: Path) -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=4 * 96, freq="15min")
    forecast = pd.DataFrame(
        {
            "Load_Model_MW": np.full(len(index), 110.0),
            "Load_Actual_MW": np.full(len(index), 100.0),
        },
        index=index,
    )
    config = _config(
        tmp_path,
        apply_rolling_bias_correction=True,
        bias_correction_group="hour",
        bias_correction_window_days=7,
        bias_correction_min_observations=4,
        target_availability_lag_days=1,
    )

    corrected = lf.apply_rolling_bias_correction(forecast, config)

    assert corrected.loc[index[0], "Load_Model_BiasCorrected_MW"] == 110.0
    assert np.isclose(corrected.loc[pd.Timestamp("2026-01-03T12:00:00+01:00"), "Load_Model_BiasCorrected_MW"], 100.0)


def test_rolling_bias_correction_handles_duplicate_timestamps(tmp_path: Path) -> None:
    base_index = pd.date_range("2026-01-01T00:00:00+01:00", periods=4 * 96, freq="15min")
    index = base_index.insert(120, base_index[120])
    forecast = pd.DataFrame(
        {
            "Load_Model_MW": np.full(len(index), 110.0),
            "Load_Actual_MW": np.full(len(index), 100.0),
        },
        index=index,
    )
    config = _config(
        tmp_path,
        apply_rolling_bias_correction=True,
        bias_correction_group="hour",
        bias_correction_window_days=7,
        bias_correction_min_observations=4,
        target_availability_lag_days=1,
    )

    corrected = lf.apply_rolling_bias_correction(forecast, config)

    assert "Load_Model_BiasCorrected_MW" in corrected.columns
    assert corrected["Load_Model_BiasCorrected_MW"].notna().all()


def test_rolling_load_residual_ensemble_uses_only_available_history() -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=12 * 96, freq="15min")
    benchmark = pd.Series(50_000.0, index=index)
    actual = pd.Series(50_000.0, index=index)
    source_a = pd.DataFrame(
        {
            "Load_Model_MW": benchmark + 100.0,
            "Load_Benchmark_MW": benchmark,
            "Load_Actual_MW": actual,
        },
        index=index,
    )
    source_b = pd.DataFrame(
        {
            "Load_Model_MW": benchmark + 500.0,
            "Load_Benchmark_MW": benchmark,
            "Load_Actual_MW": actual,
        },
        index=index,
    )
    # Make source B look artificially perfect on D-1. A D forecast may not use that day's actual errors.
    d_minus_1 = index.normalize() == pd.Timestamp("2026-01-09T00:00:00+01:00")
    source_b.loc[d_minus_1, "Load_Model_MW"] = source_b.loc[d_minus_1, "Load_Actual_MW"]

    forecast, calibration = lf.build_rolling_load_residual_ensemble(
        {"a": source_a.loc["2026-01-01":"2026-01-10"], "b": source_b.loc["2026-01-01":"2026-01-10"]},
        train_days=7,
        min_train_days=2,
        target_availability_lag_days=1,
        weight_grid=[0.0, 1.0],
        shrink_grid=[1.0],
    )

    row = calibration.loc[calibration["forecast_day"].eq(pd.Timestamp("2026-01-10T00:00:00+01:00"))].iloc[0]
    assert row["weight_a"] == 1.0
    assert np.isclose(forecast.loc[pd.Timestamp("2026-01-10T12:00:00+01:00"), "Load_Model_MW"], 50_100.0)


def test_rolling_load_forecast_outputs_complete_day(tmp_path: Path) -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=20 * 96, freq="15min")
    mtu = index.hour * 4 + index.minute // 15
    load = 45_000.0 + 1_000.0 * np.sin(2 * np.pi * mtu / 96) + np.arange(len(index), dtype=float) * 0.5
    dataset = pd.DataFrame(index=index)
    dataset["feature_trend"] = np.arange(len(index), dtype=float)
    dataset["feature_mtu_sin"] = np.sin(2 * np.pi * mtu / 96)
    dataset["Load_Actual_MW"] = load

    config = _config(
        tmp_path,
        test_start=pd.Timestamp("2026-01-12").date(),
        test_end=pd.Timestamp("2026-01-12").date(),
    )
    forecast, runtime = lf.rolling_load_forecast(dataset, config)

    assert len(forecast) == 96
    assert forecast.columns.tolist() == ["Load_Model_MW", "Load_Actual_MW"]
    assert forecast["Load_Model_MW"].notna().all()
    assert runtime.loc[0, "train_rows"] >= 2 * 96


def test_rolling_load_forecast_predicts_future_day_without_actual_target(tmp_path: Path) -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=20 * 96, freq="15min")
    mtu = index.hour * 4 + index.minute // 15
    load = 45_000.0 + 1_000.0 * np.sin(2 * np.pi * mtu / 96) + np.arange(len(index), dtype=float) * 0.5
    dataset = pd.DataFrame(index=index)
    dataset["feature_trend"] = np.arange(len(index), dtype=float)
    dataset["feature_mtu_sin"] = np.sin(2 * np.pi * mtu / 96)
    dataset["Load_Actual_MW"] = load
    target_day = pd.Timestamp("2026-01-12T00:00:00+01:00")
    dataset.loc[dataset.index.normalize() == target_day, "Load_Actual_MW"] = np.nan

    config = _config(
        tmp_path,
        test_start=target_day.date(),
        test_end=target_day.date(),
    )
    forecast, _ = lf.rolling_load_forecast(dataset, config)

    assert len(forecast) == 96
    assert forecast["Load_Model_MW"].notna().all()
    assert forecast["Load_Actual_MW"].isna().all()


def test_rolling_load_forecast_supports_hour_block_models(tmp_path: Path) -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=20 * 96, freq="15min")
    mtu = index.hour * 4 + index.minute // 15
    load = 45_000.0 + 1_000.0 * np.sin(2 * np.pi * mtu / 96) + np.arange(len(index), dtype=float) * 0.5
    dataset = pd.DataFrame(index=index)
    dataset["feature_trend"] = np.arange(len(index), dtype=float)
    dataset["feature_mtu_sin"] = np.sin(2 * np.pi * mtu / 96)
    dataset["Load_Actual_MW"] = load

    config = _config(
        tmp_path,
        test_start=pd.Timestamp("2026-01-12").date(),
        test_end=pd.Timestamp("2026-01-12").date(),
        model_granularity="hour_block",
        hour_block_boundaries=[0, 12, 24],
    )
    forecast, runtime = lf.rolling_load_forecast(dataset, config)

    assert len(forecast) == 96
    assert forecast["Load_Model_MW"].notna().all()
    assert runtime.loc[0, "model_granularity"] == "hour_block"
    assert runtime.loc[0, "n_models"] == 2


def test_evaluate_load_forecast_includes_r2() -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=4, freq="15min")
    forecast = pd.DataFrame(
        {
            "Load_Model_MW": [1.0, 2.0, 3.0, 4.0],
            "Load_Actual_MW": [1.0, 2.0, 3.0, 5.0],
        },
        index=index,
    )

    metrics = lf.evaluate_load_forecast(forecast)

    assert "r2" in metrics.columns
    assert np.isclose(metrics.loc[metrics["period"].eq("full"), "r2"].iloc[0], 0.8857142857142857)


def test_entsoe_load_forecast_benchmark_frame_and_metrics(monkeypatch, tmp_path: Path) -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=2, freq="15min")
    actual = pd.DataFrame({"load_actual": [10.0, 20.0]}, index=index)
    forecast = pd.DataFrame({"load_fc": [11.0, 19.0]}, index=index)

    monkeypatch.setattr(lf, "_load_or_fetch_actual_load", lambda config: actual)
    monkeypatch.setattr(lf, "_load_or_fetch_entsoe_load_forecast", lambda config: forecast)

    config = EntsoeLoadForecastBenchmarkConfig(
        repo_root=tmp_path,
        actual_load_file=tmp_path / "actual.csv",
        forecast_file=tmp_path / "forecast.csv",
        export_dir=tmp_path / "out",
        test_start=None,
        test_end=None,
    )

    result = lf.build_entsoe_load_forecast_benchmark(config)
    metrics = lf.evaluate_load_forecast(result)

    assert result.columns.tolist() == ["Load_Benchmark_MW", "Load_Actual_MW"]
    assert result.loc[index[0], "Load_Benchmark_MW"] == 11.0
    assert metrics.loc[metrics["period"].eq("full"), "model"].iloc[0] == "Benchmark"
    assert metrics.loc[metrics["period"].eq("full"), "mae"].iloc[0] == 1.0


def test_entsoe_load_forecast_benchmark_respects_test_window(monkeypatch, tmp_path: Path) -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=3 * 96, freq="15min")
    actual = pd.DataFrame({"load_actual": np.arange(len(index), dtype=float)}, index=index)
    forecast = pd.DataFrame({"load_fc": np.arange(len(index), dtype=float)}, index=index)

    monkeypatch.setattr(lf, "_load_or_fetch_actual_load", lambda config: actual)
    monkeypatch.setattr(lf, "_load_or_fetch_entsoe_load_forecast", lambda config: forecast)

    config = EntsoeLoadForecastBenchmarkConfig(
        repo_root=tmp_path,
        actual_load_file=tmp_path / "actual.csv",
        forecast_file=tmp_path / "forecast.csv",
        export_dir=tmp_path / "out",
        test_start=pd.Timestamp("2026-01-02").date(),
        test_end=pd.Timestamp("2026-01-02").date(),
    )

    result = lf.build_entsoe_load_forecast_benchmark(config)

    assert len(result) == 96
    assert result.index.min() == pd.Timestamp("2026-01-02T00:00:00+01:00")
    assert result.index.max() == pd.Timestamp("2026-01-02T23:45:00+01:00")
