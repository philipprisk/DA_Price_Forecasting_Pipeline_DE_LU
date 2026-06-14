from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from da_price_forecasting.config import RenewableGenerationModelConfig, RunConfig, validate_config_payload
from da_price_forecasting.pipelines.common import load_timestamp_csv, save_timestamp_csv
from da_price_forecasting.scripts import energy_arena_renewable_daily as daily


def test_renewable_daily_submission_payload_uses_forecast_file_source(tmp_path: Path) -> None:
    forecast_path = tmp_path / "forecast.csv"
    payload = daily.build_renewable_submission_payload(
        forecast_path=forecast_path,
        forecast_date=date(2026, 5, 26),
        challenge_id=44,
        submit=False,
        target_tz="Europe/Berlin",
        value_column="Solar_Model_MW",
        source_name="renewable_generation_solar",
        approach_name="solar_model",
        approach_description=None,
    )

    assert payload["submit"] is False
    assert payload["config"]["forecast_date"] == "2026-05-26"
    assert payload["config"]["challenge_id"] == 44
    assert payload["config"]["source"]["kind"] == "forecast_file"
    assert payload["config"]["source"]["value_column"] == "Solar_Model_MW"
    assert payload["config"]["value_column"] == "Solar_Model_MW"
    validate_config_payload(payload, RunConfig, repo_root=Path.cwd())


def test_renewable_daily_rewrites_feature_and_model_dates(tmp_path: Path) -> None:
    repo_root = Path.cwd()
    forecast_day = date(2026, 5, 26)
    feature_payload = daily.build_renewable_feature_payload(
        feature_config_path=daily.DEFAULT_FEATURE_CONFIG,
        forecast_date=forecast_day,
        repo_root=repo_root,
    )
    feature_config = feature_payload["config"]

    assert feature_config["open_meteo_end_date"] == "2026-05-26"

    proxy_file = tmp_path / "regional_renewable_features.csv"
    model_payload = daily.build_renewable_model_payload(
        model_config_path=daily.DEFAULT_MODEL_CONFIG,
        forecast_date=forecast_day,
        forecast_dir=tmp_path / "forecast_run",
        renewable_proxy_file=proxy_file,
        repo_root=repo_root,
    )
    model_config = model_payload["config"]

    assert model_config["test_start"] == "2026-05-26"
    assert model_config["test_end"] == "2026-05-26"
    assert model_config["entsoe_end_date"] == "2026-05-25"
    assert model_config["renewable_proxy_file"] == str(proxy_file)
    assert model_config["export_dir"] == str(tmp_path / "forecast_run")
    validate_config_payload(model_payload, RunConfig, repo_root=repo_root)


def test_renewable_daily_leaves_dwd_feature_config_dates_unchanged() -> None:
    payload = daily.build_renewable_feature_payload(
        feature_config_path=Path(
            "configs/regional_renewable_features_dwd_icon_mastr_solar_tso_c25_run06_solar_spread.yaml"
        ),
        forecast_date=date(2026, 5, 26),
        repo_root=Path.cwd(),
    )

    assert payload["config"]["weather_source"] == "dwd_icon"
    assert "open_meteo_end_date" not in payload["config"]
    validate_config_payload(payload, RunConfig, repo_root=Path.cwd())


def test_dated_renewable_work_paths_are_per_forecast_day(tmp_path: Path) -> None:
    paths = daily.dated_renewable_work_paths(tmp_path, date(2026, 5, 26), work_root=tmp_path / "work")

    assert paths.work_dir == tmp_path / "work" / "2026-05-26"
    assert paths.forecast_file == paths.work_dir / "forecast_run" / "forecast.csv"
    assert paths.solar_submission_config == paths.work_dir / "generated_configs" / "energy_arena_solar_submission.generated.yaml"
    assert paths.wind_submission_config == paths.work_dir / "generated_configs" / "energy_arena_wind_submission.generated.yaml"
    assert paths.extra_feature_config(2) == paths.work_dir / "generated_configs" / "regional_renewable_features_extra_2.generated.yaml"


def test_update_actual_generation_cache_refreshes_overlap(monkeypatch, tmp_path: Path) -> None:
    cache = tmp_path / "actual_generation.csv"
    existing_index = pd.date_range("2026-05-19T00:00:00+02:00", periods=2, freq="D")
    save_timestamp_csv(
        pd.DataFrame(
            {
                "Solar_Actual_MW": [1.0, 2.0],
                "Wind_Onshore_Actual_MW": [3.0, 4.0],
                "Wind_Offshore_Actual_MW": [5.0, 6.0],
                "Wind_Total_Actual_MW": [8.0, 10.0],
                "Renewable_Total_Actual_MW": [9.0, 12.0],
            },
            index=existing_index,
        ),
        cache,
    )

    captured = {}
    fetched_index = pd.date_range("2026-05-19T00:00:00+02:00", periods=4, freq="D")

    def fake_fetch_actual_renewable_generation(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            {
                "Solar_Actual_MW": np.arange(10.0, 14.0),
                "Wind_Onshore_Actual_MW": np.arange(20.0, 24.0),
                "Wind_Offshore_Actual_MW": np.arange(30.0, 34.0),
                "Wind_Total_Actual_MW": np.arange(50.0, 54.0),
                "Renewable_Total_Actual_MW": np.arange(60.0, 64.0),
            },
            index=fetched_index,
        )

    monkeypatch.setattr(daily, "fetch_actual_renewable_generation", fake_fetch_actual_renewable_generation)

    config = RenewableGenerationModelConfig(
        repo_root=tmp_path,
        entsoe_start_date=date(2026, 5, 1),
        entsoe_end_date=date(2026, 5, 22),
        actual_generation_file=cache,
        renewable_proxy_file=tmp_path / "proxy.csv",
        unavailability_file=tmp_path / "unavailability.csv",
        icon_dir=tmp_path / "icon",
        export_dir=tmp_path / "export",
        target_availability_lag_days=1,
        target_availability_cutoff_hour=10,
        test_start=date(2026, 5, 23),
        test_end=date(2026, 5, 23),
    )

    daily.update_actual_generation_cache(config, forecast_date=date(2026, 5, 23))

    assert captured["start_day"].date().isoformat() == "2026-05-19"
    assert captured["end_day"].date().isoformat() == "2026-05-22"

    updated = load_timestamp_csv(cache, "Europe/Berlin")
    assert updated.index.max().date().isoformat() == "2026-05-22"
    assert updated.loc[pd.Timestamp("2026-05-19T00:00:00+02:00"), "Solar_Actual_MW"] == 10.0
    assert updated.loc[pd.Timestamp("2026-05-22T00:00:00+02:00"), "Renewable_Total_Actual_MW"] == 63.0


def test_update_actual_generation_cache_refreshes_solar_control_area_targets(monkeypatch, tmp_path: Path) -> None:
    cache = tmp_path / "actual_generation.csv"
    fetched_index = pd.date_range("2026-05-21T00:00:00+02:00", periods=2, freq="D")

    def fake_fetch_actual_renewable_generation(**kwargs):
        return pd.DataFrame(
            {
                "Solar_Actual_MW": [1.0, 2.0],
                "Wind_Onshore_Actual_MW": [3.0, 4.0],
                "Wind_Offshore_Actual_MW": [5.0, 6.0],
                "Wind_Total_Actual_MW": [8.0, 10.0],
                "Renewable_Total_Actual_MW": [9.0, 12.0],
            },
            index=fetched_index,
        )

    captured = {}

    def fake_fetch_actual_solar_generation_by_control_area(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            {
                "Solar_50Hertz_Actual_MW": [10.0, 11.0],
                "Solar_Amprion_Actual_MW": [20.0, 21.0],
                "Solar_TenneT_Actual_MW": [30.0, 31.0],
                "Solar_TransnetBW_Actual_MW": [40.0, 41.0],
            },
            index=fetched_index,
        )

    monkeypatch.setattr(daily, "fetch_actual_renewable_generation", fake_fetch_actual_renewable_generation)
    monkeypatch.setattr(
        daily,
        "fetch_actual_solar_generation_by_control_area",
        fake_fetch_actual_solar_generation_by_control_area,
    )

    config = RenewableGenerationModelConfig(
        repo_root=tmp_path,
        entsoe_start_date=date(2026, 5, 21),
        entsoe_end_date=date(2026, 5, 22),
        include_solar_control_area_targets=True,
        target_columns=[
            "Solar_50Hertz_Actual_MW",
            "Solar_Amprion_Actual_MW",
            "Solar_TenneT_Actual_MW",
            "Solar_TransnetBW_Actual_MW",
        ],
        actual_generation_file=cache,
        renewable_proxy_file=tmp_path / "proxy.csv",
        unavailability_file=tmp_path / "unavailability.csv",
        icon_dir=tmp_path / "icon",
        export_dir=tmp_path / "export",
        target_availability_lag_days=1,
        target_availability_cutoff_hour=10,
        test_start=date(2026, 5, 23),
        test_end=date(2026, 5, 23),
    )

    daily.update_actual_generation_cache(config, forecast_date=date(2026, 5, 23))

    assert captured["control_area_targets"] == config.solar_control_area_targets
    updated = load_timestamp_csv(cache, "Europe/Berlin")
    assert updated.loc[pd.Timestamp("2026-05-22T00:00:00+02:00"), "Solar_50Hertz_Actual_MW"] == 11.0
    assert updated.loc[pd.Timestamp("2026-05-22T00:00:00+02:00"), "Solar_TransnetBW_Actual_MW"] == 41.0
