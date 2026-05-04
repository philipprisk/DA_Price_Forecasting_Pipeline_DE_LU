from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from da_price_forecasting.config import RunConfig, validate_config_payload
from da_price_forecasting.scripts.energy_arena_daily import (
    _parse_retry_until,
    _retry_deadline,
    _validate_point_base_dataset,
    build_point_submission_payload,
    build_quantile_submission_payload,
    dated_work_paths,
    day_bounds,
    prepare_lear_config_for_point_base,
    sqra_train_days,
    tomorrow_in_tz,
)


def _point_payload() -> dict:
    return {
        "kind": "energy_arena_submit",
        "submit": True,
        "config": {
            "repo_root": ".",
            "source": {
                "kind": "lear_operational",
                "run_before_submit": True,
                "config": {
                    "repo_root": ".",
                    "target_tz": "Europe/Berlin",
                    "entsoe_api_key_env": "ENTSOE_API_KEY",
                    "entsoe_start_date": "2025-01-01T00:00:00+01:00",
                    "entsoe_end_date": "2026-04-28T00:00:00+02:00",
                    "variant": "exaa_only",
                    "lars_start_date": "2025-12-01T00:00:00+01:00",
                    "test_start": "2026-04-28T00:00:00+02:00",
                    "test_end": "2026-04-28T23:45:00+02:00",
                    "train_days_rolling": 56,
                    "n_clusters": 5,
                },
            },
            "forecast_date": "2026-04-28",
            "challenge_id": 2,
            "target_tz": "Europe/Berlin",
            "objective": "point",
            "value_column": "y_pred",
            "approach_name": "lear_exaa_only",
        },
    }


def _quantile_payload() -> dict:
    return {
        "kind": "energy_arena_submit",
        "submit": True,
        "config": {
            "repo_root": ".",
            "source": {
                "kind": "sqra",
                "run_before_submit": True,
                "config": {
                    "repo_root": ".",
                    "import_paths": ["results/old/forecast.csv"],
                    "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
                    "test_start": "2026-04-28T00:00:00+02:00",
                    "test_end": "2026-04-28T23:45:00+02:00",
                    "train_days_rolling": 60,
                    "experiment_name": "sqra_exaa_only_d60_energy_arena",
                    "export_dir": "results/old/sqra_forecast",
                },
            },
            "forecast_date": "2026-04-28",
            "challenge_id": 8,
            "target_tz": "Europe/Berlin",
            "objective": "quantile",
            "quantile_columns": ["q0.100", "q0.250", "q0.500", "q0.750", "q0.900"],
            "approach_name": "sqra_exaa_only",
        },
    }


def test_tomorrow_and_day_bounds_use_target_timezone() -> None:
    now = datetime(2026, 4, 28, 11, 30, tzinfo=ZoneInfo("Europe/Berlin"))

    assert tomorrow_in_tz("Europe/Berlin", now=now) == date(2026, 4, 29)
    assert day_bounds(date(2026, 4, 29), "Europe/Berlin") == (
        "2026-04-29T00:00:00+02:00",
        "2026-04-29T23:45:00+02:00",
    )


def test_daily_payloads_rewrite_dates_and_point_forecast_paths(tmp_path: Path) -> None:
    forecast_date = date(2026, 4, 29)
    point_forecast = tmp_path / "point_base" / "forecast.csv"
    sqra_export = tmp_path / "sqra_forecast"

    point_payload = build_point_submission_payload(
        point_payload=_point_payload(),
        forecast_date=forecast_date,
        point_forecast_path=point_forecast,
        point_source_name="lear_exaa_only_d56",
        submit=False,
    )
    quantile_payload = build_quantile_submission_payload(
        quantile_payload=_quantile_payload(),
        forecast_date=forecast_date,
        point_forecast_path=point_forecast,
        sqra_export_dir=sqra_export,
        submit=False,
        repo_root=Path.cwd(),
    )

    assert point_payload["submit"] is False
    assert point_payload["config"]["forecast_date"] == "2026-04-29"
    assert point_payload["config"]["source"] == {
        "kind": "forecast_file",
        "path": str(point_forecast),
        "name": "lear_exaa_only_d56",
        "value_column": "y_pred",
    }
    validate_config_payload(point_payload, RunConfig, repo_root=Path.cwd())

    sqra_config = quantile_payload["config"]["source"]["config"]
    assert quantile_payload["submit"] is False
    assert quantile_payload["config"]["forecast_date"] == "2026-04-29"
    assert sqra_config["import_paths"] == [str(point_forecast)]
    assert sqra_config["test_start"] == "2026-04-29T00:00:00+02:00"
    assert sqra_config["test_end"] == "2026-04-29T23:45:00+02:00"
    assert sqra_config["export_dir"] == str(sqra_export)
    validate_config_payload(quantile_payload, RunConfig, repo_root=Path.cwd())


def test_prepare_lear_config_for_point_base_uses_dated_work_dir(tmp_path: Path) -> None:
    forecast_date = date(2026, 4, 29)
    paths = dated_work_paths(Path.cwd(), forecast_date, work_root=tmp_path / "work")

    config = prepare_lear_config_for_point_base(
        point_payload=_point_payload(),
        forecast_date=forecast_date,
        paths=paths,
        repo_root=Path.cwd(),
    )

    assert config.entsoe_end_date.isoformat() == "2026-04-29T00:00:00+02:00"
    assert config.test_end.isoformat() == "2026-04-29T23:45:00+02:00"
    assert config.resolved_export_dir == paths.point_base_dir


def test_sqra_train_days_are_read_from_quantile_config() -> None:
    assert sqra_train_days(_quantile_payload(), Path.cwd()) == 60


def test_retry_deadline_uses_same_target_timezone_day() -> None:
    now = datetime(2026, 4, 28, 11, 30, tzinfo=ZoneInfo("Europe/Berlin"))

    deadline = _retry_deadline(now, _parse_retry_until("12:30"), "Europe/Berlin")

    assert deadline is not None
    assert deadline.isoformat() == "2026-04-28T12:30:00+02:00"


def test_missing_target_feature_day_raises_actionable_error() -> None:
    forecast_day = pd.Timestamp("2026-04-29", tz="Europe/Berlin")
    available_index = pd.DatetimeIndex(["2026-04-28"], tz="Europe/Berlin")
    X = pd.DataFrame({"exaa_d0_mtu_00": [42.0]}, index=available_index)
    forecast_days = pd.date_range("2026-04-28", "2026-04-29", freq="D", tz="Europe/Berlin")

    with pytest.raises(RuntimeError, match="No complete EXAA-only feature row"):
        _validate_point_base_dataset(X=X, forecast_days=forecast_days, forecast_day=forecast_day, train_days=56)


def test_target_feature_day_without_training_rows_raises_actionable_error() -> None:
    forecast_day = pd.Timestamp("2026-04-29", tz="Europe/Berlin")
    X = pd.DataFrame({"exaa_d0_mtu_00": [42.0]}, index=pd.DatetimeIndex([forecast_day]))

    with pytest.raises(RuntimeError, match="not enough historical EXAA feature days"):
        _validate_point_base_dataset(
            X=X,
            forecast_days=pd.DatetimeIndex([forecast_day]),
            forecast_day=forecast_day,
            train_days=56,
        )
