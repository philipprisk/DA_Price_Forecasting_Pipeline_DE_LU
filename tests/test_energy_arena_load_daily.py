from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from da_price_forecasting.config import EnergyArenaSubmissionConfig, RunConfig, validate_config_payload
from da_price_forecasting.integrations.energy_arena.sources import load_or_run_forecast_source
from da_price_forecasting.scripts.energy_arena_load_daily import (
    build_load_submission_payload,
    dated_load_work_paths,
)


def test_load_daily_payload_uses_load_forecast_source() -> None:
    payload = build_load_submission_payload(
        model_config_path=Path(
            "configs/load_forecast_hybrid_entsoe_residual_c25_run06_rich_temp_daily_weather_hgb_smooth_febapr.yaml"
        ),
        forecast_date=date(2026, 5, 16),
        challenge_id=42,
        submit=False,
        target_tz="Europe/Berlin",
        value_column="Load_Model_MW",
        approach_name=None,
        approach_description=None,
    )

    assert payload["submit"] is False
    assert payload["config"]["forecast_date"] == "2026-05-16"
    assert payload["config"]["challenge_id"] == 42
    assert payload["config"]["source"]["kind"] == "load_forecast_model"
    assert payload["config"]["value_column"] == "Load_Model_MW"
    validate_config_payload(payload, RunConfig, repo_root=Path.cwd())


def test_dated_load_work_paths_are_per_forecast_day(tmp_path: Path) -> None:
    paths = dated_load_work_paths(tmp_path, date(2026, 5, 16), work_root=tmp_path / "work")

    assert paths.work_dir == tmp_path / "work" / "2026-05-16"
    assert paths.submission_config == paths.work_dir / "generated_configs" / "energy_arena_load_submission.generated.yaml"


def test_load_energy_arena_source_rewrites_forecast_window(monkeypatch, tmp_path: Path) -> None:
    from da_price_forecasting.pipelines import load_forecast as lf

    captured = {}
    index = pd.date_range("2026-05-16T00:00:00+02:00", periods=96, freq="15min")

    def fake_run_load_forecast_pipeline(config, save_outputs):
        captured["config"] = config
        captured["save_outputs"] = save_outputs
        return {
            "forecast": pd.DataFrame({"Load_Model_MW": np.arange(96, dtype=float)}, index=index),
        }

    monkeypatch.setattr(lf, "run_load_forecast_pipeline", fake_run_load_forecast_pipeline)

    submission_config = validate_config_payload(
        {
            "repo_root": tmp_path,
            "source": {
                "kind": "load_forecast_model",
                "run_before_submit": True,
                "config": {
                    "repo_root": str(tmp_path),
                    "actual_load_file": str(tmp_path / "actual_load.csv"),
                    "entsoe_load_forecast_file": str(tmp_path / "load_forecast.csv"),
                    "icon_dir": str(tmp_path / "icon"),
                    "export_dir": str(tmp_path / "old_export"),
                    "include_weather_features": False,
                    "include_holiday_features": False,
                },
            },
            "forecast_date": "2026-05-16",
            "challenge_id": 42,
            "target_tz": "Europe/Berlin",
            "objective": "point",
            "value_column": "Load_Model_MW",
        },
        EnergyArenaSubmissionConfig,
        repo_root=tmp_path,
    )

    forecast_dir = tmp_path / "forecast_run"
    result = load_or_run_forecast_source(submission_config, forecast_dir=forecast_dir)

    assert result.default_value_column == "Load_Model_MW"
    assert len(result.forecast) == 96
    assert captured["save_outputs"] is True
    assert captured["config"].test_start.isoformat() == "2026-05-16"
    assert captured["config"].test_end.isoformat() == "2026-05-16"
    assert captured["config"].entsoe_end_date.isoformat() == "2026-05-16"
    assert captured["config"].export_dir == forecast_dir


def test_load_energy_arena_source_accepts_runner_config_path(monkeypatch, tmp_path: Path) -> None:
    from da_price_forecasting.pipelines import load_forecast as lf

    captured = {}
    index = pd.date_range("2026-05-16T00:00:00+02:00", periods=96, freq="15min")

    def fake_run_load_forecast_pipeline(config, save_outputs):
        captured["config"] = config
        return {
            "forecast": pd.DataFrame({"Load_Model_MW": np.arange(96, dtype=float)}, index=index),
        }

    monkeypatch.setattr(lf, "run_load_forecast_pipeline", fake_run_load_forecast_pipeline)

    runner_config = tmp_path / "load_runner_config.json"
    runner_config.write_text(
        json.dumps(
            {
                "kind": "load_forecast_model",
                "config": {
                    "repo_root": str(tmp_path),
                    "actual_load_file": "actual_load.csv",
                    "entsoe_load_forecast_file": "entsoe_load_forecast.csv",
                    "icon_dir": "icon_aggregated_c25_run06",
                    "export_dir": "old_export",
                    "include_weather_features": False,
                    "include_holiday_features": False,
                },
            }
        ),
        encoding="utf-8",
    )

    submission_config = validate_config_payload(
        {
            "repo_root": tmp_path,
            "source": {
                "kind": "load_forecast_model",
                "run_before_submit": True,
                "config_path": str(runner_config),
            },
            "forecast_date": "2026-05-16",
            "challenge_id": 42,
            "target_tz": "Europe/Berlin",
            "objective": "point",
            "value_column": "Load_Model_MW",
        },
        EnergyArenaSubmissionConfig,
        repo_root=tmp_path,
    )

    forecast_dir = tmp_path / "forecast_run"
    result = load_or_run_forecast_source(submission_config, forecast_dir=forecast_dir)

    assert len(result.forecast) == 96
    assert captured["config"].icon_dir == tmp_path / "icon_aggregated_c25_run06"
    assert captured["config"].export_dir == forecast_dir
