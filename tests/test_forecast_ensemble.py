from __future__ import annotations

import pandas as pd

from da_price_forecasting.config.forecast_ensemble import ForecastEnsembleConfig
from da_price_forecasting.pipelines.forecast_ensemble import run_forecast_ensemble_pipeline


def test_forecast_ensemble_averages_common_timestamps(tmp_path) -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=2, freq="15min")
    first = pd.DataFrame({"y_pred": [1.0, 3.0], "y_true": [2.0, 4.0]}, index=index)
    second = pd.DataFrame({"y_pred": [3.0, 5.0], "y_true": [2.0, 4.0]}, index=index)
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    first.to_csv(first_path)
    second.to_csv(second_path)

    config = ForecastEnsembleConfig.model_validate(
        {
            "repo_root": ".",
            "sources": [
                {"name": "first", "path": first_path, "weight": 1.0},
                {"name": "second", "path": second_path, "weight": 3.0},
            ],
            "export_dir": tmp_path / "ensemble",
        }
    )

    forecast = run_forecast_ensemble_pipeline(config, save_outputs=False)

    assert forecast["y_pred"].tolist() == [2.5, 4.5]
    assert forecast["y_true"].tolist() == [2.0, 4.0]


def test_forecast_ensemble_rolling_inverse_mae_learns_from_past_days(tmp_path) -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=3 * 96, freq="15min")
    y_true = pd.Series(10.0, index=index)
    first = pd.DataFrame({"y_pred": 10.0, "y_true": y_true}, index=index)
    second = pd.DataFrame({"y_pred": 20.0, "y_true": y_true}, index=index)
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    first.to_csv(first_path)
    second.to_csv(second_path)

    config = ForecastEnsembleConfig.model_validate(
        {
            "repo_root": ".",
            "method": "rolling_inverse_mae",
            "train_days": 1,
            "min_train_days": 1,
            "sources": [
                {"name": "first", "path": first_path, "weight": 1.0},
                {"name": "second", "path": second_path, "weight": 1.0},
            ],
            "export_dir": tmp_path / "ensemble",
        }
    )

    forecast = run_forecast_ensemble_pipeline(config, save_outputs=False)

    assert forecast["y_pred"].iloc[:96].eq(15.0).all()
    assert forecast["y_pred"].iloc[96:].lt(10.01).all()


def test_forecast_ensemble_applies_rolling_bias_correction(tmp_path) -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=3 * 96, freq="15min")
    y_true = pd.Series(10.0, index=index)
    first = pd.DataFrame({"y_pred": 8.0, "y_true": y_true}, index=index)
    second = pd.DataFrame({"y_pred": 8.0, "y_true": y_true}, index=index)
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    first.to_csv(first_path)
    second.to_csv(second_path)

    config = ForecastEnsembleConfig.model_validate(
        {
            "repo_root": ".",
            "bias_correction": "rolling_mean_error",
            "bias_train_days": 1,
            "bias_min_train_days": 1,
            "sources": [
                {"name": "first", "path": first_path, "weight": 1.0},
                {"name": "second", "path": second_path, "weight": 1.0},
            ],
            "export_dir": tmp_path / "ensemble",
        }
    )

    forecast = run_forecast_ensemble_pipeline(config, save_outputs=False)

    assert forecast["y_pred"].iloc[:96].eq(8.0).all()
    assert forecast["y_pred"].iloc[96:].eq(10.0).all()


def test_forecast_ensemble_applies_rolling_hour_bias_correction(tmp_path) -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=3 * 96, freq="15min")
    y_true = pd.Series(10.0, index=index)
    y_pred = pd.Series(10.0, index=index)
    y_pred.loc[y_pred.index.hour == 17] = 8.0
    first = pd.DataFrame({"y_pred": y_pred, "y_true": y_true}, index=index)
    second = first.copy()
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    first.to_csv(first_path)
    second.to_csv(second_path)

    config = ForecastEnsembleConfig.model_validate(
        {
            "repo_root": ".",
            "bias_correction": "rolling_hour_mean_error",
            "bias_train_days": 1,
            "bias_min_train_days": 1,
            "sources": [
                {"name": "first", "path": first_path, "weight": 1.0},
                {"name": "second", "path": second_path, "weight": 1.0},
            ],
            "export_dir": tmp_path / "ensemble",
        }
    )

    forecast = run_forecast_ensemble_pipeline(config, save_outputs=False)

    assert forecast["y_pred"].iloc[:96][forecast.index[:96].hour == 17].eq(8.0).all()
    assert forecast["y_pred"].iloc[96:][forecast.index[96:].hour == 17].eq(10.0).all()
    assert forecast["y_pred"].iloc[96:][forecast.index[96:].hour == 16].eq(10.0).all()


def test_forecast_ensemble_rolling_regime_inverse_mae_runs(tmp_path) -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=4 * 96, freq="15min")
    y_true = pd.Series(10.0, index=index)
    first = pd.DataFrame({"y_pred": 10.0, "y_true": y_true}, index=index)
    second = pd.DataFrame({"y_pred": 20.0, "y_true": y_true}, index=index)
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    first.to_csv(first_path)
    second.to_csv(second_path)

    config = ForecastEnsembleConfig.model_validate(
        {
            "repo_root": ".",
            "method": "rolling_regime_inverse_mae",
            "train_days": 1,
            "min_train_days": 1,
            "sources": [
                {"name": "first", "path": first_path, "weight": 1.0},
                {"name": "second", "path": second_path, "weight": 1.0},
            ],
            "export_dir": tmp_path / "ensemble",
        }
    )

    forecast = run_forecast_ensemble_pipeline(config, save_outputs=False)

    assert len(forecast) == len(index)
    assert forecast["y_pred"].iloc[:96].eq(15.0).all()
    assert forecast["y_pred"].iloc[96:].lt(10.01).all()
