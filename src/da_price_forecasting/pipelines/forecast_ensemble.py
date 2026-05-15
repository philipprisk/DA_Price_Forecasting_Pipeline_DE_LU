from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd

from ..config.forecast_ensemble import ForecastEnsembleCalendarSwitchConfig, ForecastEnsembleConfig
from ..models.lear import save_experiment_outputs


def _load_forecast(path) -> pd.DataFrame:
    forecast = pd.read_csv(path, index_col=0)
    forecast.index = pd.to_datetime(forecast.index, utc=True).tz_convert("Europe/Berlin")
    required_columns = {"y_pred", "y_true"}
    missing = required_columns - set(forecast.columns)
    if missing:
        raise ValueError(f"Forecast file {path} is missing required columns: {sorted(missing)}")
    return forecast[["y_pred", "y_true"]].sort_index()


def _validate_y_true(joined: pd.DataFrame, true_columns: list[str]) -> pd.Series:
    y_true = joined[true_columns[0]].copy()
    for column in true_columns[1:]:
        if not np.allclose(y_true.to_numpy(dtype=float), joined[column].to_numpy(dtype=float), equal_nan=True):
            raise ValueError("Forecast ensemble sources have different y_true values on common timestamps.")
    return y_true


def _fixed_weight_predictions(joined: pd.DataFrame, pred_columns: list[str], weights: np.ndarray) -> np.ndarray:
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Forecast ensemble weights must be non-negative and sum to a positive value.")
    return np.average(joined[pred_columns].to_numpy(dtype=float), axis=1, weights=weights)


def _rolling_inverse_mae_predictions(
    joined: pd.DataFrame,
    pred_columns: list[str],
    y_true: pd.Series,
    *,
    train_days: int,
    min_train_days: int,
    weight_power: float,
    epsilon: float,
    fallback_weights: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    prediction_parts = []
    weight_records = []
    daily_index = joined.index.normalize()

    for day in daily_index.unique().sort_values():
        train_start = day - pd.Timedelta(days=train_days)
        train_end = day - pd.Timedelta(minutes=15)
        train_mask = (joined.index >= train_start) & (joined.index <= train_end)
        test_mask = daily_index == day

        train_days_available = joined.index[train_mask].normalize().nunique()
        if train_days_available >= min_train_days:
            errors = joined.loc[train_mask, pred_columns].sub(y_true.loc[train_mask], axis=0).abs()
            model_mae = errors.mean(axis=0).to_numpy(dtype=float)
            weights = 1.0 / np.power(model_mae + epsilon, weight_power)
            if not np.isfinite(weights).all() or weights.sum() <= 0:
                weights = fallback_weights.copy()
        else:
            weights = fallback_weights.copy()

        weights = weights / weights.sum()
        day_predictions = joined.loc[test_mask, pred_columns].to_numpy(dtype=float) @ weights
        prediction_parts.append(pd.Series(day_predictions, index=joined.index[test_mask]))
        weight_records.append(
            {
                "forecast_day": day,
                "train_days_available": int(train_days_available),
                **{f"weight_{column.removeprefix('y_pred_')}": float(weight) for column, weight in zip(pred_columns, weights)},
            }
        )

    return pd.concat(prediction_parts).sort_index().to_numpy(dtype=float), pd.DataFrame(weight_records)


def _regime_mask(
    reference_prediction: pd.Series,
    *,
    price_threshold: float,
    ramp_threshold: float,
) -> pd.DataFrame:
    ramp_1h = reference_prediction.diff(4).abs()
    return pd.DataFrame(
        {
            "is_peak_hour": reference_prediction.index.hour.isin([7, 8, 9, 10, 15, 16, 17, 18]),
            "is_high_price": reference_prediction >= price_threshold,
            "is_high_ramp": ramp_1h >= ramp_threshold,
        },
        index=reference_prediction.index,
    )


def _rolling_regime_inverse_mae_predictions(
    joined: pd.DataFrame,
    pred_columns: list[str],
    y_true: pd.Series,
    *,
    train_days: int,
    min_train_days: int,
    weight_power: float,
    epsilon: float,
    fallback_weights: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    prediction_parts = []
    weight_records = []
    daily_index = joined.index.normalize()
    reference_prediction = joined[pred_columns].mean(axis=1)

    for day in daily_index.unique().sort_values():
        train_start = day - pd.Timedelta(days=train_days)
        train_end = day - pd.Timedelta(minutes=15)
        train_mask = (joined.index >= train_start) & (joined.index <= train_end)
        test_mask = daily_index == day

        train_days_available = joined.index[train_mask].normalize().nunique()
        day_pred = joined.loc[test_mask, pred_columns].copy()
        day_weights = pd.DataFrame(index=day_pred.index, columns=pred_columns, dtype=float)

        if train_days_available >= min_train_days:
            train_reference = reference_prediction.loc[train_mask]
            price_threshold = float(train_reference.quantile(0.75))
            ramp_threshold = float(train_reference.diff(4).abs().quantile(0.80))
            train_regimes = _regime_mask(
                reference_prediction.loc[train_mask],
                price_threshold=price_threshold,
                ramp_threshold=ramp_threshold,
            )
            test_regimes = _regime_mask(
                reference_prediction.loc[test_mask],
                price_threshold=price_threshold,
                ramp_threshold=ramp_threshold,
            )
            global_errors = joined.loc[train_mask, pred_columns].sub(y_true.loc[train_mask], axis=0).abs()
            global_mae = global_errors.mean(axis=0).to_numpy(dtype=float)
            global_weights = 1.0 / np.power(global_mae + epsilon, weight_power)
            if not np.isfinite(global_weights).all() or global_weights.sum() <= 0:
                global_weights = fallback_weights.copy()
            global_weights = global_weights / global_weights.sum()

            for regime_values, regime_test_rows in test_regimes.groupby(
                ["is_peak_hour", "is_high_price", "is_high_ramp"], sort=False
            ):
                regime_train_mask = (
                    (train_regimes["is_peak_hour"] == regime_values[0])
                    & (train_regimes["is_high_price"] == regime_values[1])
                    & (train_regimes["is_high_ramp"] == regime_values[2])
                )
                if int(regime_train_mask.sum()) >= min_train_days * 4:
                    regime_index = train_regimes.index[regime_train_mask]
                    errors = joined.loc[regime_index, pred_columns].sub(y_true.loc[regime_index], axis=0).abs()
                    model_mae = errors.mean(axis=0).to_numpy(dtype=float)
                    weights = 1.0 / np.power(model_mae + epsilon, weight_power)
                    if not np.isfinite(weights).all() or weights.sum() <= 0:
                        weights = global_weights.copy()
                else:
                    weights = global_weights.copy()
                weights = weights / weights.sum()
                day_weights.loc[regime_test_rows.index, pred_columns] = weights
        else:
            day_weights.loc[:, pred_columns] = fallback_weights

        day_weights = day_weights.astype(float)
        day_predictions = (day_pred.to_numpy(dtype=float) * day_weights.to_numpy(dtype=float)).sum(axis=1)
        prediction_parts.append(pd.Series(day_predictions, index=day_pred.index))
        mean_weights = day_weights.mean(axis=0)
        weight_records.append(
            {
                "forecast_day": day,
                "train_days_available": int(train_days_available),
                **{
                    f"weight_{column.removeprefix('y_pred_')}": float(mean_weights[column])
                    for column in pred_columns
                },
            }
        )

    return pd.concat(prediction_parts).sort_index().to_numpy(dtype=float), pd.DataFrame(weight_records)


def _apply_rolling_bias_correction(
    forecast_df: pd.DataFrame,
    *,
    train_days: int,
    min_train_days: int,
    by_hour: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    corrected_parts = []
    bias_records = []
    daily_index = forecast_df.index.normalize()

    for day in daily_index.unique().sort_values():
        train_start = day - pd.Timedelta(days=train_days)
        train_end = day - pd.Timedelta(minutes=15)
        train_mask = (forecast_df.index >= train_start) & (forecast_df.index <= train_end)
        test_mask = daily_index == day

        train_days_available = forecast_df.index[train_mask].normalize().nunique()
        correction = 0.0
        if train_days_available >= min_train_days:
            past_error = forecast_df.loc[train_mask, "y_pred"] - forecast_df.loc[train_mask, "y_true"]
            correction = -float(past_error.mean())

        day_forecast = forecast_df.loc[test_mask].copy()
        if by_hour and train_days_available >= min_train_days:
            day_corrections = pd.Series(0.0, index=day_forecast.index)
            train_hours = forecast_df.index[train_mask].hour
            for hour in range(24):
                hour_train_mask = train_mask & (forecast_df.index.hour == hour)
                hour_days_available = forecast_df.index[hour_train_mask].normalize().nunique()
                hour_correction = correction
                if hour_days_available >= min_train_days:
                    hour_error = forecast_df.loc[hour_train_mask, "y_pred"] - forecast_df.loc[hour_train_mask, "y_true"]
                    hour_correction = -float(hour_error.mean())
                day_corrections.loc[day_corrections.index.hour == hour] = hour_correction
            day_forecast["y_pred"] = day_forecast["y_pred"] + day_corrections
        else:
            day_forecast["y_pred"] = day_forecast["y_pred"] + correction
        corrected_parts.append(day_forecast)
        bias_records.append(
            {
                "forecast_day": day,
                "train_days_available": int(train_days_available),
                "bias_correction": correction,
                "by_hour": bool(by_hour),
            }
        )

    return pd.concat(corrected_parts).sort_index(), pd.DataFrame(bias_records)


def _calendar_switch_mask(index: pd.DatetimeIndex, condition: str) -> pd.Series:
    if condition.startswith("weekday_"):
        weekday = int(condition.removeprefix("weekday_"))
        return pd.Series(index.dayofweek == weekday, index=index)

    is_weekend = pd.Series(index.dayofweek >= 5, index=index)
    if condition == "weekend":
        return is_weekend

    if condition in {"holiday", "nonworkday"}:
        import holidays

        de_holidays = holidays.Germany(years=sorted(index.year.unique()))
        is_holiday = pd.Series([timestamp.date() in de_holidays for timestamp in index], index=index)
        if condition == "holiday":
            return is_holiday
        return is_weekend | is_holiday

    raise ValueError(f"Unsupported calendar switch condition: {condition!r}")


def _apply_calendar_switches(
    forecast_df: pd.DataFrame,
    joined: pd.DataFrame,
    switches: list[ForecastEnsembleCalendarSwitchConfig],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not switches:
        return forecast_df, pd.DataFrame()

    out = forecast_df.copy()
    switch_records = []
    for switch in switches:
        source_column = f"y_pred_{switch.source}"
        if source_column not in joined.columns:
            raise ValueError(f"Calendar switch source {switch.source!r} is not available in the joined forecasts.")
        mask = _calendar_switch_mask(out.index, switch.condition)
        out.loc[mask, "y_pred"] = joined.loc[mask, source_column].to_numpy(dtype=float)
        switch_records.append(
            {
                "condition": switch.condition,
                "source": switch.source,
                "n_obs": int(mask.sum()),
            }
        )

    return out, pd.DataFrame(switch_records)


def run_forecast_ensemble_pipeline(config: ForecastEnsembleConfig, save_outputs: bool = True) -> pd.DataFrame:
    start_time = time.perf_counter()

    joined = None
    weights = {}
    source_paths = {}
    for source in config.sources:
        forecast = _load_forecast(source.path)
        weights[source.name] = float(source.weight)
        source_paths[source.name] = str(source.path)
        renamed = forecast.rename(columns={"y_pred": f"y_pred_{source.name}", "y_true": f"y_true_{source.name}"})
        joined = renamed if joined is None else joined.join(renamed, how="inner")

    assert joined is not None
    pred_columns = [f"y_pred_{source.name}" for source in config.sources]
    true_columns = [f"y_true_{source.name}" for source in config.sources]
    weight_values = np.asarray([weights[source.name] for source in config.sources], dtype=float)
    y_true = _validate_y_true(joined, true_columns)

    weight_records_df = pd.DataFrame()
    if config.method == "fixed":
        y_pred = _fixed_weight_predictions(joined, pred_columns, weight_values)
    elif config.method == "rolling_inverse_mae":
        if np.any(weight_values < 0) or weight_values.sum() <= 0:
            raise ValueError("Fallback forecast ensemble weights must be non-negative and sum to a positive value.")
        fallback_weights = weight_values / weight_values.sum()
        y_pred, weight_records_df = _rolling_inverse_mae_predictions(
            joined,
            pred_columns,
            y_true,
            train_days=config.train_days,
            min_train_days=config.min_train_days,
            weight_power=config.weight_power,
            epsilon=config.epsilon,
            fallback_weights=fallback_weights,
        )
    elif config.method == "rolling_regime_inverse_mae":
        if np.any(weight_values < 0) or weight_values.sum() <= 0:
            raise ValueError("Fallback forecast ensemble weights must be non-negative and sum to a positive value.")
        fallback_weights = weight_values / weight_values.sum()
        y_pred, weight_records_df = _rolling_regime_inverse_mae_predictions(
            joined,
            pred_columns,
            y_true,
            train_days=config.train_days,
            min_train_days=config.min_train_days,
            weight_power=config.weight_power,
            epsilon=config.epsilon,
            fallback_weights=fallback_weights,
        )
    else:
        raise ValueError(f"Unsupported ensemble method: {config.method!r}")

    forecast_df = pd.DataFrame({"y_pred": y_pred, "y_true": y_true.to_numpy(dtype=float)}, index=joined.index)
    forecast_df.index.name = None
    forecast_df, switch_records_df = _apply_calendar_switches(forecast_df, joined, config.calendar_switches)
    bias_records_df = pd.DataFrame()
    if config.bias_correction in {"rolling_mean_error", "rolling_hour_mean_error"}:
        forecast_df, bias_records_df = _apply_rolling_bias_correction(
            forecast_df,
            train_days=config.bias_train_days,
            min_train_days=config.bias_min_train_days,
            by_hour=config.bias_correction == "rolling_hour_mean_error",
        )
    elif config.bias_correction != "none":
        raise ValueError(f"Unsupported bias correction: {config.bias_correction!r}")

    runtime_df = pd.DataFrame(
        [
            {
                "runtime_seconds": time.perf_counter() - start_time,
                "n_sources": len(config.sources),
                "n_obs": len(forecast_df),
                "method": config.method,
            }
        ]
    )
    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "sources": source_paths,
        "weights": weights,
        "join": config.join,
        "method": config.method,
        "train_days": config.train_days,
        "min_train_days": config.min_train_days,
        "weight_power": config.weight_power,
        "bias_correction": config.bias_correction,
        "bias_train_days": config.bias_train_days,
        "bias_min_train_days": config.bias_min_train_days,
        "calendar_switches": [
            {"condition": switch.condition, "source": switch.source} for switch in config.calendar_switches
        ],
    }
    if save_outputs:
        save_experiment_outputs(
            name="forecast_ensemble",
            forecast_df=forecast_df,
            runtime_df=runtime_df,
            config=payload,
            export_dir=config.export_dir,
        )
        if not weight_records_df.empty:
            weight_records_df.to_csv(config.export_dir / "weights.csv", index=False)
        if not bias_records_df.empty:
            bias_records_df.to_csv(config.export_dir / "bias_corrections.csv", index=False)
        if not switch_records_df.empty:
            switch_records_df.to_csv(config.export_dir / "calendar_switches.csv", index=False)
    return forecast_df
