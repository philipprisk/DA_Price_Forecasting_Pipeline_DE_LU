from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ..config import TabpfnLocalConfig
from .tabpfn_ts import _as_target_tz_timestamp, _build_tabpfn_covariates, _compute_metrics


def _load_env(repo_root: Path) -> None:
    load_dotenv(repo_root / ".env")


def _require_tabpfn_regressor():
    try:
        from tabpfn import TabPFNRegressor
    except ImportError as exc:
        raise ImportError(
            "tabpfn is not installed. Install the TabPFN environment with "
            "`pixi install -e tabpfn` or run the command via `pixi run -e tabpfn ...`."
        ) from exc

    return TabPFNRegressor


def _require_entsoe_fetchers():
    try:
        from ..data.entsoe import fetch_prices
    except ImportError as exc:
        raise ImportError(
            "entsoe-py is not installed. Install the TabPFN environment with "
            "`pixi install -e tabpfn` or run the command via `pixi run -e tabpfn ...`."
        ) from exc

    return fetch_prices


def _forecast_days(config: TabpfnLocalConfig) -> pd.DatetimeIndex:
    start = _as_target_tz_timestamp(config.test_start, config.target_tz).normalize()
    end = _as_target_tz_timestamp(config.test_end, config.target_tz).normalize()
    return pd.date_range(start=start, end=end, freq="D")


def _build_calendar_features(index: pd.DatetimeIndex, config: TabpfnLocalConfig) -> pd.DataFrame:
    local_index = index.tz_convert(config.target_tz) if index.tz is not None else index.tz_localize(config.target_tz)
    minute_of_day = local_index.hour * 60 + local_index.minute
    mtu = local_index.hour * 4 + local_index.minute // 15
    dayofweek = local_index.dayofweek
    month = local_index.month
    dayofyear = local_index.dayofyear
    post_regime_start = _as_target_tz_timestamp(config.post_regime_start, config.target_tz)

    features = pd.DataFrame(index=index)
    features["mtu"] = mtu.astype(float)
    features["hour"] = local_index.hour.astype(float)
    features["dayofweek"] = dayofweek.astype(float)
    features["month"] = month.astype(float)
    features["is_weekend"] = (dayofweek >= 5).astype(float)
    features["is_15min_market"] = (local_index >= post_regime_start).astype(float)
    features["mtu_sin"] = np.sin(2 * np.pi * mtu / 96)
    features["mtu_cos"] = np.cos(2 * np.pi * mtu / 96)
    features["minute_of_day_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    features["minute_of_day_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    features["dow_sin"] = np.sin(2 * np.pi * dayofweek / 7)
    features["dow_cos"] = np.cos(2 * np.pi * dayofweek / 7)
    features["month_sin"] = np.sin(2 * np.pi * month / 12)
    features["month_cos"] = np.cos(2 * np.pi * month / 12)
    features["dayofyear_sin"] = np.sin(2 * np.pi * dayofyear / 366)
    features["dayofyear_cos"] = np.cos(2 * np.pi * dayofyear / 366)
    return features


def _build_price_lag_features(
    prices: pd.DataFrame,
    index: pd.DatetimeIndex,
    lag_days: list[int],
    target_tz: str,
) -> pd.DataFrame:
    if "price_da" not in prices.columns:
        raise ValueError("Price DataFrame must contain a 'price_da' column.")

    base = prices["price_da"].sort_index().tz_convert("UTC")
    features = pd.DataFrame(index=index)
    for lag_day in lag_days:
        lagged = base.shift(freq=f"{int(lag_day)}D").tz_convert(target_tz)
        features[f"price_lag{int(lag_day)}d"] = lagged.reindex(index).astype(float)
    return features


def _build_price_daily_stat_features(
    prices: pd.DataFrame,
    index: pd.DatetimeIndex,
    target_tz: str,
) -> pd.DataFrame:
    if "price_da" not in prices.columns:
        raise ValueError("Price DataFrame must contain a 'price_da' column.")

    local_prices = prices["price_da"].sort_index()
    if local_prices.index.tz is None:
        local_prices.index = local_prices.index.tz_localize(target_tz)
    else:
        local_prices.index = local_prices.index.tz_convert(target_tz)

    frame = local_prices.to_frame("price_da")
    frame["date_local"] = frame.index.date
    daily = frame.groupby("date_local")["price_da"].agg(
        close="last",
        min="min",
        max="max",
    )

    local_index = index.tz_convert(target_tz) if index.tz is not None else index.tz_localize(target_tz)
    previous_dates = [day - timedelta(days=1) for day in local_index.date]
    features = pd.DataFrame(index=index)
    features["price_lastday_close"] = [daily["close"].get(day, np.nan) for day in previous_dates]
    features["price_lastday_min"] = [daily["min"].get(day, np.nan) for day in previous_dates]
    features["price_lastday_max"] = [daily["max"].get(day, np.nan) for day in previous_dates]
    return features


def _build_price_rolling_stat_features(
    prices: pd.DataFrame,
    index: pd.DatetimeIndex,
    windows_days: list[int],
    target_tz: str,
) -> pd.DataFrame:
    if "price_da" not in prices.columns:
        raise ValueError("Price DataFrame must contain a 'price_da' column.")

    base = prices["price_da"].sort_index().tz_convert("UTC")
    lagged = base.shift(freq="1D")
    features = pd.DataFrame(index=index)
    for window_days in windows_days:
        window = f"{int(window_days)}D"
        prefix = f"price_roll_{int(window_days)}d"
        features[f"{prefix}_mean"] = lagged.rolling(window, min_periods=1).mean().tz_convert(target_tz).reindex(index)
        features[f"{prefix}_std"] = lagged.rolling(window, min_periods=2).std().tz_convert(target_tz).reindex(index)
    return features.astype(float)


def _build_holiday_feature(index: pd.DatetimeIndex, target_tz: str) -> pd.DataFrame:
    try:
        import holidays
    except ImportError as exc:
        raise ImportError("The optional TabPFN-local holiday feature requires the `holidays` package.") from exc

    local_index = index.tz_convert(target_tz) if index.tz is not None else index.tz_localize(target_tz)
    de_holidays = holidays.Germany(years=local_index.year.unique())
    return pd.DataFrame(
        {"is_holiday": pd.Series(local_index.date, index=index).isin(set(de_holidays.keys())).astype(float)},
        index=index,
    )


def _daily_mtu_matrix(series: pd.Series, target_tz: str) -> pd.DataFrame:
    local_series = series.sort_index()
    if local_series.index.tz is None:
        local_series.index = local_series.index.tz_localize(target_tz)
    else:
        local_series.index = local_series.index.tz_convert(target_tz)

    frame = local_series.to_frame("value")
    frame["date_local"] = frame.index.normalize()
    frame["mtu"] = frame.index.hour * 4 + frame.index.minute // 15
    return (
        frame.pivot_table(index="date_local", columns="mtu", values="value", aggfunc="mean")
        .reindex(columns=range(96))
        .interpolate(axis=1, limit=4, limit_area="inside")
    )


def _covariate_source_column(source: str) -> str:
    return {
        "exaa": "price_exaa",
        "load_forecast": "load_fc",
        "ntc": "NTC_Net_MW",
        "generation_unavailability": "Unavail_Total_MW",
        "renewable_generation_proxy": "Renewable_Total_Proxy_MW",
        "wind_generation_proxy": "Renewable_Wind_Proxy_MW",
        "solar_generation_proxy": "Renewable_Solar_Proxy_MW",
        "residual_load_proxy": "Residual_Load_Proxy_MW",
        "gas_ttf": "gas_ttf",
        "co2_eua": "co2_eua",
        "coal_api2": "coal_api2",
    }.get(source, source)


def _build_covariate_ramp_features(
    covariates: pd.DataFrame,
    index: pd.DatetimeIndex,
    sources: list[str],
    horizons_mtu: list[int],
) -> pd.DataFrame:
    frames = []
    aligned = covariates.reindex(index).sort_index()
    for source_name in sources:
        source = _covariate_source_column(source_name)
        if source not in aligned.columns:
            continue
        for horizon_mtu in horizons_mtu:
            horizon = int(horizon_mtu)
            if horizon <= 0:
                continue
            frames.append(aligned[source].diff(horizon).rename(f"{source}_ramp_{horizon}mtu"))

    if not frames:
        return pd.DataFrame(index=index)
    return pd.concat(frames, axis=1).astype(float)


def _build_engineered_tabpfn_local_features(
    index: pd.DatetimeIndex,
    prices: pd.DataFrame,
    covariates: pd.DataFrame,
    price_lag_features: pd.DataFrame,
    config: TabpfnLocalConfig,
) -> pd.DataFrame:
    frames = []
    engineered = config.engineered_features

    if engineered.holiday:
        frames.append(_build_holiday_feature(index, config.target_tz))

    if engineered.price_daily_stats:
        frames.append(_build_price_daily_stat_features(prices, index, config.target_tz))

    if engineered.price_rolling_stats.enabled and engineered.price_rolling_stats.windows_days:
        frames.append(
            _build_price_rolling_stat_features(
                prices,
                index,
                engineered.price_rolling_stats.windows_days,
                config.target_tz,
            )
        )

    if not covariates.empty and engineered.covariate_ramps.sources and engineered.covariate_ramps.horizons_mtu:
        frames.append(
            _build_covariate_ramp_features(
                covariates,
                index,
                engineered.covariate_ramps.sources,
                engineered.covariate_ramps.horizons_mtu,
            )
        )

    if engineered.interactions.seq2_minus_price_lag_d1 and "price_exaa" in covariates.columns:
        seq2 = covariates["price_exaa"].reindex(index).astype(float)
        if "price_lag1d" in price_lag_features.columns:
            frames.append((seq2 - price_lag_features["price_lag1d"]).to_frame("seq2_minus_price_lag1d"))

    clean_frames = [frame for frame in frames if not frame.empty]
    if not clean_frames:
        return pd.DataFrame(index=index)
    return pd.concat(clean_frames, axis=1)


def _build_curve_features(
    covariates: pd.DataFrame,
    index: pd.DatetimeIndex,
    config: TabpfnLocalConfig,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    local_index = index.tz_convert(config.target_tz) if index.tz is not None else index.tz_localize(config.target_tz)
    dates = local_index.normalize()
    mtu = local_index.hour * 4 + local_index.minute // 15

    for source_name in config.curve_feature_sources:
        source = _covariate_source_column(source_name)
        if source not in covariates.columns:
            continue

        matrix = _daily_mtu_matrix(covariates[source], config.target_tz)
        if matrix.empty:
            continue

        if config.add_curve_features:
            full = matrix.reindex(dates)
            full.index = index
            full.columns = [f"{source}_curve_mtu_{int(col):02d}" for col in full.columns]
            frames.append(full.astype(float))

        if config.curve_neighbor_offsets:
            neighbor = pd.DataFrame(index=index)
            day_matrix = matrix.reindex(dates)
            values = day_matrix.to_numpy(dtype=float)
            for offset in config.curve_neighbor_offsets:
                cols = np.clip(mtu + int(offset), 0, 95)
                neighbor[f"{source}_offset_{int(offset):+d}mtu"] = values[np.arange(len(index)), cols]
            frames.append(neighbor)

        if config.add_curve_summary_features:
            summary = pd.DataFrame(index=matrix.index)
            summary[f"{source}_daily_mean"] = matrix.mean(axis=1)
            summary[f"{source}_daily_min"] = matrix.min(axis=1)
            summary[f"{source}_daily_max"] = matrix.max(axis=1)
            summary[f"{source}_daily_spread"] = summary[f"{source}_daily_max"] - summary[f"{source}_daily_min"]
            summary[f"{source}_morning_peak_mean"] = matrix.loc[:, 24:39].mean(axis=1)
            summary[f"{source}_evening_peak_mean"] = matrix.loc[:, 64:79].mean(axis=1)
            summary[f"{source}_max_abs_ramp"] = matrix.diff(axis=1).abs().max(axis=1)
            repeated = summary.reindex(dates)
            repeated.index = index
            frames.append(repeated.astype(float))

    if not frames:
        return pd.DataFrame(index=index)

    return pd.concat(frames, axis=1)


def build_tabpfn_local_features(
    index: pd.DatetimeIndex,
    prices: pd.DataFrame,
    covariates: pd.DataFrame,
    config: TabpfnLocalConfig,
) -> pd.DataFrame:
    """Build supervised tabular rows for the local TabPFN regressor."""
    frames: list[pd.DataFrame] = []
    if config.add_calendar_features:
        frames.append(_build_calendar_features(index, config))
    price_lag_features = pd.DataFrame(index=index)
    if config.price_lag_days:
        price_lag_features = _build_price_lag_features(prices, index, config.price_lag_days, config.target_tz)
        frames.append(price_lag_features)
    if not covariates.empty:
        frames.append(covariates.reindex(index).astype(float))
        curve_features = _build_curve_features(covariates, index, config)
        if not curve_features.empty:
            frames.append(curve_features)
    engineered_features = _build_engineered_tabpfn_local_features(index, prices, covariates, price_lag_features, config)
    if not engineered_features.empty:
        frames.append(engineered_features)

    if not frames:
        raise ValueError("No features configured for local TabPFN.")

    features = pd.concat(frames, axis=1)
    features = features.loc[:, ~features.columns.duplicated()].sort_index()
    features.index.name = "timestamp"
    return features


def _build_covariates(config: TabpfnLocalConfig, start_day: pd.Timestamp, end_day: pd.Timestamp) -> pd.DataFrame:
    return _build_tabpfn_covariates(config, start_day, end_day)


def _training_frame_for_day(
    feature_frame: pd.DataFrame,
    prices: pd.DataFrame,
    forecast_day: pd.Timestamp,
    config: TabpfnLocalConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    y = prices["price_da"].reindex(feature_frame.index).astype(float)
    mask = feature_frame.index < forecast_day
    if config.train_window_days is not None:
        mask &= feature_frame.index >= forecast_day - pd.Timedelta(days=config.train_window_days)

    train = feature_frame.loc[mask].copy()
    y_train = y.loc[train.index].copy()
    valid = y_train.notna()
    train = train.loc[valid]
    y_train = y_train.loc[valid]

    if train.empty:
        raise ValueError(f"No local TabPFN training rows available before {forecast_day.date()}.")

    train, y_train = _select_training_context(train, y_train, forecast_day, config)
    return train, y_train


def _evenly_spaced_index(index: pd.Index, n_rows: int) -> pd.Index:
    if n_rows <= 0 or len(index) == 0:
        return index[:0]
    if len(index) <= n_rows:
        return index
    positions = np.linspace(0, len(index) - 1, num=n_rows, dtype=int)
    return index[np.unique(positions)]


def _select_training_context(
    train: pd.DataFrame,
    y_train: pd.Series,
    forecast_day: pd.Timestamp,
    config: TabpfnLocalConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    if config.max_train_rows is None or len(train) <= config.max_train_rows:
        return train, y_train

    if config.context_selection == "tail":
        selected_index = train.tail(config.max_train_rows).index
    else:
        selected_index = _recent_weekday_even_context_index(train.index, forecast_day, config)

    return train.loc[selected_index], y_train.loc[selected_index]


def _recent_weekday_even_context_index(
    index: pd.DatetimeIndex,
    forecast_day: pd.Timestamp,
    config: TabpfnLocalConfig,
) -> pd.DatetimeIndex:
    max_rows = int(config.max_train_rows or len(index))
    if len(index) <= max_rows:
        return index

    recent_cutoff = forecast_day - pd.Timedelta(days=config.context_recent_days)
    recent_index = index[index >= recent_cutoff]
    recent_index = recent_index[-max_rows:]

    remaining = max_rows - len(recent_index)
    if remaining <= 0:
        return pd.DatetimeIndex(recent_index).sort_values()

    older_index = index.difference(recent_index).sort_values()
    selected: list[pd.DatetimeIndex] = [pd.DatetimeIndex(recent_index)]

    weekday_budget = int(round(max_rows * config.context_same_weekday_fraction))
    weekday_budget = min(max(weekday_budget, 0), remaining)
    if weekday_budget > 0 and len(older_index) > 0:
        same_weekday = older_index[older_index.dayofweek == forecast_day.dayofweek]
        weekday_index = _evenly_spaced_index(same_weekday, weekday_budget)
        selected.append(pd.DatetimeIndex(weekday_index))
        remaining -= len(weekday_index)
        older_index = older_index.difference(weekday_index).sort_values()

    even_budget = int(round(max_rows * config.context_even_history_fraction))
    even_budget = min(max(even_budget, 0), remaining)
    if even_budget > 0 and len(older_index) > 0:
        even_index = _evenly_spaced_index(older_index, even_budget)
        selected.append(pd.DatetimeIndex(even_index))
        remaining -= len(even_index)
        older_index = older_index.difference(even_index).sort_values()

    if remaining > 0 and len(older_index) > 0:
        fill_index = _evenly_spaced_index(older_index, remaining)
        selected.append(pd.DatetimeIndex(fill_index))

    combined = selected[0]
    for extra in selected[1:]:
        combined = combined.union(extra)
    combined = combined.sort_values()
    if len(combined) > max_rows:
        combined = combined[-max_rows:]
    return combined


def _prepare_model_matrices(
    train: pd.DataFrame,
    future: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.replace([np.inf, -np.inf], np.nan)
    future = future.replace([np.inf, -np.inf], np.nan)
    medians = train.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train_prepared = train.fillna(medians).fillna(0.0)
    future_prepared = future.fillna(medians).fillna(0.0)
    return train_prepared.astype(float), future_prepared.astype(float)


def _categorical_feature_indices(config: TabpfnLocalConfig, columns: pd.Index) -> list[int] | None:
    if not config.use_categorical_features:
        return None
    indices = [idx for idx, name in enumerate(columns) if name in set(config.categorical_feature_names)]
    return indices or None


def _new_regressor(config: TabpfnLocalConfig, feature_columns: pd.Index):
    TabPFNRegressor = _require_tabpfn_regressor()
    return TabPFNRegressor(
        n_estimators=config.n_estimators,
        categorical_features_indices=_categorical_feature_indices(config, feature_columns),
        device=config.device,
        ignore_pretraining_limits=config.ignore_pretraining_limits,
        inference_precision=config.inference_precision,
        fit_mode=config.fit_mode,
        random_state=config.random_state,
        n_preprocessing_jobs=config.n_preprocessing_jobs,
    )


def _robust_params(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    center = float(np.median(finite))
    q25, q75 = np.percentile(finite, [25, 75])
    scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(finite))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return center, scale


def _transform_target(y_train: pd.Series, config: TabpfnLocalConfig) -> tuple[np.ndarray, dict[str, float | str]]:
    values = y_train.to_numpy(dtype=float)
    if config.target_transform == "none":
        return values, {"kind": "none"}

    center, scale = _robust_params(values)
    transformed = np.arcsinh((values - center) / scale)
    return transformed, {"kind": "asinh_robust", "center": center, "scale": scale}


def _inverse_target(values: np.ndarray, params: dict[str, float | str]) -> np.ndarray:
    if params.get("kind") != "asinh_robust":
        return np.asarray(values, dtype=float)
    center = float(params["center"])
    scale = float(params["scale"])
    return center + scale * np.sinh(np.asarray(values, dtype=float))


def _is_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    name = exc.__class__.__name__.lower()
    return "outofmemory" in name or "out of memory" in message or "oom" in message


def _predict_main(
    model: Any,
    x_future: pd.DataFrame,
    quantiles: list[float],
    batch_size: int | None,
) -> dict[str, Any]:
    if batch_size is None or batch_size <= 0 or batch_size >= len(x_future):
        return model.predict(
            x_future,
            output_type="main",
            quantiles=[float(tau) for tau in quantiles],
        )

    chunks = []
    for start in range(0, len(x_future), batch_size):
        chunks.append(
            model.predict(
                x_future.iloc[start : start + batch_size],
                output_type="main",
                quantiles=[float(tau) for tau in quantiles],
            )
        )

    return {
        "mean": np.concatenate([np.asarray(chunk["mean"], dtype=float) for chunk in chunks]),
        "median": np.concatenate([np.asarray(chunk["median"], dtype=float) for chunk in chunks]),
        "mode": np.concatenate([np.asarray(chunk["mode"], dtype=float) for chunk in chunks]),
        "quantiles": [
            np.concatenate([np.asarray(chunk["quantiles"][idx], dtype=float) for chunk in chunks])
            for idx, _ in enumerate(quantiles)
        ],
    }


def _local_tabpfn_oom_message() -> str:
    return (
        "Local TabPFN ran out of memory even when predicting one row at a time. "
        "That means the training/context set is too large for the selected device. "
        "Reduce `max_train_rows`, set `device` to `cpu`, lower `n_estimators`, or shorten "
        "`train_window_days` in the local TabPFN config."
    )


def _is_cpu_large_dataset_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "running on cpu with more than 1000 samples" in message


def _local_tabpfn_fit_error_message() -> str:
    return (
        "Local TabPFN refused the configured CPU training set because it has more than "
        "1000 samples. Set `max_train_rows` to 1000 or lower for the conservative CPU "
        "path. If you intentionally want a slower large-CPU run, set "
        "`ignore_pretraining_limits` to true or export TABPFN_ALLOW_CPU_LARGE_DATASET=1."
    )


def _prediction_frame(
    model: Any,
    x_future: pd.DataFrame,
    future_index: pd.DatetimeIndex,
    y_true: pd.Series | None,
    config: TabpfnLocalConfig,
    target_params: dict[str, float | str],
) -> pd.DataFrame:
    try:
        predictions = _predict_main(model, x_future, config.quantiles, config.predict_batch_size)
    except Exception as exc:
        if config.predict_batch_size is not None and config.predict_batch_size > 1 and _is_oom_error(exc):
            try:
                predictions = _predict_main(model, x_future, config.quantiles, 1)
            except Exception as retry_exc:
                if _is_oom_error(retry_exc):
                    raise RuntimeError(_local_tabpfn_oom_message()) from retry_exc
                raise
        elif _is_oom_error(exc):
            raise RuntimeError(_local_tabpfn_oom_message()) from exc
        else:
            raise

    result = pd.DataFrame(index=future_index)
    result["y_pred"] = _inverse_target(np.asarray(predictions[config.point_output_type], dtype=float), target_params)

    quantile_predictions = predictions["quantiles"]
    for tau, values in zip(config.quantiles, quantile_predictions, strict=True):
        result[f"q{tau:.3f}"] = _inverse_target(np.asarray(values, dtype=float), target_params)

    if y_true is not None:
        result["y_true"] = y_true.reindex(future_index).astype(float)
    else:
        result["y_true"] = np.nan
    return result


def run_tabpfn_local_pipeline(config: TabpfnLocalConfig, save_outputs: bool = True) -> dict[str, Any]:
    """Run a local TabPFN tabular regressor as a rate-limit-free benchmark."""
    _load_env(config.repo_root)
    fetch_prices = _require_entsoe_fetchers()

    entsoe_start = _as_target_tz_timestamp(config.entsoe_start_date, config.target_tz)
    price_end = _as_target_tz_timestamp(config.entsoe_end_date, config.target_tz)
    test_end = _as_target_tz_timestamp(config.test_end, config.target_tz)
    covariate_end = max(price_end, test_end)

    prices = fetch_prices(
        start_day=entsoe_start,
        end_day=price_end,
        country_code=config.country_code_entsoe,
        api_key_env=config.entsoe_api_key_env,
        target_tz=config.target_tz,
    )
    covariates = _build_covariates(config, entsoe_start, covariate_end)

    feature_start = prices.index.min()
    feature_end = max(prices.index.max(), covariates.index.max() if not covariates.empty else prices.index.max(), test_end)
    feature_index = pd.date_range(start=feature_start, end=feature_end, freq=config.frequency, tz=config.target_tz)
    feature_frame = build_tabpfn_local_features(feature_index, prices, covariates, config)

    all_forecasts = []
    runtime_records = []
    for forecast_day in _forecast_days(config):
        start_time = time.perf_counter()
        future_index = pd.date_range(start=forecast_day, periods=config.forecast_horizon, freq=config.frequency)

        train, y_train = _training_frame_for_day(feature_frame, prices, forecast_day, config)
        future = feature_frame.reindex(future_index)
        x_train, x_future = _prepare_model_matrices(train, future)
        y_model, target_params = _transform_target(y_train, config)

        model = _new_regressor(config, x_train.columns)
        try:
            model.fit(x_train, y_model)
        except Exception as exc:
            if _is_cpu_large_dataset_error(exc):
                raise RuntimeError(_local_tabpfn_fit_error_message()) from exc
            if _is_oom_error(exc):
                raise RuntimeError(
                    "Local TabPFN ran out of memory while fitting. Reduce `max_train_rows`, set "
                    "`device` to `cpu`, lower `n_estimators`, or shorten `train_window_days` in "
                    "the local TabPFN config."
                ) from exc
            raise
        y_true = prices["price_da"] if "price_da" in prices.columns else None
        forecast_df = _prediction_frame(model, x_future, future_index, y_true, config, target_params)
        all_forecasts.append(forecast_df)

        runtime_seconds = time.perf_counter() - start_time
        runtime_records.append(
            {
                "forecast_day": forecast_day,
                "runtime_seconds": runtime_seconds,
                "train_rows": len(x_train),
                "n_features": len(x_train.columns),
                "forecast_horizon": config.forecast_horizon,
                "n_estimators": config.n_estimators,
                "device": config.device,
                "predict_batch_size": config.predict_batch_size,
                "use_exaa": config.use_exaa,
                "use_load_forecast": config.use_load_forecast,
                "covariates": ",".join(config.features.covariates if config.features is not None else []),
                "context_selection": config.context_selection,
                "target_transform": config.target_transform,
                "use_categorical_features": config.use_categorical_features,
            }
        )
        print(f"  {forecast_day.date()}  TabPFN-local  {runtime_seconds:.1f}s")

    forecast_all = pd.concat(all_forecasts).sort_index() if all_forecasts else pd.DataFrame()
    runtime_df = pd.DataFrame(runtime_records)
    metrics_df = _compute_metrics(forecast_all, config.quantile_columns)

    if save_outputs:
        save_tabpfn_local_outputs(
            export_dir=config.resolved_export_dir,
            forecast_df=forecast_all,
            runtime_df=runtime_df,
            config=config,
            metrics_df=metrics_df,
            feature_columns=list(feature_frame.columns),
        )

    return {
        "forecast": forecast_all,
        "runtime": runtime_df,
        "metrics": metrics_df,
        "feature_columns": list(feature_frame.columns),
    }


def save_tabpfn_local_outputs(
    export_dir: Path,
    forecast_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    config: TabpfnLocalConfig,
    metrics_df: pd.DataFrame | None = None,
    feature_columns: list[str] | None = None,
) -> None:
    """Save local TabPFN outputs with the shared forecast artifact layout."""
    export_dir.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(export_dir / "forecast.csv", index=True)
    runtime_df.to_csv(export_dir / "runtime.csv", index=False)
    if metrics_df is not None and not metrics_df.empty:
        metrics_df.to_csv(export_dir / "metrics.csv", index=False)

    payload = config.model_dump(mode="json")
    payload["experiment_name"] = config.resolved_experiment_name
    payload["feature_columns"] = feature_columns or []
    payload.setdefault("created_at", datetime.now().strftime("%Y-%m-%d %H:%M"))
    with open(export_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"✓ Saved local TabPFN -> {export_dir}")
