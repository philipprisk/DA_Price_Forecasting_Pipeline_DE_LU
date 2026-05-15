from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ..config import TabpfnTsConfig
from ..features.covariates import build_timestamp_covariates, merge_timestamp_covariates


def _load_env(repo_root: Path) -> None:
    load_dotenv(repo_root / ".env")


def _require_tabpfn_ts():
    try:
        from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline
    except ImportError as exc:
        raise ImportError(
            "tabpfn-time-series is not installed. Install the TabPFN environment with "
            "`pixi install -e tabpfn` or run the command via `pixi run -e tabpfn ...`."
        ) from exc

    return TabPFNTSPipeline, TabPFNMode


def _require_entsoe_fetchers():
    try:
        from ..data.entsoe import fetch_prices
    except ImportError as exc:
        raise ImportError(
            "entsoe-py is not installed. Install the TabPFN environment with "
            "`pixi install -e tabpfn` or run the command via `pixi run -e tabpfn ...`."
        ) from exc

    return fetch_prices


def _prepared_tabpfn_model_config(config: TabpfnTsConfig) -> dict[str, Any]:
    model_config = dict(config.tabpfn_model_config or {})
    model_path = model_config.get("model_path")
    if model_path is None:
        return model_config

    resolved_model_path = Path(model_path)
    if not resolved_model_path.is_absolute():
        if resolved_model_path.parent == Path("."):
            resolved_model_path = config.repo_root / ".cache" / "models" / "tabpfn" / resolved_model_path.name
        else:
            resolved_model_path = config.repo_root / resolved_model_path
        resolved_model_path = resolved_model_path.resolve()

    if config.tabpfn_mode == "local":
        resolved_model_path.parent.mkdir(parents=True, exist_ok=True)

    model_config["model_path"] = str(resolved_model_path)
    return model_config


def _local_naive_index(index: pd.DatetimeIndex, target_tz: str) -> pd.DatetimeIndex:
    if index.tz is None:
        return index.tz_localize(target_tz).tz_localize(None)
    return index.tz_convert(target_tz).tz_localize(None)


def _as_target_tz_timestamp(value: Any, target_tz: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(target_tz)
    return timestamp.tz_convert(target_tz)


def _to_context_frame(df: pd.DataFrame, target_tz: str, item_id: str = "DE_LU") -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = _local_naive_index(out.index, target_tz)
    out["item_id"] = item_id
    out = out.rename(columns={"price_da": "target"})
    cols = ["item_id", "timestamp", "target"] + [
        col for col in out.columns if col not in {"item_id", "timestamp", "target"}
    ]
    return out[cols].reset_index(drop=True)


def _to_future_frame(index: pd.DatetimeIndex, covariates: pd.DataFrame, target_tz: str, item_id: str = "DE_LU") -> pd.DataFrame:
    out = covariates.reindex(index).copy()
    out["timestamp"] = _local_naive_index(index, target_tz)
    out["item_id"] = item_id
    out["target"] = np.nan
    cols = ["item_id", "timestamp", "target"] + [
        col for col in out.columns if col not in {"item_id", "timestamp", "target"}
    ]
    return out[cols].reset_index(drop=True)


def _normalize_predictions(
    pred_df: pd.DataFrame,
    future_index: pd.DatetimeIndex,
    y_true: pd.Series | None,
    quantiles: list[float],
    target_tz: str,
) -> pd.DataFrame:
    work = pred_df.copy()
    if isinstance(work.index, pd.MultiIndex):
        work = work.reset_index()
        if "timestamp" not in work.columns:
            work = work.rename(columns={work.columns[-1]: "timestamp"})
        work = work.set_index("timestamp")
    elif "timestamp" in work.columns:
        work = work.set_index("timestamp")

    work.index = pd.to_datetime(work.index)
    if getattr(work.index, "tz", None) is None:
        work.index = work.index.tz_localize(target_tz)
    else:
        work.index = work.index.tz_convert(target_tz)

    result = pd.DataFrame(index=future_index)
    if "target" not in work.columns:
        raise ValueError("TabPFN-TS prediction output does not contain a 'target' column.")
    result["y_pred"] = work.reindex(future_index)["target"].astype(float)

    for tau in quantiles:
        candidate_cols = [tau, str(tau), f"{tau:.1f}", f"{tau:.2f}", f"{tau:.3f}"]
        out_col = f"q{tau:.3f}"
        for candidate_col in candidate_cols:
            if candidate_col in work.columns:
                result[out_col] = work.reindex(future_index)[candidate_col].astype(float)
                break

    if y_true is not None:
        result["y_true"] = y_true.reindex(future_index).astype(float)
    else:
        result["y_true"] = np.nan

    return result


def _merge_timestamp_covariates(
    frames: list[pd.DataFrame],
    index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    return merge_timestamp_covariates(frames, index=index)


def _build_tabpfn_covariates(config: Any, start_day: pd.Timestamp, end_day: pd.Timestamp) -> pd.DataFrame:
    if config.features is None:
        return pd.DataFrame()
    return build_timestamp_covariates(
        config.features,
        start_day=start_day,
        end_day=end_day,
        country_code_entsoe=config.country_code_entsoe,
        entsoe_api_key_env=config.entsoe_api_key_env,
        target_tz=config.target_tz,
    )


def _build_covariates(config: TabpfnTsConfig, start_day: pd.Timestamp, end_day: pd.Timestamp) -> pd.DataFrame:
    return _build_tabpfn_covariates(config, start_day, end_day)


def _forecast_days(config: TabpfnTsConfig) -> pd.DatetimeIndex:
    start = _as_target_tz_timestamp(config.test_start, config.target_tz).normalize()
    end = _as_target_tz_timestamp(config.test_end, config.target_tz).normalize()
    return pd.date_range(
        start=start,
        end=end,
        freq="D",
    )


def _compute_metrics(forecast_df: pd.DataFrame, quantile_cols: list[str]) -> pd.DataFrame:
    if forecast_df.empty or "y_true" not in forecast_df.columns or not forecast_df["y_true"].notna().any():
        return pd.DataFrame()

    valid = forecast_df[forecast_df["y_true"].notna() & forecast_df["y_pred"].notna()].copy()
    records: list[dict[str, Any]] = []
    if not valid.empty:
        err = valid["y_pred"] - valid["y_true"]
        records.append(
            {
                "metric": "mae",
                "value": float(err.abs().mean()),
                "n_obs": int(len(valid)),
            }
        )
        records.append(
            {
                "metric": "rmse",
                "value": float(np.sqrt((err**2).mean())),
                "n_obs": int(len(valid)),
            }
        )
        records.append(
            {
                "metric": "bias",
                "value": float(err.mean()),
                "n_obs": int(len(valid)),
            }
        )

    for col in quantile_cols:
        if col not in forecast_df.columns:
            continue
        q = float(col.removeprefix("q"))
        q_valid = forecast_df[forecast_df["y_true"].notna() & forecast_df[col].notna()]
        if q_valid.empty:
            continue
        diff = q_valid["y_true"] - q_valid[col]
        pinball = np.maximum(q * diff, (q - 1) * diff).mean()
        records.append({"metric": f"pinball_{col}", "value": float(pinball), "n_obs": int(len(q_valid))})

    return pd.DataFrame(records)


def run_tabpfn_ts_pipeline(config: TabpfnTsConfig, save_outputs: bool = True) -> dict[str, Any]:
    """Run TabPFN-TS as a standalone benchmark model."""
    _load_env(config.repo_root)
    if config.disable_telemetry:
        os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

    TabPFNTSPipeline, TabPFNMode = _require_tabpfn_ts()
    fetch_prices = _require_entsoe_fetchers()
    mode = TabPFNMode.CLIENT if config.tabpfn_mode == "client" else TabPFNMode.LOCAL
    tabpfn_model_config = _prepared_tabpfn_model_config(config)
    pipeline = TabPFNTSPipeline(
        max_context_length=config.max_context_length,
        tabpfn_mode=mode,
        tabpfn_output_selection=config.tabpfn_output_selection,
        tabpfn_model_config=tabpfn_model_config,
    )

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
    model_input = prices.join(covariates, how="left") if not covariates.empty else prices

    all_forecasts = []
    runtime_records = []
    for forecast_day in _forecast_days(config):
        start_time = time.perf_counter()
        future_index = pd.date_range(
            start=forecast_day,
            periods=config.forecast_horizon,
            freq=config.frequency,
        )

        context_df = model_input.loc[model_input.index < forecast_day].copy()
        future_covariates = covariates if not covariates.empty else pd.DataFrame()
        context_frame = _to_context_frame(context_df, config.target_tz)
        future_frame = _to_future_frame(future_index, future_covariates, config.target_tz)

        pred_df = pipeline.predict_df(
            context_df=context_frame,
            future_df=future_frame,
            quantiles=config.quantiles,
        )
        y_true = prices["price_da"] if "price_da" in prices.columns else None
        forecast_df = _normalize_predictions(
            pred_df=pred_df,
            future_index=future_index,
            y_true=y_true,
            quantiles=config.quantiles,
            target_tz=config.target_tz,
        )
        all_forecasts.append(forecast_df)

        runtime_seconds = time.perf_counter() - start_time
        runtime_records.append(
            {
                "forecast_day": forecast_day,
                "runtime_seconds": runtime_seconds,
                "context_rows": len(context_frame),
                "forecast_horizon": config.forecast_horizon,
                "tabpfn_mode": config.tabpfn_mode,
                "tabpfn_model_config": tabpfn_model_config,
                "use_exaa": config.use_exaa,
                "use_load_forecast": config.use_load_forecast,
                "covariates": ",".join(config.features.covariates if config.features is not None else []),
            }
        )
        print(f"  {forecast_day.date()}  TabPFN-TS  {runtime_seconds:.1f}s")

    forecast_all = pd.concat(all_forecasts).sort_index() if all_forecasts else pd.DataFrame()
    runtime_df = pd.DataFrame(runtime_records)
    metrics_df = _compute_metrics(forecast_all, config.quantile_columns)

    if save_outputs:
        save_tabpfn_ts_outputs(
            export_dir=config.resolved_export_dir,
            forecast_df=forecast_all,
            runtime_df=runtime_df,
            config=config,
            metrics_df=metrics_df,
        )

    return {
        "forecast": forecast_all,
        "runtime": runtime_df,
        "metrics": metrics_df,
    }


def save_tabpfn_ts_outputs(
    export_dir: Path,
    forecast_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    config: TabpfnTsConfig,
    metrics_df: pd.DataFrame | None = None,
) -> None:
    """Save TabPFN-TS outputs with the shared forecast artifact layout."""
    export_dir.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(export_dir / "forecast.csv", index=True)
    runtime_df.to_csv(export_dir / "runtime.csv", index=False)
    if metrics_df is not None and not metrics_df.empty:
        metrics_df.to_csv(export_dir / "metrics.csv", index=False)

    payload = config.model_dump(mode="json")
    payload["experiment_name"] = config.resolved_experiment_name
    payload.setdefault("created_at", datetime.now().strftime("%Y-%m-%d %H:%M"))
    with open(export_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"✓ Saved TabPFN-TS -> {export_dir}")
