from __future__ import annotations

from typing import Any

import pandas as pd

from ..config import SqraConfig
from ..models.sqra import (
    evaluate_probabilistic_forecasts,
    load_forecast,
    plot_prob_forecast,
    rolling_sqra_forecast_mtu,
    save_sqra_outputs,
    sort_quantiles,
)


def _deduplicate_forecast_index(forecast: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted forecast frame with one row per timestamp."""
    if forecast.empty:
        return forecast.sort_index()
    ordered = forecast.sort_index()
    if ordered.index.has_duplicates:
        ordered = ordered.loc[~ordered.index.duplicated(keep="last")]
    return ordered


def build_sqra_panel(config: SqraConfig) -> tuple[pd.DataFrame, list[str]]:
    """Build the SQRA input panel from one or more point-forecast files."""
    forecasts = [_deduplicate_forecast_index(load_forecast(path)) for path in config.import_paths]
    feature_cols = [f"prediction_p{i + 1}" for i in range(len(forecasts))]

    common_index = forecasts[0].index.drop_duplicates()
    for forecast in forecasts[1:]:
        common_index = common_index.intersection(forecast.index.drop_duplicates())
    common_index = common_index.sort_values()

    df_qra = pd.DataFrame(index=common_index)
    for idx, fc in enumerate(forecasts):
        df_qra[f"prediction_p{idx + 1}"] = fc.reindex(common_index)["y_pred"].to_numpy()

    if "y_true" in forecasts[0].columns:
        df_qra["y_true"] = forecasts[0].reindex(common_index)["y_true"].to_numpy()
    else:
        df_qra["y_true"] = pd.NA
    df_qra["mtu"] = df_qra.index.hour * 4 + df_qra.index.minute // 15
    return df_qra, feature_cols


def run_sqra_pipeline(
    config: SqraConfig,
    save_outputs: bool = True,
    plot: bool = False,
) -> dict[str, Any]:
    """Run the SQRA pipeline outside of the notebook."""
    df_qra, feature_cols = build_sqra_panel(config)
    forecast_days = pd.date_range(
        start=pd.Timestamp(config.test_start).tz_convert(config.target_tz).normalize(),
        end=pd.Timestamp(config.test_end).tz_convert(config.target_tz).normalize(),
        freq="D",
    )

    forecast_df, runtime_df = rolling_sqra_forecast_mtu(
        df=df_qra,
        forecast_days=forecast_days,
        train_days=config.train_days_rolling,
        quantiles=config.quantiles,
        feature_cols=feature_cols,
    )

    quantile_cols = config.quantile_columns
    crossing_count_before = (
        forecast_df[quantile_cols].diff(axis=1).iloc[:, 1:].lt(0).any(axis=1).sum()
        if not forecast_df.empty
        else 0
    )
    forecast_sorted = sort_quantiles(df=forecast_df, quantile_cols=quantile_cols)
    crossing_count_after = (
        forecast_sorted[quantile_cols].diff(axis=1).iloc[:, 1:].lt(0).any(axis=1).sum()
        if not forecast_sorted.empty
        else 0
    )

    if "y_true" in forecast_sorted.columns and forecast_sorted["y_true"].notna().any():
        metrics_df = evaluate_probabilistic_forecasts(
            df=forecast_sorted,
            start_date=str(pd.Timestamp(config.test_start).date()),
            end_date=str(pd.Timestamp(config.test_end).date()),
            quantiles=config.quantiles,
            y_true_col="y_true",
        )
    else:
        metrics_df = pd.DataFrame()

    payload = {
        "experiment_name": config.experiment_name,
        "train_days": config.train_days_rolling,
        "quantiles": config.quantiles,
        "test_start": str(pd.Timestamp(config.test_start).date()),
        "test_end": str(pd.Timestamp(config.test_end).date()),
        "import_paths": [str(path) for path in config.import_paths],
    }

    if save_outputs:
        save_sqra_outputs(
            export_dir=config.export_dir,
            forecast_df=forecast_sorted,
            runtime_df=runtime_df,
            config=payload,
            metrics_df=metrics_df if not metrics_df.empty else None,
        )

    if plot:
        plot_prob_forecast(
            df_forecast=forecast_sorted,
            quantile_cols=quantile_cols,
            start_date=str(pd.Timestamp(config.test_start).date()),
            end_date=str(pd.Timestamp(config.test_end).date()),
            model_info=f"SQRA | d{config.train_days_rolling}",
        )

    return {
        "forecast": forecast_sorted,
        "runtime": runtime_df,
        "metrics": metrics_df,
        "panel": df_qra,
        "feature_cols": feature_cols,
        "crossings_before_sort": int(crossing_count_before),
        "crossings_after_sort": int(crossing_count_after),
    }
