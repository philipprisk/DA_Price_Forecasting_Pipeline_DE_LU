from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def _require_sqra_class():
    try:
        from remodels.qra import SQRA
    except ImportError as exc:
        raise ImportError(
            "remodels is not installed. Install the SQRA Pixi environment with `pixi install -e sqra` "
            "or run the command via `pixi run -e sqra ...`."
        ) from exc
    return SQRA


def fit_sqra_for_mtu_and_quantile(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    mtu: int,
    quantile: float,
    feature_cols: list[str],
) -> Any | None:
    """Fit SQRA model for one MTU and one quantile."""
    idx = train_mask & (df["mtu"] == mtu)
    if idx.sum() == 0:
        return None

    X_train = df.loc[idx, feature_cols]
    y_train = df.loc[idx, "y_true"]

    valid = X_train.notna().all(axis=1) & y_train.notna()
    if valid.sum() == 0:
        return None

    X_train = X_train.loc[valid].to_numpy()
    y_train = y_train.loc[valid].to_numpy()

    SQRA = _require_sqra_class()
    model = SQRA(quantile=quantile, fit_intercept=True)

    try:
        model.fit(X_train, y_train)
    except Exception:
        return None

    return model


def predict_sqra_for_mtu(
    df: pd.DataFrame,
    test_mask: np.ndarray,
    mtu: int,
    model: Any | None,
    feature_cols: list[str],
) -> pd.Series:
    """Generate SQRA predictions for a single MTU on the forecast day."""
    idx = test_mask & (df["mtu"] == mtu)
    if not idx.any():
        return pd.Series(dtype=float)

    index = df.loc[idx].index
    if model is None:
        return pd.Series(index=index, dtype=float)

    X_test = df.loc[idx, feature_cols]
    valid = X_test.notna().all(axis=1)
    if not valid.any():
        return pd.Series(index=index, dtype=float)

    preds = pd.Series(index=index, dtype=float)
    preds.loc[valid] = model.predict(X_test.loc[valid].to_numpy())
    return preds


def rolling_sqra_forecast_mtu(
    df: pd.DataFrame,
    forecast_days: list[pd.Timestamp] | pd.DatetimeIndex,
    train_days: int,
    quantiles: list[float],
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate rolling SQRA quantile forecasts for all MTUs and forecast days."""
    all_days = []
    runtime_records = []

    for day in forecast_days:
        print(day)
        start_time = time.perf_counter()

        train_start = day - pd.Timedelta(days=train_days)
        train_end = day - pd.Timedelta(minutes=15)
        test_end = day + pd.Timedelta(days=1)

        train_mask = (df.index >= train_start) & (df.index <= train_end)
        test_mask = (df.index >= day) & (df.index < test_end)

        day_index = df.loc[test_mask].index
        day_df = pd.DataFrame(index=day_index)

        for tau in quantiles:
            preds_tau = []
            for mtu in range(96):
                model = fit_sqra_for_mtu_and_quantile(
                    df=df,
                    train_mask=train_mask,
                    mtu=mtu,
                    quantile=tau,
                    feature_cols=feature_cols,
                )
                preds_mtu = predict_sqra_for_mtu(
                    df=df,
                    test_mask=test_mask,
                    mtu=mtu,
                    model=model,
                    feature_cols=feature_cols,
                )
                preds_tau.append(preds_mtu)

            day_df[f"q{tau:.3f}"] = pd.concat(preds_tau).sort_index()

        day_df["y_true"] = df.loc[test_mask, "y_true"]
        all_days.append(day_df)

        runtime_seconds = time.perf_counter() - start_time
        runtime_records.append(
            {
                "forecast_day": day,
                "computation_time_seconds": runtime_seconds,
            }
        )

    forecast_df = pd.concat(all_days).sort_index() if all_days else pd.DataFrame()
    runtime_df = pd.DataFrame(runtime_records)
    return forecast_df, runtime_df


def sort_quantiles(df: pd.DataFrame, quantile_cols: list[str]) -> pd.DataFrame:
    """Enforce monotonicity of quantile forecasts per timestamp."""
    df_sorted = df.copy()
    missing_cols = [col for col in quantile_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing quantile columns: {missing_cols}")

    sorted_values = np.sort(df_sorted[quantile_cols].to_numpy(), axis=1)
    df_sorted[quantile_cols] = sorted_values
    return df_sorted


def plot_prob_forecast(
    df_forecast: pd.DataFrame,
    quantile_cols: list[str],
    start_date: str,
    end_date: str,
    y_true_col: str = "y_true",
    title: str = "",
    model_info: str | None = None,
):
    """Plot probabilistic forecast with prediction intervals and median."""
    df = df_forecast.loc[start_date:end_date]
    if df.empty:
        raise ValueError("Selected time window is empty.")

    n_cols = len(quantile_cols)
    q_outer_low = quantile_cols[0]
    q_outer_high = quantile_cols[-1]
    q_median = quantile_cols[n_cols // 2]

    if n_cols >= 4:
        q_inner_low = quantile_cols[1]
        q_inner_high = quantile_cols[-2]
    else:
        q_inner_low = None
        q_inner_high = None

    plt.figure(figsize=(18, 6))
    if q_outer_low in df.columns and q_outer_high in df.columns:
        plt.fill_between(df.index, df[q_outer_low], df[q_outer_high], alpha=0.2, label="Outer PI")

    if q_inner_low and q_inner_high and q_inner_low in df.columns and q_inner_high in df.columns:
        plt.fill_between(df.index, df[q_inner_low], df[q_inner_high], alpha=0.4, label="Inner PI")

    if q_median in df.columns:
        plt.plot(df.index, df[q_median], linewidth=2, label="Median forecast")

    if y_true_col in df.columns:
        plt.plot(df.index, df[y_true_col], linewidth=2, label="Actual", color="#3b5b73")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Electricity Price [EUR/MWh]")
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right")

    if model_info is not None:
        plt.gca().text(
            0.01,
            0.99,
            model_info,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.show()


def mae_median(df: pd.DataFrame, q_median: float = 0.5, y_true_col: str = "y_true"):
    """Compute MAE between realized values and the median quantile forecast."""
    col_median = f"q{q_median:.3f}"
    return mean_absolute_error(df[y_true_col], df[col_median])


def empirical_coverage(
    df: pd.DataFrame,
    lower_q: float,
    upper_q: float,
    y_true_col: str = "y_true",
):
    """Compute empirical coverage of a prediction interval."""
    col_lower = f"q{lower_q:.3f}"
    col_upper = f"q{upper_q:.3f}"
    inside = (df[y_true_col] >= df[col_lower]) & (df[y_true_col] <= df[col_upper])
    return inside.mean()


def pinball_score(y_true, y_q, q):
    """Compute the pinball (quantile) loss for a single quantile."""
    diff = y_true - y_q
    return np.mean(np.maximum(q * diff, (q - 1) * diff))


def aggregate_pinball_score(df: pd.DataFrame, quantiles: list[float], y_true_col: str = "y_true"):
    """Compute the average pinball score across all quantile levels."""
    scores = {}
    y_true = df[y_true_col]
    for q in quantiles:
        col = f"q{q:.3f}"
        scores[q] = pinball_score(y_true, df[col], q)

    aps = np.mean(list(scores.values()))
    return scores, aps


def evaluate_probabilistic_forecasts(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    quantiles: list[float],
    y_true_col: str = "y_true",
):
    """Evaluate probabilistic forecasts per calendar day over a given window."""
    quantiles = sorted(quantiles)
    if 0.5 not in quantiles:
        raise ValueError("Quantiles must contain 0.5 for MAE evaluation.")
    if len(quantiles) < 3:
        raise ValueError("Need at least 3 quantiles to form prediction intervals.")

    mid_idx = quantiles.index(0.5)
    if mid_idx == 0 or mid_idx == len(quantiles) - 1:
        raise ValueError("Quantiles must be symmetric around 0.5.")

    q_inner_low = quantiles[mid_idx - 1]
    q_inner_high = quantiles[mid_idx + 1]
    q_outer_low = quantiles[0]
    q_outer_high = quantiles[-1]

    tz = df.index.tz
    date_range = pd.date_range(start=start_date, end=end_date, freq="D", tz=tz)

    results = []
    for day in date_range:
        day_end = day + pd.Timedelta(days=1)
        df_day = df.loc[(df.index >= day) & (df.index < day_end)]
        if df_day.empty:
            continue

        mae = mae_median(df_day, q_median=0.5, y_true_col=y_true_col)
        cov_inner = empirical_coverage(df_day, q_inner_low, q_inner_high, y_true_col)
        cov_outer = empirical_coverage(df_day, q_outer_low, q_outer_high, y_true_col)
        _, aps = aggregate_pinball_score(df_day, quantiles, y_true_col)

        results.append(
            {
                "Target Day": str(day.date()),
                "MAE (median)": mae,
                f"Coverage {q_inner_low}-{q_inner_high}": cov_inner,
                f"Coverage {q_outer_low}-{q_outer_high}": cov_outer,
                "APS": aps,
            }
        )

    df_res = pd.DataFrame(results)
    summary = df_res.mean(numeric_only=True)
    summary["Target Day"] = "Mean over all days"
    return pd.concat([df_res, pd.DataFrame([summary])], ignore_index=True)


def load_forecast(path: Path) -> pd.DataFrame:
    """Load a point forecast CSV and return a timezone-aware DataFrame."""
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("Europe/Berlin")
    return df


def save_sqra_outputs(
    export_dir: Path,
    forecast_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    config: dict,
    metrics_df: pd.DataFrame | None = None,
) -> None:
    """Save SQRA outputs to disk using the notebook-compatible artifact layout."""
    export_dir.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(export_dir / "forecast.csv", index=True)
    runtime_df.to_csv(export_dir / "runtime.csv", index=False)
    if metrics_df is not None:
        metrics_df.to_csv(export_dir / "metrics.csv", index=False)

    payload = dict(config)
    payload.setdefault("created_at", datetime.now().strftime("%Y-%m-%d %H:%M"))
    with open(export_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"✓ Saved  ->  {export_dir}")
