from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from ...config import EnergyArenaSubmissionConfig, LearOperationalConfig, SqraConfig, TabpfnLocalConfig, TabpfnTsConfig

PLACEHOLDER_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


def extract_point_predictions(
    forecast_df: pd.DataFrame,
    forecast_date: date,
    value_column: str = "y_pred",
    target_tz: str = "Europe/Berlin",
) -> pd.DataFrame:
    """Extract one forecast day as timestamp/value pairs for submission."""
    if value_column not in forecast_df.columns:
        raise ValueError(f"Forecast DataFrame must contain a '{value_column}' column.")

    index = forecast_df.index
    if getattr(index, "tz", None) is None:
        index = index.tz_localize(target_tz)
    else:
        index = index.tz_convert(target_tz)

    work = forecast_df.copy()
    work.index = index

    start = pd.Timestamp(forecast_date, tz=target_tz)
    end = start + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    day_df = work.loc[start:end, [value_column]].copy()

    if day_df.empty:
        raise ValueError(f"No predictions found for forecast date {forecast_date}.")
    if len(day_df) != 96:
        raise ValueError(f"Expected 96 quarter-hour predictions, found {len(day_df)}.")
    if day_df[value_column].isna().any():
        raise ValueError(f"Forecast contains NaN values in '{value_column}'.")

    day_df = day_df.rename(columns={value_column: "value"}).reset_index()
    day_df = day_df.rename(columns={"index": "timestamp"})
    day_df["timestamp"] = day_df["timestamp"].map(lambda ts: pd.Timestamp(ts).isoformat())
    day_df["value"] = day_df["value"].map(float)
    return day_df


def extract_quantile_predictions(
    forecast_df: pd.DataFrame,
    forecast_date: date,
    quantile_columns: Sequence[str],
    target_tz: str = "Europe/Berlin",
) -> pd.DataFrame:
    """Extract one forecast day as timestamp/list-of-quantiles pairs."""
    missing_cols = [col for col in quantile_columns if col not in forecast_df.columns]
    if missing_cols:
        raise ValueError(f"Forecast DataFrame is missing quantile columns: {missing_cols}")

    index = forecast_df.index
    if getattr(index, "tz", None) is None:
        index = index.tz_localize(target_tz)
    else:
        index = index.tz_convert(target_tz)

    work = forecast_df.copy()
    work.index = index

    start = pd.Timestamp(forecast_date, tz=target_tz)
    end = start + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    day_df = work.loc[start:end, list(quantile_columns)].copy()

    if day_df.empty:
        raise ValueError(f"No predictions found for forecast date {forecast_date}.")
    if len(day_df) != 96:
        raise ValueError(f"Expected 96 quarter-hour predictions, found {len(day_df)}.")
    if day_df.isna().any().any():
        raise ValueError("Forecast contains NaN values in quantile columns.")

    out = pd.DataFrame(index=day_df.index)
    out["value"] = day_df.apply(lambda row: [float(row[col]) for col in quantile_columns], axis=1)
    out = out.reset_index().rename(columns={"index": "timestamp"})
    out["timestamp"] = out["timestamp"].map(lambda ts: pd.Timestamp(ts).isoformat())
    return out


def extract_submission_predictions(
    forecast_df: pd.DataFrame,
    forecast_date: date,
    objective: str,
    target_tz: str = "Europe/Berlin",
    value_column: str | None = None,
    quantile_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Extract one forecast day in the dense Energy Arena value shape."""
    if objective == "point":
        return extract_point_predictions(
            forecast_df=forecast_df,
            forecast_date=forecast_date,
            value_column=value_column or "y_pred",
            target_tz=target_tz,
        )

    if objective == "quantile":
        if not quantile_columns:
            raise ValueError("quantile_columns must be configured for quantile submissions.")
        return extract_quantile_predictions(
            forecast_df=forecast_df,
            forecast_date=forecast_date,
            quantile_columns=quantile_columns,
            target_tz=target_tz,
        )

    raise ValueError(f"Unsupported Energy Arena objective: {objective}")


def _dense_value(value: Any) -> Any:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [float(item) for item in value]
    return float(value)


def build_model_metadata(model_config: Any | None) -> dict[str, Any]:
    """Build model metadata without coupling the payload to one model type."""
    if model_config is None:
        return {}

    if isinstance(model_config, LearOperationalConfig):
        return {
            "model_type": "lear_operational",
            "model_variant": model_config.variant.value,
            "weather_source": None if model_config.use_exaa_only else model_config.weather_source.value,
            "train_days_rolling": model_config.train_days_rolling,
            "use_vst": model_config.use_vst,
            "lars_start_date": str(pd.Timestamp(model_config.lars_start_date).date()),
        }

    if isinstance(model_config, SqraConfig):
        return {
            "model_type": "sqra",
            "quantiles": model_config.quantiles,
            "train_days_rolling": model_config.train_days_rolling,
            "import_paths": [str(path) for path in model_config.import_paths],
        }

    if isinstance(model_config, TabpfnTsConfig):
        return {
            "model_type": "tabpfn_ts",
            "quantiles": model_config.quantiles,
            "tabpfn_mode": model_config.tabpfn_mode,
            "tabpfn_output_selection": model_config.tabpfn_output_selection,
            "tabpfn_model_config": model_config.tabpfn_model_config,
            "max_context_length": model_config.max_context_length,
            "use_exaa": model_config.use_exaa,
            "use_load_forecast": model_config.use_load_forecast,
        }

    if isinstance(model_config, TabpfnLocalConfig):
        return {
            "model_type": "tabpfn_local",
            "quantiles": model_config.quantiles,
            "point_output_type": model_config.point_output_type,
            "max_train_rows": model_config.max_train_rows,
            "train_window_days": model_config.train_window_days,
            "n_estimators": model_config.n_estimators,
            "device": model_config.device,
            "fit_mode": model_config.fit_mode,
            "predict_batch_size": model_config.predict_batch_size,
            "use_exaa": model_config.use_exaa,
            "use_load_forecast": model_config.use_load_forecast,
            "price_lag_days": model_config.price_lag_days,
            "add_calendar_features": model_config.add_calendar_features,
        }

    return {}


def build_submission_context(
    predictions: pd.DataFrame,
    submission_config: EnergyArenaSubmissionConfig,
    model_config: Any | None = None,
    source_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the template context for submission payload rendering."""
    values = [_dense_value(value) for value in predictions["value"].tolist()]
    metadata = {
        "approach_name": submission_config.approach_name,
        "approach_description": submission_config.approach_description,
        "repo_url": submission_config.repo_url,
        "objective": submission_config.objective.value,
    }
    metadata.update(build_model_metadata(model_config))
    metadata.update(source_metadata or {})

    # Backward-compatible field for older metadata consumers.
    metadata.setdefault("model_variant", None)
    metadata.setdefault("weather_source", None)

    return {
        "challenge_id": submission_config.challenge_id,
        "challenge_id_str": str(submission_config.challenge_id),
        "forecast_date": submission_config.forecast_date.isoformat(),
        "target_start": predictions.iloc[0]["timestamp"],
        "values": values,
        "forecast_visibility": submission_config.forecast_visibility,
        "leaderboard_visibility": submission_config.leaderboard_visibility,
        "approach_name": submission_config.approach_name,
        "approach_description": submission_config.approach_description,
        "repo_url": submission_config.repo_url,
        "metadata": metadata,
        "predictions": predictions.to_dict(orient="records"),
    }


def _render_string_template(template: str, context: dict[str, Any]) -> Any:
    match = PLACEHOLDER_RE.fullmatch(template)
    if match:
        return context[match.group(1)]

    def _replace_token(found: re.Match[str]) -> str:
        key = found.group(1)
        value = context[key]
        return "" if value is None else str(value)

    return PLACEHOLDER_RE.sub(_replace_token, template)


def render_payload_template(template: Any, context: dict[str, Any]) -> Any:
    """Render a JSON-style object tree with {{placeholders}}."""
    if isinstance(template, str):
        return _render_string_template(template, context)
    if isinstance(template, list):
        return [render_payload_template(item, context) for item in template]
    if isinstance(template, dict):
        return {key: render_payload_template(value, context) for key, value in template.items()}
    return template


def build_candidate_payload(
    predictions: pd.DataFrame,
    submission_config: EnergyArenaSubmissionConfig,
    model_config: Any | None = None,
    source_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the dense Energy Arena payload shape from the example schema."""
    context = build_submission_context(
        predictions,
        submission_config,
        model_config=model_config,
        source_metadata=source_metadata,
    )
    return {
        "challenge_id": context["challenge_id_str"],
        "target_start": context["target_start"],
        "values": context["values"],
    }


def load_payload_template(path: Path) -> Any:
    """Load a JSON payload template from disk."""
    import json

    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
