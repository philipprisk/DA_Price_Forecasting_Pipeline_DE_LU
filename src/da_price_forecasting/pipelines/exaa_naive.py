from __future__ import annotations

import time
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from ..config import ExaaNaiveConfig
from ..data.entsoe import fetch_prices, fetch_prices_exaa
from ..models.lear import save_experiment_outputs


def _load_env(config: ExaaNaiveConfig) -> None:
    load_dotenv(config.repo_root / ".env")


def _as_target_tz(value: Any, target_tz: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(target_tz)
    return timestamp.tz_convert(target_tz)


def run_exaa_naive_pipeline(config: ExaaNaiveConfig, save_outputs: bool = True) -> dict[str, Any]:
    """Run the EXAA naive benchmark: y_pred equals same-day EXAA price."""
    _load_env(config)
    start_time = time.perf_counter()

    entsoe_start = _as_target_tz(config.entsoe_start_date, config.target_tz)
    entsoe_end = _as_target_tz(config.entsoe_end_date, config.target_tz)
    test_start = _as_target_tz(config.test_start, config.target_tz)
    test_end = _as_target_tz(config.test_end, config.target_tz)

    prices = fetch_prices(
        start_day=entsoe_start,
        end_day=entsoe_end,
        country_code=config.country_code_entsoe,
        api_key_env=config.entsoe_api_key_env,
        target_tz=config.target_tz,
    )
    prices_exaa = fetch_prices_exaa(
        start_day=entsoe_start,
        end_day=entsoe_end,
        country_code=config.country_code_entsoe,
        api_key_env=config.entsoe_api_key_env,
        target_tz=config.target_tz,
    )

    forecast_df = (
        prices_exaa.rename(columns={"price_exaa": "y_pred"})
        .join(prices.rename(columns={"price_da": "y_true"}), how="inner")
        .loc[test_start:test_end]
        .sort_index()
    )

    runtime_df = pd.DataFrame(
        [
            {
                "runtime_seconds": time.perf_counter() - start_time,
                "n_rows": len(forecast_df),
                "test_start": test_start,
                "test_end": test_end,
            }
        ]
    )

    export_payload = {
        "experiment_name": config.experiment_name,
        "model": "exaa_naive",
        "test_start": str(test_start.date()),
        "test_end": str(test_end.date()),
    }
    if save_outputs:
        save_experiment_outputs(
            name=config.experiment_name,
            forecast_df=forecast_df,
            runtime_df=runtime_df,
            config=export_payload,
            export_dir=config.resolved_export_dir,
        )

    return {
        "forecast": forecast_df,
        "runtime": runtime_df,
    }
