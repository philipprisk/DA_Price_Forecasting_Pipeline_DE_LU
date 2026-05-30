from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from da_price_forecasting.config import RenewableGenerationModelConfig
from da_price_forecasting.pipelines.common import save_timestamp_csv
from da_price_forecasting.pipelines.renewable_generation import build_renewable_generation_dataset


def test_renewable_generation_dataset_keeps_future_proxy_rows_without_actuals(tmp_path: Path) -> None:
    target_tz = "Europe/Berlin"
    proxy_index = pd.date_range("2026-05-29T00:00:00+02:00", periods=3 * 96, freq="15min")
    actual_index = proxy_index[proxy_index < pd.Timestamp("2026-05-31T00:00:00+02:00")]

    proxy_file = tmp_path / "renewable_proxy.csv"
    actual_file = tmp_path / "actual_generation.csv"
    save_timestamp_csv(
        pd.DataFrame(
            {
                "solar_proxy_mw": np.linspace(0.0, 100.0, len(proxy_index)),
                "wind_proxy_mw": np.linspace(100.0, 200.0, len(proxy_index)),
            },
            index=proxy_index,
        ),
        proxy_file,
    )
    save_timestamp_csv(
        pd.DataFrame(
            {
                "Solar_Actual_MW": np.linspace(0.0, 100.0, len(actual_index)),
                "Wind_Total_Actual_MW": np.linspace(100.0, 200.0, len(actual_index)),
                "Renewable_Total_Actual_MW": np.linspace(100.0, 300.0, len(actual_index)),
            },
            index=actual_index,
        ),
        actual_file,
    )

    config = RenewableGenerationModelConfig(
        repo_root=tmp_path,
        target_tz=target_tz,
        entsoe_start_date=date(2026, 5, 1),
        entsoe_end_date=date(2026, 5, 30),
        actual_generation_file=actual_file,
        renewable_proxy_file=proxy_file,
        unavailability_file=tmp_path / "unavailability.csv",
        icon_dir=tmp_path / "icon",
        export_dir=tmp_path / "export",
        include_dwd_cluster_features=False,
        include_unavailability=False,
        target_columns=["Solar_Actual_MW", "Wind_Total_Actual_MW"],
        test_start=date(2026, 5, 31),
        test_end=date(2026, 5, 31),
    )

    dataset = build_renewable_generation_dataset(config)
    forecast_day = pd.Timestamp("2026-05-31T00:00:00+02:00")

    assert dataset.index.max() == pd.Timestamp("2026-05-31T23:45:00+02:00")
    assert len(dataset.loc[forecast_day:]) == 96
    assert dataset.loc[forecast_day:, "Solar_Actual_MW"].isna().all()
    assert dataset.loc[forecast_day:, "solar_proxy_mw"].notna().all()
