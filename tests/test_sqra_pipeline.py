from __future__ import annotations

from pathlib import Path

import pandas as pd

from da_price_forecasting.config import SqraConfig
from da_price_forecasting.pipelines.sqra import build_sqra_panel


def test_build_sqra_panel_deduplicates_point_forecast_timestamps(tmp_path: Path) -> None:
    forecast_path = tmp_path / "forecast.csv"
    index = pd.DatetimeIndex(
        [
            "2026-05-20T00:00:00+02:00",
            "2026-05-20T00:00:00+02:00",
            "2026-05-20T00:15:00+02:00",
        ]
    )
    pd.DataFrame(
        {
            "y_pred": [10.0, 11.0, 12.0],
            "y_true": [9.0, 9.5, 13.0],
        },
        index=index,
    ).to_csv(forecast_path)

    panel, feature_cols = build_sqra_panel(
        SqraConfig(
            repo_root=tmp_path,
            import_paths=[forecast_path],
            test_start="2026-05-20T00:00:00+02:00",
            test_end="2026-05-20T00:15:00+02:00",
        )
    )

    assert feature_cols == ["prediction_p1"]
    assert panel.index.is_unique
    assert panel.loc[pd.Timestamp("2026-05-20T00:00:00+02:00"), "prediction_p1"] == 11.0
    assert panel.loc[pd.Timestamp("2026-05-20T00:00:00+02:00"), "y_true"] == 9.5


def test_build_sqra_panel_accepts_load_forecast_columns(tmp_path: Path) -> None:
    forecast_path = tmp_path / "load_forecast.csv"
    index = pd.date_range("2026-05-20T00:00:00+02:00", periods=2, freq="15min")
    pd.DataFrame(
        {
            "Load_Model_MW": [50_000.0, 51_000.0],
            "Load_Actual_MW": [49_500.0, 51_500.0],
        },
        index=index,
    ).to_csv(forecast_path)

    panel, feature_cols = build_sqra_panel(
        SqraConfig(
            repo_root=tmp_path,
            import_paths=[forecast_path],
            test_start="2026-05-20T00:00:00+02:00",
            test_end="2026-05-20T00:15:00+02:00",
        )
    )

    assert feature_cols == ["prediction_p1"]
    assert panel.loc[index[0], "prediction_p1"] == 50_000.0
    assert panel.loc[index[1], "y_true"] == 51_500.0
