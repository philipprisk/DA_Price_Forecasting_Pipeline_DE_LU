from __future__ import annotations

import pandas as pd

from da_price_forecasting.config import ExaaNaiveConfig
from da_price_forecasting.pipelines import exaa_naive


def test_exaa_naive_pipeline_uses_exaa_as_prediction(monkeypatch) -> None:
    index = pd.date_range("2025-12-01T00:00:00+01:00", periods=4, freq="15min")

    def fake_fetch_prices(**kwargs):
        return pd.DataFrame({"price_da": [10.0, 11.0, 12.0, 13.0]}, index=index)

    def fake_fetch_prices_exaa(**kwargs):
        return pd.DataFrame({"price_exaa": [9.0, 10.5, 12.5, 13.5]}, index=index)

    monkeypatch.setattr(exaa_naive, "fetch_prices", fake_fetch_prices)
    monkeypatch.setattr(exaa_naive, "fetch_prices_exaa", fake_fetch_prices_exaa)
    monkeypatch.setattr(exaa_naive, "_load_env", lambda config: None)

    config = ExaaNaiveConfig.model_validate(
        {
            "repo_root": ".",
            "entsoe_start_date": "2025-12-01T00:00:00+01:00",
            "entsoe_end_date": "2025-12-01T00:45:00+01:00",
            "test_start": "2025-12-01T00:15:00+01:00",
            "test_end": "2025-12-01T00:30:00+01:00",
        }
    )

    result = exaa_naive.run_exaa_naive_pipeline(config, save_outputs=False)
    forecast = result["forecast"]

    assert forecast.index.tolist() == [index[1], index[2]]
    assert forecast["y_pred"].tolist() == [10.5, 12.5]
    assert forecast["y_true"].tolist() == [11.0, 12.0]
