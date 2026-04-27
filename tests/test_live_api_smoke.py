from __future__ import annotations

import importlib
import os
import sys

import pandas as pd
import pytest


pytestmark = [
    pytest.mark.live_api,
    pytest.mark.skipif(
        os.getenv("RUN_LIVE_API_TESTS") != "1",
        reason="set RUN_LIVE_API_TESTS=1 to run live API smoke tests",
    ),
    pytest.mark.skipif(
        not os.getenv("ENTSOE_API_KEY"),
        reason="set ENTSOE_API_KEY to run live ENTSO-E smoke tests",
    ),
]


def _load_real_entsoe_module():
    pytest.importorskip("entsoe")
    sys.modules.pop("da_price_forecasting.data.entsoe", None)
    return importlib.import_module("da_price_forecasting.data.entsoe")


def test_live_entsoe_day_ahead_prices_short_window() -> None:
    entsoe = _load_real_entsoe_module()
    day = pd.Timestamp("2026-03-01", tz="Europe/Berlin")

    prices = entsoe.fetch_prices(start_day=day, end_day=day)

    assert len(prices) == 96
    assert prices.index[0] == pd.Timestamp("2026-03-01T00:00:00+01:00")
    assert prices.index[-1] == pd.Timestamp("2026-03-01T23:45:00+01:00")
    assert prices.columns.tolist() == ["price_da"]
    assert prices["price_da"].notna().all()


def test_live_entsoe_load_forecast_short_window() -> None:
    entsoe = _load_real_entsoe_module()
    day = pd.Timestamp("2026-03-01", tz="Europe/Berlin")

    load = entsoe.fetch_load_forecast(start_day=day, end_day=day)

    assert len(load) == 96
    assert load.index[0] == pd.Timestamp("2026-03-01T00:00:00+01:00")
    assert load.index[-1] == pd.Timestamp("2026-03-01T23:45:00+01:00")
    assert load.columns.tolist() == ["load_fc"]
    assert load["load_fc"].notna().all()
