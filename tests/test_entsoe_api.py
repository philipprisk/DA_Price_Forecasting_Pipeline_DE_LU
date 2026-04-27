from __future__ import annotations

import importlib
import sys
import types

import pandas as pd


def _load_entsoe_module(monkeypatch, pandas_client_cls, raw_client_cls, parse_prices):
    entsoe_module = types.ModuleType("entsoe")
    entsoe_module.EntsoePandasClient = pandas_client_cls
    entsoe_module.EntsoeRawClient = raw_client_cls

    parsers_module = types.ModuleType("entsoe.parsers")
    parsers_module.parse_prices = parse_prices

    monkeypatch.setitem(sys.modules, "entsoe", entsoe_module)
    monkeypatch.setitem(sys.modules, "entsoe.parsers", parsers_module)
    sys.modules.pop("da_price_forecasting.data.entsoe", None)
    return importlib.import_module("da_price_forecasting.data.entsoe")


def _hourly_series(start: pd.Timestamp, name: str) -> pd.Series:
    index = pd.date_range(start=start, periods=24, freq="h")
    return pd.Series(range(24), index=index, dtype=float, name=name)


def test_fetch_prices_queries_short_window_and_expands_to_quarter_hours(monkeypatch) -> None:
    class FakePandasClient:
        instances = []

        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.requests = []
            self.instances.append(self)

        def query_day_ahead_prices(self, country_code, start, end):
            self.requests.append((country_code, start, end))
            return _hourly_series(start, "price_da")

    class FakeRawClient:
        pass

    monkeypatch.setenv("ENTSOE_TEST_API_KEY", "secret")
    entsoe = _load_entsoe_module(monkeypatch, FakePandasClient, FakeRawClient, lambda xml: None)

    start_day = pd.Timestamp("2026-03-01", tz="Europe/Berlin")
    result = entsoe.fetch_prices(
        start_day=start_day,
        end_day=start_day,
        api_key_env="ENTSOE_TEST_API_KEY",
    )

    assert len(result) == 96
    assert result.index[0] == pd.Timestamp("2026-03-01T00:00:00+01:00")
    assert result.index[-1] == pd.Timestamp("2026-03-01T23:45:00+01:00")
    assert result["price_da"].iloc[:4].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert result["price_da"].iloc[4] == 1.0

    client = FakePandasClient.instances[0]
    assert client.api_key == "secret"
    country_code, query_start, query_end = client.requests[0]
    assert country_code == "DE_LU"
    assert query_start == pd.Timestamp("2026-03-01T00:00:00+01:00")
    assert query_end == pd.Timestamp("2026-03-02T00:00:00+01:00")


def test_fetch_load_forecast_accepts_dataframe_response(monkeypatch) -> None:
    class FakePandasClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def query_load_forecast(self, country_code, start, end):
            return _hourly_series(start, "load").to_frame()

    class FakeRawClient:
        pass

    monkeypatch.setenv("ENTSOE_TEST_API_KEY", "secret")
    entsoe = _load_entsoe_module(monkeypatch, FakePandasClient, FakeRawClient, lambda xml: None)

    start_day = pd.Timestamp("2026-03-01", tz="Europe/Berlin")
    result = entsoe.fetch_load_forecast(
        start_day=start_day,
        end_day=start_day,
        api_key_env="ENTSOE_TEST_API_KEY",
    )

    assert len(result) == 96
    assert result.columns.tolist() == ["load_fc"]
    assert result["load_fc"].iloc[0] == 0.0
    assert result["load_fc"].iloc[-1] == 23.0


def test_fetch_exaa_prices_parses_raw_api_xml(monkeypatch) -> None:
    class FakePandasClient:
        pass

    class FakeRawClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def query_day_ahead_prices(self, country_code, start, end, sequence):
            assert country_code == "DE_LU"
            assert sequence == 2
            return "<publication-market-document/>"

    def fake_parse_prices(xml):
        assert xml == "<publication-market-document/>"
        return _hourly_series(pd.Timestamp("2026-03-01T00:00:00+01:00"), "price_exaa")

    monkeypatch.setenv("ENTSOE_TEST_API_KEY", "secret")
    entsoe = _load_entsoe_module(monkeypatch, FakePandasClient, FakeRawClient, fake_parse_prices)

    start_day = pd.Timestamp("2026-03-01", tz="Europe/Berlin")
    result = entsoe.fetch_prices_exaa(
        start_day=start_day,
        end_day=start_day,
        api_key_env="ENTSOE_TEST_API_KEY",
    )

    assert len(result) == 96
    assert result.columns.tolist() == ["price_exaa"]
    assert result["price_exaa"].iloc[0] == 0.0
    assert result["price_exaa"].iloc[-1] == 23.0
