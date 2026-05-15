from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from da_price_forecasting.data.commodities import (
    InvestinyCommodityInstrument,
    YFinanceCommodityInstrument,
    fetch_investiny_commodity_prices,
    fetch_yfinance_commodity_prices,
)


def test_fetch_yfinance_commodity_prices_lags_and_expands_daily_values(monkeypatch) -> None:
    calls = []

    def fake_download(ticker, start, end, interval, auto_adjust, progress):
        calls.append(
            {
                "ticker": ticker,
                "start": start,
                "end": end,
                "interval": interval,
                "auto_adjust": auto_adjust,
                "progress": progress,
            }
        )
        index = pd.date_range("2026-02-04", periods=5, freq="D")
        return pd.DataFrame({"Close": [4.0, 5.0, 6.0, 7.0, 8.0]}, index=index)

    monkeypatch.setattr(
        "da_price_forecasting.data.commodities._require_yfinance",
        lambda: SimpleNamespace(download=fake_download),
    )

    result = fetch_yfinance_commodity_prices(
        start_day=pd.Timestamp("2026-02-08", tz="Europe/Berlin"),
        end_day=pd.Timestamp("2026-02-08", tz="Europe/Berlin"),
        instruments=[
            YFinanceCommodityInstrument(
                name="gas_ttf",
                ticker="TTF=F",
                price_field="Close",
            )
        ],
        lag_days=2,
        target_tz="Europe/Berlin",
        lookback_days=4,
        auto_adjust=False,
        progress=False,
    )

    assert result.columns.tolist() == ["gas_ttf"]
    assert len(result) == 96
    assert result.index.name == "timestamp"
    assert result.loc[pd.Timestamp("2026-02-08T00:00:00+01:00"), "gas_ttf"] == 6.0
    assert result.loc[pd.Timestamp("2026-02-08T23:45:00+01:00"), "gas_ttf"] == 6.0
    assert calls == [
        {
            "ticker": "TTF=F",
            "start": "2026-02-02",
            "end": "2026-02-09",
            "interval": "1d",
            "auto_adjust": False,
            "progress": False,
        }
    ]


def test_fetch_investiny_commodity_prices_lags_and_expands_daily_values(monkeypatch) -> None:
    calls = []

    class FakeResponse:
        def json(self):
            return {
                "s": "ok",
                "t": [
                    int(pd.Timestamp("2026-02-04", tz="UTC").timestamp()),
                    int(pd.Timestamp("2026-02-05", tz="UTC").timestamp()),
                    int(pd.Timestamp("2026-02-06", tz="UTC").timestamp()),
                    int(pd.Timestamp("2026-02-07", tz="UTC").timestamp()),
                    int(pd.Timestamp("2026-02-08", tz="UTC").timestamp()),
                ],
                "o": [3.5, 4.5, 5.5, 6.5, 7.5],
                "h": [4.5, 5.5, 6.5, 7.5, 8.5],
                "l": [3.0, 4.0, 5.0, 6.0, 7.0],
                "c": [4.0, 5.0, 6.0, 7.0, 8.0],
            }

        def raise_for_status(self):
            return None

    def fake_get(url, params, headers, timeout):
        calls.append(
            {
                "url": url,
                "symbol": params["symbol"],
                "from": params["from"],
                "to": params["to"],
                "resolution": params["resolution"],
                "referer": headers["Referer"],
                "timeout": timeout,
            }
        )
        return FakeResponse()

    monkeypatch.setattr(
        "da_price_forecasting.data.commodities.requests.get",
        fake_get,
    )

    result = fetch_investiny_commodity_prices(
        start_day=pd.Timestamp("2026-02-08", tz="Europe/Berlin"),
        end_day=pd.Timestamp("2026-02-08", tz="Europe/Berlin"),
        instruments=[
            InvestinyCommodityInstrument(
                name="co2_eua",
                investing_id=8848,
                price_field="close",
            )
        ],
        lag_days=2,
        target_tz="Europe/Berlin",
        lookback_days=4,
        request_timeout_seconds=5,
    )

    assert result.columns.tolist() == ["co2_eua"]
    assert len(result) == 96
    assert result.index.name == "timestamp"
    assert result.loc[pd.Timestamp("2026-02-08T00:00:00+01:00"), "co2_eua"] == 6.0
    assert result.loc[pd.Timestamp("2026-02-08T23:45:00+01:00"), "co2_eua"] == 6.0
    assert calls == [
        {
            "url": calls[0]["url"],
            "symbol": 8848,
            "from": int(pd.Timestamp("2026-02-02", tz="Europe/Berlin").timestamp()),
            "to": int(pd.Timestamp("2026-02-09", tz="Europe/Berlin").timestamp()),
            "resolution": "D",
            "referer": "https://tvc-invdn-com.investing.com/",
            "timeout": 5,
        }
    ]
    assert calls[0]["url"].startswith("https://tvc6.investing.com/")
