from __future__ import annotations

from datetime import date

import pytest
import requests

from da_price_forecasting.data.weather import _fetch_open_meteo_batch


class _OpenMeteoResponse:
    status_code = 200
    headers: dict[str, str] = {}
    url = "https://single-runs-api.open-meteo.com/v1/forecast"

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "hourly": {
                "time": ["2026-04-05T00:00"],
                "temperature_2m": [12.0],
            }
        }


def test_fetch_open_meteo_batch_retries_connection_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_get(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.exceptions.ConnectTimeout("timed out")
        return _OpenMeteoResponse()

    monkeypatch.setattr("da_price_forecasting.data.weather.requests.get", fake_get)
    monkeypatch.setattr("da_price_forecasting.data.weather.time.sleep", lambda seconds: None)

    items = _fetch_open_meteo_batch(
        base_url="https://single-runs-api.open-meteo.com/v1/forecast",
        latitude=[52.0],
        longitude=[13.0],
        hourly_variables=["temperature_2m"],
        model="icon_d2",
        cell_selection="nearest",
        timeout_seconds=60,
        start_date=date(2026, 4, 5),
        end_date=date(2026, 4, 5),
        retry_attempts=1,
        retry_backoff_seconds=0.0,
    )

    assert calls["count"] == 2
    assert items == [_OpenMeteoResponse().json()]


def test_fetch_open_meteo_batch_raises_after_timeout_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        raise requests.exceptions.ConnectTimeout("timed out")

    monkeypatch.setattr("da_price_forecasting.data.weather.requests.get", fake_get)
    monkeypatch.setattr("da_price_forecasting.data.weather.time.sleep", lambda seconds: None)

    with pytest.raises(requests.exceptions.ConnectTimeout):
        _fetch_open_meteo_batch(
            base_url="https://single-runs-api.open-meteo.com/v1/forecast",
            latitude=[52.0],
            longitude=[13.0],
            hourly_variables=["temperature_2m"],
            model="icon_d2",
            cell_selection="nearest",
            timeout_seconds=60,
            start_date=date(2026, 4, 5),
            end_date=date(2026, 4, 5),
            retry_attempts=1,
            retry_backoff_seconds=0.0,
        )
