from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import requests

from da_price_forecasting.data import weather
from da_price_forecasting.data.weather import _fetch_open_meteo_batch, fetch_open_meteo_cluster_weather


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


def test_single_run_backfill_preserves_existing_cache_days(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cluster_file = tmp_path / "clusters.csv"
    cluster_file.write_text("cluster_id,lat,lon\n0,52.0,13.0\n")
    cache_file = tmp_path / "open_meteo.csv"
    existing = pd.DataFrame(
        {"t2m_cluster_0": [283.15]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-20T00:00:00+01:00")], name="timestamp"),
    )
    existing.to_csv(cache_file)

    def fake_fetch_open_meteo_batch(**kwargs):  # noqa: ANN003, ARG001
        return [
            {
                "hourly": {
                    "time": ["2026-03-20T23:00"],
                    "temperature_2m": [11.0],
                }
            }
        ]

    monkeypatch.setattr(weather, "_fetch_open_meteo_batch", fake_fetch_open_meteo_batch)

    fetch_open_meteo_cluster_weather(
        cluster_file=cluster_file,
        start_date=date(2026, 3, 21),
        end_date=date(2026, 3, 21),
        output_file=cache_file,
        hourly_variables=["temperature_2m"],
        batch_size=1,
        target_tz="Europe/Berlin",
        api_mode="single_run",
        single_run_hour_utc="06:00",
    )

    cached = pd.read_csv(cache_file, index_col=0)
    cached.index = pd.to_datetime(cached.index, utc=True).tz_convert("Europe/Berlin")
    cached_days = {timestamp.date().isoformat() for timestamp in cached.index.normalize()}

    assert cached_days == {"2026-03-20", "2026-03-21"}
