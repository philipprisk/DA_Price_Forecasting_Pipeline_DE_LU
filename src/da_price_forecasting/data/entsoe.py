from __future__ import annotations

import os

import pandas as pd
from entsoe import EntsoePandasClient, EntsoeRawClient
from entsoe.parsers import parse_prices


def _require_api_key(env_var: str) -> str:
    api_key = os.getenv(env_var)
    if api_key is None:
        raise ValueError(f"Environment variable '{env_var}' is not set.")
    return api_key


def _expand_hourly_series_to_quarter_hour(series: pd.Series, target_tz: str, value_name: str) -> pd.DataFrame:
    if series.empty:
        raise ValueError(f"No data returned for '{value_name}'.")

    full_index = pd.date_range(
        start=series.index.min().normalize(),
        end=series.index.max().normalize() + pd.Timedelta(days=1) - pd.Timedelta(minutes=15),
        freq="15min",
        tz=target_tz,
    )
    series_15 = series.reindex(full_index).ffill(limit=3)

    expected_counts = pd.Series(1, index=full_index).groupby(full_index.normalize()).transform("count")
    actual_counts = series_15.groupby(series_15.index.normalize()).transform("count")
    series_15 = series_15.where(actual_counts >= expected_counts)

    return series_15.to_frame(name=value_name).rename_axis("timestamp").sort_index()


def _restrict_calendar_window(df: pd.DataFrame, start_day: pd.Timestamp, end_day: pd.Timestamp, target_tz: str) -> pd.DataFrame:
    start_cut = start_day.tz_convert(target_tz).normalize()
    end_cut = end_day.tz_convert(target_tz).normalize() + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    return df.loc[start_cut:end_cut]


def _combine_parsed_price_series(parsed) -> pd.Series | None:
    if not isinstance(parsed, dict):
        return parsed

    series_parts = [series for series in parsed.values() if series is not None and len(series) > 0]
    if not series_parts:
        return None
    return pd.concat(series_parts).sort_index()


def fetch_prices(
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    country_code: str = "DE_LU",
    api_key_env: str = "ENTSOE_API_KEY",
    target_tz: str = "Europe/Berlin",
) -> pd.DataFrame:
    """Fetch DE-LU SDAC day-ahead prices as a 15-minute Europe/Berlin series."""
    client = EntsoePandasClient(api_key=_require_api_key(api_key_env))

    query_start = start_day.tz_convert(target_tz)
    query_end = (end_day + pd.Timedelta(days=1)).tz_convert(target_tz)

    raw_series = client.query_day_ahead_prices(country_code, start=query_start, end=query_end)
    if raw_series.empty:
        raise ValueError("No day-ahead price data returned by the ENTSO-E API.")

    raw_series = raw_series.tz_convert(target_tz).rename("price_da")
    df_prices_15 = _expand_hourly_series_to_quarter_hour(raw_series, target_tz=target_tz, value_name="price_da")
    return _restrict_calendar_window(df_prices_15, start_day, end_day, target_tz)


def fetch_prices_exaa(
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    country_code: str = "DE_LU",
    api_key_env: str = "ENTSOE_API_KEY",
    target_tz: str = "Europe/Berlin",
    chunk_days: int = 90,
) -> pd.DataFrame:
    """Fetch DE-LU EXAA day-ahead prices as a 15-minute Europe/Berlin series."""
    client = EntsoeRawClient(api_key=_require_api_key(api_key_env))

    all_series = []
    current_start = start_day.normalize()

    while current_start <= end_day:
        current_end = min(current_start + pd.Timedelta(days=chunk_days - 1), end_day)

        query_start = current_start.tz_convert(target_tz)
        query_end = (current_end + pd.Timedelta(days=1)).tz_convert(target_tz)

        xml = client.query_day_ahead_prices(
            country_code,
            start=query_start,
            end=query_end,
            sequence=2,
        )
        parsed = parse_prices(xml)

        chunk_series = _combine_parsed_price_series(parsed)

        if chunk_series is None or len(chunk_series) == 0:
            current_start = current_end + pd.Timedelta(days=1)
            continue

        if getattr(chunk_series.index, "tz", None) is None:
            chunk_series = chunk_series.tz_localize("UTC")

        chunk_series = chunk_series.tz_convert(target_tz).rename("price_exaa")
        all_series.append(chunk_series)
        current_start = current_end + pd.Timedelta(days=1)

    series = pd.concat(all_series).sort_index()
    series = series[~series.index.duplicated(keep="last")]

    df_prices_exaa_15 = _expand_hourly_series_to_quarter_hour(series, target_tz=target_tz, value_name="price_exaa")
    return _restrict_calendar_window(df_prices_exaa_15, start_day, end_day, target_tz)


def fetch_load_forecast(
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    country_code: str = "DE_LU",
    api_key_env: str = "ENTSOE_API_KEY",
    target_tz: str = "Europe/Berlin",
) -> pd.DataFrame:
    """Fetch DE-LU day-ahead load forecast as a 15-minute Europe/Berlin series."""
    client = EntsoePandasClient(api_key=_require_api_key(api_key_env))

    query_start = start_day.tz_convert(target_tz)
    query_end = (end_day + pd.Timedelta(days=1)).tz_convert(target_tz)

    obj = client.query_load_forecast(country_code, start=query_start, end=query_end)

    if isinstance(obj, pd.DataFrame):
        numeric_cols = obj.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            raise ValueError(f"No numeric columns found in load forecast response: {obj.columns.tolist()}")
        series = obj[numeric_cols[0]].copy()
    else:
        series = obj.copy()

    series = series.tz_convert(target_tz)
    series.name = "load_fc"
    df_load_forecast_15 = _expand_hourly_series_to_quarter_hour(series, target_tz=target_tz, value_name="load_fc")
    return _restrict_calendar_window(df_load_forecast_15, start_day, end_day, target_tz)
