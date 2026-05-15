from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..config.features import CovariateConfig
from ..paths import find_repo_root, resolve_path


def merge_timestamp_covariates(
    frames: list[pd.DataFrame],
    index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    clean_frames = [frame.sort_index() for frame in frames if frame is not None and not frame.empty]
    if not clean_frames:
        return pd.DataFrame(index=index).rename_axis("timestamp") if index is not None else pd.DataFrame()

    covariates = pd.concat(clean_frames, axis=1).sort_index()
    covariates = covariates.loc[~covariates.index.duplicated(keep="first")]
    covariates = covariates.loc[:, ~covariates.columns.duplicated()]
    if index is not None:
        covariates = covariates.reindex(index)
    covariates.index.name = "timestamp"
    return covariates


def _require_entsoe_fetchers():
    try:
        from ..data.entsoe import (
            fetch_foreign_day_ahead_prices,
            fetch_generation_unavailability,
            fetch_load_forecast,
            fetch_ntc_data,
            fetch_prices_exaa,
        )
    except ImportError as exc:
        raise ImportError(
            "ENTSO-E covariates require entsoe-py. Install the environment that contains "
            "`entsoe-py` or disable those covariates."
        ) from exc

    return (
        fetch_prices_exaa,
        fetch_load_forecast,
        fetch_ntc_data,
        fetch_generation_unavailability,
        fetch_foreign_day_ahead_prices,
    )


def _fetch_commodity_covariates(
    covariate_config: CovariateConfig,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    target_tz: str,
) -> pd.DataFrame:
    try:
        from ..data.commodities import fetch_investiny_commodity_prices, fetch_yfinance_commodity_prices
    except ImportError as exc:
        raise ImportError("Commodity covariates require the configured commodity data fetcher.") from exc

    commodity_config = covariate_config.commodities
    instruments = [instrument.model_dump(exclude_none=True) for instrument in commodity_config.instruments]
    if commodity_config.provider == "yfinance":
        return fetch_yfinance_commodity_prices(
            start_day=start_day,
            end_day=end_day,
            instruments=instruments,
            lag_days=commodity_config.lag_days,
            lookback_days=commodity_config.lookback_days,
            target_tz=target_tz,
            auto_adjust=commodity_config.auto_adjust,
            progress=commodity_config.progress,
        )
    if commodity_config.provider == "investiny":
        return fetch_investiny_commodity_prices(
            start_day=start_day,
            end_day=end_day,
            instruments=instruments,
            lag_days=commodity_config.lag_days,
            lookback_days=commodity_config.lookback_days,
            target_tz=target_tz,
            request_timeout_seconds=commodity_config.request_timeout_seconds,
        )
    raise ValueError(f"Unsupported commodity provider: {commodity_config.provider!r}.")


def _load_renewable_proxy_covariates(
    covariate_config: CovariateConfig,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    target_tz: str,
) -> pd.DataFrame:
    repo_root = find_repo_root()
    path = resolve_path(Path(covariate_config.renewable_proxy_file), repo_root)
    if not path.exists():
        raise FileNotFoundError(
            f"Renewable proxy file not found: {path}. "
            "Run the `renewable_proxy` preprocessing config first."
        )

    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(target_tz)
    df = df.sort_index()
    df = df.loc[~df.index.duplicated(keep="last")]

    start_cut = start_day.tz_convert(target_tz).normalize()
    end_cut = end_day.tz_convert(target_tz).normalize() + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    return df.loc[start_cut:end_cut].rename_axis("timestamp").astype(float)


def _build_foreign_price_spreads(foreign_prices: pd.DataFrame, exaa_prices: pd.DataFrame) -> pd.DataFrame:
    if foreign_prices.empty or exaa_prices.empty:
        return pd.DataFrame()

    aligned = foreign_prices.join(exaa_prices[["price_exaa"]], how="inner")
    spread_columns = {}
    for column in foreign_prices.columns:
        if not column.startswith("price_da_"):
            continue
        market = column.removeprefix("price_da_")
        spread_columns[f"price_spread_{market}_minus_exaa"] = aligned[column] - aligned["price_exaa"]

    return pd.DataFrame(spread_columns, index=aligned.index).rename_axis("timestamp")


def build_timestamp_covariates(
    covariate_config: CovariateConfig,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    *,
    country_code_entsoe: str = "DE_LU",
    entsoe_api_key_env: str = "ENTSOE_API_KEY",
    target_tz: str = "Europe/Berlin",
) -> pd.DataFrame:
    """Fetch selected covariates on the shared timestamp grid."""
    frames: list[pd.DataFrame] = []
    selected_covariates = covariate_config.covariates

    if any(
        name in selected_covariates
        for name in (
            "exaa",
            "load_forecast",
            "ntc",
            "generation_unavailability",
            "foreign_day_ahead_prices",
            "foreign_day_ahead_price_spreads",
        )
    ):
        (
            fetch_prices_exaa,
            fetch_load_forecast,
            fetch_ntc_data,
            fetch_generation_unavailability,
            fetch_foreign_day_ahead_prices,
        ) = _require_entsoe_fetchers()

        if "exaa" in selected_covariates:
            frames.append(
                fetch_prices_exaa(
                    start_day=start_day,
                    end_day=end_day,
                    country_code=country_code_entsoe,
                    api_key_env=entsoe_api_key_env,
                    target_tz=target_tz,
                )
            )
        if "load_forecast" in selected_covariates:
            frames.append(
                fetch_load_forecast(
                    start_day=start_day,
                    end_day=end_day,
                    country_code=country_code_entsoe,
                    api_key_env=entsoe_api_key_env,
                    target_tz=target_tz,
                )
            )
        if "ntc" in selected_covariates:
            frames.append(
                fetch_ntc_data(
                    start_day=start_day,
                    end_day=end_day,
                    neighbors=covariate_config.ntc_neighbors,
                    country_code=country_code_entsoe,
                    api_key_env=entsoe_api_key_env,
                    target_tz=target_tz,
                )
            )
        if "generation_unavailability" in selected_covariates:
            frames.append(
                fetch_generation_unavailability(
                    start_day=start_day,
                    end_day=end_day,
                    country_code=country_code_entsoe,
                    api_key_env=entsoe_api_key_env,
                    target_tz=target_tz,
                )
            )
        if "foreign_day_ahead_prices" in selected_covariates:
            frames.append(
                fetch_foreign_day_ahead_prices(
                    start_day=start_day,
                    end_day=end_day,
                    markets=covariate_config.foreign_price_markets,
                    api_key_env=entsoe_api_key_env,
                    target_tz=target_tz,
                )
            )
        if "foreign_day_ahead_price_spreads" in selected_covariates:
            foreign_prices = fetch_foreign_day_ahead_prices(
                start_day=start_day,
                end_day=end_day,
                markets=covariate_config.foreign_price_markets,
                api_key_env=entsoe_api_key_env,
                target_tz=target_tz,
            )
            exaa_prices = fetch_prices_exaa(
                start_day=start_day,
                end_day=end_day,
                country_code=country_code_entsoe,
                api_key_env=entsoe_api_key_env,
                target_tz=target_tz,
            )
            frames.append(_build_foreign_price_spreads(foreign_prices, exaa_prices))

    if "commodities" in selected_covariates:
        frames.append(_fetch_commodity_covariates(covariate_config, start_day, end_day, target_tz))

    if "renewable_generation_proxy" in selected_covariates:
        frames.append(_load_renewable_proxy_covariates(covariate_config, start_day, end_day, target_tz))

    covariates = merge_timestamp_covariates(frames)
    if {"load_forecast", "renewable_generation_proxy"}.issubset(set(selected_covariates)):
        if {"load_fc", "Renewable_Total_Proxy_MW"}.issubset(covariates.columns):
            covariates["Residual_Load_Proxy_MW"] = covariates["load_fc"] - covariates["Renewable_Total_Proxy_MW"]
            covariates["Residual_Load_Proxy_MW_ramp_1h"] = covariates["Residual_Load_Proxy_MW"].diff(4)
            covariates["Residual_Load_Proxy_MW_ramp_3h"] = covariates["Residual_Load_Proxy_MW"].diff(12)

    return covariates


def build_daily_scalar_covariate_features(
    covariates: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    *,
    columns: list[str] | None = None,
    prefix: str = "",
) -> pd.DataFrame:
    """Convert timestamp covariates into one scalar feature per local day."""
    if covariates.empty:
        return pd.DataFrame(index=daily_index).rename_axis("date")

    selected_columns = columns or list(covariates.columns)
    selected_columns = [column for column in selected_columns if column in covariates.columns]
    if not selected_columns:
        return pd.DataFrame(index=daily_index).rename_axis("date")

    frame = covariates[selected_columns].copy().sort_index()
    frame["date_local"] = frame.index.floor("D")
    daily = frame.groupby("date_local")[selected_columns].last()
    daily = daily.reindex(daily_index)
    if prefix:
        daily = daily.rename(columns={column: f"{prefix}{column}" for column in daily.columns})
    daily.index.name = "date"
    return daily.astype(float)


def build_daily_summary_covariate_features(
    covariates: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    *,
    columns: list[str] | None = None,
    prefix: str = "",
) -> pd.DataFrame:
    """Convert timestamp covariates into compact daily scarcity-style summaries."""
    if covariates.empty:
        return pd.DataFrame(index=daily_index).rename_axis("date")

    selected_columns = columns or list(covariates.columns)
    selected_columns = [column for column in selected_columns if column in covariates.columns]
    if not selected_columns:
        return pd.DataFrame(index=daily_index).rename_axis("date")

    frame = covariates[selected_columns].copy().sort_index()
    frame["date_local"] = frame.index.floor("D")
    frame["mtu"] = frame.index.hour * 4 + frame.index.minute // 15

    feature_blocks = []
    for column in selected_columns:
        grouped = frame.groupby("date_local")[column]
        daily = pd.DataFrame(index=daily_index)
        daily[f"{prefix}{column}_mean"] = grouped.mean()
        daily[f"{prefix}{column}_min"] = grouped.min()
        daily[f"{prefix}{column}_max"] = grouped.max()
        daily[f"{prefix}{column}_range"] = daily[f"{prefix}{column}_max"] - daily[f"{prefix}{column}_min"]
        daily[f"{prefix}{column}_std"] = grouped.std()

        ramps = frame.groupby("date_local")[column].apply(lambda values: values.diff(4).abs().max())
        daily[f"{prefix}{column}_ramp_1h_abs_max"] = ramps

        morning = frame.loc[frame["mtu"].between(28, 43)].groupby("date_local")[column]
        evening = frame.loc[frame["mtu"].between(60, 75)].groupby("date_local")[column]
        daily[f"{prefix}{column}_morning_mean"] = morning.mean()
        daily[f"{prefix}{column}_morning_max"] = morning.max()
        daily[f"{prefix}{column}_evening_mean"] = evening.mean()
        daily[f"{prefix}{column}_evening_max"] = evening.max()

        feature_blocks.append(daily)

    features = pd.concat(feature_blocks, axis=1, sort=False).reindex(daily_index)
    features.index.name = "date"
    return features.astype(float)


def build_daily_vector_covariate_features(
    covariates: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    *,
    columns: list[str] | None = None,
    prefix: str = "",
) -> pd.DataFrame:
    """Convert timestamp covariates into one 96-MTU feature vector per local day."""
    if covariates.empty:
        return pd.DataFrame(index=daily_index).rename_axis("date")

    selected_columns = columns or list(covariates.columns)
    selected_columns = [column for column in selected_columns if column in covariates.columns]
    if not selected_columns:
        return pd.DataFrame(index=daily_index).rename_axis("date")

    frame = covariates[selected_columns].copy().sort_index()
    frame["date_local"] = frame.index.floor("D")
    frame["mtu"] = frame.index.hour * 4 + frame.index.minute // 15

    feature_blocks = []
    for column in selected_columns:
        daily_matrix = (
            frame.pivot_table(
                index="date_local",
                columns="mtu",
                values=column,
                aggfunc="mean",
            )
            .reindex(index=daily_index, columns=range(96))
            .interpolate(axis=1, limit=4, limit_area="inside")
        )
        daily_matrix.columns = [f"{prefix}{column}_mtu_{int(mtu):02d}" for mtu in daily_matrix.columns]
        feature_blocks.append(daily_matrix)

    daily = pd.concat(feature_blocks, axis=1, sort=False)
    daily.index.name = "date"
    return daily.astype(float)
