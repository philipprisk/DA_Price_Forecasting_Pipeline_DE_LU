from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import (
    EntsoeRenewableForecastBenchmarkConfig,
    RenewableGenerationModelConfig,
    RenewableGenerationPostprocessConfig,
)
from ..data.entsoe import (
    fetch_actual_renewable_generation,
    fetch_actual_solar_generation_by_control_area,
    fetch_generation_unavailability,
    fetch_renewable_generation_forecast,
)
from ..data.weather import load_dwd
from .common import (
    as_local_day as _as_local_day,
    load_timestamp_csv as _load_timestamp_csv,
    point_error_stats,
    save_timestamp_csv as _save_timestamp_csv,
)


def _training_target_cutoff(day: pd.Timestamp, config: RenewableGenerationModelConfig) -> pd.Timestamp:
    cutoff_day = day - pd.Timedelta(days=config.target_availability_lag_days)
    if config.target_availability_cutoff_hour is None:
        return cutoff_day - pd.Timedelta(minutes=15)
    return cutoff_day + pd.Timedelta(
        hours=config.target_availability_cutoff_hour,
        minutes=config.target_availability_cutoff_minute,
    )


def _load_renewable_proxy(config: RenewableGenerationModelConfig) -> pd.DataFrame:
    if not config.renewable_proxy_file.exists():
        raise FileNotFoundError(
            f"Renewable proxy file not found: {config.renewable_proxy_file}. "
            "Run the regional renewable feature preprocessing step first."
        )

    proxy = _load_timestamp_csv(config.renewable_proxy_file, config.target_tz)
    if config.renewable_proxy_fallback_file is not None:
        if not config.renewable_proxy_fallback_file.exists():
            raise FileNotFoundError(
                f"Renewable proxy fallback file not found: {config.renewable_proxy_fallback_file}. "
                "Run the fallback regional renewable feature preprocessing step first."
            )

        fallback = _load_timestamp_csv(config.renewable_proxy_fallback_file, config.target_tz)
        if config.renewable_proxy_fallback_end_date is not None:
            fallback_end = _as_local_day(config.renewable_proxy_fallback_end_date, config.target_tz)
            fallback = fallback.loc[fallback.index < fallback_end + pd.Timedelta(days=1)]
        proxy = proxy.combine_first(fallback).sort_index()

    prefixes = config.extra_renewable_proxy_prefixes or [f"extra_{idx + 1}_" for idx in range(len(config.extra_renewable_proxy_files))]
    for extra_file, prefix in zip(config.extra_renewable_proxy_files, prefixes, strict=True):
        if not extra_file.exists():
            raise FileNotFoundError(
                f"Extra renewable proxy file not found: {extra_file}. "
                "Run the corresponding regional renewable feature preprocessing step first."
            )
        extra = _load_timestamp_csv(extra_file, config.target_tz)
        if prefix:
            extra = extra.add_prefix(prefix)
        extra = extra.reindex(proxy.index)
        proxy = proxy.join(extra, how="left")

    proxy = proxy.loc[:, ~proxy.columns.duplicated()]
    proxy.index.name = "timestamp"
    return proxy


def _load_or_fetch_actual_generation(config: RenewableGenerationModelConfig) -> pd.DataFrame:
    if config.actual_generation_file.exists():
        actual = _load_timestamp_csv(config.actual_generation_file, config.target_tz)
        required_columns = set(config.target_columns)
        if config.actual_generation_lag_days:
            required_columns.update(config.actual_generation_lag_columns)
        if config.include_partial_actual_generation_features:
            partial_columns = config.partial_generation_columns or config.actual_generation_lag_columns
            required_columns.update(partial_columns)
        missing_columns = [column for column in required_columns if column not in actual.columns]
        if not (config.include_solar_control_area_targets and missing_columns):
            return actual

    start = _as_local_day(config.entsoe_start_date, config.target_tz)
    end = _as_local_day(config.entsoe_end_date, config.target_tz)
    df = fetch_actual_renewable_generation(
        start_day=start,
        end_day=end,
        country_code=config.country_code_entsoe,
        api_key_env=config.entsoe_api_key_env,
        target_tz=config.target_tz,
    )
    if config.include_solar_control_area_targets:
        control_area_actual = fetch_actual_solar_generation_by_control_area(
            start_day=start,
            end_day=end,
            control_area_targets=config.solar_control_area_targets,
            api_key_env=config.entsoe_api_key_env,
            target_tz=config.target_tz,
        )
        df = df.join(control_area_actual, how="outer")
    _save_timestamp_csv(df, config.actual_generation_file)
    return df


def _load_or_fetch_benchmark_actual_generation(config: EntsoeRenewableForecastBenchmarkConfig) -> pd.DataFrame:
    if config.actual_generation_file.exists():
        return _load_timestamp_csv(config.actual_generation_file, config.target_tz)

    start = _as_local_day(config.entsoe_start_date, config.target_tz)
    end = _as_local_day(config.entsoe_end_date, config.target_tz)
    df = fetch_actual_renewable_generation(
        start_day=start,
        end_day=end,
        country_code=config.country_code_entsoe,
        api_key_env=config.entsoe_api_key_env,
        target_tz=config.target_tz,
        chunk_days=config.chunk_days,
    )
    _save_timestamp_csv(df, config.actual_generation_file)
    return df


def _load_or_fetch_entsoe_renewable_forecast(config: EntsoeRenewableForecastBenchmarkConfig) -> pd.DataFrame:
    if config.forecast_file.exists():
        return _load_timestamp_csv(config.forecast_file, config.target_tz)

    start = _as_local_day(config.entsoe_start_date, config.target_tz)
    end = _as_local_day(config.entsoe_end_date, config.target_tz)
    df = fetch_renewable_generation_forecast(
        start_day=start,
        end_day=end,
        country_code=config.country_code_entsoe,
        api_key_env=config.entsoe_api_key_env,
        target_tz=config.target_tz,
        chunk_days=config.chunk_days,
        process_type=config.process_type,
    )
    _save_timestamp_csv(df, config.forecast_file)
    return df


def _load_or_fetch_unavailability(config: RenewableGenerationModelConfig) -> pd.DataFrame:
    if config.unavailability_file.exists():
        return _load_timestamp_csv(config.unavailability_file, config.target_tz)

    start = _as_local_day(config.entsoe_start_date, config.target_tz)
    end = _as_local_day(config.entsoe_end_date, config.target_tz)
    df = fetch_generation_unavailability(
        start_day=start,
        end_day=end,
        country_code=config.country_code_entsoe,
        api_key_env=config.entsoe_api_key_env,
        target_tz=config.target_tz,
        feature_mode=config.unavailability_feature_mode,
        planned_only=config.unavailability_planned_only,
    )
    _save_timestamp_csv(df, config.unavailability_file)
    return df


def _build_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    minute_of_day = index.hour * 60 + index.minute
    day_of_week = index.dayofweek
    day_of_year = index.dayofyear
    mtu = index.hour * 4 + index.minute // 15

    features = pd.DataFrame(index=index)
    features["mtu"] = mtu
    features["tod_sin"] = np.sin(2 * np.pi * minute_of_day / (24 * 60))
    features["tod_cos"] = np.cos(2 * np.pi * minute_of_day / (24 * 60))
    features["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    features["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    features["doy_sin"] = np.sin(2 * np.pi * day_of_year / 366)
    features["doy_cos"] = np.cos(2 * np.pi * day_of_year / 366)
    features["is_weekend"] = (day_of_week >= 5).astype(int)
    return features


def _region_proxy_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    return [
        column
        for column in df.columns
        if column.startswith(prefix) and column.endswith("_proxy_mw") and "_cluster_" not in column
    ]


def _safe_column(df: pd.DataFrame, column: str) -> pd.Series | None:
    return df[column] if column in df.columns else None


def _series_or_zero(index: pd.DatetimeIndex, series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(0.0, index=index)
    return series.astype(float)


def _add_summary_stats(
    source: pd.DataFrame,
    output: dict[str, pd.Series],
    *,
    name: str,
    columns: list[str],
) -> None:
    if not columns:
        return
    values = source[columns].astype(float)
    output[f"{name}_sum"] = values.sum(axis=1)
    output[f"{name}_max"] = values.max(axis=1)
    output[f"{name}_min"] = values.min(axis=1)
    output[f"{name}_spread"] = values.max(axis=1) - values.min(axis=1)
    output[f"{name}_std"] = values.std(axis=1).fillna(0.0)
    output[f"{name}_top2_sum"] = values.apply(lambda row: row.nlargest(min(2, len(row))).sum(), axis=1)


def _build_regional_summary_features(proxy: pd.DataFrame) -> pd.DataFrame:
    """Derive regional spread/gradient features from existing renewable proxy columns."""
    output: dict[str, pd.Series] = {}
    index = proxy.index

    wind_cols = _region_proxy_columns(proxy, "wind_")
    onshore_cols = _region_proxy_columns(proxy, "wind_onshore_")
    offshore_cols = _region_proxy_columns(proxy, "wind_offshore_")
    solar_cols = _region_proxy_columns(proxy, "solar_")

    _add_summary_stats(proxy, output, name="wind_region_proxy", columns=wind_cols)
    _add_summary_stats(proxy, output, name="wind_onshore_region_proxy", columns=onshore_cols)
    _add_summary_stats(proxy, output, name="wind_offshore_region_proxy", columns=offshore_cols)
    _add_summary_stats(proxy, output, name="solar_region_proxy", columns=solar_cols)

    output["wind_offshore_minus_onshore_proxy"] = (
        _series_or_zero(index, proxy[offshore_cols].sum(axis=1) if offshore_cols else None)
        - _series_or_zero(index, proxy[onshore_cols].sum(axis=1) if onshore_cols else None)
    )
    for prefix, name in [("wind_onshore", "wind_onshore"), ("solar", "solar")]:
        north = _safe_column(proxy, f"{prefix}_north_proxy_mw")
        south = _safe_column(proxy, f"{prefix}_south_proxy_mw")
        east = _safe_column(proxy, f"{prefix}_east_proxy_mw")
        west = _safe_column(proxy, f"{prefix}_west_proxy_mw")
        central = _safe_column(proxy, f"{prefix}_central_proxy_mw")
        output[f"{name}_north_minus_south_proxy"] = _series_or_zero(index, north) - _series_or_zero(index, south)
        output[f"{name}_east_minus_west_proxy"] = _series_or_zero(index, east) - _series_or_zero(index, west)
        output[f"{name}_north_minus_central_proxy"] = _series_or_zero(index, north) - _series_or_zero(index, central)

    speed_cols = [column for column in proxy.columns if column.startswith("wind_") and "speed_hub" in column]
    _add_summary_stats(proxy, output, name="wind_region_speed_hub", columns=speed_cols)

    irradiance_cols = [
        column
        for column in proxy.columns
        if column.startswith("solar_") and column.endswith("irradiance_cap_weighted_W_m2")
    ]
    _add_summary_stats(proxy, output, name="solar_region_irradiance", columns=irradiance_cols)

    summary = pd.DataFrame(output, index=index)
    ramp_cols = [column for column in summary.columns if column.endswith("_proxy") or column.endswith("_sum")]
    for step, suffix in [(4, "ramp_1h"), (12, "ramp_3h")]:
        for column in ramp_cols:
            summary[f"{column}_{suffix}"] = summary[column].diff(step)

    summary.index.name = "timestamp"
    return summary


def _build_forecast_lead_features(
    index: pd.DatetimeIndex,
    config: RenewableGenerationModelConfig,
) -> pd.DataFrame:
    """Build lead-time features for operational single-run weather forecasts."""
    run_hour_text = (
        config.forecast_run_hour_utc + ":00"
        if config.forecast_run_hour_utc.count(":") == 1
        else config.forecast_run_hour_utc
    )
    run_hour = pd.to_timedelta(run_hour_text)
    delivery_days = index.tz_localize(None).normalize()
    run_times_local = (
        delivery_days.tz_localize("UTC")
        - pd.Timedelta(days=config.forecast_run_day_offset)
        + run_hour
    ).tz_convert(config.target_tz)
    lead_hours = (index - run_times_local) / pd.Timedelta(hours=1)

    features = pd.DataFrame(index=index)
    features["forecast_lead_hours"] = lead_hours.astype(float)
    features["forecast_lead_days"] = features["forecast_lead_hours"] / 24.0
    features["forecast_lead_sin"] = np.sin(2 * np.pi * features["forecast_lead_hours"] / 24.0)
    features["forecast_lead_cos"] = np.cos(2 * np.pi * features["forecast_lead_hours"] / 24.0)
    return features


_SOLAR_GEOMETRY_POINTS = {
    "central": (51.0, 10.0),
    "east": (51.5, 13.0),
    "north": (53.5, 9.5),
    "south": (48.5, 11.5),
    "west": (51.0, 7.0),
}


def _solar_position_features(index: pd.DatetimeIndex, *, latitude: float, longitude: float) -> pd.DataFrame:
    """Approximate solar position features from NOAA-style equations."""
    output_index = index
    utc = pd.to_datetime(pd.Index(index), utc=True)
    minutes_utc = (
        utc.hour.to_numpy(dtype=float) * 60.0
        + utc.minute.to_numpy(dtype=float)
        + utc.second.to_numpy(dtype=float) / 60.0
    )
    hour_utc = minutes_utc / 60.0
    day_of_year = utc.dayofyear.to_numpy(dtype=float)

    gamma = 2.0 * np.pi / 365.0 * (day_of_year - 1.0 + (hour_utc - 12.0) / 24.0)
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )
    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.00148 * np.sin(3.0 * gamma)
    )

    true_solar_time = (minutes_utc + equation_of_time + 4.0 * longitude) % 1440.0
    hour_angle = np.deg2rad(true_solar_time / 4.0 - 180.0)
    lat_rad = np.deg2rad(latitude)

    cos_zenith = (
        np.sin(lat_rad) * np.sin(declination)
        + np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle)
    )
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    positive_cos_zenith = np.clip(cos_zenith, 0.0, None)
    elevation_deg = np.rad2deg(np.arcsin(cos_zenith))

    return pd.DataFrame(
        {
            "cos_zenith": cos_zenith,
            "positive_cos_zenith": positive_cos_zenith,
            "elevation_deg": elevation_deg,
            "clear_sky_proxy": positive_cos_zenith**1.25,
            "is_daylight": (positive_cos_zenith > 0.0).astype(float),
        },
        index=output_index,
    )


def _build_solar_geometry_features(proxy: pd.DataFrame) -> pd.DataFrame:
    """Build deterministic solar-geometry and clear-sky-index features."""
    blocks: list[pd.DataFrame] = []
    clear_sky_columns = []
    for region, (latitude, longitude) in _SOLAR_GEOMETRY_POINTS.items():
        geometry = _solar_position_features(proxy.index, latitude=latitude, longitude=longitude)
        geometry = geometry.add_prefix(f"solar_geom_{region}_")

        irradiance_col = f"solar_{region}_irradiance_cap_weighted_W_m2"
        clear_sky_col = f"solar_geom_{region}_clear_sky_proxy"
        if irradiance_col in proxy.columns:
            denominator = 1000.0 * geometry[clear_sky_col].to_numpy(dtype=float)
            clear_sky_index = np.divide(
                proxy[irradiance_col].to_numpy(dtype=float),
                denominator,
                out=np.zeros(len(proxy), dtype=float),
                where=denominator > 1e-6,
            )
            geometry[f"solar_geom_{region}_clear_sky_index"] = np.clip(clear_sky_index, 0.0, 2.0)

        blocks.append(geometry)
        clear_sky_columns.append(clear_sky_col)

    result = pd.concat(blocks, axis=1)
    result["solar_geom_clear_sky_proxy_mean"] = result[clear_sky_columns].mean(axis=1)
    result["solar_geom_clear_sky_proxy_max"] = result[clear_sky_columns].max(axis=1)
    result["solar_geom_clear_sky_proxy_spread"] = (
        result["solar_geom_clear_sky_proxy_max"] - result[clear_sky_columns].min(axis=1)
    )
    result.index.name = "timestamp"
    return result


def _build_solar_physics_features(proxy: pd.DataFrame) -> pd.DataFrame:
    """Build compact PV-oriented features from regional solar weather proxies."""
    frames: dict[str, pd.Series] = {}
    summary_columns: dict[str, list[str]] = {
        "direct_proxy": [],
        "diffuse_proxy": [],
        "temperature_corrected_proxy": [],
        "clear_sky_index": [],
    }

    for region, (latitude, longitude) in _SOLAR_GEOMETRY_POINTS.items():
        prefix = f"solar_{region}"
        proxy_col = f"{prefix}_proxy_mw"
        irradiance_col = f"{prefix}_irradiance_cap_weighted_W_m2"
        direct_col = f"{prefix}_direct_irradiance_cap_weighted_W_m2"
        t2m_col = f"{prefix}_t2m_cap_weighted_K"

        if proxy_col not in proxy.columns or irradiance_col not in proxy.columns:
            continue

        proxy_mw = proxy[proxy_col].astype(float)
        irradiance = proxy[irradiance_col].astype(float).clip(lower=0.0)
        if direct_col in proxy.columns:
            direct = proxy[direct_col].astype(float).clip(lower=0.0)
        else:
            direct = pd.Series(0.0, index=proxy.index)
        diffuse = (irradiance - direct).clip(lower=0.0)
        direct_share = pd.Series(
            np.divide(
                direct.to_numpy(dtype=float),
                irradiance.to_numpy(dtype=float),
                out=np.zeros(len(proxy), dtype=float),
                where=irradiance.to_numpy(dtype=float) > 1e-6,
            ),
            index=proxy.index,
        ).clip(lower=0.0, upper=1.0)
        diffuse_share = (1.0 - direct_share).clip(lower=0.0, upper=1.0)

        frames[f"solar_phys_{region}_direct_share"] = direct_share
        frames[f"solar_phys_{region}_diffuse_irradiance_W_m2"] = diffuse
        frames[f"solar_phys_{region}_direct_proxy_mw"] = proxy_mw * direct_share
        frames[f"solar_phys_{region}_diffuse_proxy_mw"] = proxy_mw * diffuse_share
        summary_columns["direct_proxy"].append(f"solar_phys_{region}_direct_proxy_mw")
        summary_columns["diffuse_proxy"].append(f"solar_phys_{region}_diffuse_proxy_mw")

        geometry = _solar_position_features(proxy.index, latitude=latitude, longitude=longitude)
        clear_sky = 1000.0 * geometry["clear_sky_proxy"].astype(float)
        clear_sky_index = pd.Series(
            np.divide(
                irradiance.to_numpy(dtype=float),
                clear_sky.to_numpy(dtype=float),
                out=np.zeros(len(proxy), dtype=float),
                where=clear_sky.to_numpy(dtype=float) > 1e-6,
            ),
            index=proxy.index,
        ).clip(lower=0.0, upper=2.0)
        frames[f"solar_phys_{region}_clear_sky_index"] = clear_sky_index
        frames[f"solar_phys_{region}_cloud_attenuation_proxy"] = (1.0 - clear_sky_index).clip(lower=-1.0, upper=1.0)
        summary_columns["clear_sky_index"].append(f"solar_phys_{region}_clear_sky_index")

        if t2m_col in proxy.columns:
            t2m_c = proxy[t2m_col].astype(float) - 273.15
            module_temp_c = t2m_c + 0.025 * irradiance
            efficiency_factor = (1.0 - 0.004 * (module_temp_c - 25.0)).clip(lower=0.75, upper=1.15)
            temp_corrected_proxy = proxy_mw * efficiency_factor
            frames[f"solar_phys_{region}_module_temp_proxy_C"] = module_temp_c
            frames[f"solar_phys_{region}_temperature_efficiency_factor"] = efficiency_factor
            frames[f"solar_phys_{region}_temperature_corrected_proxy_mw"] = temp_corrected_proxy
            frames[f"solar_phys_{region}_hot_irradiance_interaction"] = (module_temp_c - 25.0).clip(lower=0.0) * irradiance
            summary_columns["temperature_corrected_proxy"].append(
                f"solar_phys_{region}_temperature_corrected_proxy_mw"
            )

    if not frames:
        return pd.DataFrame(index=proxy.index)

    result = pd.DataFrame(frames, index=proxy.index)
    for name, columns in summary_columns.items():
        available = [column for column in columns if column in result.columns]
        if not available:
            continue
        result[f"solar_phys_{name}_sum"] = result[available].sum(axis=1)
        result[f"solar_phys_{name}_mean"] = result[available].mean(axis=1)
        result[f"solar_phys_{name}_spread"] = result[available].max(axis=1) - result[available].min(axis=1)

    result.index.name = "timestamp"
    return result


def _build_solar_physics_proxy_baselines(
    proxy: pd.DataFrame,
    config: RenewableGenerationModelConfig,
) -> pd.DataFrame:
    """Build temperature-corrected physical PV baseline columns.

    The raw regional renewable proxy is already capacity weighted. This helper
    keeps that structure and applies a simple module-temperature efficiency
    correction so the model can learn a residual around a physics-informed
    baseline rather than the full solar generation level.
    """
    frames: dict[str, pd.Series] = {}
    proxy_suffix = "_proxy_mw"
    proxy_columns = [
        column
        for column in proxy.columns
        if column.startswith("solar_")
        and column.endswith(proxy_suffix)
        and "_physics_" not in column
    ]
    if not proxy_columns:
        return pd.DataFrame(index=proxy.index)

    for proxy_col in proxy_columns:
        region = proxy_col.removeprefix("solar_").removesuffix(proxy_suffix)
        region_prefix = f"solar_{region}"
        base_proxy = proxy[proxy_col].astype(float).clip(lower=0.0)
        irradiance_col = f"{region_prefix}_irradiance_cap_weighted_W_m2"
        t2m_col = f"{region_prefix}_t2m_cap_weighted_K"

        if irradiance_col in proxy.columns and t2m_col in proxy.columns:
            irradiance = proxy[irradiance_col].astype(float).clip(lower=0.0)
            t2m_c = proxy[t2m_col].astype(float) - 273.15
            module_temp_c = (
                t2m_c
                + config.solar_physics_proxy_module_temperature_irradiance_coeff * irradiance
            )
            temperature_factor = (
                1.0
                + config.solar_physics_proxy_temperature_coefficient
                * (module_temp_c - 25.0)
            ).clip(
                lower=config.solar_physics_proxy_min_temperature_factor,
                upper=config.solar_physics_proxy_max_temperature_factor,
            )
        else:
            module_temp_c = pd.Series(np.nan, index=proxy.index)
            temperature_factor = pd.Series(1.0, index=proxy.index)

        frames[f"{region_prefix}_physics_proxy_mw"] = (base_proxy * temperature_factor).clip(lower=0.0)
        frames[f"{region_prefix}_physics_temperature_factor"] = temperature_factor
        frames[f"{region_prefix}_physics_module_temp_C"] = module_temp_c

    result = pd.DataFrame(frames, index=proxy.index)
    tso_physics_columns = [
        f"solar_{region}_physics_proxy_mw"
        for region in ("50hertz", "amprion", "tennet", "transnetbw")
        if f"solar_{region}_physics_proxy_mw" in result.columns
    ]
    if tso_physics_columns:
        result["Renewable_Solar_Physics_Proxy_MW"] = result[tso_physics_columns].sum(axis=1)
        result["Solar_Physics_Proxy_MW"] = result["Renewable_Solar_Physics_Proxy_MW"]

    result.index.name = "timestamp"
    return result


def _build_dwd_cluster_features(config: RenewableGenerationModelConfig) -> pd.DataFrame:
    df_hourly, df_qh = load_dwd(
        icon_dir=config.icon_dir,
        start_folder_date=config.start_folder_date,
        required_run=config.required_run,
        skip_dates=set(config.skip_dates),
        folder_offset_date=config.dwd_folder_offset_date,
        target_tz=config.target_tz,
    )

    hourly_data: dict[str, np.ndarray] = {}
    u_cols = sorted(column for column in df_hourly.columns if column.startswith("u10_cluster_"))
    for u_col in u_cols:
        cluster_id = u_col.rsplit("_", 1)[-1]
        v_col = f"v10_cluster_{cluster_id}"
        if v_col not in df_hourly.columns:
            continue

        speed_10m = np.sqrt(df_hourly[u_col].to_numpy(dtype=float) ** 2 + df_hourly[v_col].to_numpy(dtype=float) ** 2)
        speed_hub = speed_10m * (config.wind_hub_height_m / config.wind_reference_height_m) ** config.wind_shear_alpha
        hourly_data[f"wind_speed_10m_cluster_{cluster_id}"] = speed_10m
        hourly_data[f"wind_speed_hub_cluster_{cluster_id}"] = speed_hub
        hourly_data[f"wind_speed_hub_sq_cluster_{cluster_id}"] = speed_hub**2
        hourly_data[f"wind_speed_hub_cube_cluster_{cluster_id}"] = speed_hub**3

    for column in sorted(column for column in df_hourly.columns if column.startswith("t2m_cluster_")):
        hourly_data[column] = df_hourly[column].to_numpy(dtype=float)

    hourly_features = pd.DataFrame(hourly_data, index=df_hourly.index)

    qh_data: dict[str, np.ndarray] = {}
    dir_cols = sorted(column for column in df_qh.columns if column.startswith("ASWDIR_cluster_"))
    for dir_col in dir_cols:
        cluster_id = dir_col.rsplit("_", 1)[-1]
        dif_col = f"ASWDIFD_cluster_{cluster_id}"
        if dif_col not in df_qh.columns:
            continue

        direct = np.clip(df_qh[dir_col].to_numpy(dtype=float), 0.0, None)
        diffuse = np.clip(df_qh[dif_col].to_numpy(dtype=float), 0.0, None)
        qh_data[f"solar_direct_cluster_{cluster_id}"] = direct
        qh_data[f"solar_diffuse_cluster_{cluster_id}"] = diffuse
        qh_data[f"solar_global_cluster_{cluster_id}"] = direct + diffuse

    qh_features = pd.DataFrame(qh_data, index=df_qh.index)

    full_index = pd.date_range(
        start=min(hourly_features.index.min(), qh_features.index.min()),
        end=max(hourly_features.index.max(), qh_features.index.max()),
        freq="15min",
        tz=config.target_tz,
        name="timestamp",
    )
    hourly_features = hourly_features.loc[~hourly_features.index.duplicated(keep="last")].reindex(full_index).ffill(limit=3)
    qh_features = qh_features.loc[~qh_features.index.duplicated(keep="last")].reindex(full_index)
    features = pd.concat([hourly_features, qh_features], axis=1).sort_index()
    features.index.name = "timestamp"
    return features


def _add_lag_diff_features(
    features: pd.DataFrame,
    *,
    lag_steps: list[int],
    diff_steps: list[int],
) -> pd.DataFrame:
    """Add paper-style lagged and differenced NWP features without changing raw inputs."""
    if features.empty:
        return features

    blocks = [features]
    for step in lag_steps:
        if step <= 0:
            continue
        lagged = features.shift(step).add_suffix(f"_lag_{step}")
        blocks.append(lagged)
    for step in diff_steps:
        if step <= 0:
            continue
        differenced = features.diff(step).add_suffix(f"_diff_{step}")
        blocks.append(differenced)

    return pd.concat(blocks, axis=1)


def _build_actual_generation_lag_features(
    actual: pd.DataFrame,
    config: RenewableGenerationModelConfig,
) -> pd.DataFrame:
    """Build leakage-safe actual generation lags available before day-ahead gate closure."""
    lag_days = sorted({int(day) for day in config.actual_generation_lag_days if int(day) > 0})
    if not lag_days:
        return pd.DataFrame(index=actual.index)

    columns = [column for column in config.actual_generation_lag_columns if column in actual.columns]
    if not columns:
        return pd.DataFrame(index=actual.index)

    blocks = []
    for lag_day in lag_days:
        lagged = actual[columns].copy()
        lagged.index = lagged.index + pd.Timedelta(days=lag_day)
        lagged = lagged.rename(columns={column: f"{column}_lag_d{lag_day}" for column in columns})
        blocks.append(lagged)

    result = pd.concat(blocks, axis=1).sort_index()
    result = result.loc[~result.index.duplicated(keep="last")]
    result.index.name = "timestamp"
    return result


def _partial_generation_feature_label(column: str) -> str:
    return column.removesuffix("_Actual_MW").lower()


def _partial_generation_window_label(end_hour: int, end_minute: int) -> str:
    if end_minute == 0:
        return f"{end_hour:02d}00"
    return f"{end_hour:02d}{end_minute:02d}"


def _build_partial_actual_generation_features(
    actual: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    *,
    columns: list[str],
    reference_day: int,
    comparison_lag_days: int,
    morning_end_hour: int,
    morning_end_minute: int,
) -> pd.DataFrame:
    """Summarise actual generation observed on the forecast creation morning.

    For a forecast day D, source values are taken from D-reference_day between
    00:00 and the configured morning cutoff. This keeps live forecasts aligned
    with the information available before day-ahead gate closure.
    """
    if target_index.empty:
        return pd.DataFrame(index=target_index)

    available_columns = [column for column in columns if column in actual.columns]
    if not available_columns:
        return pd.DataFrame(index=target_index)

    actual = actual.sort_index()
    forecast_days = pd.DatetimeIndex(target_index.normalize().unique()).sort_values()
    end_delta = pd.Timedelta(hours=morning_end_hour, minutes=morning_end_minute)
    window_label = _partial_generation_window_label(morning_end_hour, morning_end_minute)

    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for day in forecast_days:
        source_day = day - pd.Timedelta(days=reference_day)
        comparison_day = source_day - pd.Timedelta(days=comparison_lag_days)
        source = actual.loc[source_day:source_day + end_delta, available_columns]
        comparison = actual.loc[comparison_day:comparison_day + end_delta, available_columns]

        row: dict[str, float] = {}
        for column in available_columns:
            label = _partial_generation_feature_label(column)
            prefix = f"partial_gen_{label}_d{reference_day}_00_{window_label}"

            source_values = source[column].dropna()
            comparison_values = comparison[column].dropna()

            if source_values.empty:
                source_stats = {
                    "mean": np.nan,
                    "max": np.nan,
                    "min": np.nan,
                    "last": np.nan,
                    "sum": np.nan,
                    "range": np.nan,
                    "ramp": np.nan,
                }
            else:
                source_stats = {
                    "mean": float(source_values.mean()),
                    "max": float(source_values.max()),
                    "min": float(source_values.min()),
                    "last": float(source_values.iloc[-1]),
                    "sum": float(source_values.sum()),
                    "range": float(source_values.max() - source_values.min()),
                    "ramp": float(source_values.iloc[-1] - source_values.iloc[0]),
                }

            if comparison_values.empty:
                comparison_stats = {
                    "mean": np.nan,
                    "last": np.nan,
                    "sum": np.nan,
                    "ramp": np.nan,
                }
            else:
                comparison_stats = {
                    "mean": float(comparison_values.mean()),
                    "last": float(comparison_values.iloc[-1]),
                    "sum": float(comparison_values.sum()),
                    "ramp": float(comparison_values.iloc[-1] - comparison_values.iloc[0]),
                }

            for stat, value in source_stats.items():
                row[f"{prefix}_{stat}"] = value
            for stat, comparison_value in comparison_stats.items():
                row[f"{prefix}_{stat}_diff_d{comparison_lag_days}"] = (
                    source_stats[stat] - comparison_value
                )

        rows[day] = row

    daily = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    result = daily.reindex(pd.DatetimeIndex(target_index.normalize()))
    result.index = target_index
    result.index.name = "timestamp"
    return result


def build_renewable_generation_dataset(config: RenewableGenerationModelConfig) -> pd.DataFrame:
    """Build aligned timestamp-level features and actual renewable generation targets."""
    actual = _load_or_fetch_actual_generation(config)
    proxy = _load_renewable_proxy(config)

    feature_blocks = [proxy]
    if config.include_regional_summary_features:
        feature_blocks.append(_build_regional_summary_features(proxy))
    feature_blocks.append(_build_time_features(proxy.index))
    if config.include_forecast_lead_features:
        feature_blocks.append(_build_forecast_lead_features(proxy.index, config))
    if config.include_solar_geometry_features:
        feature_blocks.append(_build_solar_geometry_features(proxy))
    if config.include_solar_physics_features:
        feature_blocks.append(_build_solar_physics_features(proxy))
    if config.include_solar_physics_features or config.target_baseline_mode == "solar_physics_proxy":
        feature_blocks.append(_build_solar_physics_proxy_baselines(proxy, config))
    if config.include_dwd_cluster_features:
        dwd_features = _build_dwd_cluster_features(config)
        if config.include_nwp_lag_diff_features:
            dwd_features = _add_lag_diff_features(
                dwd_features,
                lag_steps=config.nwp_lag_steps,
                diff_steps=config.nwp_diff_steps,
            )
        feature_blocks.append(dwd_features)
    if config.include_unavailability:
        feature_blocks.append(_load_or_fetch_unavailability(config))
    if config.actual_generation_lag_days:
        feature_blocks.append(_build_actual_generation_lag_features(actual, config))
    if config.include_partial_actual_generation_features:
        partial_columns = config.partial_generation_columns or config.actual_generation_lag_columns
        feature_blocks.append(
            _build_partial_actual_generation_features(
                actual,
                proxy.index,
                columns=partial_columns,
                reference_day=config.partial_generation_reference_day,
                comparison_lag_days=config.partial_generation_comparison_lag_days,
                morning_end_hour=config.partial_generation_morning_end_hour,
                morning_end_minute=config.partial_generation_morning_end_minute,
            )
        )

    features = pd.concat(feature_blocks, axis=1).sort_index()
    features = features.loc[:, ~features.columns.duplicated()]
    features = features.reindex(proxy.index)
    dataset = features.join(actual, how="left")
    dataset.index.name = "timestamp"
    return dataset.sort_index()


def _feature_columns(dataset: pd.DataFrame, target_columns: list[str]) -> list[str]:
    target_set = set(target_columns) | {
        "Solar_Actual_MW",
        "Wind_Onshore_Actual_MW",
        "Wind_Offshore_Actual_MW",
        "Wind_Total_Actual_MW",
        "Renewable_Total_Actual_MW",
        "Solar_Control_Area_Total_Actual_MW",
    }
    return [
        column
        for column in dataset.select_dtypes(include="number").columns
        if column not in target_set and not column.endswith("_Actual_MW")
    ]


def _read_feature_allowlist(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Feature allowlist file not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "feature" in df.columns:
            values = df["feature"]
        elif len(df.columns) >= 1:
            values = df.iloc[:, 0]
        else:
            values = pd.Series(dtype=str)
        return [str(value) for value in values.dropna().tolist() if str(value).strip()]

    with path.open(encoding="utf-8") as handle:
        return [
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]


def _feature_allowlist_for_target(
    target: str,
    config: RenewableGenerationModelConfig,
) -> list[str] | None:
    path = None
    for alias in _target_aliases(target):
        path = config.target_feature_allowlist_files.get(alias)
        if path is not None:
            break
    if path is None:
        path = config.feature_allowlist_file
    if path is None:
        return None
    return _read_feature_allowlist(path)


def _target_candidate_features(
    all_features: list[str],
    target: str,
    config: RenewableGenerationModelConfig,
) -> list[str]:
    if config.target_feature_mode == "all":
        return all_features

    target_lower = target.lower()
    time_features = {
        "mtu",
        "tod_sin",
        "tod_cos",
        "dow_sin",
        "dow_cos",
        "doy_sin",
        "doy_cos",
        "is_weekend",
        "forecast_lead_hours",
        "forecast_lead_days",
        "forecast_lead_sin",
        "forecast_lead_cos",
    }

    if "solar" in target_lower:
        technology_terms = ("solar", "t2m")
    elif "offshore" in target_lower:
        technology_terms = ("offshore", "t2m", "sp", "cloud")
    elif "onshore" in target_lower:
        technology_terms = ("onshore", "t2m", "sp", "cloud")
    elif "wind" in target_lower:
        technology_terms = ("wind", "t2m")
    else:
        return all_features

    selected = [
        feature
        for feature in all_features
        if (
            feature in time_features
            or any(term in feature.lower() for term in technology_terms)
            or (config.include_unavailability_in_target_features and feature.lower().startswith("unavail_"))
        )
    ]
    if not selected:
        return all_features
    return selected


def _target_aliases(target: str) -> tuple[str, ...]:
    return (
        target,
        target.removesuffix("_Actual_MW"),
        target.removesuffix("_Model_MW"),
        target.replace("_Actual_MW", ""),
        target.replace("_Model_MW", ""),
    )


def _target_model_overrides(config: RenewableGenerationModelConfig, target: str | None) -> dict[str, Any]:
    if target is None:
        return {}
    for key in _target_aliases(target):
        overrides = config.target_model_overrides.get(key)
        if overrides:
            return overrides
    return {}


def _model_param(config: RenewableGenerationModelConfig, overrides: dict[str, Any], name: str) -> Any:
    return overrides.get(name, getattr(config, name))


def _make_model(config: RenewableGenerationModelConfig, target: str | None = None):
    overrides = _target_model_overrides(config, target)
    model_type = _model_param(config, overrides, "model_type")
    use_pca = bool(_model_param(config, overrides, "use_pca"))
    pca_n_components = _model_param(config, overrides, "pca_n_components")
    pca_whiten = bool(_model_param(config, overrides, "pca_whiten"))

    if model_type == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=int(_model_param(config, overrides, "hgb_max_iter")),
            learning_rate=float(_model_param(config, overrides, "hgb_learning_rate")),
            max_leaf_nodes=int(_model_param(config, overrides, "hgb_max_leaf_nodes")),
            max_depth=_model_param(config, overrides, "hgb_max_depth"),
            min_samples_leaf=int(_model_param(config, overrides, "hgb_min_samples_leaf")),
            l2_regularization=float(_model_param(config, overrides, "hgb_l2_regularization")),
            max_features=float(_model_param(config, overrides, "hgb_max_features")),
            max_bins=int(_model_param(config, overrides, "hgb_max_bins")),
            random_state=int(_model_param(config, overrides, "random_state")),
        )
        if use_pca:
            return make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                PCA(n_components=pca_n_components, whiten=pca_whiten),
                estimator,
            )
        return estimator
    if model_type == "ridge":
        steps = [
            SimpleImputer(strategy="median"),
            StandardScaler(),
        ]
        if use_pca:
            steps.append(PCA(n_components=pca_n_components, whiten=pca_whiten))
        steps.append(Ridge(alpha=float(_model_param(config, overrides, "ridge_alpha"))))
        return make_pipeline(
            *steps,
        )
    raise ValueError(f"Unsupported renewable generation model_type: {model_type!r}")


def _target_output_name(target: str) -> str:
    return target.removesuffix("_Actual_MW")


def _solar_physics_proxy_baseline_candidates(target: str) -> list[str]:
    target_lower = _target_output_name(target).lower()
    if "50hertz" in target_lower:
        regions = ["50hertz"]
    elif "amprion" in target_lower:
        regions = ["amprion"]
    elif "tennet" in target_lower:
        regions = ["tennet"]
    elif "transnetbw" in target_lower:
        regions = ["transnetbw"]
    elif "solar" in target_lower:
        return [
            "Solar_Physics_Proxy_MW",
            "Renewable_Solar_Physics_Proxy_MW",
            "Renewable_Solar_Proxy_MW",
        ]
    else:
        regions = []

    candidates: list[str] = []
    for region in regions:
        candidates.extend(
            [
                f"solar_{region}_physics_proxy_mw",
                f"solar_{region}_proxy_mw",
            ]
        )
    return candidates


def _target_baseline_column(
    target: str,
    config: RenewableGenerationModelConfig,
    features: list[str],
) -> str | None:
    if config.target_baseline_mode == "none":
        return None

    feature_set = set(features)
    for alias in _target_aliases(target):
        configured = config.target_baseline_columns.get(alias)
        if configured is None:
            continue
        if configured not in feature_set:
            raise ValueError(
                f"Configured target baseline column {configured!r} for {target!r} "
                "is not available in the renewable generation dataset."
            )
        return configured

    if config.target_baseline_mode == "solar_physics_proxy":
        if "solar" not in target.lower():
            return None
        for candidate in _solar_physics_proxy_baseline_candidates(target):
            if candidate in feature_set:
                return candidate
        raise ValueError(
            f"target_baseline_mode='solar_physics_proxy' could not find a baseline "
            f"column for {target!r}. Expected one of "
            f"{_solar_physics_proxy_baseline_candidates(target)}."
        )

    raise ValueError(f"Unsupported target_baseline_mode: {config.target_baseline_mode!r}")


def _target_model_values_with_baseline(
    target: str,
    y_train: pd.Series,
    X_train_all: pd.DataFrame,
    config: RenewableGenerationModelConfig,
    installed_capacity_mw: dict[str, float],
    baseline_column: str | None,
) -> pd.Series:
    if baseline_column is None:
        return _model_target_values(target, y_train, config, installed_capacity_mw)

    baseline = X_train_all.loc[y_train.index, baseline_column].astype(float).fillna(0.0)
    return y_train - baseline


def _model_predictions_with_baseline_to_mw(
    target: str,
    predictions: np.ndarray,
    X_test: pd.DataFrame,
    config: RenewableGenerationModelConfig,
    installed_capacity_mw: dict[str, float],
    baseline_column: str | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if baseline_column is None:
        return (
            _model_predictions_to_mw(target, predictions, config, installed_capacity_mw),
            None,
        )

    baseline = X_test[baseline_column].astype(float).fillna(0.0).to_numpy(dtype=float)
    return baseline + predictions, baseline


def _target_upper_bound(
    target: str,
    y_train: pd.Series,
    config: RenewableGenerationModelConfig,
) -> float | None:
    for key in (target, _target_output_name(target)):
        if key in config.target_capacity_caps_mw:
            cap = config.target_capacity_caps_mw[key]
            return float(cap) if pd.notna(cap) and cap > 0 else None

    if not config.clip_predictions_to_training_target_range:
        return None

    valid = y_train.dropna()
    if valid.empty:
        return None
    if config.prediction_upper_quantile >= 1.0:
        return float(valid.max())
    return float(valid.quantile(config.prediction_upper_quantile))


def _select_target_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: RenewableGenerationModelConfig,
) -> list[str]:
    max_features = config.max_features_per_target
    if max_features is None or max_features <= 0 or X_train.shape[1] <= max_features:
        return list(X_train.columns)

    scores = (
        X_train.corrwith(y_train)
        .abs()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .sort_values(ascending=False)
    )
    selected = list(scores.head(max_features).index)
    if not selected:
        return list(X_train.columns)
    return selected


def _prediction_output_upper_bound(
    target: str,
    y_train: pd.Series,
    config: RenewableGenerationModelConfig,
    installed_capacity_mw: dict[str, float],
) -> float | None:
    if config.target_transform == "capacity_factor":
        capacity = _installed_capacity_for_target(target, installed_capacity_mw)
        if capacity is None:
            raise ValueError(
                f"target_transform='capacity_factor' requires installed capacity for {target!r}. "
                "Provide target_installed_capacity_mw or ensure the renewable proxy metadata exists."
            )
        return capacity
    return _target_upper_bound(target, y_train, config)


def _model_target_values(
    target: str,
    y_train: pd.Series,
    config: RenewableGenerationModelConfig,
    installed_capacity_mw: dict[str, float],
) -> pd.Series:
    if config.target_transform == "mw":
        return y_train
    if config.target_transform != "capacity_factor":
        raise ValueError(f"Unsupported target_transform: {config.target_transform!r}")

    capacity = _installed_capacity_for_target(target, installed_capacity_mw)
    if capacity is None:
        raise ValueError(
            f"target_transform='capacity_factor' requires installed capacity for {target!r}. "
            "Provide target_installed_capacity_mw or ensure the renewable proxy metadata exists."
        )
    return (y_train / capacity).clip(0.0, 1.0)


def _model_predictions_to_mw(
    target: str,
    predictions: np.ndarray,
    config: RenewableGenerationModelConfig,
    installed_capacity_mw: dict[str, float],
) -> np.ndarray:
    if config.target_transform == "mw":
        return predictions
    capacity = _installed_capacity_for_target(target, installed_capacity_mw)
    if capacity is None:
        raise ValueError(
            f"target_transform='capacity_factor' requires installed capacity for {target!r}. "
            "Provide target_installed_capacity_mw or ensure the renewable proxy metadata exists."
        )
    return np.clip(predictions, 0.0, 1.0) * capacity


def _bias_group_values(index: pd.DatetimeIndex, group: str) -> np.ndarray:
    if group == "global":
        return np.repeat("global", len(index))
    if group == "hour":
        return index.hour
    if group == "mtu":
        return index.hour * 4 + index.minute // 15
    raise ValueError(f"Unsupported rolling_bias_correction_group: {group!r}")


def _apply_rolling_bias_correction(
    day_forecast: pd.DataFrame,
    forecast_blocks: list[pd.DataFrame],
    *,
    day: pd.Timestamp,
    pred_col: str,
    true_col: str,
    upper_bound: float | None,
    config: RenewableGenerationModelConfig,
    window_days: int | None = None,
    group: str | None = None,
    min_observations: int | None = None,
    shrinkage: float | None = None,
) -> None:
    window_days = int(config.rolling_bias_correction_window_days if window_days is None else window_days)
    group = config.rolling_bias_correction_group if group is None else group
    min_observations = (
        config.rolling_bias_correction_min_observations if min_observations is None else int(min_observations)
    )
    shrinkage = config.rolling_bias_correction_shrinkage if shrinkage is None else float(shrinkage)

    if window_days <= 0 or pred_col not in day_forecast.columns or true_col not in day_forecast.columns:
        return

    history_blocks = [
        block[[pred_col, true_col]]
        for block in forecast_blocks
        if pred_col in block.columns and true_col in block.columns
    ]
    if not history_blocks:
        return

    window_start = day - pd.Timedelta(days=window_days)
    history = pd.concat(history_blocks).sort_index()
    history = history.loc[(history.index >= window_start) & (history.index < day), [pred_col, true_col]].dropna()
    if len(history) < min_observations:
        return

    errors = history[pred_col] - history[true_col]
    global_bias = float(errors.mean())

    if group == "global":
        correction = np.repeat(global_bias, len(day_forecast))
    else:
        history_keys = _bias_group_values(history.index, group)
        bias_by_group = errors.groupby(history_keys).mean()
        count_by_group = errors.groupby(history_keys).count()
        bias_by_group = bias_by_group[count_by_group >= min_observations]
        current_keys = pd.Series(_bias_group_values(day_forecast.index, group), index=day_forecast.index)
        correction_series = current_keys.map(bias_by_group).fillna(global_bias)
        correction = correction_series.to_numpy(dtype=float)

    corrected = day_forecast[pred_col].to_numpy(dtype=float) - (shrinkage * correction)
    day_forecast[pred_col] = np.clip(corrected, 0.0, upper_bound)


def rolling_renewable_generation_forecast(
    dataset: pd.DataFrame,
    config: RenewableGenerationModelConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train rolling timestamp-level models and predict complete forecast days."""
    target_columns = list(config.target_columns)
    missing_targets = [target for target in target_columns if target not in dataset.columns]
    if missing_targets:
        raise ValueError(f"Dataset is missing configured target columns: {missing_targets}")

    features = _feature_columns(dataset, target_columns)
    if not features:
        raise ValueError("No renewable generation model feature columns were built.")

    features_by_target = {
        target: _target_candidate_features(features, target, config)
        for target in target_columns
    }
    baseline_columns_by_target = {
        target: _target_baseline_column(target, config, features)
        for target in target_columns
    }
    feature_allowlists_by_target = {
        target: _feature_allowlist_for_target(target, config)
        for target in target_columns
    }

    test_start = _as_local_day(config.test_start, config.target_tz)
    test_end = _as_local_day(config.test_end, config.target_tz)
    forecast_days = pd.date_range(start=test_start, end=test_end, freq="D", tz=config.target_tz)

    forecast_blocks = []
    runtime_rows: list[dict[str, Any]] = []
    min_train_rows = config.min_train_days * 96
    installed_capacity_mw = _load_installed_capacity_denominators(config)

    for day in forecast_days:
        day_start = time.perf_counter()
        train_start = day - pd.Timedelta(days=config.train_days_rolling)
        train_end = _training_target_cutoff(day, config)
        test_end_ts = day + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)

        train_mask = (dataset.index >= train_start) & (dataset.index <= train_end)
        test_mask = (dataset.index >= day) & (dataset.index <= test_end_ts)
        X_train_all = dataset.loc[train_mask, features]
        X_test = dataset.loc[test_mask, features]

        if len(X_train_all) < min_train_rows or len(X_test) == 0:
            continue

        day_forecast = pd.DataFrame(index=X_test.index)
        for target in target_columns:
            target_features = features_by_target[target]
            baseline_column = baseline_columns_by_target[target]
            feature_allowlist = feature_allowlists_by_target[target]
            if feature_allowlist is not None:
                candidate_set = set(target_features)
                target_features = [
                    feature
                    for feature in feature_allowlist
                    if feature in candidate_set
                ]
                if not target_features:
                    raise ValueError(
                        f"Feature allowlist for {target!r} did not match any candidate features."
                    )

            train_target_mask = dataset.loc[train_mask, target].notna()
            X_train = X_train_all.loc[train_target_mask, target_features]
            y_train = dataset.loc[train_mask, target].loc[train_target_mask]

            if len(X_train) < min_train_rows:
                continue

            y_model = _target_model_values_with_baseline(
                target,
                y_train,
                X_train_all,
                config,
                installed_capacity_mw,
                baseline_column,
            )
            selected_features = _select_target_features(X_train, y_model, config)
            model = _make_model(config, target=target)
            model.fit(X_train[selected_features], y_model)

            output_name = _target_output_name(target)
            pred_col = f"{output_name}_Model_MW"
            true_col = f"{output_name}_Actual_MW"
            upper_bound = _prediction_output_upper_bound(target, y_train, config, installed_capacity_mw)
            prediction_mw, baseline_values = _model_predictions_with_baseline_to_mw(
                target,
                model.predict(X_test.loc[:, selected_features]),
                X_test,
                config,
                installed_capacity_mw,
                baseline_column,
            )
            if baseline_values is not None:
                day_forecast[f"{output_name}_Baseline_MW"] = np.clip(baseline_values, 0.0, upper_bound)
            day_forecast[pred_col] = np.clip(
                prediction_mw,
                0.0,
                upper_bound,
            )
            if target in dataset.columns:
                day_forecast[true_col] = dataset.loc[test_mask, target]

            _apply_rolling_bias_correction(
                day_forecast,
                forecast_blocks,
                day=day,
                pred_col=pred_col,
                true_col=true_col,
                upper_bound=upper_bound,
                config=config,
            )

        if {"Wind_Onshore_Model_MW", "Wind_Offshore_Model_MW"}.issubset(day_forecast.columns):
            day_forecast["Wind_Total_Model_MW"] = (
                day_forecast["Wind_Onshore_Model_MW"] + day_forecast["Wind_Offshore_Model_MW"]
            )
        if "Wind_Total_Model_MW" in day_forecast.columns and "Wind_Total_Actual_MW" in dataset.columns:
            day_forecast["Wind_Total_Actual_MW"] = dataset.loc[test_mask, "Wind_Total_Actual_MW"]

        solar_control_targets = [
            target
            for target in config.solar_control_area_targets
            if target in target_columns
        ]
        solar_control_model_columns = [
            f"{_target_output_name(target)}_Model_MW"
            for target in solar_control_targets
        ]
        if solar_control_model_columns and set(solar_control_model_columns).issubset(day_forecast.columns):
            day_forecast["Solar_Model_MW"] = day_forecast[solar_control_model_columns].sum(axis=1)
            if "Solar_Actual_MW" in dataset.columns:
                day_forecast["Solar_Actual_MW"] = dataset.loc[test_mask, "Solar_Actual_MW"]
            else:
                solar_control_actual_columns = [
                    f"{_target_output_name(target)}_Actual_MW"
                    for target in solar_control_targets
                ]
                if set(solar_control_actual_columns).issubset(day_forecast.columns):
                    day_forecast["Solar_Actual_MW"] = day_forecast[solar_control_actual_columns].sum(axis=1)
            _apply_rolling_bias_correction(
                day_forecast,
                forecast_blocks,
                day=day,
                pred_col="Solar_Model_MW",
                true_col="Solar_Actual_MW",
                upper_bound=_installed_capacity_for_target("Solar_Actual_MW", installed_capacity_mw),
                config=config,
                window_days=config.solar_total_bias_correction_window_days,
                group=config.solar_total_bias_correction_group,
                min_observations=config.solar_total_bias_correction_min_observations,
                shrinkage=config.solar_total_bias_correction_shrinkage,
            )

        if {"Solar_Model_MW", "Wind_Total_Model_MW"}.issubset(day_forecast.columns):
            day_forecast["Renewable_Total_Model_MW"] = (
                day_forecast["Solar_Model_MW"] + day_forecast["Wind_Total_Model_MW"]
            )
        if "Renewable_Total_Actual_MW" in dataset.columns:
            day_forecast["Renewable_Total_Actual_MW"] = dataset.loc[test_mask, "Renewable_Total_Actual_MW"]

        if not day_forecast.empty:
            forecast_blocks.append(day_forecast)
        runtime_rows.append(
            {
                "date": day,
                "train_rows": len(X_train_all),
                "test_rows": len(X_test),
                "n_features": len(features),
                "model_type": config.model_type,
                "runtime_seconds": time.perf_counter() - day_start,
            }
        )
        print(f"  {day.date()}  {config.model_type}  {runtime_rows[-1]['runtime_seconds']:.1f}s")

    if not forecast_blocks:
        raise ValueError("No renewable generation forecasts were produced.")

    forecast = pd.concat(forecast_blocks).sort_index()
    forecast.index.name = "timestamp"
    runtime = pd.DataFrame(runtime_rows)
    return forecast, runtime


def _capacity_aliases(target: str) -> tuple[str, ...]:
    return (
        *_target_aliases(target),
        target,
        target.removesuffix("_Model_MW"),
        target.removesuffix("_Actual_MW"),
        target.replace("_Model_MW", "_Actual_MW"),
        target.replace("_Actual_MW", "_Model_MW"),
    )


def _installed_capacity_for_target(target: str, installed_capacity_mw: dict[str, float]) -> float | None:
    for key in _capacity_aliases(target):
        if key in installed_capacity_mw:
            value = installed_capacity_mw[key]
            return float(value) if pd.notna(value) and value > 0 else None
    return None


def _load_installed_capacity_denominators(config: RenewableGenerationModelConfig) -> dict[str, float]:
    denominators: dict[str, float] = {
        str(key): float(value)
        for key, value in {**config.target_capacity_caps_mw, **config.target_installed_capacity_mw}.items()
        if pd.notna(value) and float(value) > 0
    }
    if denominators:
        return denominators

    metadata_path = config.renewable_proxy_file.with_suffix(".json")
    if not metadata_path.exists():
        return {}

    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}

    summary = metadata.get("capacity_summary_mw")
    if not isinstance(summary, list):
        return {}

    wind = 0.0
    wind_onshore = 0.0
    wind_offshore = 0.0
    solar = 0.0
    solar_region_aliases = {
        "solar_50hertz": "Solar_50Hertz",
        "solar_amprion": "Solar_Amprion",
        "solar_tennet": "Solar_TenneT",
        "solar_transnetbw": "Solar_TransnetBW",
    }
    for row in summary:
        if not isinstance(row, dict):
            continue
        technology = str(row.get("technology_group", "")).lower()
        region = str(row.get("region", "")).lower()
        capacity = float(row.get("capacity_mw", 0.0) or 0.0)
        if technology == "wind_onshore":
            wind_onshore += capacity
            wind += capacity
        elif technology == "wind_offshore":
            wind_offshore += capacity
            wind += capacity
        elif technology == "solar":
            solar += capacity
            if region in solar_region_aliases:
                denominators[solar_region_aliases[region]] = capacity

    if wind_onshore > 0:
        denominators["Wind_Onshore"] = wind_onshore
    if wind_offshore > 0:
        denominators["Wind_Offshore"] = wind_offshore
    if wind > 0:
        denominators["Wind_Total"] = wind
    if solar > 0:
        denominators["Solar"] = solar
    if wind + solar > 0:
        denominators["Renewable_Total"] = wind + solar
    return denominators


def _period_metrics(
    forecast: pd.DataFrame,
    pred_col: str,
    true_col: str,
    period: str,
    installed_capacity_mw: dict[str, float],
) -> dict[str, Any]:
    stats = point_error_stats(forecast, pred_col, true_col)
    rmse = float(stats["rmse"])
    capacity = _installed_capacity_for_target(pred_col, installed_capacity_mw)
    nrmse = float(rmse / capacity) if capacity is not None else np.nan
    return {
        "target": pred_col.removesuffix("_Model_MW"),
        "period": period,
        "mae": stats["mae"],
        "mse": stats["mse"],
        "rmse": rmse,
        "nrmse": nrmse,
        "nrmse_pct": float(nrmse * 100.0) if pd.notna(nrmse) else np.nan,
        "r2": stats["r2"],
        "bias": stats["bias"],
        "installed_capacity_mw": float(capacity) if capacity is not None else np.nan,
        "n_obs": stats["n_obs"],
    }


def evaluate_renewable_generation_forecast(
    forecast: pd.DataFrame,
    installed_capacity_mw: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pairs = []
    installed_capacity_mw = installed_capacity_mw or {}
    for pred_col in [column for column in forecast.columns if column.endswith("_Model_MW")]:
        true_col = pred_col.replace("_Model_MW", "_Actual_MW")
        if true_col in forecast.columns:
            pairs.append((pred_col, true_col))

    for pred_col, true_col in pairs:
        rows.append(_period_metrics(forecast, pred_col, true_col, "full", installed_capacity_mw))
        month_index = forecast.index.tz_localize(None).to_period("M")
        for month, group in forecast.groupby(month_index):
            rows.append(_period_metrics(group, pred_col, true_col, str(month), installed_capacity_mw))

    return pd.DataFrame(rows)


def _load_forecast_csv(path: Path, target_tz: str) -> pd.DataFrame:
    forecast = pd.read_csv(path, index_col=0)
    forecast.index = pd.to_datetime(forecast.index)
    if forecast.index.tz is None:
        forecast.index = forecast.index.tz_localize(target_tz)
    else:
        forecast.index = forecast.index.tz_convert(target_tz)
    forecast = forecast.sort_index()
    forecast = forecast.loc[~forecast.index.duplicated(keep="last")]
    forecast.index.name = "timestamp"
    return forecast


def _forecast_pairs(forecast: pd.DataFrame) -> list[tuple[str, str]]:
    return [
        (pred_col, pred_col.replace("_Model_MW", "_Actual_MW"))
        for pred_col in forecast.columns
        if pred_col.endswith("_Model_MW") and pred_col.replace("_Model_MW", "_Actual_MW") in forecast.columns
    ]


def _target_name_from_prediction(pred_col: str) -> str:
    return pred_col.removesuffix("_Model_MW")


def _postprocess_upper_bound(pred_col: str, config: RenewableGenerationPostprocessConfig) -> float | None:
    target = _target_name_from_prediction(pred_col)
    for key in (pred_col, target, target.replace("_", "")):
        if key in config.target_upper_bounds_mw:
            value = float(config.target_upper_bounds_mw[key])
            return value if pd.notna(value) and value > 0 else None
    return None


def _prediction_bin_edges(values: pd.Series, n_bins: int) -> np.ndarray | None:
    valid = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < max(2, n_bins):
        return None
    quantiles = np.linspace(0.0, 1.0, max(2, n_bins) + 1)
    edges = np.unique(valid.quantile(quantiles).to_numpy(dtype=float))
    if len(edges) < 3:
        return None
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _prediction_bins(values: pd.Series, edges: np.ndarray | None) -> pd.Series:
    if edges is None:
        return pd.Series(np.repeat("all", len(values)), index=values.index)
    bins = pd.cut(values.astype(float), bins=edges, include_lowest=True, labels=False)
    return bins.astype("Int64").astype(str).replace("<NA>", "missing")


def _postprocess_group_values(
    index: pd.DatetimeIndex,
    predictions: pd.Series,
    *,
    group: str,
    prediction_edges: np.ndarray | None,
) -> pd.Series:
    if group == "none":
        return pd.Series(np.repeat("none", len(index)), index=index)
    if group == "global":
        return pd.Series(np.repeat("global", len(index)), index=index)

    hour = pd.Series(index.hour, index=index).astype(str)
    mtu = pd.Series(index.hour * 4 + index.minute // 15, index=index).astype(str)
    daytype = pd.Series(np.where(index.dayofweek >= 5, "weekend", "weekday"), index=index)
    pred_bin = _prediction_bins(predictions, prediction_edges)

    if group == "hour":
        return hour
    if group == "mtu":
        return mtu
    if group == "daytype_hour":
        return daytype + "_h" + hour
    if group == "daytype_mtu":
        return daytype + "_m" + mtu
    if group == "prediction_bin":
        return pred_bin
    if group == "hour_prediction_bin":
        return hour + "_b" + pred_bin
    if group == "mtu_prediction_bin":
        return mtu + "_b" + pred_bin
    if group == "daytype_prediction_bin":
        return daytype + "_b" + pred_bin
    raise ValueError(f"Unsupported correction_group: {group!r}")


def _apply_postprocess_correction_for_day(
    corrected_day: pd.DataFrame,
    history: pd.DataFrame,
    *,
    pred_col: str,
    true_col: str,
    config: RenewableGenerationPostprocessConfig,
) -> None:
    if config.correction_group == "none" or config.correction_window_days <= 0:
        return
    if pred_col not in corrected_day.columns or true_col not in corrected_day.columns:
        return
    if pred_col not in history.columns or true_col not in history.columns:
        return

    hist = history[[pred_col, true_col]].dropna()
    if len(hist) < config.correction_min_observations:
        return

    errors = hist[pred_col] - hist[true_col]
    global_bias = float(errors.mean())
    edges = _prediction_bin_edges(hist[pred_col], config.correction_prediction_bins)
    hist_groups = _postprocess_group_values(
        hist.index,
        hist[pred_col],
        group=config.correction_group,
        prediction_edges=edges,
    )
    current_groups = _postprocess_group_values(
        corrected_day.index,
        corrected_day[pred_col],
        group=config.correction_group,
        prediction_edges=edges,
    )

    bias_by_group = errors.groupby(hist_groups).mean()
    count_by_group = errors.groupby(hist_groups).count()
    bias_by_group = bias_by_group[count_by_group >= config.correction_min_observations]
    correction = current_groups.map(bias_by_group).fillna(global_bias).to_numpy(dtype=float)
    values = corrected_day[pred_col].to_numpy(dtype=float) - config.correction_shrinkage * correction
    upper_bound = _postprocess_upper_bound(pred_col, config)
    corrected_day[pred_col] = np.clip(values, 0.0, upper_bound)


def apply_renewable_generation_postprocess(
    forecast: pd.DataFrame,
    config: RenewableGenerationPostprocessConfig,
) -> pd.DataFrame:
    """Apply leakage-safe rolling residual correction to an existing forecast."""
    corrected_blocks: list[pd.DataFrame] = []
    test_days = pd.date_range(
        forecast.index.min().normalize(),
        forecast.index.max().normalize(),
        freq="D",
        tz=config.target_tz,
    )
    pairs = [
        pair
        for pair in _forecast_pairs(forecast)
        if pair[0] not in {"Renewable_Total_Model_MW"}
    ]

    for day in test_days:
        day_end = day + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
        day_forecast = forecast.loc[(forecast.index >= day) & (forecast.index <= day_end)].copy()
        if day_forecast.empty:
            continue

        window_start = day - pd.Timedelta(days=config.correction_window_days)
        history = pd.concat(corrected_blocks).sort_index() if corrected_blocks else pd.DataFrame()
        if not history.empty:
            history = history.loc[(history.index >= window_start) & (history.index < day)]

        for pred_col, true_col in pairs:
            target_name = _target_name_from_prediction(pred_col)
            if config.correction_targets and target_name not in set(config.correction_targets):
                continue
            _apply_postprocess_correction_for_day(
                day_forecast,
                history,
                pred_col=pred_col,
                true_col=true_col,
                config=config,
            )

        if {"Wind_Onshore_Model_MW", "Wind_Offshore_Model_MW"}.issubset(day_forecast.columns):
            day_forecast["Wind_Total_Model_MW"] = (
                day_forecast["Wind_Onshore_Model_MW"] + day_forecast["Wind_Offshore_Model_MW"]
            )
        if {"Solar_Model_MW", "Wind_Total_Model_MW"}.issubset(day_forecast.columns):
            day_forecast["Renewable_Total_Model_MW"] = (
                day_forecast["Solar_Model_MW"] + day_forecast["Wind_Total_Model_MW"]
            )

        corrected_blocks.append(day_forecast)

    if not corrected_blocks:
        raise ValueError("No forecast rows were available for post-processing.")
    corrected = pd.concat(corrected_blocks).sort_index()
    corrected.index.name = "timestamp"
    return corrected


def _metrics_from_errors(
    errors: pd.Series,
    actual: pd.Series,
    *,
    target: str,
    group_name: str,
    group_value: str,
    installed_capacity_mw: dict[str, float],
) -> dict[str, Any]:
    valid = pd.concat([errors.rename("error"), actual.rename("actual")], axis=1).dropna()
    if valid.empty:
        return {
            "target": target,
            "group": group_name,
            "value": group_value,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "nrmse_pct": np.nan,
            "n_obs": 0,
        }
    err = valid["error"]
    rmse = float(np.sqrt((err**2).mean()))
    capacity = _installed_capacity_for_target(target, installed_capacity_mw)
    nrmse = rmse / capacity if capacity is not None else np.nan
    return {
        "target": target,
        "group": group_name,
        "value": group_value,
        "mae": float(err.abs().mean()),
        "rmse": rmse,
        "bias": float(err.mean()),
        "nrmse_pct": float(nrmse * 100.0) if pd.notna(nrmse) else np.nan,
        "n_obs": int(len(valid)),
    }


def _diagnostic_quantile_groups(series: pd.Series, n_bins: int) -> pd.Series:
    edges = _prediction_bin_edges(series, n_bins)
    return _prediction_bins(series, edges)


def build_renewable_residual_diagnostics(
    forecast: pd.DataFrame,
    *,
    installed_capacity_mw: dict[str, float] | None = None,
    n_bins: int = 5,
) -> dict[str, pd.DataFrame]:
    """Build residual diagnostics by time, level, and ramp regimes."""
    installed_capacity_mw = installed_capacity_mw or {}
    rows_by_time: list[dict[str, Any]] = []
    rows_by_level: list[dict[str, Any]] = []
    rows_by_ramp: list[dict[str, Any]] = []

    for pred_col, true_col in _forecast_pairs(forecast):
        target = _target_name_from_prediction(pred_col)
        errors = forecast[pred_col] - forecast[true_col]
        actual = forecast[true_col]

        time_groups = {
            "hour": pd.Series(forecast.index.hour, index=forecast.index).astype(str),
            "mtu": pd.Series(forecast.index.hour * 4 + forecast.index.minute // 15, index=forecast.index).astype(str),
            "month": pd.Series(forecast.index.strftime("%Y-%m"), index=forecast.index),
            "weekday": pd.Series(forecast.index.day_name(), index=forecast.index),
            "daytype": pd.Series(np.where(forecast.index.dayofweek >= 5, "weekend", "weekday"), index=forecast.index),
        }
        for group_name, groups in time_groups.items():
            for value, idx in groups.groupby(groups).groups.items():
                rows_by_time.append(
                    _metrics_from_errors(
                        errors.loc[idx],
                        actual.loc[idx],
                        target=target,
                        group_name=group_name,
                        group_value=str(value),
                        installed_capacity_mw=installed_capacity_mw,
                    )
                )

        level_groups = {
            "actual_bin": _diagnostic_quantile_groups(actual, n_bins),
            "prediction_bin": _diagnostic_quantile_groups(forecast[pred_col], n_bins),
        }
        for group_name, groups in level_groups.items():
            for value, idx in groups.groupby(groups).groups.items():
                rows_by_level.append(
                    _metrics_from_errors(
                        errors.loc[idx],
                        actual.loc[idx],
                        target=target,
                        group_name=group_name,
                        group_value=str(value),
                        installed_capacity_mw=installed_capacity_mw,
                    )
                )

        for step, label in [(4, "1h"), (12, "3h")]:
            ramp = actual.diff(step).abs()
            groups = _diagnostic_quantile_groups(ramp, n_bins)
            for value, idx in groups.groupby(groups).groups.items():
                rows_by_ramp.append(
                    _metrics_from_errors(
                        errors.loc[idx],
                        actual.loc[idx],
                        target=target,
                        group_name=f"actual_ramp_{label}_bin",
                        group_value=str(value),
                        installed_capacity_mw=installed_capacity_mw,
                    )
                )

    return {
        "diagnostics_by_time": pd.DataFrame(rows_by_time),
        "diagnostics_by_level": pd.DataFrame(rows_by_level),
        "diagnostics_by_ramp": pd.DataFrame(rows_by_ramp),
    }


def run_renewable_generation_postprocess_pipeline(
    config: RenewableGenerationPostprocessConfig,
    *,
    save_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    print("\n--- Renewable Generation Postprocess ---")
    forecast = _load_forecast_csv(config.forecast_file, config.target_tz)
    corrected = apply_renewable_generation_postprocess(forecast, config)
    installed_capacity_mw = {
        str(key): float(value)
        for key, value in config.installed_capacity_mw.items()
        if pd.notna(value) and float(value) > 0
    }
    metrics = evaluate_renewable_generation_forecast(corrected, installed_capacity_mw=installed_capacity_mw)
    diagnostics = build_renewable_residual_diagnostics(
        corrected,
        installed_capacity_mw=installed_capacity_mw,
        n_bins=config.diagnostics_bins,
    )

    print(metrics[metrics["period"].astype(str).eq("full")].to_string(index=False))

    if save_outputs:
        config.export_dir.mkdir(parents=True, exist_ok=True)
        corrected.to_csv(config.export_dir / "forecast.csv")
        metrics.to_csv(config.export_dir / "metrics.csv", index=False)
        for name, df in diagnostics.items():
            df.to_csv(config.export_dir / f"{name}.csv", index=False)
        with (config.export_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config.model_dump(mode="json"), handle, indent=2)
        print(f"Saved renewable generation postprocess outputs -> {config.export_dir}")

    return {
        "forecast": corrected,
        "metrics": metrics,
        **diagnostics,
    }


def build_entsoe_renewable_forecast_benchmark(
    config: EntsoeRenewableForecastBenchmarkConfig,
) -> pd.DataFrame:
    """Build a model-style forecast frame from ENTSO-E renewable forecast data."""
    actual = _load_or_fetch_benchmark_actual_generation(config)
    forecast = _load_or_fetch_entsoe_renewable_forecast(config)

    frame = pd.DataFrame(index=forecast.index)
    rename_map = {
        "Solar_Forecast_MW": "Solar_Model_MW",
        "Wind_Onshore_Forecast_MW": "Wind_Onshore_Model_MW",
        "Wind_Offshore_Forecast_MW": "Wind_Offshore_Model_MW",
        "Wind_Total_Forecast_MW": "Wind_Total_Model_MW",
        "Renewable_Total_Forecast_MW": "Renewable_Total_Model_MW",
    }
    for source, target in rename_map.items():
        if source in forecast.columns:
            frame[target] = forecast[source]

    for column in [
        "Solar_Actual_MW",
        "Wind_Onshore_Actual_MW",
        "Wind_Offshore_Actual_MW",
        "Wind_Total_Actual_MW",
        "Renewable_Total_Actual_MW",
    ]:
        if column in actual.columns:
            frame[column] = actual[column]

    frame = frame.sort_index()
    frame = frame.loc[:, ~frame.columns.duplicated()]
    frame.index.name = "timestamp"
    return frame


def run_entsoe_renewable_forecast_benchmark(
    config: EntsoeRenewableForecastBenchmarkConfig,
    *,
    save_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    load_dotenv(config.repo_root / ".env")
    print("\n--- ENTSO-E Renewable Forecast Benchmark ---")
    forecast = build_entsoe_renewable_forecast_benchmark(config)
    metrics = evaluate_renewable_generation_forecast(
        forecast,
        installed_capacity_mw=config.installed_capacity_mw,
    )
    print(metrics.to_string(index=False))

    if save_outputs:
        config.export_dir.mkdir(parents=True, exist_ok=True)
        forecast.to_csv(config.export_dir / "forecast.csv")
        metrics.to_csv(config.export_dir / "metrics.csv", index=False)
        with (config.export_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config.model_dump(mode="json"), handle, indent=2)
        print(f"Saved ENTSO-E renewable forecast benchmark outputs -> {config.export_dir}")

    return {
        "forecast": forecast,
        "metrics": metrics,
    }


def run_renewable_generation_pipeline(
    config: RenewableGenerationModelConfig,
    *,
    save_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    load_dotenv(config.repo_root / ".env")
    print("\n--- Building Renewable Generation Dataset ---")
    dataset = build_renewable_generation_dataset(config)
    print(f"Dataset: {dataset.shape[0]:,} rows, {dataset.shape[1]:,} columns")

    print("\n--- Rolling Renewable Generation Forecast ---")
    forecast, runtime = rolling_renewable_generation_forecast(dataset, config)
    installed_capacity_mw = _load_installed_capacity_denominators(config)
    metrics = evaluate_renewable_generation_forecast(forecast, installed_capacity_mw=installed_capacity_mw)
    print("\n--- Renewable Generation Metrics ---")
    print(metrics.to_string(index=False))

    if save_outputs:
        config.export_dir.mkdir(parents=True, exist_ok=True)
        forecast.to_csv(config.export_dir / "forecast.csv")
        runtime.to_csv(config.export_dir / "runtime.csv", index=False)
        metrics.to_csv(config.export_dir / "metrics.csv", index=False)
        with (config.export_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config.model_dump(mode="json"), handle, indent=2)
        print(f"Saved renewable generation outputs -> {config.export_dir}")

    return {
        "dataset": dataset,
        "forecast": forecast,
        "runtime": runtime,
        "metrics": metrics,
    }
