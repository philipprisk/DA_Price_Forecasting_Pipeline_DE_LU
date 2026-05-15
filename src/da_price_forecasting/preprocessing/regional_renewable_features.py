from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..config import RegionalRenewableFeatureConfig
from ..data.weather import load_dwd, load_era5, load_open_meteo


def _normalise_label(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _load_clusters(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        clusters = pd.read_parquet(path)
    else:
        clusters = pd.read_csv(path)

    required = {"lon", "lat", "cluster_id"}
    missing = required - set(clusters.columns)
    if missing:
        raise ValueError(f"Cluster file is missing required columns: {sorted(missing)}")

    clusters = clusters[["lon", "lat", "cluster_id"]].dropna().copy()
    clusters["cluster_id"] = clusters["cluster_id"].astype(int)
    return clusters


def _technology_group(raw_technology: object, config: RegionalRenewableFeatureConfig) -> str | None:
    technology = _normalise_label(raw_technology)
    if technology in {_normalise_label(value) for value in config.wind_offshore_values}:
        return "wind_offshore"
    if technology in {_normalise_label(value) for value in config.wind_onshore_values}:
        return "wind_onshore"
    if technology in {_normalise_label(value) for value in config.solar_values}:
        return "solar"
    return None


def _region_for_row(row: pd.Series, config: RegionalRenewableFeatureConfig) -> str:
    technology = row["technology_group"]
    lat = float(row[config.latitude_column])
    lon = float(row[config.longitude_column])

    if technology == "wind_offshore":
        return "offshore_north"

    if technology == "wind_onshore":
        prefix = "onshore"
    else:
        prefix = "solar"

    if lat >= config.north_latitude:
        return f"{prefix}_north"
    if lat < config.south_latitude:
        return f"{prefix}_south"
    if lon < config.west_longitude:
        return f"{prefix}_west"
    if lon >= config.east_longitude:
        return f"{prefix}_east"
    return f"{prefix}_central"


def build_capacity_map(config: RegionalRenewableFeatureConfig) -> pd.DataFrame:
    """Assign active MaStR renewable units to weather clusters and coarse regions."""
    capacity = pd.read_csv(config.capacity_file)
    required = {
        config.technology_column,
        config.capacity_column,
        config.latitude_column,
        config.longitude_column,
        config.commissioning_date_column,
    }
    missing = required - set(capacity.columns)
    if missing:
        raise ValueError(f"Capacity file is missing required columns: {sorted(missing)}")

    capacity = capacity.copy()
    capacity["technology_group"] = capacity[config.technology_column].map(lambda value: _technology_group(value, config))
    capacity["capacity_mw"] = pd.to_numeric(capacity[config.capacity_column], errors="coerce")
    capacity["commissioning_date"] = pd.to_datetime(
        capacity[config.commissioning_date_column],
        errors="coerce",
    ).dt.date
    if config.decommissioning_date_column in capacity.columns:
        capacity["decommissioning_date"] = pd.to_datetime(
            capacity[config.decommissioning_date_column],
            errors="coerce",
        ).dt.date
    else:
        capacity["decommissioning_date"] = pd.NaT

    capacity = capacity.dropna(
        subset=[
            "technology_group",
            "capacity_mw",
            config.latitude_column,
            config.longitude_column,
            "commissioning_date",
        ]
    )
    capacity = capacity.loc[capacity["capacity_mw"] > 0].copy()

    if config.operating_status_column in capacity.columns and config.active_status_codes:
        active_codes = {str(code) for code in config.active_status_codes}
        capacity = capacity.loc[capacity[config.operating_status_column].astype(str).isin(active_codes)].copy()

    clusters = _load_clusters(config.cluster_file)
    tree = cKDTree(clusters[["lon", "lat"]].to_numpy(dtype=float))
    points = capacity[[config.longitude_column, config.latitude_column]].to_numpy(dtype=float)
    _, nearest_idx = tree.query(points)
    capacity["cluster_id"] = clusters.iloc[nearest_idx]["cluster_id"].to_numpy(dtype=int)
    capacity["region"] = capacity.apply(lambda row: _region_for_row(row, config), axis=1)

    columns = [
        "technology_group",
        "region",
        "cluster_id",
        "capacity_mw",
        "commissioning_date",
        "decommissioning_date",
        config.latitude_column,
        config.longitude_column,
    ]
    return capacity[columns].sort_values(["technology_group", "region", "cluster_id"]).reset_index(drop=True)


def _normalised_wind_power_curve(speed_m_s: np.ndarray) -> np.ndarray:
    cut_in = 3.0
    rated = 12.0
    cut_out = 25.0
    values = np.zeros_like(speed_m_s, dtype=float)
    ramp = (speed_m_s >= cut_in) & (speed_m_s < rated)
    rated_mask = (speed_m_s >= rated) & (speed_m_s <= cut_out)
    values[ramp] = ((speed_m_s[ramp] - cut_in) / (rated - cut_in)) ** 3
    values[rated_mask] = 1.0
    return np.clip(values, 0.0, 1.0)


def _active_capacity_by_cluster(
    capacity_map: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    *,
    technology: str,
    region: str | None,
) -> pd.DataFrame:
    subset = capacity_map.loc[capacity_map["technology_group"] == technology].copy()
    if region is not None:
        subset = subset.loc[subset["region"] == region].copy()
    clusters = sorted(capacity_map["cluster_id"].unique())
    days = pd.DatetimeIndex(timestamps.normalize().unique()).sort_values()
    result = pd.DataFrame(0.0, index=days, columns=clusters)
    if subset.empty:
        return result

    first_day = days.min().tz_localize(None).normalize()
    last_day = days.max().tz_localize(None).normalize()
    subset["_commissioning_ts"] = pd.to_datetime(subset["commissioning_date"], errors="coerce").dt.normalize()
    subset["_decommissioning_ts"] = pd.to_datetime(subset["decommissioning_date"], errors="coerce").dt.normalize()
    commissioned = subset["_commissioning_ts"]
    decommissioned = subset["_decommissioning_ts"]

    initial_mask = (commissioned <= first_day) & (decommissioned.isna() | (decommissioned > first_day))
    initial = subset.loc[initial_mask].groupby("cluster_id")["capacity_mw"].sum()
    result.loc[:, initial.index] = initial.to_numpy(dtype=float)

    event_rows = []
    commissioning_mask = (commissioned > first_day) & (commissioned <= last_day)
    for _, row in subset.loc[commissioning_mask].iterrows():
        event_rows.append((
            pd.Timestamp(row["_commissioning_ts"].date(), tz=timestamps.tz),
            row["cluster_id"],
            row["capacity_mw"],
        ))

    decommissioning_mask = decommissioned.notna() & (decommissioned > first_day) & (decommissioned <= last_day)
    for _, row in subset.loc[decommissioning_mask].iterrows():
        event_rows.append((
            pd.Timestamp(row["_decommissioning_ts"].date(), tz=timestamps.tz),
            row["cluster_id"],
            -row["capacity_mw"],
        ))

    if event_rows:
        events = pd.DataFrame(event_rows, columns=["date", "cluster_id", "capacity_delta_mw"])
        event_pivot = events.pivot_table(
            index="date",
            columns="cluster_id",
            values="capacity_delta_mw",
            aggfunc="sum",
            fill_value=0.0,
        )
        event_pivot = event_pivot.reindex(days, fill_value=0.0)
        for column in event_pivot.columns:
            if column in result.columns:
                result[column] = result[column] + event_pivot[column].cumsum()

    return result.clip(lower=0.0)


def _columns_for_clusters(df: pd.DataFrame, prefix: str, clusters: list[int]) -> list[str]:
    return [f"{prefix}_cluster_{cluster_id}" for cluster_id in clusters if f"{prefix}_cluster_{cluster_id}" in df.columns]


def _capacity_for_timestamps(capacity_daily: pd.DataFrame, timestamps: pd.DatetimeIndex, clusters: list[int]) -> np.ndarray:
    capacity = capacity_daily.reindex(timestamps.normalize()).loc[:, clusters].fillna(0.0)
    capacity.index = timestamps
    return capacity.to_numpy(dtype=float)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    denominator = weights.sum(axis=1)
    numerator = (values * weights).sum(axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0)


def _air_density_kg_m3(pressure_pa: np.ndarray, temperature_k: np.ndarray) -> np.ndarray:
    return np.divide(
        pressure_pa,
        287.05 * temperature_k,
        out=np.full_like(pressure_pa, np.nan, dtype=float),
        where=temperature_k > 0,
    )


def _weighted_wind_direction_components(u: np.ndarray, v: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed = np.sqrt(u**2 + v**2)
    unit_u = np.divide(u, speed, out=np.zeros_like(u, dtype=float), where=speed > 0)
    unit_v = np.divide(v, speed, out=np.zeros_like(v, dtype=float), where=speed > 0)
    return _weighted_mean(unit_u, weights), _weighted_mean(unit_v, weights)


def _add_ramps(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return df
    blocks = [df]
    for step, suffix in ((4, "ramp_1h"), (12, "ramp_3h")):
        ramp = df[columns].diff(step).rename(columns={column: f"{column}_{suffix}" for column in columns})
        blocks.append(ramp)
    return pd.concat(blocks, axis=1)


def _add_wind_cluster_features(
    frames: dict[str, pd.Series],
    ramp_columns: list[str],
    capacity_map: pd.DataFrame,
    weather: pd.DataFrame,
    config: RegionalRenewableFeatureConfig,
    *,
    technology: str,
    u_prefix: str,
    v_prefix: str,
    wind_factor: float,
) -> None:
    capacity_daily = _active_capacity_by_cluster(capacity_map, weather.index, technology=technology, region=None)
    for cluster_id in sorted(capacity_daily.columns):
        u_col = f"{u_prefix}_cluster_{cluster_id}"
        v_col = f"{v_prefix}_cluster_{cluster_id}"
        if u_col not in weather.columns or v_col not in weather.columns:
            continue

        capacity = capacity_daily.reindex(weather.index.normalize())[cluster_id].fillna(0.0).to_numpy(dtype=float)
        if np.nanmax(capacity) <= 0:
            continue

        u = weather[u_col].to_numpy(dtype=float)
        v = weather[v_col].to_numpy(dtype=float)
        speed = np.sqrt(u**2 + v**2) * wind_factor
        proxy = capacity * _normalised_wind_power_curve(speed)
        proxy_v3 = capacity * speed**3

        prefix = f"{technology}_cluster_{cluster_id}"
        frames[f"{prefix}_capacity_mw"] = pd.Series(capacity, index=weather.index)
        frames[f"{prefix}_proxy_mw"] = pd.Series(proxy, index=weather.index)
        frames[f"{prefix}_proxy_v3"] = pd.Series(proxy_v3, index=weather.index)
        frames[f"{prefix}_speed_hub_m_s"] = pd.Series(speed, index=weather.index)
        frames[f"{prefix}_u_m_s"] = pd.Series(u, index=weather.index)
        frames[f"{prefix}_v_m_s"] = pd.Series(v, index=weather.index)
        speed_safe = np.divide(1.0, speed, out=np.zeros_like(speed, dtype=float), where=speed > 0)
        frames[f"{prefix}_direction_cos"] = pd.Series(u * speed_safe, index=weather.index)
        frames[f"{prefix}_direction_sin"] = pd.Series(v * speed_safe, index=weather.index)
        ramp_columns.append(f"{prefix}_proxy_mw")

        for variable, suffix in (("t2m", "t2m_K"), ("sp", "sp_Pa"), ("tcc", "cloud_cover")):
            column = f"{variable}_cluster_{cluster_id}"
            if column in weather.columns:
                frames[f"{prefix}_{suffix}"] = pd.Series(weather[column].to_numpy(dtype=float), index=weather.index)


def _add_solar_cluster_features(
    frames: dict[str, pd.Series],
    ramp_columns: list[str],
    capacity_map: pd.DataFrame,
    weather: pd.DataFrame,
    config: RegionalRenewableFeatureConfig,
) -> None:
    capacity_daily = _active_capacity_by_cluster(capacity_map, weather.index, technology="solar", region=None)
    for cluster_id in sorted(capacity_daily.columns):
        ssrd_col = f"ssrd_cluster_{cluster_id}"
        if ssrd_col not in weather.columns:
            continue

        capacity = capacity_daily.reindex(weather.index.normalize())[cluster_id].fillna(0.0).to_numpy(dtype=float)
        if np.nanmax(capacity) <= 0:
            continue

        ssrd = np.clip(weather[ssrd_col].to_numpy(dtype=float), 0.0, None)
        proxy = capacity * ssrd / 1000.0 * config.solar_performance_ratio

        prefix = f"solar_cluster_{cluster_id}"
        frames[f"{prefix}_capacity_mw"] = pd.Series(capacity, index=weather.index)
        frames[f"{prefix}_proxy_mw"] = pd.Series(proxy, index=weather.index)
        frames[f"{prefix}_irradiance_W_m2"] = pd.Series(ssrd, index=weather.index)
        ramp_columns.append(f"{prefix}_proxy_mw")

        for variable, suffix in (
            ("fdir", "direct_irradiance_W_m2"),
            ("t2m", "t2m_K"),
            ("tcc", "cloud_cover"),
        ):
            column = f"{variable}_cluster_{cluster_id}"
            if column in weather.columns:
                values = weather[column].to_numpy(dtype=float)
                if variable == "fdir":
                    values = np.clip(values, 0.0, None)
                frames[f"{prefix}_{suffix}"] = pd.Series(values, index=weather.index)


def _deduplicate_mean(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.has_duplicates:
        return df.groupby(level=0).mean()
    return df.sort_index()


def _load_dwd_icon_weather(config: RegionalRenewableFeatureConfig) -> pd.DataFrame:
    df_hourly, df_qh = load_dwd(
        icon_dir=config.icon_dir,
        start_folder_date=config.start_folder_date,
        required_run=config.required_run,
        skip_dates=set(config.skip_dates),
        folder_offset_date=config.dwd_folder_offset_date,
        target_tz=config.target_tz,
    )
    df_hourly = _deduplicate_mean(df_hourly)
    df_qh = _deduplicate_mean(df_qh)

    full_index = pd.date_range(
        start=min(df_hourly.index.min(), df_qh.index.min()),
        end=max(df_hourly.index.max(), df_qh.index.max()),
        freq="15min",
        tz=config.target_tz,
        name="timestamp",
    )

    hourly = df_hourly.reindex(full_index).ffill(limit=3)
    qh = df_qh.reindex(full_index)
    weather = pd.concat([hourly, qh], axis=1).sort_index()

    direct_cols = sorted(column for column in weather.columns if column.startswith("ASWDIR_cluster_"))
    for direct_col in direct_cols:
        cluster_id = direct_col.rsplit("_", 1)[-1]
        diffuse_col = f"ASWDIFD_cluster_{cluster_id}"
        if diffuse_col not in weather.columns:
            continue
        direct = np.clip(weather[direct_col].to_numpy(dtype=float), 0.0, None)
        diffuse = np.clip(weather[diffuse_col].to_numpy(dtype=float), 0.0, None)
        weather[f"fdir_cluster_{cluster_id}"] = direct
        weather[f"ssrd_cluster_{cluster_id}"] = direct + diffuse

    weather.index.name = "timestamp"
    return weather


def _load_weather(config: RegionalRenewableFeatureConfig) -> pd.DataFrame:
    if config.weather_source == "era5":
        return load_era5(config.era5_dirs, target_tz=config.target_tz)
    if config.weather_source == "dwd_icon":
        return _load_dwd_icon_weather(config)
    if config.weather_source == "open_meteo":
        return load_open_meteo(
            cluster_file=config.cluster_file,
            start_date=config.open_meteo_start_date,
            end_date=config.open_meteo_end_date,
            cache_file=config.open_meteo_weather_file,
            base_url=config.open_meteo_base_url,
            model=config.open_meteo_model,
            hourly_variables=config.open_meteo_hourly_variables,
            batch_size=config.open_meteo_batch_size,
            cell_selection=config.open_meteo_cell_selection,
            timeout_seconds=config.open_meteo_timeout_seconds,
            target_tz=config.target_tz,
            force_download=config.open_meteo_force_download,
            point_selection=config.open_meteo_point_selection,
            max_points_per_cluster=config.open_meteo_max_points_per_cluster,
            api_mode=config.open_meteo_api_mode,
            single_run_hour_utc=config.open_meteo_single_run_hour_utc,
            single_run_forecast_days=config.open_meteo_single_run_forecast_days,
            request_pause_seconds=config.open_meteo_request_pause_seconds,
            retry_attempts=config.open_meteo_retry_attempts,
            retry_backoff_seconds=config.open_meteo_retry_backoff_seconds,
        )
    raise ValueError(f"Unsupported regional renewable weather_source: {config.weather_source!r}")


def build_regional_renewable_features(config: RegionalRenewableFeatureConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    capacity_map = build_capacity_map(config)
    weather = _load_weather(config)

    u_prefix = "u100" if any(column.startswith("u100_cluster_") for column in weather.columns) else "u10"
    v_prefix = "v100" if u_prefix == "u100" else "v10"
    wind_factor = 1.0
    if u_prefix == "u10":
        wind_factor = (config.wind_hub_height_m / config.wind_reference_height_m) ** config.wind_shear_alpha

    frames: dict[str, pd.Series] = {}
    ramp_columns: list[str] = []
    wind_total = pd.Series(0.0, index=weather.index)
    solar_total = pd.Series(0.0, index=weather.index)

    for technology in ["wind_offshore", "wind_onshore"]:
        for region in sorted(capacity_map.loc[capacity_map["technology_group"] == technology, "region"].unique()):
            capacity_daily = _active_capacity_by_cluster(capacity_map, weather.index, technology=technology, region=region)
            clusters = [
                cluster_id
                for cluster_id in capacity_daily.columns
                if f"{u_prefix}_cluster_{cluster_id}" in weather.columns and f"{v_prefix}_cluster_{cluster_id}" in weather.columns
            ]
            if not clusters:
                continue

            capacity = _capacity_for_timestamps(capacity_daily, weather.index, clusters)
            u = weather[_columns_for_clusters(weather, u_prefix, clusters)].to_numpy(dtype=float)
            v = weather[_columns_for_clusters(weather, v_prefix, clusters)].to_numpy(dtype=float)
            speed = np.sqrt(u**2 + v**2) * wind_factor
            proxy_curve = (capacity * _normalised_wind_power_curve(speed)).sum(axis=1)
            proxy_v3 = (capacity * speed**3).sum(axis=1)

            prefix = f"wind_{region}"
            frames[f"{prefix}_proxy_mw"] = pd.Series(proxy_curve, index=weather.index)
            frames[f"{prefix}_proxy_v3"] = pd.Series(proxy_v3, index=weather.index)
            frames[f"{prefix}_speed_hub_cap_weighted_m_s"] = pd.Series(_weighted_mean(speed, capacity), index=weather.index)
            frames[f"{prefix}_u_cap_weighted_m_s"] = pd.Series(_weighted_mean(u, capacity), index=weather.index)
            frames[f"{prefix}_v_cap_weighted_m_s"] = pd.Series(_weighted_mean(v, capacity), index=weather.index)
            direction_cos, direction_sin = _weighted_wind_direction_components(u, v, capacity)
            frames[f"{prefix}_direction_cos_cap_weighted"] = pd.Series(direction_cos, index=weather.index)
            frames[f"{prefix}_direction_sin_cap_weighted"] = pd.Series(direction_sin, index=weather.index)
            wind_total = wind_total + frames[f"{prefix}_proxy_mw"]
            ramp_columns.append(f"{prefix}_proxy_mw")

            t2m = None
            sp = None
            if "t2m_cluster_0" in weather.columns:
                t2m = weather[_columns_for_clusters(weather, "t2m", clusters)].to_numpy(dtype=float)
                frames[f"{prefix}_t2m_cap_weighted_K"] = pd.Series(_weighted_mean(t2m, capacity), index=weather.index)
            if "td2m_cluster_0" in weather.columns:
                td2m = weather[_columns_for_clusters(weather, "td2m", clusters)].to_numpy(dtype=float)
                frames[f"{prefix}_td2m_cap_weighted_K"] = pd.Series(_weighted_mean(td2m, capacity), index=weather.index)
            if "sp_cluster_0" in weather.columns:
                sp = weather[_columns_for_clusters(weather, "sp", clusters)].to_numpy(dtype=float)
                frames[f"{prefix}_sp_cap_weighted_Pa"] = pd.Series(_weighted_mean(sp, capacity), index=weather.index)
            if t2m is not None and sp is not None:
                density = _air_density_kg_m3(sp, t2m)
                density_factor = density / 1.225
                frames[f"{prefix}_air_density_cap_weighted_kg_m3"] = pd.Series(
                    _weighted_mean(density, capacity),
                    index=weather.index,
                )
                frames[f"{prefix}_proxy_v3_air_density"] = pd.Series(
                    (capacity * speed**3 * density_factor).sum(axis=1),
                    index=weather.index,
                )
            if "vmax10m_cluster_0" in weather.columns:
                vmax10 = np.clip(weather[_columns_for_clusters(weather, "vmax10m", clusters)].to_numpy(dtype=float), 0.0, None)
                vmax_hub = vmax10 * (config.wind_hub_height_m / config.wind_reference_height_m) ** config.wind_shear_alpha
                frames[f"{prefix}_vmax10m_cap_weighted_m_s"] = pd.Series(
                    _weighted_mean(vmax10, capacity),
                    index=weather.index,
                )
                frames[f"{prefix}_vmax_hub_cap_weighted_m_s"] = pd.Series(
                    _weighted_mean(vmax_hub, capacity),
                    index=weather.index,
                )
                frames[f"{prefix}_proxy_vmax_v3"] = pd.Series((capacity * vmax_hub**3).sum(axis=1), index=weather.index)
            if "tp_cluster_0" in weather.columns:
                tp = weather[_columns_for_clusters(weather, "tp", clusters)].to_numpy(dtype=float)
                frames[f"{prefix}_tot_prec_cap_weighted"] = pd.Series(_weighted_mean(tp, capacity), index=weather.index)
            if "sde_cluster_0" in weather.columns:
                sde = weather[_columns_for_clusters(weather, "sde", clusters)].to_numpy(dtype=float)
                frames[f"{prefix}_snow_depth_cap_weighted_m"] = pd.Series(_weighted_mean(sde, capacity), index=weather.index)
            if "snow_gsp_cluster_0" in weather.columns:
                snow_gsp = weather[_columns_for_clusters(weather, "snow_gsp", clusters)].to_numpy(dtype=float)
                frames[f"{prefix}_snow_gsp_cap_weighted"] = pd.Series(_weighted_mean(snow_gsp, capacity), index=weather.index)
            if "tcc_cluster_0" in weather.columns:
                tcc = weather[_columns_for_clusters(weather, "tcc", clusters)].to_numpy(dtype=float)
                frames[f"{prefix}_cloud_cover_cap_weighted"] = pd.Series(_weighted_mean(tcc, capacity), index=weather.index)

    if config.include_cluster_features:
        for technology in ["wind_offshore", "wind_onshore"]:
            _add_wind_cluster_features(
                frames,
                ramp_columns,
                capacity_map,
                weather,
                config,
                technology=technology,
                u_prefix=u_prefix,
                v_prefix=v_prefix,
                wind_factor=wind_factor,
            )

    for region in sorted(capacity_map.loc[capacity_map["technology_group"] == "solar", "region"].unique()):
        capacity_daily = _active_capacity_by_cluster(capacity_map, weather.index, technology="solar", region=region)
        clusters = [cluster_id for cluster_id in capacity_daily.columns if f"ssrd_cluster_{cluster_id}" in weather.columns]
        if not clusters:
            continue

        capacity = _capacity_for_timestamps(capacity_daily, weather.index, clusters)
        ssrd = np.clip(weather[_columns_for_clusters(weather, "ssrd", clusters)].to_numpy(dtype=float), 0.0, None)
        proxy = (capacity * ssrd / 1000.0 * config.solar_performance_ratio).sum(axis=1)

        prefix = f"solar_{region.removeprefix('solar_')}"
        frames[f"{prefix}_proxy_mw"] = pd.Series(proxy, index=weather.index)
        frames[f"{prefix}_irradiance_cap_weighted_W_m2"] = pd.Series(_weighted_mean(ssrd, capacity), index=weather.index)
        solar_total = solar_total + frames[f"{prefix}_proxy_mw"]
        ramp_columns.append(f"{prefix}_proxy_mw")

        if "fdir_cluster_0" in weather.columns:
            fdir = np.clip(weather[_columns_for_clusters(weather, "fdir", clusters)].to_numpy(dtype=float), 0.0, None)
            frames[f"{prefix}_direct_irradiance_cap_weighted_W_m2"] = pd.Series(_weighted_mean(fdir, capacity), index=weather.index)
            diffuse = np.clip(ssrd - fdir, 0.0, None)
            diffuse_share = np.divide(
                diffuse,
                ssrd,
                out=np.zeros_like(diffuse, dtype=float),
                where=ssrd > 0,
            )
            frames[f"{prefix}_diffuse_share_cap_weighted"] = pd.Series(
                _weighted_mean(diffuse_share, capacity),
                index=weather.index,
            )
        if "t2m_cluster_0" in weather.columns:
            t2m = weather[_columns_for_clusters(weather, "t2m", clusters)].to_numpy(dtype=float)
            frames[f"{prefix}_t2m_cap_weighted_K"] = pd.Series(_weighted_mean(t2m, capacity), index=weather.index)
        if "td2m_cluster_0" in weather.columns:
            td2m = weather[_columns_for_clusters(weather, "td2m", clusters)].to_numpy(dtype=float)
            frames[f"{prefix}_td2m_cap_weighted_K"] = pd.Series(_weighted_mean(td2m, capacity), index=weather.index)
        if "tp_cluster_0" in weather.columns:
            tp = weather[_columns_for_clusters(weather, "tp", clusters)].to_numpy(dtype=float)
            frames[f"{prefix}_tot_prec_cap_weighted"] = pd.Series(_weighted_mean(tp, capacity), index=weather.index)
        if "sde_cluster_0" in weather.columns:
            sde = weather[_columns_for_clusters(weather, "sde", clusters)].to_numpy(dtype=float)
            frames[f"{prefix}_snow_depth_cap_weighted_m"] = pd.Series(_weighted_mean(sde, capacity), index=weather.index)
        if "snow_gsp_cluster_0" in weather.columns:
            snow_gsp = weather[_columns_for_clusters(weather, "snow_gsp", clusters)].to_numpy(dtype=float)
            frames[f"{prefix}_snow_gsp_cap_weighted"] = pd.Series(_weighted_mean(snow_gsp, capacity), index=weather.index)
        if "tcc_cluster_0" in weather.columns:
            tcc = weather[_columns_for_clusters(weather, "tcc", clusters)].to_numpy(dtype=float)
            frames[f"{prefix}_cloud_cover_cap_weighted"] = pd.Series(_weighted_mean(tcc, capacity), index=weather.index)

    if config.include_cluster_features:
        _add_solar_cluster_features(frames, ramp_columns, capacity_map, weather, config)

    result = pd.DataFrame(frames, index=weather.index)
    result["Renewable_Wind_Proxy_MW"] = wind_total
    result["Renewable_Solar_Proxy_MW"] = solar_total
    result["Renewable_Total_Proxy_MW"] = wind_total + solar_total
    ramp_columns.extend(["Renewable_Wind_Proxy_MW", "Renewable_Solar_Proxy_MW", "Renewable_Total_Proxy_MW"])
    if config.include_ramps:
        result = _add_ramps(result, ramp_columns)

    end_time = result.index.max()
    if end_time.minute == 0 and len(result.index) > 1:
        observed_steps = result.index.to_series().diff().dropna()
        if not observed_steps.empty and observed_steps.min() >= pd.Timedelta(hours=1):
            end_time = end_time + pd.Timedelta(minutes=45)

    full_index = pd.date_range(
        start=result.index.min(),
        end=end_time,
        freq="15min",
        tz=config.target_tz,
        name="timestamp",
    )
    result = result.loc[~result.index.duplicated(keep="last")].reindex(full_index).ffill(limit=3)
    result.index.name = "timestamp"
    return result.astype(float), capacity_map


def run_regional_renewable_features(config: RegionalRenewableFeatureConfig) -> pd.DataFrame:
    features, capacity_map = build_regional_renewable_features(config)
    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    config.capacity_map_file.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(config.output_file)
    capacity_map.to_csv(config.capacity_map_file, index=False)

    summary = (
        capacity_map
        .groupby(["technology_group", "region"], as_index=False)["capacity_mw"]
        .sum()
        .sort_values(["technology_group", "region"])
    )
    metadata = {
        "era5_dirs": [str(path) for path in config.era5_dirs],
        "weather_source": config.weather_source,
        "icon_dir": str(config.icon_dir) if config.weather_source == "dwd_icon" else None,
        "required_run": config.required_run if config.weather_source == "dwd_icon" else None,
        "open_meteo_weather_file": str(config.open_meteo_weather_file) if config.weather_source == "open_meteo" else None,
        "open_meteo_base_url": config.open_meteo_base_url if config.weather_source == "open_meteo" else None,
        "open_meteo_model": config.open_meteo_model if config.weather_source == "open_meteo" else None,
        "open_meteo_api_mode": config.open_meteo_api_mode if config.weather_source == "open_meteo" else None,
        "open_meteo_single_run_hour_utc": (
            config.open_meteo_single_run_hour_utc if config.weather_source == "open_meteo" else None
        ),
        "open_meteo_single_run_forecast_days": (
            config.open_meteo_single_run_forecast_days if config.weather_source == "open_meteo" else None
        ),
        "open_meteo_request_pause_seconds": (
            config.open_meteo_request_pause_seconds if config.weather_source == "open_meteo" else None
        ),
        "open_meteo_point_selection": config.open_meteo_point_selection if config.weather_source == "open_meteo" else None,
        "open_meteo_max_points_per_cluster": (
            config.open_meteo_max_points_per_cluster if config.weather_source == "open_meteo" else None
        ),
        "open_meteo_start_date": (
            config.open_meteo_start_date.isoformat() if config.weather_source == "open_meteo" else None
        ),
        "open_meteo_end_date": (
            config.open_meteo_end_date.isoformat() if config.weather_source == "open_meteo" else None
        ),
        "capacity_file": str(config.capacity_file),
        "cluster_file": str(config.cluster_file),
        "include_cluster_features": config.include_cluster_features,
        "capacity_map_file": str(config.capacity_map_file),
        "output_file": str(config.output_file),
        "capacity_summary_mw": summary.to_dict(orient="records"),
    }
    with open(config.output_file.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(summary.to_string(index=False))
    print(f"Saved regional renewable features: {config.output_file}")
    print(f"Saved regional capacity map: {config.capacity_map_file}")
    return features
