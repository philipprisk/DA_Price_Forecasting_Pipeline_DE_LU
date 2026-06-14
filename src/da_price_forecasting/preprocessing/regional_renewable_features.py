from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans

from ..config import RegionalRenewableFeatureConfig
from ..data.weather import load_dwd, load_era5, load_open_meteo, load_open_meteo_points


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

    columns = ["lon", "lat", "cluster_id"]
    if "cluster_region" in clusters.columns:
        columns.append("cluster_region")
    clusters = clusters[columns].dropna(subset=["lon", "lat", "cluster_id"]).copy()
    clusters["cluster_id"] = clusters["cluster_id"].astype(int)
    if "cluster_region" in clusters.columns:
        clusters["cluster_region"] = clusters["cluster_region"].astype(str)
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


_SOLAR_STATE_TSO_PROXY_REGIONS = {
    # MaStR federal state codes. This is a state-level proxy for the four
    # German TSO solar reporting areas, not an exact control-area polygon.
    "1400": "solar_50hertz",  # Brandenburg
    "1401": "solar_50hertz",  # Berlin
    "1406": "solar_50hertz",  # Hamburg
    "1407": "solar_50hertz",  # Mecklenburg-Vorpommern
    "1413": "solar_50hertz",  # Sachsen
    "1414": "solar_50hertz",  # Sachsen-Anhalt
    "1415": "solar_50hertz",  # Thueringen
    "1409": "solar_amprion",  # Nordrhein-Westfalen
    "1410": "solar_amprion",  # Rheinland-Pfalz
    "1412": "solar_amprion",  # Saarland
    "1403": "solar_tennet",  # Bayern
    "1404": "solar_tennet",  # Bremen
    "1405": "solar_tennet",  # Hessen
    "1408": "solar_tennet",  # Niedersachsen
    "1411": "solar_tennet",  # Schleswig-Holstein
    "1402": "solar_transnetbw",  # Baden-Wuerttemberg
}


def _solar_tso_proxy_region_for_row(row: pd.Series, config: RegionalRenewableFeatureConfig) -> str | None:
    if config.federal_state_column not in row.index:
        return None
    value = row[config.federal_state_column]
    if pd.isna(value):
        return None
    try:
        state_code = str(int(float(value)))
    except (TypeError, ValueError):
        state_code = str(value).strip()
    return _SOLAR_STATE_TSO_PROXY_REGIONS.get(state_code)


def _region_for_row(row: pd.Series, config: RegionalRenewableFeatureConfig) -> str:
    technology = row["technology_group"]
    lat = float(row[config.latitude_column])
    lon = float(row[config.longitude_column])

    if technology == "wind_offshore":
        return "offshore_north"

    if technology == "solar" and config.solar_region_strategy == "state_tso_proxy":
        tso_region = _solar_tso_proxy_region_for_row(row, config)
        if tso_region is not None:
            return tso_region

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

    capacity["region"] = capacity.apply(lambda row: _region_for_row(row, config), axis=1)
    clusters = _load_clusters(config.cluster_file)
    capacity["cluster_id"] = -1

    if "cluster_region" in clusters.columns:
        for region, region_capacity in capacity.groupby("region", sort=False):
            region_clusters = clusters.loc[clusters["cluster_region"] == region].copy()
            if region_clusters.empty:
                continue
            tree = cKDTree(region_clusters[["lon", "lat"]].to_numpy(dtype=float))
            points = region_capacity[[config.longitude_column, config.latitude_column]].to_numpy(dtype=float)
            _, nearest_idx = tree.query(points)
            capacity.loc[region_capacity.index, "cluster_id"] = (
                region_clusters.iloc[nearest_idx]["cluster_id"].to_numpy(dtype=int)
            )

    missing_cluster = capacity["cluster_id"] < 0
    if missing_cluster.any():
        tree = cKDTree(clusters[["lon", "lat"]].to_numpy(dtype=float))
        points = capacity.loc[missing_cluster, [config.longitude_column, config.latitude_column]].to_numpy(dtype=float)
        _, nearest_idx = tree.query(points)
        capacity.loc[missing_cluster, "cluster_id"] = clusters.iloc[nearest_idx]["cluster_id"].to_numpy(dtype=int)

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
    if config.federal_state_column in capacity.columns:
        columns.append(config.federal_state_column)
    return capacity[columns].sort_values(["technology_group", "region", "cluster_id"]).reset_index(drop=True)


def _capacity_weather_point_count(
    technology: str,
    region: str,
    config: RegionalRenewableFeatureConfig,
) -> int:
    if technology == "solar":
        return config.capacity_weather_solar_points_per_region
    if technology == "wind_onshore":
        return config.capacity_weather_onshore_points_per_region
    if technology == "wind_offshore":
        return config.capacity_weather_offshore_points
    raise ValueError(f"Unsupported technology group for capacity weather points: {technology!r}")


def _representative_capacity_points(
    subset: pd.DataFrame,
    *,
    n_points: int,
    latitude_column: str,
    longitude_column: str,
    random_state: int,
) -> pd.DataFrame:
    grouped = (
        subset
        .groupby([latitude_column, longitude_column], as_index=False)
        .agg(capacity_mw=("capacity_mw", "sum"), source_unit_count=("capacity_mw", "size"))
    )
    grouped = grouped.loc[grouped["capacity_mw"] > 0].copy()
    if grouped.empty:
        return grouped

    n_points = min(int(n_points), len(grouped))
    if len(grouped) <= n_points:
        return grouped.rename(columns={latitude_column: "lat", longitude_column: "lon"})

    coords = grouped[[longitude_column, latitude_column]].to_numpy(dtype=float)
    weights = grouped["capacity_mw"].to_numpy(dtype=float)
    model = MiniBatchKMeans(
        n_clusters=n_points,
        random_state=random_state,
        n_init=5,
        batch_size=max(1024, n_points * 20),
        reassignment_ratio=0.0,
    )
    model.fit(coords, sample_weight=weights)
    labels = model.predict(coords)

    rows = []
    for label in range(n_points):
        mask = labels == label
        if not mask.any():
            continue
        label_weights = weights[mask]
        label_coords = coords[mask]
        rows.append(
            {
                "lat": float(np.average(label_coords[:, 1], weights=label_weights)),
                "lon": float(np.average(label_coords[:, 0], weights=label_weights)),
                "capacity_mw": float(label_weights.sum()),
                "source_unit_count": int(grouped.loc[mask, "source_unit_count"].sum()),
            }
        )

    return pd.DataFrame(rows).sort_values(["lon", "lat"]).reset_index(drop=True)


def _representative_capacity_cells(
    subset: pd.DataFrame,
    *,
    n_points: int,
    latitude_column: str,
    longitude_column: str,
    cell_size_degrees: float,
    min_cell_capacity_mw: float,
) -> pd.DataFrame:
    grouped_units = (
        subset
        .groupby([latitude_column, longitude_column], as_index=False)
        .agg(capacity_mw=("capacity_mw", "sum"), source_unit_count=("capacity_mw", "size"))
    )
    grouped_units = grouped_units.loc[grouped_units["capacity_mw"] > 0].copy()
    if grouped_units.empty:
        return grouped_units.rename(columns={latitude_column: "lat", longitude_column: "lon"})

    grouped_units["_cell_lat"] = np.round(grouped_units[latitude_column] / cell_size_degrees) * cell_size_degrees
    grouped_units["_cell_lon"] = np.round(grouped_units[longitude_column] / cell_size_degrees) * cell_size_degrees

    rows = []
    for _, cell in grouped_units.groupby(["_cell_lat", "_cell_lon"], sort=True):
        weights = cell["capacity_mw"].to_numpy(dtype=float)
        rows.append(
            {
                "lat": float(np.average(cell[latitude_column].to_numpy(dtype=float), weights=weights)),
                "lon": float(np.average(cell[longitude_column].to_numpy(dtype=float), weights=weights)),
                "capacity_mw": float(weights.sum()),
                "source_unit_count": int(cell["source_unit_count"].sum()),
            }
        )

    cells = pd.DataFrame(rows)
    if cells.empty:
        return cells

    eligible = cells.loc[cells["capacity_mw"] >= min_cell_capacity_mw].copy()
    if eligible.empty:
        eligible = cells

    n_points = min(int(n_points), len(eligible))
    return (
        eligible
        .sort_values(["capacity_mw", "source_unit_count", "lon", "lat"], ascending=[False, False, True, True])
        .head(n_points)
        .sort_values(["lon", "lat"])
        .reset_index(drop=True)
    )


def build_capacity_weather_points(
    capacity_map: pd.DataFrame,
    config: RegionalRenewableFeatureConfig,
) -> pd.DataFrame:
    """Build technology/region-specific representative weather points weighted by installed capacity."""
    rows = []
    point_id = 0
    technologies = [technology for technology in config.capacity_weather_technologies if technology]
    if not technologies:
        raise ValueError("capacity_weather_technologies must contain at least one technology.")

    for technology in technologies:
        technology_map = capacity_map.loc[capacity_map["technology_group"] == technology]
        for region in sorted(technology_map["region"].unique()):
            subset = technology_map.loc[technology_map["region"] == region].copy()
            if subset.empty:
                continue
            n_points = _capacity_weather_point_count(technology, region, config)
            if config.capacity_weather_point_strategy == "kmeans":
                representatives = _representative_capacity_points(
                    subset,
                    n_points=n_points,
                    latitude_column=config.latitude_column,
                    longitude_column=config.longitude_column,
                    random_state=config.capacity_weather_random_state + point_id,
                )
            elif config.capacity_weather_point_strategy == "capacity_cells":
                representatives = _representative_capacity_cells(
                    subset,
                    n_points=n_points,
                    latitude_column=config.latitude_column,
                    longitude_column=config.longitude_column,
                    cell_size_degrees=config.capacity_weather_cell_size_degrees,
                    min_cell_capacity_mw=config.capacity_weather_min_cell_capacity_mw,
                )
            else:
                raise ValueError(f"Unsupported capacity weather point strategy: {config.capacity_weather_point_strategy!r}")
            for _, representative in representatives.iterrows():
                rows.append(
                    {
                        "weather_point_id": point_id,
                        "technology_group": technology,
                        "region": region,
                        "lat": float(representative["lat"]),
                        "lon": float(representative["lon"]),
                        "capacity_mw": float(representative["capacity_mw"]),
                        "source_unit_count": int(representative["source_unit_count"]),
                    }
                )
                point_id += 1

    if not rows:
        raise ValueError("No capacity-weighted Open-Meteo weather points could be built.")
    return pd.DataFrame(rows).sort_values("weather_point_id").reset_index(drop=True)


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


def _assign_capacity_to_weather_points(
    capacity_map: pd.DataFrame,
    points: pd.DataFrame,
    config: RegionalRenewableFeatureConfig,
) -> pd.DataFrame:
    """Assign every unit to its nearest representative point within the same technology/region."""
    mapped = capacity_map.copy()
    mapped["weather_point_id"] = -1
    for (technology, region), subset in mapped.groupby(["technology_group", "region"], sort=True):
        group_points = points.loc[
            (points["technology_group"] == technology) & (points["region"] == region)
        ].copy()
        if group_points.empty:
            continue
        tree = cKDTree(group_points[["lon", "lat"]].to_numpy(dtype=float))
        coordinates = subset[[config.longitude_column, config.latitude_column]].to_numpy(dtype=float)
        _, nearest_idx = tree.query(coordinates)
        mapped.loc[subset.index, "weather_point_id"] = (
            group_points.iloc[nearest_idx]["weather_point_id"].to_numpy(dtype=int)
        )

    mapped = mapped.loc[mapped["weather_point_id"] >= 0].copy()
    mapped["weather_point_id"] = mapped["weather_point_id"].astype(int)
    return mapped


def _active_capacity_by_weather_point(
    capacity_map: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    *,
    technology: str,
    region: str,
) -> pd.DataFrame:
    subset = capacity_map.loc[
        (capacity_map["technology_group"] == technology) & (capacity_map["region"] == region)
    ].copy()
    point_ids = sorted(subset["weather_point_id"].dropna().astype(int).unique())
    days = pd.DatetimeIndex(timestamps.normalize().unique()).sort_values()
    result = pd.DataFrame(0.0, index=days, columns=point_ids)
    if subset.empty or not point_ids:
        return result

    first_day = days.min().tz_localize(None).normalize()
    last_day = days.max().tz_localize(None).normalize()
    subset["_commissioning_ts"] = pd.to_datetime(subset["commissioning_date"], errors="coerce").dt.normalize()
    subset["_decommissioning_ts"] = pd.to_datetime(subset["decommissioning_date"], errors="coerce").dt.normalize()
    commissioned = subset["_commissioning_ts"]
    decommissioned = subset["_decommissioning_ts"]

    initial_mask = (commissioned <= first_day) & (decommissioned.isna() | (decommissioned > first_day))
    initial = subset.loc[initial_mask].groupby("weather_point_id")["capacity_mw"].sum()
    result.loc[:, initial.index] = initial.to_numpy(dtype=float)

    event_rows = []
    commissioning_mask = (commissioned > first_day) & (commissioned <= last_day)
    for _, row in subset.loc[commissioning_mask].iterrows():
        event_rows.append((
            pd.Timestamp(row["_commissioning_ts"].date(), tz=timestamps.tz),
            row["weather_point_id"],
            row["capacity_mw"],
        ))

    decommissioning_mask = decommissioned.notna() & (decommissioned > first_day) & (decommissioned <= last_day)
    for _, row in subset.loc[decommissioning_mask].iterrows():
        event_rows.append((
            pd.Timestamp(row["_decommissioning_ts"].date(), tz=timestamps.tz),
            row["weather_point_id"],
            -row["capacity_mw"],
        ))

    if event_rows:
        events = pd.DataFrame(event_rows, columns=["date", "weather_point_id", "capacity_delta_mw"])
        event_pivot = events.pivot_table(
            index="date",
            columns="weather_point_id",
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


def _columns_for_points(df: pd.DataFrame, prefix: str, point_ids: list[int]) -> list[str]:
    return [f"{prefix}_point_{point_id}" for point_id in point_ids if f"{prefix}_point_{point_id}" in df.columns]


def _capacity_for_timestamps(capacity_daily: pd.DataFrame, timestamps: pd.DatetimeIndex, clusters: list[int]) -> np.ndarray:
    capacity = capacity_daily.reindex(timestamps.normalize()).loc[:, clusters].fillna(0.0)
    capacity.index = timestamps
    return capacity.to_numpy(dtype=float)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    denominator = weights.sum(axis=1)
    numerator = (values * weights).sum(axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0)


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean = _weighted_mean(values, weights)
    centered = values - mean[:, None]
    variance = _weighted_mean(centered**2, weights)
    return np.sqrt(np.clip(variance, 0.0, None))


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
    solar_columns: dict[str, np.ndarray] = {}
    for direct_col in direct_cols:
        cluster_id = direct_col.rsplit("_", 1)[-1]
        diffuse_col = f"ASWDIFD_cluster_{cluster_id}"
        if diffuse_col not in weather.columns:
            continue
        direct = np.clip(weather[direct_col].to_numpy(dtype=float), 0.0, None)
        diffuse = np.clip(weather[diffuse_col].to_numpy(dtype=float), 0.0, None)
        solar_columns[f"fdir_cluster_{cluster_id}"] = direct
        solar_columns[f"ssrd_cluster_{cluster_id}"] = direct + diffuse
    if solar_columns:
        weather = pd.concat([weather, pd.DataFrame(solar_columns, index=weather.index)], axis=1)

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


def _load_open_meteo_capacity_point_weather(
    config: RegionalRenewableFeatureConfig,
    points: pd.DataFrame,
) -> pd.DataFrame:
    return load_open_meteo_points(
        points=points,
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
        api_mode=config.open_meteo_api_mode,
        single_run_hour_utc=config.open_meteo_single_run_hour_utc,
        single_run_forecast_days=config.open_meteo_single_run_forecast_days,
        request_pause_seconds=config.open_meteo_request_pause_seconds,
        retry_attempts=config.open_meteo_retry_attempts,
        retry_backoff_seconds=config.open_meteo_retry_backoff_seconds,
    )


def _build_capacity_point_renewable_features(
    capacity_map: pd.DataFrame,
    points: pd.DataFrame,
    weather: pd.DataFrame,
    config: RegionalRenewableFeatureConfig,
) -> pd.DataFrame:
    assigned_capacity = _assign_capacity_to_weather_points(capacity_map, points, config)
    u_prefix = "u100" if any(column.startswith("u100_point_") for column in weather.columns) else "u10"
    v_prefix = "v100" if u_prefix == "u100" else "v10"
    wind_factor = 1.0
    if u_prefix == "u10":
        wind_factor = (config.wind_hub_height_m / config.wind_reference_height_m) ** config.wind_shear_alpha

    frames: dict[str, pd.Series] = {}
    ramp_columns: list[str] = []
    wind_total = pd.Series(0.0, index=weather.index)
    solar_total = pd.Series(0.0, index=weather.index)

    for technology in ["wind_offshore", "wind_onshore"]:
        regions = sorted(points.loc[points["technology_group"] == technology, "region"].unique())
        for region in regions:
            capacity_daily = _active_capacity_by_weather_point(
                assigned_capacity,
                weather.index,
                technology=technology,
                region=region,
            )
            point_ids = [
                int(point_id)
                for point_id in capacity_daily.columns
                if f"{u_prefix}_point_{point_id}" in weather.columns and f"{v_prefix}_point_{point_id}" in weather.columns
            ]
            if not point_ids:
                continue

            capacity = _capacity_for_timestamps(capacity_daily, weather.index, point_ids)
            u = weather[_columns_for_points(weather, u_prefix, point_ids)].to_numpy(dtype=float)
            v = weather[_columns_for_points(weather, v_prefix, point_ids)].to_numpy(dtype=float)
            speed = np.sqrt(u**2 + v**2) * wind_factor
            proxy_curve = (capacity * _normalised_wind_power_curve(speed)).sum(axis=1)
            proxy_v3 = (capacity * speed**3).sum(axis=1)

            prefix = f"wind_{region}"
            frames[f"{prefix}_proxy_mw"] = pd.Series(proxy_curve, index=weather.index)
            frames[f"{prefix}_proxy_v3"] = pd.Series(proxy_v3, index=weather.index)
            frames[f"{prefix}_speed_hub_cap_weighted_m_s"] = pd.Series(_weighted_mean(speed, capacity), index=weather.index)
            frames[f"{prefix}_speed_hub_cap_weighted_std_m_s"] = pd.Series(_weighted_std(speed, capacity), index=weather.index)
            frames[f"{prefix}_u_cap_weighted_m_s"] = pd.Series(_weighted_mean(u, capacity), index=weather.index)
            frames[f"{prefix}_v_cap_weighted_m_s"] = pd.Series(_weighted_mean(v, capacity), index=weather.index)
            direction_cos, direction_sin = _weighted_wind_direction_components(u, v, capacity)
            frames[f"{prefix}_direction_cos_cap_weighted"] = pd.Series(direction_cos, index=weather.index)
            frames[f"{prefix}_direction_sin_cap_weighted"] = pd.Series(direction_sin, index=weather.index)
            wind_total = wind_total + frames[f"{prefix}_proxy_mw"]
            ramp_columns.append(f"{prefix}_proxy_mw")

            if all(f"{wind_column}_point_{point_ids[0]}" in weather.columns for wind_column in ["u80", "v80", "u120", "v120"]):
                u80 = weather[_columns_for_points(weather, "u80", point_ids)].to_numpy(dtype=float)
                v80 = weather[_columns_for_points(weather, "v80", point_ids)].to_numpy(dtype=float)
                u120 = weather[_columns_for_points(weather, "u120", point_ids)].to_numpy(dtype=float)
                v120 = weather[_columns_for_points(weather, "v120", point_ids)].to_numpy(dtype=float)
                speed80 = np.sqrt(u80**2 + v80**2)
                speed120 = np.sqrt(u120**2 + v120**2)
                speed_ratio = np.divide(
                    speed120,
                    speed80,
                    out=np.ones_like(speed120, dtype=float),
                    where=speed80 > 1e-6,
                )
                frames[f"{prefix}_speed_80m_cap_weighted_m_s"] = pd.Series(
                    _weighted_mean(speed80, capacity),
                    index=weather.index,
                )
                frames[f"{prefix}_speed_120m_cap_weighted_m_s"] = pd.Series(
                    _weighted_mean(speed120, capacity),
                    index=weather.index,
                )
                frames[f"{prefix}_speed_120m_minus_80m_cap_weighted_m_s"] = pd.Series(
                    _weighted_mean(speed120 - speed80, capacity),
                    index=weather.index,
                )
                frames[f"{prefix}_speed_120m_to_80m_ratio_cap_weighted"] = pd.Series(
                    _weighted_mean(speed_ratio, capacity),
                    index=weather.index,
                )
                frames[f"{prefix}_proxy_v3_80m"] = pd.Series((capacity * speed80**3).sum(axis=1), index=weather.index)
                frames[f"{prefix}_proxy_v3_120m"] = pd.Series((capacity * speed120**3).sum(axis=1), index=weather.index)

                if all(f"{wind_column}_point_{point_ids[0]}" in weather.columns for wind_column in ["u180", "v180"]):
                    u180 = weather[_columns_for_points(weather, "u180", point_ids)].to_numpy(dtype=float)
                    v180 = weather[_columns_for_points(weather, "v180", point_ids)].to_numpy(dtype=float)
                    speed180 = np.sqrt(u180**2 + v180**2)
                    speed180_ratio = np.divide(
                        speed180,
                        speed120,
                        out=np.ones_like(speed180, dtype=float),
                        where=speed120 > 1e-6,
                    )
                    frames[f"{prefix}_speed_180m_cap_weighted_m_s"] = pd.Series(
                        _weighted_mean(speed180, capacity),
                        index=weather.index,
                    )
                    frames[f"{prefix}_speed_180m_cap_weighted_std_m_s"] = pd.Series(
                        _weighted_std(speed180, capacity),
                        index=weather.index,
                    )
                    frames[f"{prefix}_speed_180m_minus_120m_cap_weighted_m_s"] = pd.Series(
                        _weighted_mean(speed180 - speed120, capacity),
                        index=weather.index,
                    )
                    frames[f"{prefix}_speed_180m_to_120m_ratio_cap_weighted"] = pd.Series(
                        _weighted_mean(speed180_ratio, capacity),
                        index=weather.index,
                    )
                    frames[f"{prefix}_proxy_v3_180m"] = pd.Series(
                        (capacity * speed180**3).sum(axis=1),
                        index=weather.index,
                    )

            t2m = None
            sp = None
            if f"t2m_point_{point_ids[0]}" in weather.columns:
                t2m = weather[_columns_for_points(weather, "t2m", point_ids)].to_numpy(dtype=float)
                frames[f"{prefix}_t2m_cap_weighted_K"] = pd.Series(_weighted_mean(t2m, capacity), index=weather.index)
                frames[f"{prefix}_t2m_cap_weighted_std_K"] = pd.Series(_weighted_std(t2m, capacity), index=weather.index)
            if f"td2m_point_{point_ids[0]}" in weather.columns:
                td2m = weather[_columns_for_points(weather, "td2m", point_ids)].to_numpy(dtype=float)
                frames[f"{prefix}_td2m_cap_weighted_K"] = pd.Series(_weighted_mean(td2m, capacity), index=weather.index)
            if f"sp_point_{point_ids[0]}" in weather.columns:
                sp = weather[_columns_for_points(weather, "sp", point_ids)].to_numpy(dtype=float)
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
            if f"tp_point_{point_ids[0]}" in weather.columns:
                tp = weather[_columns_for_points(weather, "tp", point_ids)].to_numpy(dtype=float)
                frames[f"{prefix}_tot_prec_cap_weighted"] = pd.Series(_weighted_mean(tp, capacity), index=weather.index)
            if f"sde_point_{point_ids[0]}" in weather.columns:
                sde = weather[_columns_for_points(weather, "sde", point_ids)].to_numpy(dtype=float)
                frames[f"{prefix}_snow_depth_cap_weighted_m"] = pd.Series(_weighted_mean(sde, capacity), index=weather.index)
            if f"pblh_point_{point_ids[0]}" in weather.columns:
                pblh = weather[_columns_for_points(weather, "pblh", point_ids)].to_numpy(dtype=float)
                frames[f"{prefix}_boundary_layer_height_cap_weighted_m"] = pd.Series(
                    _weighted_mean(pblh, capacity),
                    index=weather.index,
                )
                frames[f"{prefix}_boundary_layer_height_cap_weighted_std_m"] = pd.Series(
                    _weighted_std(pblh, capacity),
                    index=weather.index,
                )
            if f"tcc_point_{point_ids[0]}" in weather.columns:
                tcc = weather[_columns_for_points(weather, "tcc", point_ids)].to_numpy(dtype=float)
                frames[f"{prefix}_cloud_cover_cap_weighted"] = pd.Series(_weighted_mean(tcc, capacity), index=weather.index)

    for region in sorted(points.loc[points["technology_group"] == "solar", "region"].unique()):
        capacity_daily = _active_capacity_by_weather_point(
            assigned_capacity,
            weather.index,
            technology="solar",
            region=region,
        )
        point_ids = [
            int(point_id)
            for point_id in capacity_daily.columns
            if f"ssrd_point_{point_id}" in weather.columns
        ]
        if not point_ids:
            continue

        capacity = _capacity_for_timestamps(capacity_daily, weather.index, point_ids)
        ssrd = np.clip(weather[_columns_for_points(weather, "ssrd", point_ids)].to_numpy(dtype=float), 0.0, None)
        proxy = (capacity * ssrd / 1000.0 * config.solar_performance_ratio).sum(axis=1)

        prefix = f"solar_{region.removeprefix('solar_')}"
        frames[f"{prefix}_proxy_mw"] = pd.Series(proxy, index=weather.index)
        frames[f"{prefix}_irradiance_cap_weighted_W_m2"] = pd.Series(_weighted_mean(ssrd, capacity), index=weather.index)
        frames[f"{prefix}_irradiance_cap_weighted_std_W_m2"] = pd.Series(_weighted_std(ssrd, capacity), index=weather.index)
        solar_total = solar_total + frames[f"{prefix}_proxy_mw"]
        ramp_columns.append(f"{prefix}_proxy_mw")

        fdir = None
        if f"fdir_point_{point_ids[0]}" in weather.columns:
            fdir = np.clip(weather[_columns_for_points(weather, "fdir", point_ids)].to_numpy(dtype=float), 0.0, None)
            frames[f"{prefix}_direct_irradiance_cap_weighted_W_m2"] = pd.Series(_weighted_mean(fdir, capacity), index=weather.index)
        if f"diffuse_point_{point_ids[0]}" in weather.columns:
            diffuse = np.clip(weather[_columns_for_points(weather, "diffuse", point_ids)].to_numpy(dtype=float), 0.0, None)
            frames[f"{prefix}_diffuse_irradiance_cap_weighted_W_m2"] = pd.Series(
                _weighted_mean(diffuse, capacity),
                index=weather.index,
            )
        elif fdir is not None:
            diffuse = np.clip(ssrd - fdir, 0.0, None)
        else:
            diffuse = None
        if diffuse is not None:
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
        if f"t2m_point_{point_ids[0]}" in weather.columns:
            t2m = weather[_columns_for_points(weather, "t2m", point_ids)].to_numpy(dtype=float)
            frames[f"{prefix}_t2m_cap_weighted_K"] = pd.Series(_weighted_mean(t2m, capacity), index=weather.index)
            frames[f"{prefix}_t2m_cap_weighted_std_K"] = pd.Series(_weighted_std(t2m, capacity), index=weather.index)
        if f"td2m_point_{point_ids[0]}" in weather.columns:
            td2m = weather[_columns_for_points(weather, "td2m", point_ids)].to_numpy(dtype=float)
            frames[f"{prefix}_td2m_cap_weighted_K"] = pd.Series(_weighted_mean(td2m, capacity), index=weather.index)
        if f"tp_point_{point_ids[0]}" in weather.columns:
            tp = weather[_columns_for_points(weather, "tp", point_ids)].to_numpy(dtype=float)
            frames[f"{prefix}_tot_prec_cap_weighted"] = pd.Series(_weighted_mean(tp, capacity), index=weather.index)
        if f"sde_point_{point_ids[0]}" in weather.columns:
            sde = weather[_columns_for_points(weather, "sde", point_ids)].to_numpy(dtype=float)
            frames[f"{prefix}_snow_depth_cap_weighted_m"] = pd.Series(_weighted_mean(sde, capacity), index=weather.index)
        if f"tcc_point_{point_ids[0]}" in weather.columns:
            tcc = weather[_columns_for_points(weather, "tcc", point_ids)].to_numpy(dtype=float)
            frames[f"{prefix}_cloud_cover_cap_weighted"] = pd.Series(_weighted_mean(tcc, capacity), index=weather.index)

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
        tz=result.index.tz,
        name="timestamp",
    )
    result = result.loc[~result.index.duplicated(keep="last")].reindex(full_index).ffill(limit=3)
    result.index.name = "timestamp"
    return result.astype(float)


def _build_cluster_cloud_cover_features(
    capacity_map: pd.DataFrame,
    weather: pd.DataFrame,
    config: RegionalRenewableFeatureConfig,
) -> pd.DataFrame:
    """Build compact capacity-weighted regional cloud-cover features from cluster weather."""
    cloud_prefixes = [
        ("tcc", "cloud_cover"),
        ("lcc", "low_cloud_cover"),
        ("mcc", "mid_cloud_cover"),
        ("hcc", "high_cloud_cover"),
    ]
    available_cloud_prefixes = [
        (cloud_prefix, feature_name)
        for cloud_prefix, feature_name in cloud_prefixes
        if any(column.startswith(f"{cloud_prefix}_cluster_") for column in weather.columns)
    ]
    if not available_cloud_prefixes:
        raise ValueError("Cloud-cover feature mode requires at least one *_cluster_* cloud-cover weather column.")

    frames: dict[str, pd.Series] = {}
    ramp_columns: list[str] = []
    for technology in sorted(capacity_map["technology_group"].unique()):
        if technology not in {"solar", "wind_onshore", "wind_offshore"}:
            continue
        for region in sorted(capacity_map.loc[capacity_map["technology_group"] == technology, "region"].unique()):
            capacity_daily = _active_capacity_by_cluster(capacity_map, weather.index, technology=technology, region=region)
            clusters = [
                cluster_id
                for cluster_id in capacity_daily.columns
                if any(f"{cloud_prefix}_cluster_{cluster_id}" in weather.columns for cloud_prefix, _ in available_cloud_prefixes)
            ]
            if not clusters:
                continue

            capacity = _capacity_for_timestamps(capacity_daily, weather.index, clusters)
            prefix = f"{technology}_{region.removeprefix('solar_').removeprefix('onshore_').removeprefix('offshore_')}"
            if technology == "wind_offshore":
                prefix = f"wind_{region}"
            elif technology == "wind_onshore":
                prefix = f"wind_{region}"

            for cloud_prefix, feature_name in available_cloud_prefixes:
                cloud_clusters = [cluster_id for cluster_id in clusters if f"{cloud_prefix}_cluster_{cluster_id}" in weather.columns]
                if not cloud_clusters:
                    continue
                cloud_capacity = _capacity_for_timestamps(capacity_daily, weather.index, cloud_clusters)
                cloud = weather[_columns_for_clusters(weather, cloud_prefix, cloud_clusters)].to_numpy(dtype=float)
                mean_name = f"{prefix}_{feature_name}_cap_weighted"
                frames[mean_name] = pd.Series(_weighted_mean(cloud, cloud_capacity), index=weather.index)
                frames[f"{prefix}_{feature_name}_cap_weighted_std"] = pd.Series(
                    _weighted_std(cloud, cloud_capacity),
                    index=weather.index,
                )
                frames[f"{prefix}_{feature_name}_cap_weighted_min"] = pd.Series(np.nanmin(cloud, axis=1), index=weather.index)
                frames[f"{prefix}_{feature_name}_cap_weighted_max"] = pd.Series(np.nanmax(cloud, axis=1), index=weather.index)
                ramp_columns.append(mean_name)

    if not frames:
        raise ValueError("No cloud-cover regional features could be built.")

    result = pd.DataFrame(frames, index=weather.index).sort_index()
    if config.include_ramps:
        result = _add_ramps(result, ramp_columns)
    if len(result.index) > 1:
        observed_steps = result.index.to_series().diff().dropna()
        if not observed_steps.empty and observed_steps.min() >= pd.Timedelta(hours=1):
            full_index = pd.date_range(
                start=result.index.min(),
                end=result.index.max() + pd.Timedelta(minutes=45),
                freq="15min",
                tz=result.index.tz,
                name="timestamp",
            )
            result = result.loc[~result.index.duplicated(keep="last")].reindex(full_index).ffill(limit=3)
    result.index.name = "timestamp"
    return result.astype(float)


def build_regional_renewable_features(
    config: RegionalRenewableFeatureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    capacity_map = build_capacity_map(config)
    if config.weather_source == "open_meteo" and config.open_meteo_point_source == "capacity":
        capacity_points = build_capacity_weather_points(capacity_map, config)
        weather = _load_open_meteo_capacity_point_weather(config, capacity_points)
        features = _build_capacity_point_renewable_features(capacity_map, capacity_points, weather, config)
        return features, capacity_map, capacity_points

    weather = _load_weather(config)
    if config.feature_mode == "cloud_cover":
        features = _build_cluster_cloud_cover_features(capacity_map, weather, config)
        return features, capacity_map, None

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
        if config.include_spatial_spread_features:
            frames[f"{prefix}_irradiance_cap_weighted_std_W_m2"] = pd.Series(
                _weighted_std(ssrd, capacity),
                index=weather.index,
            )
        solar_total = solar_total + frames[f"{prefix}_proxy_mw"]
        ramp_columns.append(f"{prefix}_proxy_mw")

        if "fdir_cluster_0" in weather.columns:
            fdir = np.clip(weather[_columns_for_clusters(weather, "fdir", clusters)].to_numpy(dtype=float), 0.0, None)
            frames[f"{prefix}_direct_irradiance_cap_weighted_W_m2"] = pd.Series(_weighted_mean(fdir, capacity), index=weather.index)
            diffuse = np.clip(ssrd - fdir, 0.0, None)
            if config.include_spatial_spread_features:
                frames[f"{prefix}_direct_irradiance_cap_weighted_std_W_m2"] = pd.Series(
                    _weighted_std(fdir, capacity),
                    index=weather.index,
                )
                frames[f"{prefix}_diffuse_irradiance_cap_weighted_W_m2"] = pd.Series(
                    _weighted_mean(diffuse, capacity),
                    index=weather.index,
                )
                frames[f"{prefix}_diffuse_irradiance_cap_weighted_std_W_m2"] = pd.Series(
                    _weighted_std(diffuse, capacity),
                    index=weather.index,
                )
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
            if config.include_spatial_spread_features:
                frames[f"{prefix}_diffuse_share_cap_weighted_std"] = pd.Series(
                    _weighted_std(diffuse_share, capacity),
                    index=weather.index,
                )
        if "t2m_cluster_0" in weather.columns:
            t2m = weather[_columns_for_clusters(weather, "t2m", clusters)].to_numpy(dtype=float)
            frames[f"{prefix}_t2m_cap_weighted_K"] = pd.Series(_weighted_mean(t2m, capacity), index=weather.index)
            if config.include_spatial_spread_features:
                frames[f"{prefix}_t2m_cap_weighted_std_K"] = pd.Series(_weighted_std(t2m, capacity), index=weather.index)
        if "td2m_cluster_0" in weather.columns:
            td2m = weather[_columns_for_clusters(weather, "td2m", clusters)].to_numpy(dtype=float)
            frames[f"{prefix}_td2m_cap_weighted_K"] = pd.Series(_weighted_mean(td2m, capacity), index=weather.index)
            if config.include_spatial_spread_features:
                frames[f"{prefix}_td2m_cap_weighted_std_K"] = pd.Series(_weighted_std(td2m, capacity), index=weather.index)
        if "tp_cluster_0" in weather.columns:
            tp = weather[_columns_for_clusters(weather, "tp", clusters)].to_numpy(dtype=float)
            frames[f"{prefix}_tot_prec_cap_weighted"] = pd.Series(_weighted_mean(tp, capacity), index=weather.index)
            if config.include_spatial_spread_features:
                frames[f"{prefix}_tot_prec_cap_weighted_std"] = pd.Series(_weighted_std(tp, capacity), index=weather.index)
        if "sde_cluster_0" in weather.columns:
            sde = weather[_columns_for_clusters(weather, "sde", clusters)].to_numpy(dtype=float)
            frames[f"{prefix}_snow_depth_cap_weighted_m"] = pd.Series(_weighted_mean(sde, capacity), index=weather.index)
            if config.include_spatial_spread_features:
                frames[f"{prefix}_snow_depth_cap_weighted_std_m"] = pd.Series(_weighted_std(sde, capacity), index=weather.index)
        if "snow_gsp_cluster_0" in weather.columns:
            snow_gsp = weather[_columns_for_clusters(weather, "snow_gsp", clusters)].to_numpy(dtype=float)
            frames[f"{prefix}_snow_gsp_cap_weighted"] = pd.Series(_weighted_mean(snow_gsp, capacity), index=weather.index)
            if config.include_spatial_spread_features:
                frames[f"{prefix}_snow_gsp_cap_weighted_std"] = pd.Series(
                    _weighted_std(snow_gsp, capacity),
                    index=weather.index,
                )
        if "tcc_cluster_0" in weather.columns:
            tcc = weather[_columns_for_clusters(weather, "tcc", clusters)].to_numpy(dtype=float)
            frames[f"{prefix}_cloud_cover_cap_weighted"] = pd.Series(_weighted_mean(tcc, capacity), index=weather.index)
            if config.include_spatial_spread_features:
                frames[f"{prefix}_cloud_cover_cap_weighted_std"] = pd.Series(_weighted_std(tcc, capacity), index=weather.index)
        for cloud_prefix, feature_suffix in [
            ("lcc", "low_cloud_cover"),
            ("mcc", "mid_cloud_cover"),
            ("hcc", "high_cloud_cover"),
        ]:
            if f"{cloud_prefix}_cluster_0" not in weather.columns:
                continue
            cloud = weather[_columns_for_clusters(weather, cloud_prefix, clusters)].to_numpy(dtype=float)
            frames[f"{prefix}_{feature_suffix}_cap_weighted"] = pd.Series(_weighted_mean(cloud, capacity), index=weather.index)
            if config.include_spatial_spread_features:
                frames[f"{prefix}_{feature_suffix}_cap_weighted_std"] = pd.Series(
                    _weighted_std(cloud, capacity),
                    index=weather.index,
                )

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
        tz=result.index.tz,
        name="timestamp",
    )
    result = result.loc[~result.index.duplicated(keep="last")].reindex(full_index).ffill(limit=3)
    result.index.name = "timestamp"
    return result.astype(float), capacity_map, None


def run_regional_renewable_features(config: RegionalRenewableFeatureConfig) -> pd.DataFrame:
    features, capacity_map, capacity_points = build_regional_renewable_features(config)
    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    config.capacity_map_file.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(config.output_file)
    capacity_map.to_csv(config.capacity_map_file, index=False)
    if capacity_points is not None:
        config.capacity_weather_point_file.parent.mkdir(parents=True, exist_ok=True)
        capacity_points.to_csv(config.capacity_weather_point_file, index=False)

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
        "open_meteo_point_source": config.open_meteo_point_source if config.weather_source == "open_meteo" else None,
        "capacity_weather_point_strategy": (
            config.capacity_weather_point_strategy
            if config.weather_source == "open_meteo" and config.open_meteo_point_source == "capacity"
            else None
        ),
        "capacity_weather_cell_size_degrees": (
            config.capacity_weather_cell_size_degrees
            if config.weather_source == "open_meteo" and config.open_meteo_point_source == "capacity"
            else None
        ),
        "capacity_weather_min_cell_capacity_mw": (
            config.capacity_weather_min_cell_capacity_mw
            if config.weather_source == "open_meteo" and config.open_meteo_point_source == "capacity"
            else None
        ),
        "open_meteo_max_points_per_cluster": (
            config.open_meteo_max_points_per_cluster if config.weather_source == "open_meteo" else None
        ),
        "capacity_weather_point_file": (
            str(config.capacity_weather_point_file)
            if config.weather_source == "open_meteo" and config.open_meteo_point_source == "capacity"
            else None
        ),
        "capacity_weather_point_count": int(len(capacity_points)) if capacity_points is not None else None,
        "open_meteo_start_date": (
            config.open_meteo_start_date.isoformat() if config.weather_source == "open_meteo" else None
        ),
        "open_meteo_end_date": (
            config.open_meteo_end_date.isoformat() if config.weather_source == "open_meteo" else None
        ),
        "capacity_file": str(config.capacity_file),
        "cluster_file": str(config.cluster_file),
        "solar_region_strategy": config.solar_region_strategy,
        "federal_state_column": config.federal_state_column,
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
    if capacity_points is not None:
        print(f"Saved capacity-weighted weather points: {config.capacity_weather_point_file}")
    return features
