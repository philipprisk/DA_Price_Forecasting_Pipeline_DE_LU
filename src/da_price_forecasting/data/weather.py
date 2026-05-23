from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests


def _read_dwd_variable_name(path: Path) -> str:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            if line.startswith("# Variable:"):
                return line.split(":", 1)[1].strip()
    return path.name.replace(".csv", "").split("_")[0]


def _canonical_dwd_variable_name(raw_name: str) -> str:
    key = raw_name.strip().lower().replace("-", "_")
    compact = key.replace("_", "")

    if compact in {"aswdirs", "aswdir"}:
        return "ASWDIR"
    if compact in {"aswdifds", "aswdifd"}:
        return "ASWDIFD"
    if compact in {"t2m", "2t"}:
        return "t2m"
    if compact in {"td2m", "d2m", "2d"}:
        return "td2m"
    if compact in {"u10", "10u"}:
        return "u10"
    if compact in {"v10", "10v"}:
        return "v10"
    if compact in {"vmax10m", "vmax10", "10fg", "fg10"}:
        return "vmax10m"
    if compact in {"p", "sp", "pres", "pressure"}:
        return "sp"
    if compact in {"tp", "totprec", "totalprecipitation"}:
        return "tp"
    if compact in {"sde", "hsnow", "snowdepth"}:
        return "sde"
    if compact in {"snowgsp", "snow", "snowfall", "lsfwe"}:
        return "snow_gsp"
    return key


def load_era5(
    dirs: list[Path],
    target_tz: str = "Europe/Berlin",
) -> pd.DataFrame:
    """Load clustered ERA5 CSV files from multiple yearly directories."""
    yearly_dfs = []

    for base_dir in dirs:
        csv_files = sorted(filename for filename in os.listdir(base_dir) if filename.endswith(".csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {base_dir}")

        variable_dfs = []
        for filename in csv_files:
            var_name = filename.split("_")[0]
            df_var = pd.read_csv(
                base_dir / filename,
                comment="#",
                parse_dates=["timestamp"],
            )
            df_var = df_var.set_index("timestamp")
            df_var = df_var.rename(columns={col: f"{var_name}_{col}" for col in df_var.columns})
            variable_dfs.append(df_var)

        df_year = variable_dfs[0].copy()
        for df_next in variable_dfs[1:]:
            df_year = df_year.join(df_next, how="inner")

        yearly_dfs.append(df_year)

    df_era5 = pd.concat(yearly_dfs).sort_index()
    df_era5 = df_era5.loc[~df_era5.index.duplicated(keep="first")]

    if df_era5.index.tz is None:
        df_era5.index = df_era5.index.tz_localize("UTC")

    df_era5.index = df_era5.index.tz_convert(target_tz)
    df_era5.index.name = "timestamp"
    return df_era5


def load_dwd(
    icon_dir: Path,
    start_folder_date: date,
    required_run: str,
    skip_dates: set[date] = frozenset(),
    folder_offset_date: date = date(2025, 10, 26),
    target_tz: str = "Europe/Berlin",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load clustered DWD ICON-D2 forecast data from the daily folder structure."""
    if not icon_dir.exists():
        raise FileNotFoundError(f"ICON directory not found: {icon_dir}")

    hourly_dfs = []
    qh_dfs = []

    for folder_name in sorted(os.listdir(icon_dir)):
        folder_path = icon_dir / folder_name
        if not folder_path.is_dir():
            continue

        try:
            date_str = folder_name.split("_")[3]
            folder_date = datetime.strptime(date_str, "%Y%m%d").date()
        except Exception:
            continue

        if folder_date < start_folder_date or folder_date in skip_dates:
            continue

        forecast_date = folder_date if folder_date < folder_offset_date else folder_date + timedelta(days=1)
        variable_dfs = []

        for filename in sorted(os.listdir(folder_path)):
            if not filename.endswith(".csv"):
                continue

            parts = filename.replace(".csv", "").split("_")
            run_hour = next((token[-2:] for token in parts if token.isdigit() and len(token) == 10), "")
            if run_hour != required_run:
                continue

            file_path = folder_path / filename
            var_name = _canonical_dwd_variable_name(_read_dwd_variable_name(file_path))
            df_var = pd.read_csv(file_path, comment="#", sep=",", engine="python")
            if "timestamp" not in df_var.columns:
                raise ValueError(f"Missing 'timestamp' column in: {filename}")

            df_var = df_var.rename(columns={col: f"{var_name}_{col}" for col in df_var.columns if col.startswith("cluster_")})
            df_var["timestamp"] = pd.to_datetime(df_var["timestamp"], utc=True).dt.tz_convert(target_tz)
            variable_dfs.append(df_var)

        if not variable_dfs:
            raise ValueError(f"No CSVs with run={required_run} found in: {folder_name}")

        df_day = variable_dfs[0].copy()
        for df_next in variable_dfs[1:]:
            df_day = df_day.merge(df_next, on="timestamp", how="outer")
        df_day = df_day.sort_values("timestamp")

        start = pd.Timestamp(forecast_date, tz=target_tz)
        end = start + pd.Timedelta(days=1)

        hourly_cols = ["timestamp"] + [
            col
            for col in df_day.columns
            if col.startswith(
                (
                    "t2m_",
                    "td2m_",
                    "u10_",
                    "v10_",
                    "vmax10m_",
                    "sp_",
                    "tp_",
                    "sde_",
                    "snow_gsp_",
                )
            )
        ]
        qh_cols = ["timestamp"] + [col for col in df_day.columns if col.startswith(("ASWDIR_", "ASWDIFD_"))]

        df_hourly_day = (
            df_day[hourly_cols]
            .loc[
                (df_day["timestamp"] >= start)
                & (df_day["timestamp"] < end)
                & (df_day["timestamp"].dt.minute == 0)
            ]
            .copy()
        )

        df_qh_day = df_day[qh_cols].copy()
        df_qh_day["timestamp"] = df_qh_day["timestamp"] - pd.Timedelta(minutes=15)
        df_qh_day = df_qh_day.loc[
            (df_qh_day["timestamp"] >= start)
            & (df_qh_day["timestamp"] < end)
        ]

        hourly_dfs.append(df_hourly_day)
        qh_dfs.append(df_qh_day)

    if not hourly_dfs:
        raise ValueError("No hourly DWD data loaded.")
    if not qh_dfs:
        raise ValueError("No quarter-hourly DWD data loaded.")

    df_hourly = pd.concat(hourly_dfs, ignore_index=True).sort_values("timestamp").set_index("timestamp")
    df_qh = pd.concat(qh_dfs, ignore_index=True).sort_values("timestamp").set_index("timestamp")

    df_hourly.index.name = "timestamp"
    df_qh.index.name = "timestamp"
    return df_hourly, df_qh


def _direction_to_uv(speed_m_s: pd.Series, direction_degrees: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Convert meteorological wind direction to eastward/northward components."""
    radians = np.deg2rad(direction_degrees.astype(float))
    speed = speed_m_s.astype(float)
    u = -speed * np.sin(radians)
    v = -speed * np.cos(radians)
    return u, v


def _interpolate_100m_wind(df: pd.DataFrame) -> pd.DataFrame:
    """Build u100/v100 from Open-Meteo's 80m and 120m wind speed/direction."""
    has_80 = {"wind_speed_80m", "wind_direction_80m"} <= set(df.columns)
    has_120 = {"wind_speed_120m", "wind_direction_120m"} <= set(df.columns)

    if has_80 and has_120:
        u80, v80 = _direction_to_uv(df["wind_speed_80m"], df["wind_direction_80m"])
        u120, v120 = _direction_to_uv(df["wind_speed_120m"], df["wind_direction_120m"])
        weight = (100.0 - 80.0) / (120.0 - 80.0)
        df["u100"] = u80 + weight * (u120 - u80)
        df["v100"] = v80 + weight * (v120 - v80)
        return df

    if {"wind_speed_10m", "wind_direction_10m"} <= set(df.columns):
        df["u10"], df["v10"] = _direction_to_uv(df["wind_speed_10m"], df["wind_direction_10m"])
    return df


def _normalise_open_meteo_units(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Open-Meteo units to the conventions used by the regional feature builder."""
    df = df.copy()
    if "temperature_2m" in df.columns:
        df["t2m"] = df["temperature_2m"] + 273.15
    if "dew_point_2m" in df.columns:
        df["td2m"] = df["dew_point_2m"] + 273.15
    if "surface_pressure" in df.columns:
        # Open-Meteo returns surface pressure in hPa.
        df["sp"] = df["surface_pressure"] * 100.0
    if "shortwave_radiation" in df.columns:
        df["ssrd"] = df["shortwave_radiation"]
    elif {"direct_radiation", "diffuse_radiation"} <= set(df.columns):
        df["ssrd"] = df["direct_radiation"] + df["diffuse_radiation"]
    if "direct_radiation" in df.columns:
        df["fdir"] = df["direct_radiation"]
    if "cloud_cover" in df.columns:
        df["tcc"] = df["cloud_cover"]
    if "precipitation" in df.columns:
        df["tp"] = df["precipitation"]
    if "snow_depth" in df.columns:
        df["sde"] = df["snow_depth"]

    df = _interpolate_100m_wind(df)
    return df


def _open_meteo_response_items(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unexpected Open-Meteo response type: {type(payload)!r}")


def _fetch_open_meteo_batch(
    *,
    base_url: str,
    latitude: list[float],
    longitude: list[float],
    hourly_variables: list[str],
    model: str | None,
    cell_selection: str,
    timeout_seconds: int,
    start_date: date | None = None,
    end_date: date | None = None,
    run: str | None = None,
    forecast_days: int | None = None,
    retry_attempts: int = 5,
    retry_backoff_seconds: float = 30.0,
) -> list[dict]:
    params: dict[str, str] = {
        "latitude": ",".join(f"{value:.6f}" for value in latitude),
        "longitude": ",".join(f"{value:.6f}" for value in longitude),
        "hourly": ",".join(hourly_variables),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
        "cell_selection": cell_selection,
    }
    if run is not None:
        params["run"] = run
        if forecast_days is not None:
            params["forecast_days"] = str(forecast_days)
    else:
        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date are required unless a single model run is requested.")
        params["start_date"] = start_date.isoformat()
        params["end_date"] = end_date.isoformat()

    if model:
        params["models"] = model

    last_error: Exception | None = None
    response: requests.Response | None = None
    for attempt in range(retry_attempts + 1):
        try:
            response = requests.get(base_url, params=params, timeout=timeout_seconds)
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= retry_attempts:
                raise

            sleep_seconds = retry_backoff_seconds * (attempt + 1)
            print(
                f"[OPEN-METEO] Request failed with {type(exc).__name__}. "
                f"Retrying in {sleep_seconds:.0f}s ({attempt + 1}/{retry_attempts})..."
            )
            time.sleep(sleep_seconds)
            continue

        if response.status_code != 429:
            response.raise_for_status()
            break

        last_error = requests.HTTPError(
            f"429 Client Error: Too Many Requests for url: {response.url}",
            response=response,
        )
        if attempt >= retry_attempts:
            raise last_error

        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                sleep_seconds = float(retry_after)
            except ValueError:
                sleep_seconds = retry_backoff_seconds * (attempt + 1)
        else:
            sleep_seconds = retry_backoff_seconds * (attempt + 1)

        print(
            f"[OPEN-METEO] Rate limited (429). "
            f"Retrying in {sleep_seconds:.0f}s ({attempt + 1}/{retry_attempts})..."
        )
        time.sleep(sleep_seconds)
    else:
        if last_error is not None:
            raise last_error

    if response is None:
        raise RuntimeError("Open-Meteo request failed without a response.")

    payload = response.json()
    if isinstance(payload, dict) and "error" in payload:
        raise RuntimeError(f"Open-Meteo error: {payload.get('reason', payload)}")
    return _open_meteo_response_items(payload)


def fetch_open_meteo_cluster_weather(
    *,
    cluster_file: Path,
    start_date: date,
    end_date: date,
    output_file: Path | None = None,
    base_url: str = "https://historical-forecast-api.open-meteo.com/v1/forecast",
    model: str | None = "icon_d2",
    hourly_variables: list[str] | None = None,
    batch_size: int = 10,
    cell_selection: str = "nearest",
    timeout_seconds: int = 60,
    target_tz: str = "Europe/Berlin",
    point_selection: str = "centroid",
    max_points_per_cluster: int | None = None,
    api_mode: str = "historical_forecast",
    single_run_hour_utc: str = "06:00",
    single_run_forecast_days: int = 2,
    request_pause_seconds: float = 0.0,
    retry_attempts: int = 5,
    retry_backoff_seconds: float = 30.0,
) -> pd.DataFrame:
    """Fetch Open-Meteo weather at cluster centroids and return ICON-style columns."""
    if hourly_variables is None:
        hourly_variables = [
            "wind_speed_80m",
            "wind_direction_80m",
            "wind_speed_120m",
            "wind_direction_120m",
            "temperature_2m",
            "dew_point_2m",
            "surface_pressure",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "cloud_cover",
            "precipitation",
            "snow_depth",
        ]

    if cluster_file.suffix.lower() == ".parquet":
        clusters = pd.read_parquet(cluster_file)
    else:
        clusters = pd.read_csv(cluster_file)

    required = {"cluster_id", "lat", "lon"}
    missing = required - set(clusters.columns)
    if missing:
        raise ValueError(f"Cluster file is missing required columns: {sorted(missing)}")

    clusters = clusters[["cluster_id", "lat", "lon"]].dropna().copy()
    clusters["cluster_id"] = clusters["cluster_id"].astype(int)
    clusters = clusters.sort_values(["cluster_id", "lat", "lon"]).reset_index(drop=True)

    if clusters.empty:
        raise ValueError(f"No cluster coordinates found in: {cluster_file}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if point_selection == "centroid":
        points = (
            clusters
            .groupby("cluster_id", as_index=False)[["lat", "lon"]]
            .mean()
            .sort_values("cluster_id")
            .reset_index(drop=True)
        )
        points["point_id"] = 0
    elif point_selection == "grid_mean":
        points = clusters.copy()
        if max_points_per_cluster is not None:
            if max_points_per_cluster <= 0:
                raise ValueError("max_points_per_cluster must be positive when provided.")
            sampled_groups = []
            for _, group in points.groupby("cluster_id", sort=True):
                selected = np.linspace(
                    0,
                    len(group) - 1,
                    min(max_points_per_cluster, len(group)),
                    dtype=int,
                )
                sampled_groups.append(group.iloc[selected].copy())
            points = pd.concat(sampled_groups, ignore_index=True)
        points["point_id"] = points.groupby("cluster_id").cumcount()
    else:
        raise ValueError("point_selection must be either 'centroid' or 'grid_mean'.")

    frames = []
    if api_mode == "historical_forecast":
        request_start_date = start_date - timedelta(days=1)
        request_end_date = end_date + timedelta(days=1)
        requests_to_make: list[tuple[pd.Timestamp | None, str | None]] = [(None, None)]
    elif api_mode == "single_run":
        request_start_date = None
        request_end_date = None
        target_days = pd.date_range(
            start=pd.Timestamp(start_date, tz=target_tz),
            end=pd.Timestamp(end_date, tz=target_tz),
            freq="D",
        )
        requests_to_make = []
        for target_day in target_days:
            run_date = target_day.date() - timedelta(days=1)
            requests_to_make.append((target_day, f"{run_date.isoformat()}T{single_run_hour_utc}"))
    else:
        raise ValueError("api_mode must be either 'historical_forecast' or 'single_run'.")

    completed_weather: pd.DataFrame | None = None
    completed_days: set[pd.Timestamp] = set()
    if api_mode == "single_run" and output_file is not None and output_file.exists():
        completed_weather = pd.read_csv(output_file, index_col=0)
        completed_weather.index = pd.to_datetime(completed_weather.index, utc=True).tz_convert(target_tz)
        completed_weather = completed_weather.sort_index()
        completed_weather = completed_weather.loc[~completed_weather.index.duplicated(keep="last")]
        completed_days = {pd.Timestamp(value).normalize() for value in completed_weather.index.normalize().unique()}

    for target_day, run in requests_to_make:
        if target_day is not None and target_day.normalize() in completed_days:
            print(f"[OPEN-METEO] Skipping cached day: {target_day.date()}")
            continue

        day_frames_start = len(frames)
        for start in range(0, len(points), batch_size):
            batch = points.iloc[start:start + batch_size]
            items = _fetch_open_meteo_batch(
                base_url=base_url,
                latitude=batch["lat"].astype(float).tolist(),
                longitude=batch["lon"].astype(float).tolist(),
                start_date=request_start_date,
                end_date=request_end_date,
                run=run,
                forecast_days=single_run_forecast_days if run is not None else None,
                hourly_variables=hourly_variables,
                model=model,
                cell_selection=cell_selection,
                timeout_seconds=timeout_seconds,
                retry_attempts=retry_attempts,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            if request_pause_seconds > 0:
                time.sleep(request_pause_seconds)

            if len(items) != len(batch):
                raise ValueError(
                    "Open-Meteo returned a different number of locations than requested "
                    f"({len(items)} returned, {len(batch)} requested)."
                )

            for (_, point_row), item in zip(batch.iterrows(), items, strict=True):
                hourly = item.get("hourly")
                if not isinstance(hourly, dict) or "time" not in hourly:
                    raise ValueError(
                        f"Open-Meteo response is missing hourly time series for cluster {point_row.cluster_id}."
                    )

                df_cluster = pd.DataFrame(hourly)
                df_cluster["timestamp"] = pd.to_datetime(df_cluster["time"], utc=True).dt.tz_convert(target_tz)
                df_cluster = df_cluster.drop(columns=["time"]).set_index("timestamp")
                if target_day is not None:
                    target_start = target_day
                    target_end = target_start + pd.Timedelta(days=1)
                    df_cluster = df_cluster.loc[(df_cluster.index >= target_start) & (df_cluster.index < target_end)]
                df_cluster = _normalise_open_meteo_units(df_cluster)

                cluster_id = int(point_row.cluster_id)
                point_id = int(point_row.point_id)
                keep_columns = {
                    "u100": f"u100_cluster_{cluster_id}_point_{point_id}",
                    "v100": f"v100_cluster_{cluster_id}_point_{point_id}",
                    "u10": f"u10_cluster_{cluster_id}_point_{point_id}",
                    "v10": f"v10_cluster_{cluster_id}_point_{point_id}",
                    "t2m": f"t2m_cluster_{cluster_id}_point_{point_id}",
                    "td2m": f"td2m_cluster_{cluster_id}_point_{point_id}",
                    "sp": f"sp_cluster_{cluster_id}_point_{point_id}",
                    "ssrd": f"ssrd_cluster_{cluster_id}_point_{point_id}",
                    "fdir": f"fdir_cluster_{cluster_id}_point_{point_id}",
                    "tcc": f"tcc_cluster_{cluster_id}_point_{point_id}",
                    "tp": f"tp_cluster_{cluster_id}_point_{point_id}",
                    "sde": f"sde_cluster_{cluster_id}_point_{point_id}",
                }
                available = [column for column in keep_columns if column in df_cluster.columns]
                frames.append(df_cluster[available].rename(columns=keep_columns))

        if api_mode == "single_run" and target_day is not None and output_file is not None:
            day_weather = _average_open_meteo_point_frames(frames[day_frames_start:], points)
            if completed_weather is not None:
                completed_weather = pd.concat([completed_weather, day_weather]).sort_index()
                completed_weather = completed_weather.loc[~completed_weather.index.duplicated(keep="last")]
            else:
                completed_weather = day_weather
            output_file.parent.mkdir(parents=True, exist_ok=True)
            completed_weather.to_csv(output_file)
            completed_days.add(target_day.normalize())
            print(f"[OPEN-METEO] Cached single-run day: {target_day.date()}")

    if not frames:
        if completed_weather is not None:
            start_ts = pd.Timestamp(start_date, tz=target_tz)
            end_ts = pd.Timestamp(end_date + timedelta(days=1), tz=target_tz)
            return completed_weather.loc[(completed_weather.index >= start_ts) & (completed_weather.index < end_ts)]
        raise ValueError("No Open-Meteo weather data was fetched.")

    weather = _average_open_meteo_point_frames(frames, points)
    if completed_weather is not None:
        weather = pd.concat([completed_weather, weather]).sort_index()
        weather = weather.loc[~weather.index.duplicated(keep="last")]

    start_ts = pd.Timestamp(start_date, tz=target_tz)
    end_ts = pd.Timestamp(end_date + timedelta(days=1), tz=target_tz)
    weather = weather.loc[(weather.index >= start_ts) & (weather.index < end_ts)]

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        weather.to_csv(output_file)

    return weather


def _average_open_meteo_point_frames(frames: list[pd.DataFrame], points: pd.DataFrame) -> pd.DataFrame:
    """Average point-level Open-Meteo columns back to cluster-level weather columns."""
    weather = pd.concat(frames, axis=1).sort_index()
    weather = weather.loc[~weather.index.duplicated(keep="last")]

    averaged = {}
    for base_name in ["u100", "v100", "u10", "v10", "t2m", "td2m", "sp", "ssrd", "fdir", "tcc", "tp", "sde"]:
        for cluster_id in sorted(points["cluster_id"].unique()):
            prefix = f"{base_name}_cluster_{cluster_id}_point_"
            columns = [column for column in weather.columns if column.startswith(prefix)]
            if columns:
                averaged[f"{base_name}_cluster_{cluster_id}"] = weather[columns].mean(axis=1)

    weather = pd.DataFrame(averaged, index=weather.index)
    weather.index.name = "timestamp"
    return weather


def load_open_meteo(
    *,
    cluster_file: Path,
    start_date: date,
    end_date: date,
    cache_file: Path,
    base_url: str = "https://historical-forecast-api.open-meteo.com/v1/forecast",
    model: str | None = "icon_d2",
    hourly_variables: list[str] | None = None,
    batch_size: int = 10,
    cell_selection: str = "nearest",
    timeout_seconds: int = 60,
    target_tz: str = "Europe/Berlin",
    force_download: bool = False,
    point_selection: str = "centroid",
    max_points_per_cluster: int | None = None,
    api_mode: str = "historical_forecast",
    single_run_hour_utc: str = "06:00",
    single_run_forecast_days: int = 2,
    request_pause_seconds: float = 0.0,
    retry_attempts: int = 5,
    retry_backoff_seconds: float = 30.0,
) -> pd.DataFrame:
    """Load cached Open-Meteo cluster weather or fetch it from the API."""
    if cache_file.exists() and not force_download:
        df = pd.read_csv(cache_file, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(target_tz)
        df = df.sort_index()
        df = df.loc[~df.index.duplicated(keep="last")]
        df.index.name = "timestamp"
        if api_mode != "single_run":
            return df

        expected_days = pd.date_range(
            start=pd.Timestamp(start_date, tz=target_tz),
            end=pd.Timestamp(end_date, tz=target_tz),
            freq="D",
        )
        cached_days = pd.DatetimeIndex(df.index.normalize().unique()).sort_values()
        missing_days = expected_days.difference(cached_days)
        if missing_days.empty:
            return df

        print(
            "[OPEN-METEO] Single-run cache is incomplete: "
            f"{len(cached_days)}/{len(expected_days)} days cached. "
            f"Resuming from {missing_days.min().date()}..."
        )

    return fetch_open_meteo_cluster_weather(
        cluster_file=cluster_file,
        start_date=start_date,
        end_date=end_date,
        output_file=cache_file,
        base_url=base_url,
        model=model,
        hourly_variables=hourly_variables,
        batch_size=batch_size,
        cell_selection=cell_selection,
        timeout_seconds=timeout_seconds,
        target_tz=target_tz,
        point_selection=point_selection,
        max_points_per_cluster=max_points_per_cluster,
        api_mode=api_mode,
        single_run_hour_utc=single_run_hour_utc,
        single_run_forecast_days=single_run_forecast_days,
        request_pause_seconds=request_pause_seconds,
        retry_attempts=retry_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
    )
