from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd


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

            var_name = parts[0]
            df_var = pd.read_csv(folder_path / filename, comment="#", sep=",", engine="python")
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

        hourly_cols = ["timestamp"] + [col for col in df_day.columns if col.startswith(("t2m_", "u10_", "v10_"))]
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

