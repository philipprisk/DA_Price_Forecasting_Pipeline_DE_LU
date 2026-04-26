from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from ..config import Era5AggregationConfig, load_config


def safe_unit_for_filename(units: str) -> str:
    """Convert a physical unit string into a filesystem-safe representation."""
    return str(units).replace("/", "-per-").replace(" ", "_").replace("*", "").replace("²", "2")


def build_era5_cluster_grid(cluster_df, lats, lons):
    """Map every ERA5 grid point to the nearest ICON-D2 cluster centroid."""
    icon_coords = cluster_df[["lon", "lat"]].to_numpy()
    icon_ids = cluster_df["cluster_id"].to_numpy()
    tree = cKDTree(icon_coords)

    lon_g, lat_g = np.meshgrid(lons, lats)
    era5_coords = np.column_stack([lon_g.ravel(), lat_g.ravel()])
    _, idx = tree.query(era5_coords)
    return icon_ids[idx].reshape(lat_g.shape)


def extract_main_values_and_times(da: xr.DataArray):
    """Extract raw values and timestamps from a standard ERA5 DataArray."""
    return da.values, pd.to_datetime(da.time.values)


def extract_solar_values_and_times(ds: xr.Dataset, da: xr.DataArray):
    """Flatten the (time, step) dimensions of an ERA5 solar DataArray."""
    vals = da.values
    t_count, s_count, y_count, x_count = vals.shape
    values = vals.reshape(t_count * s_count, y_count, x_count)
    times = pd.to_datetime(ds["valid_time"].values.reshape(t_count * s_count)) - pd.Timedelta(hours=1)
    return values, times


def aggregate_to_clusters(values_tll, cluster_grid, n_clusters):
    """Average ERA5 grid-point values within each cluster for every timestep."""
    t_count = values_tll.shape[0]
    out = np.full((t_count, n_clusters), np.nan)

    for cluster_id in range(n_clusters):
        mask = cluster_grid == cluster_id
        if np.any(mask):
            out[:, cluster_id] = np.nanmean(values_tll[:, mask], axis=1)

    return out


def write_icon_style_csv(outpath, times, out, var, long_name, units, n_clusters, transformation):
    """Write cluster-aggregated ERA5 data to a CSV file with a metadata header."""
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(out, index=times, columns=[f"cluster_{i}" for i in range(n_clusters)])
    df.index.name = "timestamp"

    header = [
        "# Source: ERA5",
        f"# Variable: {var}",
        f"# Long name: {long_name}",
        f"# Units: {units}",
        f"# Transformation: {transformation}",
        f"# Clusters: {n_clusters}",
    ]

    with open(outpath, "w", encoding="utf-8") as handle:
        handle.write("\n".join(header) + "\n")
        df.to_csv(handle)

    print(f"Saved {outpath.name} ({df.shape[0]} x {df.shape[1]})")


def process_dataset(ds, variables, cluster_df, n_clusters, solar_mode, start_time, end_time, output_parent: Path):
    """Process one ERA5 dataset: aggregate all variables to clusters and save CSVs."""
    lats = ds.latitude.values
    lons = ds.longitude.values
    cluster_grid = build_era5_cluster_grid(cluster_df, lats, lons)

    for var in variables:
        print(f"\nVariable: {var}")
        da = ds[var]

        if solar_mode:
            values_tll, times = extract_solar_values_and_times(ds, da)
        else:
            values_tll, times = extract_main_values_and_times(da)

        out = aggregate_to_clusters(values_tll, cluster_grid, n_clusters)
        mask = (times >= start_time) & (times <= end_time)

        if not np.any(mask):
            raise RuntimeError(f"No timestamps >= {start_time} found in variable {var}.")

        times = times[mask]
        out = out[mask]

        if solar_mode:
            out = np.nan_to_num(out) / 3600.0
            units = "W m-2"
            transformation = "hourly mean flux (ERA5 interval energy)"
        else:
            units = da.attrs.get("units", "")
            transformation = "raw"

        outname = f"{var}_{safe_unit_for_filename(units)}_{times[0].strftime('%Y%m%d%H')}.csv"
        write_icon_style_csv(
            output_parent / outname,
            times,
            out,
            var,
            da.attrs.get("long_name", var),
            units,
            n_clusters,
            transformation,
        )

        del values_tll, out
        gc.collect()


def run_aggregation(config: Era5AggregationConfig) -> None:
    """Load inputs, run processing for main and solar datasets."""
    config.output_parent.mkdir(parents=True, exist_ok=True)

    cluster_df = pd.read_parquet(config.cluster_file)
    n_clusters = int(cluster_df["cluster_id"].max()) + 1
    start_time = pd.Timestamp(config.start_time)
    end_time = pd.Timestamp(config.end_time)

    ds_main = xr.open_dataset(config.main_file, engine="cfgrib")
    process_dataset(
        ds_main,
        config.main_variables,
        cluster_df,
        n_clusters,
        solar_mode=False,
        start_time=start_time,
        end_time=end_time,
        output_parent=config.output_parent,
    )
    ds_main.close()

    ds_solar = xr.open_dataset(config.solar_file, engine="cfgrib")
    process_dataset(
        ds_solar,
        config.solar_variables,
        cluster_df,
        n_clusters,
        solar_mode=True,
        start_time=start_time,
        end_time=end_time,
        output_parent=config.output_parent,
    )
    ds_solar.close()

    print("\nERA5 -> ICON aggregation finished.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Aggregate ERA5 weather data to ICON-D2 clusters.")
    parser.add_argument("--config", type=Path, help="Optional JSON config file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for ERA5 aggregation."""
    args = parse_args(argv)
    config = load_config(args.config, Era5AggregationConfig) if args.config else Era5AggregationConfig()
    run_aggregation(config)


if __name__ == "__main__":
    main()

