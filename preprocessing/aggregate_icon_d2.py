#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_icon_d2.py
====================

Aggregates DWD ICON-D2 weather forecast data to spatial clusters and saves
the results as CSV files.

For each forecast day and model run, the script loads the raw GRIB2 files,
maps each grid point to the nearest ICON-D2 cluster centroid, and computes
cluster-averaged values for wind, temperature, and solar radiation variables.

Note: raw ICON-D2 data is not included in the repository due to file size.
Adjust LSDF_BASE and OUTPUT_PARENT in the CONFIG section to your local paths.
"""

# ====================================================================
# IMPORTS
# ====================================================================
import os
import bz2
import glob
import gc
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import KMeans
from datetime import datetime

# Optional plotting imports (only used if plot=True)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ====================================================================
# CONFIG
# ====================================================================

# Path to the raw ICON-D2 data source (SMB mount or local copy).
# Must be adjusted by the user to point to the correct location.
LSDF_BASE = Path("/Volumes/iip-projects/energy/climateData/icon_by_Max_Kleinebrahm")
ROOT_DIR_PATTERN = str(LSDF_BASE / "dwd_icon_daily_*")

# Output directory for aggregated CSVs.
# Note: aggregated data is stored outside the repository due to file size.
# Adjust to your preferred output location.
OUTPUT_PARENT = str(Path.home() / "icon_aggregated_output")

# Path to the Natural Earth 10m admin-0 shapefile.
# The shapefile is included in the repository under data/shapefile/.
SHAPEFILE_PATH = str(Path(__file__).parent.parent / "data" / "shapefile" / "ne_10m_admin_0_countries.shp")

N_CLUSTERS = 5
BUFFER_KM = 50
PLOT_CLUSTERS = False          # keep False for RAM friendliness
SKIP_EXISTING_OUTPUT = True    # skip if CSV already exists

# Optional: restrict to one day for testing (set e.g. "20250801" or None)
ONLY_DAY = None  # e.g. "20250801" or None

# Optional: restrict to one run hour (e.g. "12") or None
ONLY_RUN_HOUR = "09"  # e.g. "12" or None

START_DATE = datetime(2025, 11, 13)

VARIABLES = ["t_2m", "p", "u_10m", "v_10m", "aswdir_s", "aswdifd_s", "h_snow"]


# ====================================================================
# HELPERS
# ====================================================================

def compute_step_flux(timestamps, flux_values):
    """Convert cumulative mean/avg flux into instantaneous flux [W/m²]."""
    t = pd.to_datetime(timestamps)
    Fbar = np.asarray(flux_values, dtype=float)
    if len(Fbar) < 2:
        return Fbar

    tsec = np.asarray((t - t[0]) / np.timedelta64(1, "s"), dtype=float)
    E = Fbar * tsec
    F_inst = np.full_like(Fbar, np.nan, dtype=float)

    dt = np.diff(tsec)
    valid = dt != 0
    if np.any(valid):
        F_inst[1:][valid] = np.diff(E)[valid] / dt[valid]
    return np.clip(F_inst, 0, None)


def safe_unit_for_filename(units: str) -> str:
    """Make unit string filename-safe."""
    if units is None:
        units = ""
    return (
        str(units)
        .replace("/", "-per-")
        .replace("*", "")
        .replace(" ", "_")
        .replace("²", "2")
        .replace("°", "")
    )


def filter_points_in_germany(latitudes, longitudes, shapefile_path, buffer_km=50):
    """Return boolean mask for points inside Germany (+buffer)."""
    world = gpd.read_file(shapefile_path)
    germany = world[world["NAME"].str.lower() == "germany"]
    if germany.empty:
        raise ValueError("Germany not found in shapefile!")

    germany_m = germany.to_crs(epsg=3035)
    germany_buffered = gpd.GeoSeries(
        germany_m.buffer(buffer_km * 1000), crs=germany_m.crs
    ).to_crs(epsg=4326)

    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    points_flat = [Point(lon, lat) for lon, lat in zip(lon_grid.ravel(), lat_grid.ravel())]
    points_gdf = gpd.GeoDataFrame(geometry=points_flat, crs="EPSG:4326")

    mask = points_gdf.within(germany_buffered.unary_union)
    mask_grid = mask.values.reshape(lat_grid.shape)

    print(f"✅ {mask.sum():,} of {mask.size:,} grid points inside Germany (+{buffer_km} km)")
    return mask_grid


def cluster_german_coordinates(latitudes, longitudes, mask_grid, shapefile_path,
                              n_clusters=50, random_state=42, plot=False):
    """Cluster German grid points using KMeans and optionally plot."""
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    coords_germany = np.column_stack([lon_grid[mask_grid], lat_grid[mask_grid]])
    print(f"🟢 Clustering {len(coords_germany):,} German grid points into {n_clusters} clusters...")

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(coords_germany)
    centroids = km.cluster_centers_

    # Sort by latitude (north→south)
    sort_idx = np.argsort(centroids[:, 1])[::-1]
    centroids_sorted = centroids[sort_idx]
    label_mapping = {old: new for new, old in enumerate(sort_idx)}
    labels_ordered = np.array([label_mapping[l] for l in labels])

    print("✅ KMeans completed and ordered by latitude.")
    print(f"Cluster sizes: {np.bincount(labels_ordered)}")

    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        world = gpd.read_file(shapefile_path)
        world[world["NAME"] == "Germany"].plot(ax=ax, color="white", edgecolor="black", linewidth=0.5)

        sc = ax.scatter(
            coords_germany[:, 0], coords_germany[:, 1],
            c=labels_ordered, cmap="tab20", s=4, alpha=0.6
        )

        for i, (cx, cy) in enumerate(centroids_sorted):
            ax.text(cx, cy, str(i), fontsize=8, fontweight="bold", color="red",
                    ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, boxstyle="circle,pad=0.3"))

        plt.colorbar(sc, ax=ax, label="Cluster ID (north→south)")
        plt.title(f"KMeans Clustering of German Grid Points ({n_clusters} clusters)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

    return coords_germany, labels_ordered, centroids_sorted


def find_first_grib2_bz2(icon_run_dir: str) -> str | None:
    """Find any first GRIB2.bz2 file inside a run directory to read grid info."""
    var_dirs = sorted([d for d in glob.glob(os.path.join(icon_run_dir, "*")) if os.path.isdir(d)])
    for vdir in var_dirs:
        files = sorted(glob.glob(os.path.join(vdir, "icon-d2_germany_regular-lat-lon_single-level_*.grib2.bz2")))
        if files:
            return files[0]
    return None


def read_grid_from_one_file(grib2_bz2_path: str):
    """Read latitude/longitude arrays from a single compressed GRIB2 file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_grib = os.path.join(tmpdir, os.path.basename(grib2_bz2_path).replace(".bz2", ""))
        with bz2.open(grib2_bz2_path, "rb") as src, open(tmp_grib, "wb") as dst:
            dst.write(src.read())

        ds = xr.open_dataset(tmp_grib, engine="cfgrib")  # NOTE: no .load()
        lat = ds.latitude.values
        lon = ds.longitude.values
        ds.close()

    return np.array(lat), np.array(lon)


# ====================================================================
# STREAMING VARIABLE PROCESSING
# ====================================================================

def process_variable_streaming(
    var_dir: str,
    mask_germany: np.ndarray,
    cluster_labels: np.ndarray,
    output_dir: str,
    skip_existing: bool = True,
):
    """
    Stream one variable folder and write a single aggregated CSV.

    Iterates over all compressed GRIB2 files in *var_dir*, decompresses
    each file in a temporary directory, reads one field, aggregates
    immediately to cluster means, and discards the full grid.  Only the
    compact ``(timestamps, cluster_means)`` arrays are held in memory
    across iterations.  Cumulative-type variables (stepType avg / acc /
    mean) are converted to instantaneous flux before writing.

    Parameters
    ----------
    var_dir : str
        Path to the variable folder containing
        ``icon-d2_germany_regular-lat-lon_single-level_*.grib2.bz2`` files.
    mask_germany : numpy.ndarray of shape (lat, lon), dtype bool
        Boolean mask selecting ICON-D2 grid points inside Germany
        (plus buffer), as returned by :func:`filter_points_in_germany`.
    cluster_labels : numpy.ndarray of shape (n_germany_points,), dtype int
        Cluster ID (0-based) for every ``True`` point in the flattened
        *mask_germany*, as returned by :func:`cluster_german_coordinates`.
    output_dir : str
        Directory where the output CSV is written.  Created automatically
        if it does not exist.
    skip_existing : bool, optional
        If ``True`` (default), skip writing if the output CSV already
        exists.  Set to ``False`` to overwrite.

    Returns
    -------
    None
    """
    os.makedirs(output_dir, exist_ok=True)

    pattern = os.path.join(var_dir, "icon-d2_germany_regular-lat-lon_single-level_*.grib2.bz2")
    files = sorted(glob.glob(pattern))
    if not files:
        return

    folder_name = os.path.basename(var_dir)

    # For aggregation
    mask_flat = mask_germany.ravel()
    labels = cluster_labels
    n_clusters = int(np.max(labels)) + 1

    timestamps_list: list[pd.Timestamp] = []
    cluster_series: list[np.ndarray] = []

    # Metadata from first file
    field_name = None
    units = ""
    long_name = ""
    step_type = ""

    print(f"📦 Streaming variable folder: {folder_name} ({len(files)} files)")

    with tempfile.TemporaryDirectory() as tmpdir:
        for f in files:
            tmp_grib = os.path.join(tmpdir, os.path.basename(f).replace(".bz2", ""))
            with bz2.open(f, "rb") as src, open(tmp_grib, "wb") as dst:
                dst.write(src.read())

            ds = xr.open_dataset(tmp_grib, engine="cfgrib")  # no .load()
            fn = list(ds.data_vars)[0]
            if field_name is None:
                field_name = fn
                attrs = ds[field_name].attrs
                long_name = attrs.get("long_name", "") or ""
                units = attrs.get("units", "") or ""
                step_type = (attrs.get("GRIB_stepType", "") or "").lower()

            arr = ds[field_name].values  # loads the data array only
            vt = ds.get("valid_time", None)

            # valid_time can be scalar or array-like
            if vt is None:
                # fallback: try time coordinate
                tvals = ds.coords.get("time", None)
                if tvals is None:
                    raise ValueError(f"No valid_time/time coordinate in {f}")
                tvals = np.atleast_1d(tvals.values)
            else:
                tvals = np.atleast_1d(vt.values)

            ds.close()

            # arr can be 2D (lat, lon) or 3D (time, lat, lon)
            if arr.ndim == 2:
                arr_3d = arr[np.newaxis, ...]
            elif arr.ndim == 3:
                arr_3d = arr
            else:
                raise ValueError(f"Unexpected array shape {arr.shape} in {f}")

            if arr_3d.shape[0] != len(tvals):
                # Some cfgrib configs may yield mismatch; handle conservatively
                # Use min length to stay consistent.
                min_len = min(arr_3d.shape[0], len(tvals))
                arr_3d = arr_3d[:min_len, ...]
                tvals = tvals[:min_len]

            # Aggregate each timestep slice
            for i in range(arr_3d.shape[0]):
                ts = pd.to_datetime(tvals[i])
                timestamps_list.append(ts)

                slice_2d = arr_3d[i, :, :]
                slice_flat_germany = slice_2d.reshape(-1)[mask_flat]  # only Germany points

                means = np.full(n_clusters, np.nan, dtype=float)
                for c in range(n_clusters):
                    cmask = labels == c
                    if np.any(cmask):
                        means[c] = np.nanmean(slice_flat_germany[cmask])

                cluster_series.append(means)

                # per-timestep cleanup
                del slice_2d, slice_flat_germany, means
                gc.collect()

            # per-file cleanup
            del arr, arr_3d, tvals
            gc.collect()

    # Sort by timestamp to be safe
    ts = pd.to_datetime(timestamps_list)
    values = np.vstack(cluster_series)  # shape (T, n_clusters)

    order = np.argsort(ts.values)
    ts = ts.values[order]
    values = values[order, :]

    ts = pd.to_datetime(ts)

    # Compute instantaneous flux if needed
    if any(x in step_type for x in ["avg", "acc", "mean"]):
        print(f"⚡ Computing instantaneous flux for {field_name} (stepType={step_type})")
        values = np.apply_along_axis(lambda y: compute_step_flux(ts, y), 0, values)
        flux_suffix = "_instantaneous"
    else:
        flux_suffix = "_raw"

    unit_safe = safe_unit_for_filename(units)
    first_ts = ts[0].strftime("%Y%m%d%H") if len(ts) else "unknown"

    outname = f"{field_name}_{unit_safe}_{first_ts}{flux_suffix}.csv"
    outpath = Path(output_dir) / outname

    if skip_existing and outpath.exists():
        print(f"↩️  Skipping existing: {outname}")
        return

    df = pd.DataFrame(values, index=ts, columns=[f"cluster_{i}" for i in range(n_clusters)])
    df.index.name = "timestamp"

    header_lines = [
        f"# Folder: {folder_name}",
        f"# Variable: {field_name}",
        f"# Long name: {long_name}",
        f"# Units: {units}",
        f"# Step type: {step_type}",
        f"# Transformation: {'instantaneous flux' if flux_suffix == '_instantaneous' else 'raw'}",
        f"# Clusters: {n_clusters}",
        f"# First timestamp: {first_ts}",
    ]

    with open(outpath, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(header_lines) + "\n")
        df.to_csv(f)

    print(f"✅ Saved {outname} ({df.shape[0]} timesteps × {df.shape[1]} clusters)")

    # Final cleanup
    del df, values, ts, timestamps_list, cluster_series
    gc.collect()


# ====================================================================
# MAIN
# ====================================================================

if __name__ == "__main__":

    # Collect daily directories on or after START_DATE
    daily_dirs = []
    for d in glob.glob(ROOT_DIR_PATTERN):
        if not os.path.isdir(d):
            continue

        # Expected format: dwd_icon_daily_YYYYMMDD
        name = os.path.basename(d)
        try:
            date_str = name.split("_")[-1]
            day_date = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            print(f"⚠️  Could not parse date from folder name: {name}, skipping.")
            continue

        if day_date >= START_DATE:
            daily_dirs.append(d)

    daily_dirs = sorted(daily_dirs)



    if ONLY_DAY is not None:
        daily_dirs = [d for d in daily_dirs if os.path.basename(d).endswith(ONLY_DAY)]

    print(f"\n📅 Found {len(daily_dirs)} daily ICON folders:")
    for d in daily_dirs:
        print(f"  - {os.path.basename(d)}")

    # Shared state (computed once, reused across all days and run hours)
    mask_germany = None
    cluster_labels = None
    cluster_centroids = None
    latitude = None
    longitude = None

    for daily_dir in daily_dirs:
        day_name = os.path.basename(daily_dir)
        print(f"\n{'='*80}\n🗓️  Processing day: {day_name}\n{'='*80}")

        icon_dirs = sorted(glob.glob(os.path.join(daily_dir, "icon-d2", "*")))
        icon_dirs = [d for d in icon_dirs if os.path.isdir(d)]
        if ONLY_RUN_HOUR is not None:
            icon_dirs = [d for d in icon_dirs if os.path.basename(d) == ONLY_RUN_HOUR]

        if not icon_dirs:
            print(f"⚠️ No ICON run-hour subdirectories found in {daily_dir}, skipping.")
            continue

        for icon_dir in icon_dirs:
            run_hour = os.path.basename(icon_dir)
            print(f"\n🕐 Forecast run hour: {run_hour}")

            # Initialize grid/mask/clusters once (using any available file)
            if mask_germany is None or cluster_labels is None:
                first_file = find_first_grib2_bz2(icon_dir)
                if first_file is None:
                    print(f"⚠️ No GRIB2.bz2 files found in {icon_dir}, skipping run hour.")
                    continue

                print("🌍 Reading grid from first available file (one-time init)...")
                latitude, longitude = read_grid_from_one_file(first_file)

                print("🌍 Computing Germany mask (one-time init)...")
                mask_germany = filter_points_in_germany(
                    latitude, longitude, SHAPEFILE_PATH, buffer_km=BUFFER_KM
                )

                print("🧩 Computing KMeans clusters (one-time init)...")
                _, cluster_labels, cluster_centroids = cluster_german_coordinates(
                    latitude, longitude, mask_germany, SHAPEFILE_PATH,
                    n_clusters=N_CLUSTERS, random_state=42, plot=PLOT_CLUSTERS
                )
            else:
                print("♻️ Reusing existing grid, mask, and clusters.")

            # Output directory per day + run hour
            output_dir = os.path.join(OUTPUT_PARENT, f"{day_name}_{run_hour}")
            os.makedirs(output_dir, exist_ok=True)

            # Variable folders under this run hour
            var_dirs = sorted([d for d in glob.glob(os.path.join(icon_dir, "*")) if os.path.isdir(d)])
            if VARIABLES is not None:
                var_dirs = [d for d in var_dirs if os.path.basename(d) in set(VARIABLES)]

            print(f"📁 Found {len(var_dirs)} variable folders to process in {day_name}/{run_hour}.")

            # Stream each variable independently (RAM-friendly)
            for var_dir in var_dirs:
                try:
                    process_variable_streaming(
                        var_dir=var_dir,
                        mask_germany=mask_germany,
                        cluster_labels=cluster_labels,
                        output_dir=output_dir,
                        skip_existing=SKIP_EXISTING_OUTPUT
                    )
                except Exception as e:
                    print(f"❌ Error processing variable folder '{os.path.basename(var_dir)}' in {day_name}/{run_hour}: {e}")
                finally:
                    gc.collect()

            print(f"✅ Finished processing {day_name} — {run_hour}")

            # Optional: after each run-hour, force cleanup
            gc.collect()

    print("\n🎉 All requested ICON-D2 directories processed successfully (streaming mode).")
