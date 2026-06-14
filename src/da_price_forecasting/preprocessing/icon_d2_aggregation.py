from __future__ import annotations

import argparse
import bz2
from datetime import datetime
import gc
import glob
import os
from pathlib import Path
import tempfile

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point, box
from sklearn.cluster import KMeans
import xarray as xr

from ..config import IconAggregationConfig, load_config


def compute_step_flux(timestamps, flux_values):
    """Convert cumulative mean/avg flux into instantaneous flux [W/m2]."""
    times = pd.to_datetime(timestamps)
    fbar = np.asarray(flux_values, dtype=float)
    if len(fbar) < 2:
        return fbar

    tsec = np.asarray((times - times[0]) / np.timedelta64(1, "s"), dtype=float)
    energy = fbar * tsec
    f_inst = np.full_like(fbar, np.nan, dtype=float)

    dt = np.diff(tsec)
    valid = dt != 0
    if np.any(valid):
        f_inst[1:][valid] = np.diff(energy)[valid] / dt[valid]
    return np.clip(f_inst, 0, None)


def safe_unit_for_filename(units: str) -> str:
    """Make unit string filename-safe."""
    if units is None:
        units = ""
    return str(units).replace("/", "-per-").replace("*", "").replace(" ", "_").replace("²", "2").replace("°", "")


def load_world_geometries(shapefile_path):
    """Load shapefile geometries, restoring a missing .shx index if needed."""
    shapefile_path = Path(shapefile_path)
    if not shapefile_path.exists():
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")

    previous_restore = os.environ.get("SHAPE_RESTORE_SHX")
    os.environ["SHAPE_RESTORE_SHX"] = "YES"
    try:
        world = gpd.read_file(shapefile_path)
    finally:
        if previous_restore is None:
            os.environ.pop("SHAPE_RESTORE_SHX", None)
        else:
            os.environ["SHAPE_RESTORE_SHX"] = previous_restore

    if world.empty:
        raise ValueError(f"No geometries found in shapefile: {shapefile_path}")
    if world.crs is None:
        world = world.set_crs(epsg=4326)
    return world


def load_germany_geometry(shapefile_path):
    """Load Germany geometry, with a geometry-only fallback if attributes are missing."""
    world = load_world_geometries(shapefile_path)

    if "NAME" in world.columns:
        germany = world[world["NAME"].fillna("").str.lower() == "germany"].copy()
        if not germany.empty:
            return germany

    print("Germany shapefile attributes unavailable; using geometry-based fallback.")

    germany_probe = Point(10.4515, 51.1657)
    germany_bbox = box(5.5, 47.0, 15.5, 55.5)

    germany = world[world.geometry.contains(germany_probe)].copy()
    if germany.empty:
        germany = world[world.geometry.intersects(germany_probe.buffer(0.25))].copy()

    if germany.empty:
        overlaps = world.geometry.intersection(germany_bbox).area.fillna(0)
        germany = world.loc[[overlaps.idxmax()]].copy() if (overlaps > 0).any() else world.iloc[0:0].copy()

    if germany.empty:
        raise ValueError(
            "Germany not found in shapefile. Please provide the full Natural Earth country shapefile, "
            "including its attribute files."
        )

    if len(germany) > 1:
        overlaps = germany.geometry.intersection(germany_bbox).area.fillna(0)
        germany = germany.loc[[overlaps.idxmax()]].copy()

    return germany


def filter_points_in_germany(latitudes, longitudes, shapefile_path, buffer_km=50):
    """Return boolean mask for points inside Germany (+buffer)."""
    germany = load_germany_geometry(shapefile_path)
    germany_m = germany.to_crs(epsg=3035)
    germany_buffered = gpd.GeoSeries(germany_m.buffer(buffer_km * 1000), crs=germany_m.crs).to_crs(epsg=4326)

    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    points_flat = [Point(lon, lat) for lon, lat in zip(lon_grid.ravel(), lat_grid.ravel())]
    points_gdf = gpd.GeoDataFrame(geometry=points_flat, crs="EPSG:4326")

    mask = points_gdf.within(germany_buffered.unary_union)
    mask_grid = mask.values.reshape(lat_grid.shape)
    print(f"{mask.sum():,} of {mask.size:,} grid points inside Germany (+{buffer_km} km)")
    return mask_grid


def cluster_german_coordinates(
    latitudes,
    longitudes,
    mask_grid,
    shapefile_path,
    n_clusters=50,
    random_state=42,
    plot=False,
):
    """Cluster German grid points using KMeans and optionally plot."""
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    coords_germany = np.column_stack([lon_grid[mask_grid], lat_grid[mask_grid]])
    print(f"Clustering {len(coords_germany):,} German grid points into {n_clusters} clusters...")

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(coords_germany)
    centroids = km.cluster_centers_

    sort_idx = np.argsort(centroids[:, 1])[::-1]
    centroids_sorted = centroids[sort_idx]
    label_mapping = {old: new for new, old in enumerate(sort_idx)}
    labels_ordered = np.array([label_mapping[label] for label in labels])

    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        germany = load_germany_geometry(shapefile_path)
        germany.plot(ax=ax, color="white", edgecolor="black", linewidth=0.5)

        sc = ax.scatter(coords_germany[:, 0], coords_germany[:, 1], c=labels_ordered, cmap="tab20", s=4, alpha=0.6)
        for idx, (cx, cy) in enumerate(centroids_sorted):
            ax.text(
                cx,
                cy,
                str(idx),
                fontsize=8,
                fontweight="bold",
                color="red",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, boxstyle="circle,pad=0.3"),
            )
        plt.colorbar(sc, ax=ax, label="Cluster ID (north->south)")
        plt.title(f"KMeans Clustering of German Grid Points ({n_clusters} clusters)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

    return coords_germany, labels_ordered, centroids_sorted


def _normalise_label(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


_SOLAR_STATE_TSO_PROXY_REGIONS = {
    # MaStR federal state codes. This is a state-level proxy for German
    # solar control-area boundaries until a true TSO polygon source is added.
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

_SOLAR_TSO_REGION_ORDER = [
    "solar_50hertz",
    "solar_amprion",
    "solar_tennet",
    "solar_transnetbw",
]


def _solar_tso_proxy_region(value: object) -> str | None:
    if pd.isna(value):
        return None
    try:
        state_code = str(int(float(value)))
    except (TypeError, ValueError):
        state_code = str(value).strip()
    return _SOLAR_STATE_TSO_PROXY_REGIONS.get(state_code)


def _load_capacity_points(
    config: IconAggregationConfig,
    *,
    technology_values: list[str],
    label: str,
) -> pd.DataFrame:
    capacity = pd.read_csv(config.capacity_file)
    required = {
        config.technology_column,
        config.capacity_column,
        config.latitude_column,
        config.longitude_column,
    }
    missing = required - set(capacity.columns)
    if missing:
        raise ValueError(f"Capacity file is missing required columns: {sorted(missing)}")

    selected_values = {_normalise_label(value) for value in technology_values}
    capacity = capacity.copy()
    capacity["_technology"] = capacity[config.technology_column].map(_normalise_label)
    capacity = capacity.loc[capacity["_technology"].isin(selected_values)].copy()
    capacity["capacity_mw"] = pd.to_numeric(capacity[config.capacity_column], errors="coerce")
    capacity = capacity.dropna(subset=["capacity_mw", config.latitude_column, config.longitude_column])
    capacity = capacity.loc[capacity["capacity_mw"] > 0].copy()

    if config.operating_status_column in capacity.columns and config.active_status_codes:
        active_codes = {str(code) for code in config.active_status_codes}
        capacity = capacity.loc[capacity[config.operating_status_column].astype(str).isin(active_codes)].copy()

    if capacity.empty:
        raise ValueError(f"No active {label} capacity rows found for MaStR clustering.")

    grouped = (
        capacity
        .groupby([config.latitude_column, config.longitude_column], as_index=False)
        .agg(capacity_mw=("capacity_mw", "sum"))
        .rename(columns={config.latitude_column: "lat", config.longitude_column: "lon"})
    )
    grouped = grouped.loc[grouped["capacity_mw"] > 0].copy()
    return grouped.reset_index(drop=True)


def _load_solar_capacity_points(config: IconAggregationConfig) -> pd.DataFrame:
    return _load_capacity_points(config, technology_values=config.solar_values, label="solar")


def _load_solar_tso_capacity_points(config: IconAggregationConfig) -> pd.DataFrame:
    capacity = pd.read_csv(config.capacity_file)
    required = {
        config.technology_column,
        config.capacity_column,
        config.latitude_column,
        config.longitude_column,
        config.federal_state_column,
    }
    missing = required - set(capacity.columns)
    if missing:
        raise ValueError(f"Capacity file is missing required columns for TSO solar clustering: {sorted(missing)}")

    selected_values = {_normalise_label(value) for value in config.solar_values}
    capacity = capacity.copy()
    capacity["_technology"] = capacity[config.technology_column].map(_normalise_label)
    capacity = capacity.loc[capacity["_technology"].isin(selected_values)].copy()
    capacity["capacity_mw"] = pd.to_numeric(capacity[config.capacity_column], errors="coerce")
    capacity["tso_region"] = capacity[config.federal_state_column].map(_solar_tso_proxy_region)
    capacity = capacity.dropna(
        subset=["capacity_mw", config.latitude_column, config.longitude_column, "tso_region"]
    )
    capacity = capacity.loc[capacity["capacity_mw"] > 0].copy()

    if config.operating_status_column in capacity.columns and config.active_status_codes:
        active_codes = {str(code) for code in config.active_status_codes}
        capacity = capacity.loc[capacity[config.operating_status_column].astype(str).isin(active_codes)].copy()

    if capacity.empty:
        raise ValueError("No active solar capacity rows with TSO proxy regions found for MaStR clustering.")

    grouped = (
        capacity
        .groupby(["tso_region", config.latitude_column, config.longitude_column], as_index=False)
        .agg(capacity_mw=("capacity_mw", "sum"))
        .rename(columns={config.latitude_column: "lat", config.longitude_column: "lon"})
    )
    grouped = grouped.loc[grouped["capacity_mw"] > 0].copy()
    return grouped.reset_index(drop=True)


def _load_wind_capacity_points(config: IconAggregationConfig) -> pd.DataFrame:
    wind_values = [*config.wind_onshore_values, *config.wind_offshore_values]
    return _load_capacity_points(config, technology_values=wind_values, label="wind")


def cluster_mastr_capacity_coordinates(
    latitudes,
    longitudes,
    mask_grid,
    config: IconAggregationConfig,
    *,
    technology: str,
    capacity: pd.DataFrame,
):
    """Cluster DWD grid cells around MaStR capacity, weighted by installed MW."""
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    coords_germany = np.column_stack([lon_grid[mask_grid], lat_grid[mask_grid]])
    n_clusters = min(int(config.n_clusters), len(capacity))

    unit_coords = capacity[["lon", "lat"]].to_numpy(dtype=float)
    unit_weights = capacity["capacity_mw"].to_numpy(dtype=float)

    print(
        "Clustering "
        f"{len(capacity):,} {technology} capacity coordinates into {n_clusters} MaStR-weighted clusters..."
    )
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    km.fit(unit_coords, sample_weight=unit_weights)
    centroids = km.cluster_centers_

    sort_idx = np.argsort(centroids[:, 1])[::-1]
    centroids_sorted = centroids[sort_idx]
    center_tree = cKDTree(centroids_sorted)
    _, labels_ordered = center_tree.query(coords_germany)
    labels_ordered = labels_ordered.astype(int)

    grid_tree = cKDTree(coords_germany)
    _, nearest_grid_idx = grid_tree.query(unit_coords)
    capacity_weights = np.bincount(
        nearest_grid_idx,
        weights=unit_weights,
        minlength=len(coords_germany),
    ).astype(float)

    print(
        "Mapped "
        f"{capacity_weights.sum():,.1f} MW {technology} capacity to {np.count_nonzero(capacity_weights):,} "
        "DWD grid cells for weighted aggregation."
    )
    return coords_germany, labels_ordered, centroids_sorted, capacity_weights


def cluster_mastr_solar_capacity_coordinates(
    latitudes,
    longitudes,
    mask_grid,
    config: IconAggregationConfig,
):
    """Cluster DWD grid cells around MaStR solar capacity, weighted by installed MW."""
    return cluster_mastr_capacity_coordinates(
        latitudes,
        longitudes,
        mask_grid,
        config,
        technology="solar",
        capacity=_load_solar_capacity_points(config),
    )


def cluster_mastr_solar_tso_capacity_coordinates(
    latitudes,
    longitudes,
    mask_grid,
    config: IconAggregationConfig,
):
    """Build separate MaStR solar-capacity clusters inside each TSO proxy region."""
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    coords_germany = np.column_stack([lon_grid[mask_grid], lat_grid[mask_grid]])
    capacity = _load_solar_tso_capacity_points(config)
    n_per_region = int(config.n_clusters)

    centroids_by_region: list[np.ndarray] = []
    regions_by_cluster: list[str] = []
    cluster_offset = 0
    for region in _SOLAR_TSO_REGION_ORDER:
        region_capacity = capacity.loc[capacity["tso_region"] == region].copy()
        if region_capacity.empty:
            continue
        n_clusters = min(n_per_region, len(region_capacity))
        unit_coords = region_capacity[["lon", "lat"]].to_numpy(dtype=float)
        unit_weights = region_capacity["capacity_mw"].to_numpy(dtype=float)
        print(
            f"Clustering {len(region_capacity):,} {region} solar capacity coordinates into "
            f"{n_clusters} MaStR-weighted clusters..."
        )
        km = KMeans(n_clusters=n_clusters, random_state=42 + cluster_offset, n_init="auto")
        km.fit(unit_coords, sample_weight=unit_weights)
        centroids = km.cluster_centers_
        sort_idx = np.lexsort((centroids[:, 0], -centroids[:, 1]))
        centroids_sorted = centroids[sort_idx]
        centroids_by_region.append(centroids_sorted)
        regions_by_cluster.extend([region] * len(centroids_sorted))
        cluster_offset += len(centroids_sorted)

    if not centroids_by_region:
        raise ValueError("No TSO solar clusters could be built.")

    centroids_all = np.vstack(centroids_by_region)
    center_tree = cKDTree(centroids_all)
    _, labels_ordered = center_tree.query(coords_germany)
    labels_ordered = labels_ordered.astype(int)

    grid_tree = cKDTree(coords_germany)
    capacity_weights = np.zeros(len(coords_germany), dtype=float)
    region_capacity_by_grid = {
        region: np.zeros(len(coords_germany), dtype=float)
        for region in _SOLAR_TSO_REGION_ORDER
    }
    centroids_by_region_with_ids = {
        region: np.array(
            [idx for idx, cluster_region in enumerate(regions_by_cluster) if cluster_region == region],
            dtype=int,
        )
        for region in _SOLAR_TSO_REGION_ORDER
    }

    for region, region_capacity in capacity.groupby("tso_region", sort=False):
        unit_coords = region_capacity[["lon", "lat"]].to_numpy(dtype=float)
        unit_weights = region_capacity["capacity_mw"].to_numpy(dtype=float)
        _, nearest_grid_idx = grid_tree.query(unit_coords)
        region_capacity_by_grid[region] += np.bincount(
            nearest_grid_idx,
            weights=unit_weights,
            minlength=len(coords_germany),
        ).astype(float)
        capacity_weights += np.bincount(
            nearest_grid_idx,
            weights=unit_weights,
            minlength=len(coords_germany),
        ).astype(float)

    region_weight_matrix = np.vstack([region_capacity_by_grid[region] for region in _SOLAR_TSO_REGION_ORDER])
    dominant_region_idx = region_weight_matrix.argmax(axis=0)
    has_capacity = region_weight_matrix.sum(axis=0) > 0
    for region_idx, region in enumerate(_SOLAR_TSO_REGION_ORDER):
        grid_idx = np.flatnonzero(has_capacity & (dominant_region_idx == region_idx))
        cluster_ids = centroids_by_region_with_ids.get(region)
        if len(grid_idx) == 0 or cluster_ids is None or len(cluster_ids) == 0:
            continue
        tree = cKDTree(centroids_all[cluster_ids])
        _, local_labels = tree.query(coords_germany[grid_idx])
        labels_ordered[grid_idx] = cluster_ids[local_labels]

    print(
        "Mapped "
        f"{capacity_weights.sum():,.1f} MW solar capacity to {np.count_nonzero(capacity_weights):,} "
        f"DWD grid cells across {len(regions_by_cluster)} TSO-specific clusters."
    )
    return coords_germany, labels_ordered, centroids_all, capacity_weights, np.asarray(regions_by_cluster)


def cluster_mastr_wind_capacity_coordinates(
    latitudes,
    longitudes,
    mask_grid,
    config: IconAggregationConfig,
):
    """Cluster DWD grid cells around MaStR wind capacity, weighted by installed MW."""
    return cluster_mastr_capacity_coordinates(
        latitudes,
        longitudes,
        mask_grid,
        config,
        technology="wind",
        capacity=_load_wind_capacity_points(config),
    )


def save_cluster_map(
    output_file: Path,
    coords_germany: np.ndarray,
    cluster_labels: np.ndarray,
    capacity_weights: np.ndarray | None = None,
    cluster_regions: np.ndarray | None = None,
) -> None:
    """Save the grid-cell-to-cluster mapping used for later renewable feature construction."""
    data = {
        "lon": coords_germany[:, 0],
        "lat": coords_germany[:, 1],
        "cluster_id": cluster_labels.astype(int),
    }
    if capacity_weights is not None:
        data["capacity_weight_mw"] = capacity_weights
    if cluster_regions is not None:
        data["cluster_region"] = [
            str(cluster_regions[int(cluster_id)])
            for cluster_id in cluster_labels
        ]
    df = pd.DataFrame(data).sort_values(["cluster_id", "lat", "lon"]).reset_index(drop=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.suffix.lower() == ".parquet":
        df.to_parquet(output_file, index=False)
    else:
        df.to_csv(output_file, index=False)
    print(f"Saved cluster map: {output_file} ({len(df):,} grid cells)")


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

        ds = xr.open_dataset(tmp_grib, engine="cfgrib")
        lat = ds.latitude.values
        lon = ds.longitude.values
        ds.close()

    return np.array(lat), np.array(lon)


def aggregate_cluster_values(
    values: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    cluster_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Aggregate one flattened weather field to clusters, optionally capacity-weighted."""
    means = np.full(n_clusters, np.nan, dtype=float)
    for cluster_id in range(n_clusters):
        cmask = labels == cluster_id
        if not np.any(cmask):
            continue

        cluster_values = values[cmask]
        if cluster_weights is not None:
            weights = cluster_weights[cmask]
            valid = np.isfinite(cluster_values) & np.isfinite(weights) & (weights > 0)
            if np.any(valid):
                means[cluster_id] = np.average(cluster_values[valid], weights=weights[valid])
                continue

        means[cluster_id] = np.nanmean(cluster_values)
    return means


def process_variable_streaming(
    var_dir: str,
    mask_germany: np.ndarray,
    cluster_labels: np.ndarray,
    output_dir: str,
    cluster_weights: np.ndarray | None = None,
    skip_existing: bool = True,
):
    """Stream one variable folder and write a single aggregated CSV."""
    os.makedirs(output_dir, exist_ok=True)

    pattern = os.path.join(var_dir, "icon-d2_germany_regular-lat-lon_single-level_*.grib2.bz2")
    files = sorted(glob.glob(pattern))
    if not files:
        return

    folder_name = os.path.basename(var_dir)
    if skip_existing:
        existing_outpath, existing_outname = _infer_variable_output_path(files[0], output_dir)
        if existing_outpath.exists():
            print(f"Skipping existing: {existing_outname}")
            return

    mask_flat = mask_germany.ravel()
    labels = cluster_labels
    n_clusters = int(np.max(labels)) + 1

    timestamps_list: list[pd.Timestamp] = []
    cluster_series: list[np.ndarray] = []

    field_name = None
    units = ""
    long_name = ""
    step_type = ""

    print(f"Streaming variable folder: {folder_name} ({len(files)} files)")

    with tempfile.TemporaryDirectory() as tmpdir:
        for file_path in files:
            tmp_grib = os.path.join(tmpdir, os.path.basename(file_path).replace(".bz2", ""))
            with bz2.open(file_path, "rb") as src, open(tmp_grib, "wb") as dst:
                dst.write(src.read())

            ds = xr.open_dataset(tmp_grib, engine="cfgrib")
            current_field_name = list(ds.data_vars)[0]
            if field_name is None:
                field_name = current_field_name
                attrs = ds[field_name].attrs
                long_name = attrs.get("long_name", "") or ""
                units = attrs.get("units", "") or ""
                step_type = (attrs.get("GRIB_stepType", "") or "").lower()

            arr = ds[field_name].values
            valid_time = ds.get("valid_time", None)
            if valid_time is None:
                tvals = ds.coords.get("time", None)
                if tvals is None:
                    raise ValueError(f"No valid_time/time coordinate in {file_path}")
                tvals = np.atleast_1d(tvals.values)
            else:
                tvals = np.atleast_1d(valid_time.values)

            ds.close()

            if arr.ndim == 2:
                arr_3d = arr[np.newaxis, ...]
            elif arr.ndim == 3:
                arr_3d = arr
            else:
                raise ValueError(f"Unexpected array shape {arr.shape} in {file_path}")

            if arr_3d.shape[0] != len(tvals):
                min_len = min(arr_3d.shape[0], len(tvals))
                arr_3d = arr_3d[:min_len, ...]
                tvals = tvals[:min_len]

            for idx in range(arr_3d.shape[0]):
                ts = pd.to_datetime(tvals[idx])
                timestamps_list.append(ts)

                slice_2d = arr_3d[idx, :, :]
                slice_flat_germany = slice_2d.reshape(-1)[mask_flat]
                means = aggregate_cluster_values(
                    slice_flat_germany,
                    labels,
                    n_clusters,
                    cluster_weights=cluster_weights,
                )

                cluster_series.append(means)
                del slice_2d, slice_flat_germany, means
                gc.collect()

            del arr, arr_3d, tvals
            gc.collect()

    timestamps = pd.to_datetime(timestamps_list)
    values = np.vstack(cluster_series)

    order = np.argsort(timestamps.values)
    timestamps = pd.to_datetime(timestamps.values[order])
    values = values[order, :]

    if any(token in step_type for token in ["avg", "acc", "mean"]):
        print(f"Computing instantaneous flux for {field_name} (stepType={step_type})")
        values = np.apply_along_axis(lambda y: compute_step_flux(timestamps, y), 0, values)
        flux_suffix = "_instantaneous"
    else:
        flux_suffix = "_raw"

    unit_safe = safe_unit_for_filename(units)
    first_ts = timestamps[0].strftime("%Y%m%d%H") if len(timestamps) else "unknown"

    outname = f"{field_name}_{unit_safe}_{first_ts}{flux_suffix}.csv"
    outpath = Path(output_dir) / outname

    if skip_existing and outpath.exists():
        print(f"Skipping existing: {outname}")
        return

    df = pd.DataFrame(values, index=timestamps, columns=[f"cluster_{i}" for i in range(n_clusters)])
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

    with open(outpath, "w", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(header_lines) + "\n")
        df.to_csv(handle)

    print(f"Saved {outname} ({df.shape[0]} timesteps x {df.shape[1]} clusters)")

    del df, values, timestamps, timestamps_list, cluster_series
    gc.collect()


def _infer_variable_output_path(first_file: str, output_dir: str) -> tuple[Path, str]:
    """Infer the aggregated CSV filename from the first GRIB file before streaming all timesteps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_grib = os.path.join(tmpdir, os.path.basename(first_file).replace(".bz2", ""))
        with bz2.open(first_file, "rb") as src, open(tmp_grib, "wb") as dst:
            dst.write(src.read())

        ds = xr.open_dataset(tmp_grib, engine="cfgrib")
        field_name = list(ds.data_vars)[0]
        attrs = ds[field_name].attrs
        units = attrs.get("units", "") or ""
        step_type = (attrs.get("GRIB_stepType", "") or "").lower()
        valid_time = ds.get("valid_time", None)
        if valid_time is None:
            tvals = ds.coords.get("time", None)
            if tvals is None:
                raise ValueError(f"No valid_time/time coordinate in {first_file}")
            tvals = np.atleast_1d(tvals.values)
        else:
            tvals = np.atleast_1d(valid_time.values)
        ds.close()

    flux_suffix = "_instantaneous" if any(token in step_type for token in ["avg", "acc", "mean"]) else "_raw"
    unit_safe = safe_unit_for_filename(units)
    first_ts = pd.to_datetime(tvals[0]).strftime("%Y%m%d%H") if len(tvals) else "unknown"
    outname = f"{field_name}_{unit_safe}_{first_ts}{flux_suffix}.csv"
    return Path(output_dir) / outname, outname


def run_aggregation(config: IconAggregationConfig) -> None:
    """Process all requested ICON-D2 directories in streaming mode."""
    daily_dirs = []
    for directory in glob.glob(config.root_dir_pattern):
        if not os.path.isdir(directory):
            continue

        name = os.path.basename(directory)
        try:
            date_str = name.split("_")[-1]
            day_date = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            print(f"Could not parse date from folder name: {name}, skipping.")
            continue

        if day_date >= datetime.combine(config.start_date, datetime.min.time()):
            daily_dirs.append(directory)

    daily_dirs = sorted(daily_dirs)
    if config.only_day is not None:
        daily_dirs = [directory for directory in daily_dirs if os.path.basename(directory).endswith(config.only_day)]

    print(f"\nFound {len(daily_dirs)} daily ICON folders.")

    mask_germany = None
    cluster_labels = None
    cluster_weights = None
    coords_germany = None
    cluster_regions = None
    latitude = None
    longitude = None

    for daily_dir in daily_dirs:
        day_name = os.path.basename(daily_dir)
        print(f"\n{'=' * 80}\nProcessing day: {day_name}\n{'=' * 80}")

        icon_dirs = sorted(glob.glob(os.path.join(daily_dir, "icon-d2", "*")))
        icon_dirs = [directory for directory in icon_dirs if os.path.isdir(directory)]
        if config.only_run_hour is not None:
            icon_dirs = [directory for directory in icon_dirs if os.path.basename(directory) == config.only_run_hour]

        if not icon_dirs:
            print(f"No ICON run-hour subdirectories found in {daily_dir}, skipping.")
            continue

        for icon_dir in icon_dirs:
            run_hour = os.path.basename(icon_dir)
            print(f"\nForecast run hour: {run_hour}")

            if mask_germany is None or cluster_labels is None:
                first_file = find_first_grib2_bz2(icon_dir)
                if first_file is None:
                    print(f"No GRIB2.bz2 files found in {icon_dir}, skipping run hour.")
                    continue

                print("Reading grid from first available file...")
                latitude, longitude = read_grid_from_one_file(first_file)

                print("Computing Germany mask...")
                mask_germany = filter_points_in_germany(
                    latitude,
                    longitude,
                    config.shapefile_path,
                    buffer_km=config.buffer_km,
                )

                print("Computing KMeans clusters...")
                if config.cluster_source == "grid":
                    coords_germany, cluster_labels, _ = cluster_german_coordinates(
                        latitude,
                        longitude,
                        mask_germany,
                        config.shapefile_path,
                        n_clusters=config.n_clusters,
                        random_state=42,
                        plot=config.plot_clusters,
                    )
                    cluster_weights = None
                elif config.cluster_source == "mastr_solar":
                    coords_germany, cluster_labels, _, capacity_weights = cluster_mastr_solar_capacity_coordinates(
                        latitude,
                        longitude,
                        mask_germany,
                        config,
                    )
                    cluster_weights = capacity_weights if config.capacity_weighted_aggregation else None
                    cluster_regions = None
                elif config.cluster_source == "mastr_solar_tso":
                    (
                        coords_germany,
                        cluster_labels,
                        _,
                        capacity_weights,
                        cluster_regions,
                    ) = cluster_mastr_solar_tso_capacity_coordinates(
                        latitude,
                        longitude,
                        mask_germany,
                        config,
                    )
                    cluster_weights = capacity_weights if config.capacity_weighted_aggregation else None
                elif config.cluster_source == "mastr_wind":
                    coords_germany, cluster_labels, _, capacity_weights = cluster_mastr_wind_capacity_coordinates(
                        latitude,
                        longitude,
                        mask_germany,
                        config,
                    )
                    cluster_weights = capacity_weights if config.capacity_weighted_aggregation else None
                    cluster_regions = None
                else:
                    raise ValueError(f"Unsupported ICON cluster_source: {config.cluster_source!r}")

                if config.cluster_output_file is not None and coords_germany is not None:
                    save_cluster_map(
                        config.cluster_output_file,
                        coords_germany,
                        cluster_labels,
                        capacity_weights if config.cluster_source in {"mastr_solar", "mastr_solar_tso", "mastr_wind"} else None,
                        cluster_regions=cluster_regions,
                    )
            else:
                print("Reusing existing grid, mask, and clusters.")

            output_dir = os.path.join(str(config.output_parent), f"{day_name}_{run_hour}")
            os.makedirs(output_dir, exist_ok=True)

            var_dirs = sorted([directory for directory in glob.glob(os.path.join(icon_dir, "*")) if os.path.isdir(directory)])
            if config.variables is not None:
                requested_variables = set(config.variables)
                available_variables = {os.path.basename(directory) for directory in var_dirs}
                missing_variables = sorted(requested_variables - available_variables)
                if missing_variables:
                    print(
                        "Requested variable folders not found in "
                        f"{day_name}/{run_hour}: {', '.join(missing_variables)}"
                    )
                var_dirs = [directory for directory in var_dirs if os.path.basename(directory) in requested_variables]

            print(f"Found {len(var_dirs)} variable folders to process in {day_name}/{run_hour}.")
            for var_dir in var_dirs:
                try:
                    process_variable_streaming(
                        var_dir=var_dir,
                        mask_germany=mask_germany,
                        cluster_labels=cluster_labels,
                        output_dir=output_dir,
                        cluster_weights=cluster_weights,
                        skip_existing=config.skip_existing_output,
                    )
                except Exception as exc:
                    print(
                        f"Error processing variable folder '{os.path.basename(var_dir)}' in {day_name}/{run_hour}: {exc}"
                    )
                finally:
                    gc.collect()

            print(f"Finished processing {day_name} - {run_hour}")
            gc.collect()

    print("\nAll requested ICON-D2 directories processed successfully.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Aggregate DWD ICON-D2 weather data to spatial clusters.")
    parser.add_argument("--config", type=Path, help="Optional JSON config file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for ICON-D2 aggregation."""
    args = parse_args(argv)
    config = load_config(args.config, IconAggregationConfig) if args.config else IconAggregationConfig()
    run_aggregation(config)


if __name__ == "__main__":
    main()
