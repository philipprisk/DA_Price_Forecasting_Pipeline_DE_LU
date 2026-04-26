from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
import tempfile

import pandas as pd

from ..config import Era5DownloadConfig, load_config

DATASET = "reanalysis-era5-single-levels"
MAIN_FILENAME = "era5_main.grib"
SOLAR_FILENAME = "era5_solar.grib"

MAIN_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "surface_pressure",
    "snow_depth",
]

SOLAR_VARIABLES = [
    "surface_solar_radiation_downwards",
    "total_sky_direct_solar_radiation_at_surface",
]


def iter_dates(start_date: date, end_date: date):
    """Yield all dates from start to end, inclusive."""
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def build_monthly_requests(
    start_date: date,
    end_date: date,
    variables: list[str],
    area: list[float] | None,
) -> list[dict]:
    """Build one CDS request per month for ERA5 single-level hourly GRIB data."""
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")

    return build_requests_for_dates(list(iter_dates(start_date, end_date)), variables=variables, area=area)


def build_requests_for_dates(
    request_dates: list[date],
    variables: list[str],
    area: list[float] | None,
) -> list[dict]:
    """Build CDS requests for an explicit set of dates, grouped by month."""
    by_year_month: dict[tuple[int, int], list[str]] = defaultdict(list)
    for current in sorted(set(request_dates)):
        by_year_month[(current.year, current.month)].append(f"{current.day:02d}")

    requests = []
    for (year, month), days in sorted(by_year_month.items()):
        request = {
            "product_type": ["reanalysis"],
            "variable": variables,
            "year": [f"{year:04d}"],
            "month": [f"{month:02d}"],
            "day": sorted(days),
            "time": [f"{hour:02d}:00" for hour in range(24)],
            "data_format": "grib",
        }

        if area is not None:
            request["area"] = area

        requests.append(request)

    return requests


def covered_dates_from_timestamps(timestamps: pd.DatetimeIndex, expected_count_per_day: int = 24) -> set[date]:
    """Return the set of fully covered dates in an hourly timestamp series."""
    ts = pd.to_datetime(timestamps).sort_values()
    counts = pd.Series(1, index=ts).groupby(ts.normalize()).sum()
    return {idx.date() for idx, count in counts.items() if int(count) >= expected_count_per_day}


def find_missing_dates(start_date: date, end_date: date, covered_dates: set[date]) -> list[date]:
    """Return requested dates that are not fully covered in the existing file."""
    return [current for current in iter_dates(start_date, end_date) if current not in covered_dates]


def _load_existing_covered_dates(target: Path, solar_mode: bool) -> set[date]:
    """Inspect an existing GRIB file and return the set of fully covered dates."""
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "xarray is required to inspect existing ERA5 GRIB files for append mode."
        ) from exc

    ds = xr.open_dataset(target, engine="cfgrib")
    try:
        if solar_mode:
            timestamps = pd.to_datetime(ds["valid_time"].values.reshape(-1)) - pd.Timedelta(hours=1)
        else:
            timestamps = pd.to_datetime(ds.time.values)
    finally:
        ds.close()

    return covered_dates_from_timestamps(pd.DatetimeIndex(timestamps))


def _missing_dates_for_append(
    target: Path,
    start_date: date,
    end_date: date,
    solar_mode: bool,
) -> list[date]:
    """Compute the trailing missing dates that can be safely appended to a GRIB file."""
    covered_dates = _load_existing_covered_dates(target, solar_mode=solar_mode)
    missing_dates = find_missing_dates(start_date, end_date, covered_dates)
    if not missing_dates:
        return []

    requested_dates = list(iter_dates(start_date, end_date))
    suffix_start = requested_dates.index(missing_dates[0])
    expected_suffix = requested_dates[suffix_start:]
    if missing_dates != expected_suffix:
        raise ValueError(
            "Append mode only supports trailing missing dates. "
            "This file appears to have gaps in the middle of the requested range. "
            "Download the missing range into a separate folder or rebuild the combined GRIB."
        )

    return missing_dates


def _write_requests_to_target(
    client,
    dataset: str,
    requests: list[dict],
    target: Path,
    file_mode: str,
) -> None:
    """Execute grouped CDS requests and stream the partial GRIB files into target."""
    with tempfile.TemporaryDirectory(prefix="era5_download_", dir=target.parent) as tmpdir:
        partials = []
        for idx, request in enumerate(requests, start=1):
            month = request["month"][0]
            year = request["year"][0]
            partial_target = Path(tmpdir) / f"{target.stem}_{year}{month}_{idx:02d}.grib"
            print(f"Requesting {target.name} chunk {idx}/{len(requests)} ({year}-{month})...")
            client.retrieve(dataset, request, str(partial_target))
            partials.append(partial_target)

        with open(target, file_mode) as dst:
            for partial in partials:
                with open(partial, "rb") as src:
                    dst.write(src.read())

    print(f"Downloaded {target}")


def retrieve_era5(
    dataset: str,
    request_dates: list[date],
    target: Path,
    force: bool,
    append_missing: bool,
    solar_mode: bool,
    variables: list[str],
    area: list[float] | None,
) -> None:
    """Submit CDS requests and write or append GRIB messages to the target file."""
    if not request_dates:
        print(f"No requested dates supplied for {target}")
        return

    if target.exists():
        if force and append_missing:
            raise ValueError("Choose either force=True or append_missing=True, not both.")
        if force:
            effective_dates = request_dates
            file_mode = "wb"
        elif append_missing:
            effective_dates = _missing_dates_for_append(
                target=target,
                start_date=min(request_dates),
                end_date=max(request_dates),
                solar_mode=solar_mode,
            )
            if not effective_dates:
                print(f"Target already fully covers requested dates: {target}")
                return
            file_mode = "ab"
            print(
                f"Appending {len(effective_dates)} missing day(s) to {target.name}: "
                f"{effective_dates[0].isoformat()} -> {effective_dates[-1].isoformat()}"
            )
        else:
            print(f"Skipping existing download: {target}")
            return
    else:
        effective_dates = request_dates
        file_mode = "wb"

    try:
        import cdsapi
    except ImportError as exc:
        raise ImportError(
            "cdsapi is not installed. Install the core Pixi environment first, "
            "for example with `pixi install -e core`."
        ) from exc

    target.parent.mkdir(parents=True, exist_ok=True)

    requests = build_requests_for_dates(effective_dates, variables=variables, area=area)
    client = cdsapi.Client()
    _write_requests_to_target(client, dataset, requests, target, file_mode=file_mode)


def run_download(config: Era5DownloadConfig) -> None:
    """Download the ERA5 inputs needed by the ERA5 aggregation pipeline."""
    main_target = config.output_dir / MAIN_FILENAME
    solar_target = config.output_dir / SOLAR_FILENAME
    request_dates = list(iter_dates(config.start_date, config.end_date))

    print(
        "Preparing ERA5 download requests for "
        f"{config.start_date.isoformat()} to {config.end_date.isoformat()}."
    )
    print(f"Target directory: {config.output_dir}")
    print(f"Area subset (N/W/S/E): {config.area}")
    if config.append_missing:
        print("Append mode: missing trailing dates will be downloaded and appended in place.")

    retrieve_era5(
        DATASET,
        request_dates=request_dates,
        target=main_target,
        force=config.force,
        append_missing=config.append_missing,
        solar_mode=False,
        variables=MAIN_VARIABLES,
        area=config.area,
    )
    retrieve_era5(
        DATASET,
        request_dates=request_dates,
        target=solar_target,
        force=config.force,
        append_missing=config.append_missing,
        solar_mode=True,
        variables=SOLAR_VARIABLES,
        area=config.area,
    )
    print("\nERA5 raw downloads finished.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Download raw ERA5 single-level GRIB files from CDS.")
    parser.add_argument("--config", type=Path, help="Optional JSON config file.")
    parser.add_argument("--start-date", type=date.fromisoformat, help="Inclusive start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", type=date.fromisoformat, help="Inclusive end date in YYYY-MM-DD format.")
    parser.add_argument("--output-dir", type=Path, help="Directory for downloaded GRIB files.")
    parser.add_argument(
        "--area",
        nargs=4,
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
        type=float,
        help="Optional geographic subset in CDS order: NORTH WEST SOUTH EAST.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing target files.")
    parser.add_argument(
        "--append-missing",
        action="store_true",
        help="Append only trailing missing days to existing GRIB targets instead of overwriting them.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for ERA5 downloads."""
    args = parse_args(argv)
    if args.config:
        config = load_config(args.config, Era5DownloadConfig)
    else:
        config = Era5DownloadConfig()

    updates = {}
    for field in ["start_date", "end_date", "output_dir", "area"]:
        value = getattr(args, field)
        if value is not None:
            updates[field] = value
    if args.force:
        updates["force"] = True
    if args.append_missing:
        updates["append_missing"] = True

    if updates:
        config = config.model_copy(update=updates)

    run_download(config)


if __name__ == "__main__":
    main()
