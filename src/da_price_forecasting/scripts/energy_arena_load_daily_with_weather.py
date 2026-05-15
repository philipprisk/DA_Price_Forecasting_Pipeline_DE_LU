from __future__ import annotations

import argparse
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from ..config import (
    IconAggregationConfig,
    LoadForecastModelConfig,
    RunConfig,
    load_config_payload,
    validate_config_payload,
)
from ..paths import find_repo_root, resolve_path
from .energy_arena_load_daily import (
    DEFAULT_CHALLENGE_ID_ENV,
    DEFAULT_MODEL_CONFIG,
    _parse_retry_until,
    run_daily_load_energy_arena_with_retries,
    tomorrow_in_tz,
)


DEFAULT_LSDF_BASE_ENV = "DWD_ICON_LSDF_BASE"
DEFAULT_WEATHER_VARIABLES = [
    "t_2m",
    "td_2m",
    "p",
    "u_10m",
    "v_10m",
    "vmax_10m",
    "aswdir_s",
    "aswdifd_s",
    "tot_prec",
    "h_snow",
    "snow_gsp",
]


def _load_model_config(path: Path, repo_root: Path) -> LoadForecastModelConfig:
    config_path = resolve_path(path, repo_root)
    payload = load_config_payload(config_path)
    if payload.get("kind") is not None:
        run_config = validate_config_payload(payload, RunConfig, repo_root=repo_root)
        if run_config.config_path is not None:
            nested_payload = load_config_payload(run_config.config_path)
            return validate_config_payload(nested_payload, LoadForecastModelConfig, repo_root=run_config.repo_root)
        return validate_config_payload(run_config.config, LoadForecastModelConfig, repo_root=run_config.repo_root)
    return validate_config_payload(payload, LoadForecastModelConfig, repo_root=repo_root)


def dwd_issue_day_for_forecast(forecast_date: date, folder_offset_date: date) -> date:
    """Return the DWD folder date that provides weather for a target forecast day."""
    if forecast_date > folder_offset_date:
        return forecast_date - timedelta(days=1)
    return forecast_date


def processed_weather_folder(icon_dir: Path, issue_day: date, run_hour: str) -> Path:
    return icon_dir / f"dwd_icon_daily_{issue_day:%Y%m%d}_{run_hour}"


def processed_weather_folder_is_ready(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.csv"))


def _processed_issue_days(icon_dir: Path, run_hour: str) -> list[date]:
    if not icon_dir.exists():
        return []

    pattern = re.compile(rf"^dwd_icon_daily_(\d{{8}})_{re.escape(run_hour)}$")
    issue_days: list[date] = []
    for folder in icon_dir.iterdir():
        if not folder.is_dir():
            continue
        if not processed_weather_folder_is_ready(folder):
            continue
        match = pattern.match(folder.name)
        if not match:
            continue
        try:
            issue_days.append(datetime.strptime(match.group(1), "%Y%m%d").date())
        except ValueError:
            continue
    return sorted(issue_days)


def _infer_n_clusters(model_config: LoadForecastModelConfig, explicit_n_clusters: int | None) -> int:
    if explicit_n_clusters is not None:
        return explicit_n_clusters

    match = re.search(r"c(\d+)", model_config.icon_dir.name)
    if match:
        return int(match.group(1))

    if model_config.weather_cluster_weight_file is not None and model_config.weather_cluster_weight_file.exists():
        import pandas as pd

        weights = pd.read_csv(model_config.weather_cluster_weight_file)
        if model_config.weather_cluster_id_column in weights.columns:
            return int(weights[model_config.weather_cluster_id_column].nunique())

    return 25


def _date_range(start: date, end: date) -> list[date]:
    if start > end:
        return []
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def missing_dwd_issue_days_to_process(
    *,
    icon_dir: Path,
    run_hour: str,
    target_issue_day: date,
    catch_up_missing_days: bool = True,
    force: bool = False,
) -> list[date]:
    """Return DWD issue days that need aggregation before a target issue day is usable."""
    if not catch_up_missing_days:
        target_folder = processed_weather_folder(icon_dir, target_issue_day, run_hour)
        return [target_issue_day] if force or not processed_weather_folder_is_ready(target_folder) else []

    processed_days = [day for day in _processed_issue_days(icon_dir, run_hour) if day <= target_issue_day]
    if processed_days:
        start_issue_day = min(max(processed_days) + timedelta(days=1), target_issue_day)
    else:
        start_issue_day = target_issue_day

    return [
        day
        for day in _date_range(start_issue_day, target_issue_day)
        if force or not processed_weather_folder_is_ready(processed_weather_folder(icon_dir, day, run_hour))
    ]


def ensure_dwd_weather_for_forecast(
    *,
    model_config: LoadForecastModelConfig,
    forecast_date: date,
    lsdf_base: Path | None,
    lsdf_base_env: str = DEFAULT_LSDF_BASE_ENV,
    n_clusters: int | None = None,
    buffer_km: int = 50,
    shapefile_path: Path = Path("data/shapefile/ne_10m_admin_0_countries.shp"),
    variables: list[str] | None = None,
    catch_up_missing_days: bool = True,
    force: bool = False,
) -> None:
    """Aggregate missing processed DWD ICON data needed for one forecast day."""
    if model_config.weather_source != "dwd_icon" or not model_config.include_weather_features:
        print("DWD weather aggregation skipped: model does not use DWD ICON weather features.")
        return

    issue_day = dwd_issue_day_for_forecast(forecast_date, model_config.dwd_folder_offset_date)
    target_folder = processed_weather_folder(model_config.icon_dir, issue_day, model_config.required_run)
    days_to_process = missing_dwd_issue_days_to_process(
        icon_dir=model_config.icon_dir,
        run_hour=model_config.required_run,
        target_issue_day=issue_day,
        catch_up_missing_days=catch_up_missing_days,
        force=force,
    )

    if not days_to_process:
        print(f"DWD weather already available through required issue day: {issue_day.isoformat()}")
        return

    raw_lsdf_base = os.getenv(lsdf_base_env)
    resolved_lsdf_base = lsdf_base or (Path(raw_lsdf_base) if raw_lsdf_base else None)
    if resolved_lsdf_base is None:
        raise ValueError(
            f"Set {lsdf_base_env}=<mounted LSDF icon_by_Max_Kleinebrahm path> "
            "or pass --lsdf-base before using the with-weather daily runner."
        )
    if not resolved_lsdf_base.exists():
        raise FileNotFoundError(
            f"LSDF base is not reachable: {resolved_lsdf_base}. "
            "Mount LSDF/VPN first, or sync processed DWD data to the VM."
        )

    print(
        "Aggregating DWD ICON weather for issue days: "
        + ", ".join(day.isoformat() for day in days_to_process)
    )

    from ..preprocessing.icon_d2_aggregation import run_aggregation

    repo_root = model_config.repo_root
    for day in days_to_process:
        raw_day = resolved_lsdf_base / f"dwd_icon_daily_{day:%Y%m%d}"
        if not raw_day.exists():
            raise FileNotFoundError(
                f"Raw LSDF DWD folder not found: {raw_day}. "
                f"For forecast_date={forecast_date.isoformat()}, the required issue day is {issue_day.isoformat()}."
            )

        aggregation_config = IconAggregationConfig(
            repo_root=repo_root,
            lsdf_base=resolved_lsdf_base,
            output_parent=model_config.icon_dir,
            shapefile_path=shapefile_path,
            n_clusters=_infer_n_clusters(model_config, n_clusters),
            buffer_km=buffer_km,
            plot_clusters=False,
            skip_existing_output=not force,
            only_day=day.strftime("%Y%m%d"),
            only_run_hour=model_config.required_run,
            start_date=day,
            variables=variables or DEFAULT_WEATHER_VARIABLES,
        )
        run_aggregation(aggregation_config)

    if not processed_weather_folder_is_ready(target_folder):
        raise FileNotFoundError(
            f"Processed DWD weather is still missing after aggregation: {target_folder}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate missing DWD ICON weather, then run daily Energy Arena load submission."
    )
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--challenge-id", type=int, default=None)
    parser.add_argument("--challenge-id-env", default=DEFAULT_CHALLENGE_ID_ENV)
    parser.add_argument("--forecast-date", type=date.fromisoformat, default=None, help="Target date; defaults to tomorrow.")
    parser.add_argument("--target-tz", default="Europe/Berlin")
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--value-column", default="Load_Model_MW")
    parser.add_argument("--approach-name", default=None)
    parser.add_argument("--approach-description", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Generate payloads but do not submit to Energy Arena.")
    parser.add_argument("--retry-until", type=_parse_retry_until, default=None, help="Retry failed attempts until HH:MM in target timezone.")
    parser.add_argument("--retry-interval-minutes", type=float, default=10.0)

    parser.add_argument("--skip-weather-aggregation", action="store_true")
    parser.add_argument("--weather-only", action="store_true", help="Aggregate missing DWD weather and exit before forecasting.")
    parser.add_argument("--force-weather-aggregation", action="store_true")
    parser.add_argument("--lsdf-base", type=Path, default=None)
    parser.add_argument("--lsdf-base-env", default=DEFAULT_LSDF_BASE_ENV)
    parser.add_argument("--weather-n-clusters", type=int, default=None)
    parser.add_argument("--weather-buffer-km", type=int, default=50)
    parser.add_argument("--weather-shapefile", type=Path, default=Path("data/shapefile/ne_10m_admin_0_countries.shp"))
    parser.add_argument("--no-weather-catch-up", action="store_true")
    parser.add_argument("--weather-variable", action="append", dest="weather_variables", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    repo_root = find_repo_root()
    load_dotenv(repo_root / ".env")

    forecast_date = args.forecast_date or tomorrow_in_tz(args.target_tz)
    model_config = _load_model_config(args.model_config, repo_root)

    if not args.skip_weather_aggregation:
        ensure_dwd_weather_for_forecast(
            model_config=model_config,
            forecast_date=forecast_date,
            lsdf_base=args.lsdf_base,
            lsdf_base_env=args.lsdf_base_env,
            n_clusters=args.weather_n_clusters,
            buffer_km=args.weather_buffer_km,
            shapefile_path=args.weather_shapefile,
            variables=args.weather_variables,
            catch_up_missing_days=not args.no_weather_catch_up,
            force=args.force_weather_aggregation,
        )
        if args.weather_only:
            print("Weather-only run complete.")
            return

    run_daily_load_energy_arena_with_retries(
        model_config_path=args.model_config,
        challenge_id=args.challenge_id,
        challenge_id_env=args.challenge_id_env,
        forecast_date=forecast_date,
        target_tz=args.target_tz,
        work_root=args.work_root,
        submit=not args.dry_run,
        value_column=args.value_column,
        approach_name=args.approach_name,
        approach_description=args.approach_description,
        retry_until=args.retry_until,
        retry_interval_minutes=args.retry_interval_minutes,
    )


if __name__ == "__main__":
    main()
