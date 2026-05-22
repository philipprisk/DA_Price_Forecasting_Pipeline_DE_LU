from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from ..config import LoadForecastModelConfig, RunConfig, load_config_payload, validate_config_payload
from ..paths import find_repo_root, resolve_path
from ..preprocessing.dwd_icon_operational import ensure_dwd_icon_weather


DEFAULT_CONFIG = Path(
    "configs/load_forecast_hybrid_entsoe_residual_c25_run06_weighted_temp_morning_daily_weather_hgb_smooth_febapr.yaml"
)


def tomorrow_in_tz(target_tz: str, now: datetime | None = None) -> date:
    tz = ZoneInfo(target_tz)
    current = now.astimezone(tz) if now is not None and now.tzinfo is not None else (now or datetime.now(tz))
    if current.tzinfo is None:
        current = current.replace(tzinfo=tz)
    return current.date() + timedelta(days=1)


def load_model_config(path: Path, repo_root: Path) -> LoadForecastModelConfig:
    payload = load_config_payload(resolve_path(path, repo_root))
    if payload.get("kind") is not None:
        run_config = validate_config_payload(payload, RunConfig, repo_root=repo_root)
        if run_config.config_path is not None:
            payload = load_config_payload(run_config.config_path)
            return validate_config_payload(payload, LoadForecastModelConfig, repo_root=run_config.repo_root)
        return validate_config_payload(run_config.config, LoadForecastModelConfig, repo_root=run_config.repo_root)
    return validate_config_payload(payload, LoadForecastModelConfig, repo_root=repo_root)


def run_dwd_icon_daily_update(
    *,
    config_path: Path = DEFAULT_CONFIG,
    forecast_date: date | None = None,
    target_tz: str = "Europe/Berlin",
    catch_up_missing_days: bool | None = None,
    force: bool = False,
) -> list[date]:
    repo_root = find_repo_root()
    config = load_model_config(config_path, repo_root)
    day = forecast_date or tomorrow_in_tz(target_tz)
    catch_up = config.dwd_icon_catch_up_missing_days if catch_up_missing_days is None else catch_up_missing_days

    print(f"[DWD] Daily ICON-D2 update for forecast_date={day.isoformat()} run={config.required_run}")
    print(f"[DWD] Raw directory: {config.dwd_icon_raw_dir}")
    print(f"[DWD] Processed directory: {config.icon_dir}")

    processed_days = ensure_dwd_icon_weather(
        repo_root=config.repo_root,
        icon_dir=config.icon_dir,
        forecast_start=day,
        forecast_end=day,
        run_hour=config.required_run,
        folder_offset_date=config.dwd_folder_offset_date,
        raw_base_dir=config.dwd_icon_raw_dir,
        shapefile_path=config.dwd_icon_aggregation_shapefile_path,
        n_clusters=config.dwd_icon_aggregation_n_clusters,
        buffer_km=config.dwd_icon_aggregation_buffer_km,
        variables=config.dwd_icon_download_variables,
        base_url=config.dwd_icon_base_url,
        timeout_seconds=config.dwd_icon_download_timeout_seconds,
        request_pause_seconds=config.dwd_icon_request_pause_seconds,
        catch_up_missing_days=catch_up,
        force=force or config.dwd_icon_force_update,
    )
    if processed_days:
        print("[DWD] Updated issue days: " + ", ".join(day.isoformat() for day in processed_days))
    else:
        print("[DWD] Nothing to update.")
    return processed_days


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and aggregate the daily live DWD ICON-D2 run.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--forecast-date", type=date.fromisoformat, default=None, help="Target forecast date; defaults to tomorrow.")
    parser.add_argument("--target-tz", default="Europe/Berlin")
    parser.add_argument("--catch-up-missing-days", action="store_true", default=None)
    parser.add_argument("--no-catch-up-missing-days", action="store_false", dest="catch_up_missing_days")
    parser.add_argument("--force", action="store_true", help="Redownload/reaggregate even if outputs already exist.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_dwd_icon_daily_update(
        config_path=args.config,
        forecast_date=args.forecast_date,
        target_tz=args.target_tz,
        catch_up_missing_days=args.catch_up_missing_days,
        force=args.force,
    )


if __name__ == "__main__":
    main()
