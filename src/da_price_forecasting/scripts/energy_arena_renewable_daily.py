from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

from ..config import (
    RegionalRenewableFeatureConfig,
    RenewableGenerationModelConfig,
    RunConfig,
    load_config_payload,
    validate_config_payload,
)
from ..data.entsoe import fetch_actual_renewable_generation
from ..paths import find_repo_root, resolve_path
from ..pipelines.common import load_timestamp_csv, save_timestamp_csv
from .run import run_from_config


DEFAULT_FEATURE_CONFIG = Path(
    "configs/regional_renewable_features_open_meteo_icon_d2_single_run06_c25_grid_mean_p10_through_20260517.yaml"
)
DEFAULT_MODEL_CONFIG = Path(
    "configs/renewable_generation_open_meteo_icon_d2_single_run06_grid_mean_p10_regional_c25_hgb_bias30_hour_s05_d90_cutoff1000_febmay22.yaml"
)
DEFAULT_WORK_ROOT = Path("results/energy_arena_work/renewable_generation")
DEFAULT_SOLAR_CHALLENGE_ID_ENV = "ENERGY_ARENA_SOLAR_CHALLENGE_ID"
DEFAULT_WIND_CHALLENGE_ID_ENV = "ENERGY_ARENA_WIND_CHALLENGE_ID"
DEFAULT_SOLAR_VALUE_COLUMN = "Solar_Model_MW"
DEFAULT_WIND_VALUE_COLUMN = "Wind_Total_Model_MW"
DEFAULT_SOLAR_APPROACH_NAME = "renewable_open_meteo_p10_regional_c25_hgb_solar"
DEFAULT_WIND_APPROACH_NAME = "renewable_open_meteo_p10_regional_c25_hgb_wind"
DEFAULT_APPROACH_DESCRIPTION = (
    "Joint Open-Meteo ICON-D2 p10 regional renewable generation model with rolling bias correction."
)


@dataclass(frozen=True)
class DailyRenewablePaths:
    work_dir: Path
    generated_config_dir: Path

    @property
    def feature_config(self) -> Path:
        return self.generated_config_dir / "regional_renewable_features.generated.yaml"

    @property
    def model_config(self) -> Path:
        return self.generated_config_dir / "renewable_generation.generated.yaml"

    @property
    def solar_submission_config(self) -> Path:
        return self.generated_config_dir / "energy_arena_solar_submission.generated.yaml"

    @property
    def wind_submission_config(self) -> Path:
        return self.generated_config_dir / "energy_arena_wind_submission.generated.yaml"

    @property
    def forecast_dir(self) -> Path:
        return self.work_dir / "forecast_run"

    @property
    def forecast_file(self) -> Path:
        return self.forecast_dir / "forecast.csv"


def tomorrow_in_tz(target_tz: str, now: datetime | None = None) -> date:
    tz = ZoneInfo(target_tz)
    current = now.astimezone(tz) if now is not None and now.tzinfo is not None else (now or datetime.now(tz))
    if current.tzinfo is None:
        current = current.replace(tzinfo=tz)
    return current.date() + timedelta(days=1)


def dated_renewable_work_paths(
    repo_root: Path,
    forecast_date: date,
    work_root: Path | None = None,
) -> DailyRenewablePaths:
    root = resolve_path(work_root or DEFAULT_WORK_ROOT, repo_root)
    work_dir = root / forecast_date.isoformat()
    return DailyRenewablePaths(
        work_dir=work_dir,
        generated_config_dir=work_dir / "generated_configs",
    )


def _write_yaml(path: Path, payload: dict) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _resolve_challenge_id(challenge_id: int | None, env_var: str, repo_root: Path, label: str) -> int:
    if challenge_id is not None:
        return int(challenge_id)

    load_dotenv(repo_root / ".env")
    raw_value = os.getenv(env_var)
    if not raw_value:
        raise ValueError(
            f"Pass --{label}-challenge-id or set {env_var}=<Energy Arena challenge id> in the VM .env file."
        )

    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{env_var} must be an integer challenge id, got {raw_value!r}.") from exc


def _load_run_payload(path: Path, repo_root: Path) -> dict:
    return load_config_payload(resolve_path(path, repo_root))


def _nested_config(payload: dict) -> dict:
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("Expected a top-level run config with an embedded 'config' mapping.")
    return config


def _feature_config_from_payload(payload: dict, repo_root: Path) -> RegionalRenewableFeatureConfig:
    run_config = validate_config_payload(payload, RunConfig, repo_root=repo_root)
    return validate_config_payload(
        run_config.config or {},
        RegionalRenewableFeatureConfig,
        repo_root=run_config.repo_root,
    )


def _model_config_from_payload(payload: dict, repo_root: Path) -> RenewableGenerationModelConfig:
    run_config = validate_config_payload(payload, RunConfig, repo_root=repo_root)
    return validate_config_payload(
        run_config.config or {},
        RenewableGenerationModelConfig,
        repo_root=run_config.repo_root,
    )


def build_renewable_feature_payload(
    *,
    feature_config_path: Path,
    forecast_date: date,
    repo_root: Path,
) -> dict:
    payload = _load_run_payload(feature_config_path, repo_root)
    if payload.get("kind") != "regional_renewable_features":
        raise ValueError(f"Expected regional_renewable_features config, got {payload.get('kind')!r}.")

    config = _nested_config(payload)
    config["open_meteo_end_date"] = forecast_date.isoformat()
    return payload


def build_renewable_model_payload(
    *,
    model_config_path: Path,
    forecast_date: date,
    forecast_dir: Path,
    renewable_proxy_file: Path,
    repo_root: Path,
) -> dict:
    payload = _load_run_payload(model_config_path, repo_root)
    if payload.get("kind") != "renewable_generation_model":
        raise ValueError(f"Expected renewable_generation_model config, got {payload.get('kind')!r}.")

    config = _nested_config(payload)
    target_lag_days = int(config.get("target_availability_lag_days", 0))
    config["test_start"] = forecast_date.isoformat()
    config["test_end"] = forecast_date.isoformat()
    config["entsoe_end_date"] = (forecast_date - timedelta(days=target_lag_days)).isoformat()
    config["renewable_proxy_file"] = str(renewable_proxy_file)
    config["export_dir"] = str(forecast_dir)
    return payload


def update_actual_generation_cache(
    config: RenewableGenerationModelConfig,
    *,
    forecast_date: date,
    refresh_overlap_days: int = 1,
) -> Path:
    """Refresh actual renewable generation through the latest target-available day.

    The renewable pipeline deliberately reuses an existing cache file. For operational
    runs, we extend that cache first so training and rolling bias correction can use
    newly available actual solar/wind generation.
    """
    load_dotenv(config.repo_root / ".env")
    target_end_day = forecast_date - timedelta(days=config.target_availability_lag_days)
    if target_end_day < config.entsoe_start_date:
        return config.actual_generation_file

    existing: pd.DataFrame | None = None
    fetch_start_day = config.entsoe_start_date
    if config.actual_generation_file.exists():
        existing = load_timestamp_csv(config.actual_generation_file, config.target_tz)
        if not existing.empty:
            latest_day = existing.index.max().tz_convert(config.target_tz).date()
            fetch_start_day = max(
                config.entsoe_start_date,
                latest_day - timedelta(days=max(refresh_overlap_days, 0)),
            )

    if fetch_start_day > target_end_day:
        print(f"[Renewables] Actual generation cache already available through {target_end_day}.")
        return config.actual_generation_file

    print(f"[Renewables] Updating actual generation cache: {fetch_start_day} -> {target_end_day}")
    try:
        fetched = fetch_actual_renewable_generation(
            start_day=pd.Timestamp(fetch_start_day, tz=config.target_tz),
            end_day=pd.Timestamp(target_end_day, tz=config.target_tz),
            country_code=config.country_code_entsoe,
            api_key_env=config.entsoe_api_key_env,
            target_tz=config.target_tz,
        )
    except Exception as exc:
        if existing is not None and not existing.empty:
            print(
                f"[Renewables] Actual generation refresh failed, continuing with existing cache: {exc}",
                flush=True,
            )
            return config.actual_generation_file
        raise

    if existing is not None and not existing.empty:
        fetched_start = fetched.index.min()
        combined = pd.concat([existing.loc[existing.index < fetched_start], fetched]).sort_index()
        combined = combined.loc[~combined.index.duplicated(keep="last")]
    else:
        combined = fetched.sort_index()

    save_timestamp_csv(combined, config.actual_generation_file)
    print(f"[Renewables] Saved actual generation cache -> {config.actual_generation_file}")
    return config.actual_generation_file


def build_renewable_submission_payload(
    *,
    forecast_path: Path,
    forecast_date: date,
    challenge_id: int,
    submit: bool,
    target_tz: str,
    value_column: str,
    source_name: str,
    approach_name: str,
    approach_description: str | None,
) -> dict:
    return {
        "kind": "energy_arena_submit",
        "submit": submit,
        "config": {
            "repo_root": ".",
            "source": {
                "kind": "forecast_file",
                "path": str(forecast_path),
                "name": source_name,
                "value_column": value_column,
            },
            "forecast_date": forecast_date.isoformat(),
            "challenge_id": challenge_id,
            "target_tz": target_tz,
            "objective": "point",
            "value_column": value_column,
            "forecast_visibility": "closed",
            "leaderboard_visibility": "public",
            "approach_name": approach_name,
            "approach_description": approach_description or DEFAULT_APPROACH_DESCRIPTION,
            "artifacts_dir": "results/energy_arena_submissions",
        },
    }


def run_daily_renewable_energy_arena(
    *,
    feature_config_path: Path = DEFAULT_FEATURE_CONFIG,
    model_config_path: Path = DEFAULT_MODEL_CONFIG,
    solar_challenge_id: int | None = None,
    wind_challenge_id: int | None = None,
    solar_challenge_id_env: str = DEFAULT_SOLAR_CHALLENGE_ID_ENV,
    wind_challenge_id_env: str = DEFAULT_WIND_CHALLENGE_ID_ENV,
    forecast_date: date | None = None,
    target_tz: str = "Europe/Berlin",
    work_root: Path | None = None,
    submit: bool = True,
    submit_solar: bool = True,
    submit_wind: bool = True,
    solar_value_column: str = DEFAULT_SOLAR_VALUE_COLUMN,
    wind_value_column: str = DEFAULT_WIND_VALUE_COLUMN,
    solar_approach_name: str = DEFAULT_SOLAR_APPROACH_NAME,
    wind_approach_name: str = DEFAULT_WIND_APPROACH_NAME,
    approach_description: str | None = None,
    update_actual_generation: bool = True,
) -> DailyRenewablePaths:
    if not submit_solar and not submit_wind:
        raise ValueError("At least one of submit_solar or submit_wind must be enabled.")

    repo_root = find_repo_root()
    day = forecast_date or tomorrow_in_tz(target_tz)
    paths = dated_renewable_work_paths(repo_root=repo_root, forecast_date=day, work_root=work_root)

    resolved_solar_challenge_id = (
        _resolve_challenge_id(solar_challenge_id, solar_challenge_id_env, repo_root, "solar")
        if submit_solar
        else None
    )
    resolved_wind_challenge_id = (
        _resolve_challenge_id(wind_challenge_id, wind_challenge_id_env, repo_root, "wind")
        if submit_wind
        else None
    )

    print(f"Daily Energy Arena renewable run for forecast_date={day.isoformat()}")
    print(f"Working directory: {paths.work_dir}")

    feature_payload = build_renewable_feature_payload(
        feature_config_path=feature_config_path,
        forecast_date=day,
        repo_root=repo_root,
    )
    feature_config = _feature_config_from_payload(feature_payload, repo_root)
    _write_yaml(paths.feature_config, feature_payload)
    run_from_config(validate_config_payload(feature_payload, RunConfig, repo_root=repo_root))

    model_payload = build_renewable_model_payload(
        model_config_path=model_config_path,
        forecast_date=day,
        forecast_dir=paths.forecast_dir,
        renewable_proxy_file=feature_config.output_file,
        repo_root=repo_root,
    )
    model_config = _model_config_from_payload(model_payload, repo_root)
    if update_actual_generation:
        update_actual_generation_cache(model_config, forecast_date=day)
    _write_yaml(paths.model_config, model_payload)
    run_from_config(validate_config_payload(model_payload, RunConfig, repo_root=repo_root))

    if not paths.forecast_file.exists():
        raise FileNotFoundError(f"Renewable forecast CSV was not created: {paths.forecast_file}")

    submissions: dict[str, dict[str, int | str | bool | None]] = {}
    if submit_solar and resolved_solar_challenge_id is not None:
        solar_payload = build_renewable_submission_payload(
            forecast_path=paths.forecast_file,
            forecast_date=day,
            challenge_id=resolved_solar_challenge_id,
            submit=submit,
            target_tz=target_tz,
            value_column=solar_value_column,
            source_name="renewable_generation_solar",
            approach_name=solar_approach_name,
            approach_description=approach_description,
        )
        _write_yaml(paths.solar_submission_config, solar_payload)
        run_from_config(validate_config_payload(solar_payload, RunConfig, repo_root=repo_root), submit_override=submit)
        submissions["solar"] = {
            "challenge_id": resolved_solar_challenge_id,
            "value_column": solar_value_column,
            "submission_config": str(paths.solar_submission_config),
        }

    if submit_wind and resolved_wind_challenge_id is not None:
        wind_payload = build_renewable_submission_payload(
            forecast_path=paths.forecast_file,
            forecast_date=day,
            challenge_id=resolved_wind_challenge_id,
            submit=submit,
            target_tz=target_tz,
            value_column=wind_value_column,
            source_name="renewable_generation_wind",
            approach_name=wind_approach_name,
            approach_description=approach_description,
        )
        _write_yaml(paths.wind_submission_config, wind_payload)
        run_from_config(validate_config_payload(wind_payload, RunConfig, repo_root=repo_root), submit_override=submit)
        submissions["wind"] = {
            "challenge_id": resolved_wind_challenge_id,
            "value_column": wind_value_column,
            "submission_config": str(paths.wind_submission_config),
        }

    paths.work_dir.mkdir(parents=True, exist_ok=True)
    with open(paths.work_dir / "daily_run.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "forecast_date": day.isoformat(),
                "submit": submit,
                "feature_config_path": str(feature_config_path),
                "model_config_path": str(model_config_path),
                "generated_feature_config": str(paths.feature_config),
                "generated_model_config": str(paths.model_config),
                "forecast_file": str(paths.forecast_file),
                "submissions": submissions,
            },
            handle,
            indent=2,
        )
    return paths


def _parse_retry_until(value: str | None) -> datetime_time | None:
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--retry-until must use HH:MM, for example 11:55.") from exc


def _retry_deadline(now: datetime, retry_until: datetime_time | None, target_tz: str) -> datetime | None:
    if retry_until is None:
        return None
    tz = ZoneInfo(target_tz)
    current = now.astimezone(tz) if now.tzinfo is not None else now.replace(tzinfo=tz)
    return datetime.combine(current.date(), retry_until, tzinfo=tz)


def run_daily_renewable_energy_arena_with_retries(
    *,
    feature_config_path: Path,
    model_config_path: Path,
    solar_challenge_id: int | None,
    wind_challenge_id: int | None,
    solar_challenge_id_env: str,
    wind_challenge_id_env: str,
    forecast_date: date | None,
    target_tz: str,
    work_root: Path | None,
    submit: bool,
    submit_solar: bool,
    submit_wind: bool,
    solar_value_column: str,
    wind_value_column: str,
    solar_approach_name: str,
    wind_approach_name: str,
    approach_description: str | None,
    update_actual_generation: bool,
    retry_until: datetime_time | None,
    retry_interval_minutes: float,
) -> DailyRenewablePaths:
    deadline = _retry_deadline(datetime.now(ZoneInfo(target_tz)), retry_until, target_tz)
    interval_seconds = max(retry_interval_minutes, 0.1) * 60
    attempt = 1

    while True:
        try:
            return run_daily_renewable_energy_arena(
                feature_config_path=feature_config_path,
                model_config_path=model_config_path,
                solar_challenge_id=solar_challenge_id,
                wind_challenge_id=wind_challenge_id,
                solar_challenge_id_env=solar_challenge_id_env,
                wind_challenge_id_env=wind_challenge_id_env,
                forecast_date=forecast_date,
                target_tz=target_tz,
                work_root=work_root,
                submit=submit,
                submit_solar=submit_solar,
                submit_wind=submit_wind,
                solar_value_column=solar_value_column,
                wind_value_column=wind_value_column,
                solar_approach_name=solar_approach_name,
                wind_approach_name=wind_approach_name,
                approach_description=approach_description,
                update_actual_generation=update_actual_generation,
            )
        except Exception as exc:
            now = datetime.now(ZoneInfo(target_tz))
            if deadline is None or now + timedelta(seconds=interval_seconds) > deadline:
                print(f"Daily Energy Arena renewable attempt {attempt} failed and no retries remain: {exc}", flush=True)
                raise

            next_attempt = now + timedelta(seconds=interval_seconds)
            print(
                f"Daily Energy Arena renewable attempt {attempt} failed: {exc}\n"
                f"Retrying at {next_attempt.isoformat()} until {deadline.isoformat()}.",
                flush=True,
            )
            attempt += 1
            time.sleep(interval_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the daily Energy Arena solar/wind renewable generation submission workflow."
    )
    parser.add_argument("--feature-config", type=Path, default=DEFAULT_FEATURE_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--solar-challenge-id", type=int, default=None)
    parser.add_argument("--wind-challenge-id", type=int, default=None)
    parser.add_argument("--solar-challenge-id-env", default=DEFAULT_SOLAR_CHALLENGE_ID_ENV)
    parser.add_argument("--wind-challenge-id-env", default=DEFAULT_WIND_CHALLENGE_ID_ENV)
    parser.add_argument("--forecast-date", type=date.fromisoformat, default=None, help="Target date; defaults to tomorrow.")
    parser.add_argument("--target-tz", default="Europe/Berlin")
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--solar-value-column", default=DEFAULT_SOLAR_VALUE_COLUMN)
    parser.add_argument("--wind-value-column", default=DEFAULT_WIND_VALUE_COLUMN)
    parser.add_argument("--solar-approach-name", default=DEFAULT_SOLAR_APPROACH_NAME)
    parser.add_argument("--wind-approach-name", default=DEFAULT_WIND_APPROACH_NAME)
    parser.add_argument("--approach-description", default=None)
    parser.add_argument("--skip-solar", action="store_true")
    parser.add_argument("--skip-wind", action="store_true")
    parser.add_argument("--skip-actual-generation-update", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Generate payloads but do not submit to Energy Arena.")
    parser.add_argument("--retry-until", type=_parse_retry_until, default=None, help="Retry failed attempts until HH:MM in target timezone.")
    parser.add_argument("--retry-interval-minutes", type=float, default=10.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_daily_renewable_energy_arena_with_retries(
        feature_config_path=args.feature_config,
        model_config_path=args.model_config,
        solar_challenge_id=args.solar_challenge_id,
        wind_challenge_id=args.wind_challenge_id,
        solar_challenge_id_env=args.solar_challenge_id_env,
        wind_challenge_id_env=args.wind_challenge_id_env,
        forecast_date=args.forecast_date,
        target_tz=args.target_tz,
        work_root=args.work_root,
        submit=not args.dry_run,
        submit_solar=not args.skip_solar,
        submit_wind=not args.skip_wind,
        solar_value_column=args.solar_value_column,
        wind_value_column=args.wind_value_column,
        solar_approach_name=args.solar_approach_name,
        wind_approach_name=args.wind_approach_name,
        approach_description=args.approach_description,
        update_actual_generation=not args.skip_actual_generation_update,
        retry_until=args.retry_until,
        retry_interval_minutes=args.retry_interval_minutes,
    )


if __name__ == "__main__":
    main()
