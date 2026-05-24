from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from ..config import RunConfig, validate_config_payload
from ..paths import find_repo_root, resolve_path
from .run import run_from_config


DEFAULT_MODEL_CONFIG = Path(
    "configs/load_forecast_hybrid_entsoe_residual_c25_run06_rich_temp_daily_weather_hgb_smooth_febapr.yaml"
)
DEFAULT_WORK_ROOT = Path("results/energy_arena_work/load")
DEFAULT_CHALLENGE_ID_ENV = "ENERGY_ARENA_LOAD_CHALLENGE_ID"
DEFAULT_QUANTILE_COLUMNS = ["q0.025", "q0.250", "q0.500", "q0.750", "q0.975"]


@dataclass(frozen=True)
class DailyLoadPaths:
    work_dir: Path
    generated_config_dir: Path

    @property
    def submission_config(self) -> Path:
        return self.generated_config_dir / "energy_arena_load_submission.generated.yaml"


def tomorrow_in_tz(target_tz: str, now: datetime | None = None) -> date:
    tz = ZoneInfo(target_tz)
    current = now.astimezone(tz) if now is not None and now.tzinfo is not None else (now or datetime.now(tz))
    if current.tzinfo is None:
        current = current.replace(tzinfo=tz)
    return current.date() + timedelta(days=1)


def dated_load_work_paths(repo_root: Path, forecast_date: date, work_root: Path | None = None) -> DailyLoadPaths:
    root = resolve_path(work_root or DEFAULT_WORK_ROOT, repo_root)
    work_dir = root / forecast_date.isoformat()
    return DailyLoadPaths(
        work_dir=work_dir,
        generated_config_dir=work_dir / "generated_configs",
    )


def _write_yaml(path: Path, payload: dict) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _resolve_challenge_id(challenge_id: int | None, env_var: str, repo_root: Path) -> int:
    if challenge_id is not None:
        return int(challenge_id)

    load_dotenv(repo_root / ".env")
    raw_value = os.getenv(env_var)
    if not raw_value:
        raise ValueError(
            f"Pass --challenge-id or set {env_var}=<load challenge id> in the VM .env file."
        )

    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{env_var} must be an integer challenge id, got {raw_value!r}.") from exc


def build_load_submission_payload(
    *,
    model_config_path: Path,
    forecast_date: date,
    challenge_id: int,
    submit: bool,
    target_tz: str,
    objective: str,
    value_column: str,
    quantile_columns: list[str] | None,
    approach_name: str | None,
    approach_description: str | None,
) -> dict:
    model_name = model_config_path.stem
    return {
        "kind": "energy_arena_submit",
        "submit": submit,
        "config": {
            "repo_root": ".",
            "source": {
                "kind": "load_forecast_model",
                "config_path": str(model_config_path),
                "run_before_submit": True,
                "value_column": value_column,
            },
            "forecast_date": forecast_date.isoformat(),
            "challenge_id": challenge_id,
            "target_tz": target_tz,
            "objective": objective,
            "value_column": value_column,
            "quantile_columns": quantile_columns,
            "forecast_visibility": "closed",
            "leaderboard_visibility": "public",
            "approach_name": approach_name or model_name,
            "approach_description": approach_description
            or "Hybrid ENTSO-E residual load forecast with DWD ICON weather features.",
            "artifacts_dir": "results/energy_arena_submissions",
        },
    }


def run_daily_load_energy_arena(
    *,
    model_config_path: Path = DEFAULT_MODEL_CONFIG,
    challenge_id: int | None = None,
    challenge_id_env: str = DEFAULT_CHALLENGE_ID_ENV,
    forecast_date: date | None = None,
    target_tz: str = "Europe/Berlin",
    work_root: Path | None = None,
    submit: bool = True,
    objective: str = "point",
    value_column: str = "Load_Model_MW",
    quantile_columns: list[str] | None = None,
    approach_name: str | None = None,
    approach_description: str | None = None,
) -> DailyLoadPaths:
    repo_root = find_repo_root()
    day = forecast_date or tomorrow_in_tz(target_tz)
    resolved_challenge_id = _resolve_challenge_id(challenge_id, challenge_id_env, repo_root)
    paths = dated_load_work_paths(repo_root=repo_root, forecast_date=day, work_root=work_root)

    print(f"Daily Energy Arena load run for forecast_date={day.isoformat()}")
    print(f"Working directory: {paths.work_dir}")

    payload = build_load_submission_payload(
        model_config_path=model_config_path,
        forecast_date=day,
        challenge_id=resolved_challenge_id,
        submit=submit,
        target_tz=target_tz,
        objective=objective,
        value_column=value_column,
        quantile_columns=quantile_columns,
        approach_name=approach_name,
        approach_description=approach_description,
    )
    _write_yaml(paths.submission_config, payload)

    run_config = validate_config_payload(payload, RunConfig, repo_root=repo_root)
    run_from_config(run_config, submit_override=submit)

    with open(paths.work_dir / "daily_run.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "forecast_date": day.isoformat(),
                "submit": submit,
                "challenge_id": resolved_challenge_id,
                "model_config_path": str(model_config_path),
                "submission_config": str(paths.submission_config),
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
        raise argparse.ArgumentTypeError("--retry-until must use HH:MM, for example 12:30.") from exc


def _retry_deadline(now: datetime, retry_until: datetime_time | None, target_tz: str) -> datetime | None:
    if retry_until is None:
        return None
    tz = ZoneInfo(target_tz)
    current = now.astimezone(tz) if now.tzinfo is not None else now.replace(tzinfo=tz)
    return datetime.combine(current.date(), retry_until, tzinfo=tz)


def run_daily_load_energy_arena_with_retries(
    *,
    model_config_path: Path,
    challenge_id: int | None,
    challenge_id_env: str,
    forecast_date: date | None,
    target_tz: str,
    work_root: Path | None,
    submit: bool,
    objective: str,
    value_column: str,
    quantile_columns: list[str] | None,
    approach_name: str | None,
    approach_description: str | None,
    retry_until: datetime_time | None,
    retry_interval_minutes: float,
) -> DailyLoadPaths:
    deadline = _retry_deadline(datetime.now(ZoneInfo(target_tz)), retry_until, target_tz)
    interval_seconds = max(retry_interval_minutes, 0.1) * 60
    attempt = 1

    while True:
        try:
            return run_daily_load_energy_arena(
                model_config_path=model_config_path,
                challenge_id=challenge_id,
                challenge_id_env=challenge_id_env,
                forecast_date=forecast_date,
                target_tz=target_tz,
                work_root=work_root,
                submit=submit,
                objective=objective,
                value_column=value_column,
                quantile_columns=quantile_columns,
                approach_name=approach_name,
                approach_description=approach_description,
            )
        except Exception as exc:
            now = datetime.now(ZoneInfo(target_tz))
            if deadline is None or now + timedelta(seconds=interval_seconds) > deadline:
                print(f"Daily Energy Arena load attempt {attempt} failed and no retries remain: {exc}", flush=True)
                raise

            next_attempt = now + timedelta(seconds=interval_seconds)
            print(
                f"Daily Energy Arena load attempt {attempt} failed: {exc}\n"
                f"Retrying at {next_attempt.isoformat()} until {deadline.isoformat()}.",
                flush=True,
            )
            attempt += 1
            time.sleep(interval_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the daily Energy Arena load forecast submission workflow.")
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--challenge-id", type=int, default=None)
    parser.add_argument("--challenge-id-env", default=DEFAULT_CHALLENGE_ID_ENV)
    parser.add_argument("--forecast-date", type=date.fromisoformat, default=None, help="Target date; defaults to tomorrow.")
    parser.add_argument("--target-tz", default="Europe/Berlin")
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--objective", choices=["point", "quantile"], default="point")
    parser.add_argument("--value-column", default="Load_Model_MW")
    parser.add_argument("--quantile-columns", nargs="+", default=None)
    parser.add_argument("--approach-name", default=None)
    parser.add_argument("--approach-description", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Generate payloads but do not submit to Energy Arena.")
    parser.add_argument("--retry-until", type=_parse_retry_until, default=None, help="Retry failed attempts until HH:MM in target timezone.")
    parser.add_argument("--retry-interval-minutes", type=float, default=10.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_daily_load_energy_arena_with_retries(
        model_config_path=args.model_config,
        challenge_id=args.challenge_id,
        challenge_id_env=args.challenge_id_env,
        forecast_date=args.forecast_date,
        target_tz=args.target_tz,
        work_root=args.work_root,
        submit=not args.dry_run,
        objective=args.objective,
        value_column=args.value_column,
        quantile_columns=args.quantile_columns or (DEFAULT_QUANTILE_COLUMNS if args.objective == "quantile" else None),
        approach_name=args.approach_name,
        approach_description=args.approach_description,
        retry_until=args.retry_until,
        retry_interval_minutes=args.retry_interval_minutes,
    )


if __name__ == "__main__":
    main()
