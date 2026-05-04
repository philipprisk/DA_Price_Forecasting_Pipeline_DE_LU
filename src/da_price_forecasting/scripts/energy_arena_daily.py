from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from ..config import LearOperationalConfig, RunConfig, SqraConfig, load_config_payload, validate_config_payload
from ..integrations.energy_arena.formatters import build_model_metadata
from ..paths import find_repo_root, resolve_path
from .run import run_from_config


DEFAULT_POINT_CONFIG = Path("configs/energy_arena_point_submission.yaml")
DEFAULT_QUANTILE_CONFIG = Path("configs/energy_arena_sqra_quantile_submission.yaml")


@dataclass(frozen=True)
class DailyPaths:
    work_dir: Path
    point_base_dir: Path
    sqra_export_dir: Path
    generated_config_dir: Path

    @property
    def point_base_forecast(self) -> Path:
        return self.point_base_dir / "forecast.csv"

    @property
    def point_submission_config(self) -> Path:
        return self.generated_config_dir / "energy_arena_point_submission.generated.yaml"

    @property
    def quantile_submission_config(self) -> Path:
        return self.generated_config_dir / "energy_arena_sqra_quantile_submission.generated.yaml"


def tomorrow_in_tz(target_tz: str, now: datetime | None = None) -> date:
    tz = ZoneInfo(target_tz)
    current = now.astimezone(tz) if now is not None and now.tzinfo is not None else (now or datetime.now(tz))
    if current.tzinfo is None:
        current = current.replace(tzinfo=tz)
    return current.date() + timedelta(days=1)


def day_bounds(forecast_date: date, target_tz: str) -> tuple[str, str]:
    start = pd.Timestamp(forecast_date, tz=target_tz)
    end = start + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    return start.isoformat(), end.isoformat()


def dated_work_paths(repo_root: Path, forecast_date: date, work_root: Path | None = None) -> DailyPaths:
    root = resolve_path(work_root or Path("results/energy_arena_work/exaa_only"), repo_root)
    work_dir = root / forecast_date.isoformat()
    return DailyPaths(
        work_dir=work_dir,
        point_base_dir=work_dir / "sqra_point_base",
        sqra_export_dir=work_dir / "sqra_forecast",
        generated_config_dir=work_dir / "generated_configs",
    )


def _config_body(run_payload: dict[str, Any]) -> dict[str, Any]:
    config = run_payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("Daily Energy Arena configs must contain an embedded 'config' mapping.")
    return config


def _source_config_dict(source: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    if isinstance(source.get("config"), dict):
        return dict(source["config"])
    if source.get("config_path"):
        path = resolve_path(Path(source["config_path"]), repo_root)
        return load_config_payload(path)
    raise ValueError(f"Source '{source.get('kind')}' requires an embedded config or config_path.")


def _point_source_config(point_payload: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    source = _config_body(point_payload).get("source")
    if not isinstance(source, dict) or source.get("kind") != "lear_operational":
        raise ValueError("Point submission config must use source.kind='lear_operational'.")
    return _source_config_dict(source, repo_root)


def _sqra_source_config(quantile_payload: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    source = _config_body(quantile_payload).get("source")
    if not isinstance(source, dict) or source.get("kind") != "sqra":
        raise ValueError("Quantile submission config must use source.kind='sqra'.")
    return _source_config_dict(source, repo_root)


def _set_run_submit(run_payload: dict[str, Any], submit: bool) -> None:
    run_payload["submit"] = bool(submit)


def _set_submission_date(config: dict[str, Any], forecast_date: date) -> None:
    config["forecast_date"] = forecast_date.isoformat()


def _normalise_common_submission_fields(config: dict[str, Any], forecast_date: date) -> None:
    _set_submission_date(config, forecast_date)
    config.setdefault("target_tz", "Europe/Berlin")


def _model_name_from_lear_config(lear_config: LearOperationalConfig) -> str:
    return lear_config.experiment_name


def build_point_submission_payload(
    point_payload: dict[str, Any],
    forecast_date: date,
    point_forecast_path: Path,
    point_source_name: str,
    submit: bool,
) -> dict[str, Any]:
    payload = dict(point_payload)
    config = dict(_config_body(payload))
    _normalise_common_submission_fields(config, forecast_date)
    config["objective"] = "point"
    config["value_column"] = config.get("value_column") or "y_pred"
    config["source"] = {
        "kind": "forecast_file",
        "path": str(point_forecast_path),
        "name": point_source_name,
        "value_column": config["value_column"],
    }
    payload["config"] = config
    _set_run_submit(payload, submit)
    return payload


def build_quantile_submission_payload(
    quantile_payload: dict[str, Any],
    forecast_date: date,
    point_forecast_path: Path,
    sqra_export_dir: Path,
    submit: bool,
    repo_root: Path,
) -> dict[str, Any]:
    payload = dict(quantile_payload)
    config = dict(_config_body(payload))
    _normalise_common_submission_fields(config, forecast_date)

    source = dict(config.get("source") or {})
    if source.get("kind") != "sqra":
        raise ValueError("Quantile submission config must use source.kind='sqra'.")

    sqra_config = _sqra_source_config(payload, repo_root)
    start, end = day_bounds(forecast_date, config.get("target_tz", "Europe/Berlin"))
    sqra_config["import_paths"] = [str(point_forecast_path)]
    sqra_config["test_start"] = start
    sqra_config["test_end"] = end
    sqra_config["export_dir"] = str(sqra_export_dir)
    sqra_config.setdefault("experiment_name", "sqra_exaa_only_d60_energy_arena")

    source.pop("config_path", None)
    source["config"] = sqra_config
    source["run_before_submit"] = True
    config["source"] = source
    payload["config"] = config
    _set_run_submit(payload, submit)
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _load_run_payload(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = load_config_payload(resolve_path(path, repo_root))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected config mapping in {path}.")
    return payload


def load_daily_inputs(
    point_config_path: Path,
    quantile_config_path: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        _load_run_payload(point_config_path, repo_root),
        _load_run_payload(quantile_config_path, repo_root),
    )


def prepare_lear_config_for_point_base(
    point_payload: dict[str, Any],
    forecast_date: date,
    paths: DailyPaths,
    repo_root: Path,
) -> LearOperationalConfig:
    raw_config = _point_source_config(point_payload, repo_root)
    target_tz = raw_config.get("target_tz", "Europe/Berlin")
    start, end = day_bounds(forecast_date, target_tz)
    raw_config["entsoe_end_date"] = start
    raw_config["test_end"] = end
    raw_config["export_dir"] = str(paths.point_base_dir)
    return validate_config_payload(raw_config, LearOperationalConfig, repo_root=repo_root)


def sqra_train_days(quantile_payload: dict[str, Any], repo_root: Path) -> int:
    sqra_config = _sqra_source_config(quantile_payload, repo_root)
    return int(sqra_config.get("train_days_rolling", SqraConfig().train_days_rolling))


def daily_forecast_days(forecast_date: date, history_days: int, target_tz: str) -> pd.DatetimeIndex:
    start_date = forecast_date - timedelta(days=history_days)
    return pd.date_range(
        start=pd.Timestamp(start_date, tz=target_tz),
        end=pd.Timestamp(forecast_date, tz=target_tz),
        freq="D",
    )


def _format_day_index(index: pd.Index) -> str:
    if index.empty:
        return "none"
    return f"{index.min().date()}..{index.max().date()} ({len(index)} days)"


def _validate_point_base_dataset(
    X: pd.DataFrame,
    forecast_days: pd.DatetimeIndex,
    forecast_day: pd.Timestamp,
    train_days: int,
) -> None:
    if X.empty:
        raise RuntimeError(
            "No complete LEAR feature rows are available after filtering. "
            "For the EXAA-only daily run this usually means the ENTSO-E EXAA "
            "day-ahead price series is not available yet."
        )

    available_days = pd.DatetimeIndex(X.index).normalize().unique().sort_values()
    requested_days = pd.DatetimeIndex(forecast_days).normalize().unique().sort_values()
    missing_days = requested_days.difference(available_days)

    if forecast_day not in available_days:
        raise RuntimeError(
            "No complete EXAA-only feature row is available for target day "
            f"{forecast_day.date()}. Available feature days: {_format_day_index(available_days)}. "
            "Retry after the EXAA prices have been published by the upstream API."
        )

    if len(missing_days) == len(requested_days):
        raise RuntimeError(
            "None of the requested SQRA point-base forecast days has complete "
            f"LEAR features. Requested: {_format_day_index(requested_days)}; "
            f"available: {_format_day_index(available_days)}."
        )

    runnable_days = []
    for day in requested_days:
        train_start = day - pd.Timedelta(days=train_days)
        train_end = day - pd.Timedelta(days=1)
        has_test_row = day in available_days
        has_train_rows = bool(((X.index >= train_start) & (X.index <= train_end)).sum())
        if has_test_row and has_train_rows:
            runnable_days.append(day)

    if not runnable_days:
        raise RuntimeError(
            "No requested point-base forecast day has both a test feature row "
            f"and at least one prior training feature row. Requested: {_format_day_index(requested_days)}; "
            f"available: {_format_day_index(available_days)}. This often means the EXAA API response "
            "contains the target day but not enough historical EXAA feature days."
        )


def run_point_base_forecasts(
    lear_config: LearOperationalConfig,
    forecast_date: date,
    history_days: int,
    export_dir: Path,
) -> Path:
    from ..models.lear import rolling_point_forecast, save_prediction_outputs
    from ..pipelines.lear import prepare_lear_operational_prediction_dataset

    forecast_day = pd.Timestamp(forecast_date, tz=lear_config.target_tz).normalize()
    forecast_days = daily_forecast_days(forecast_date, history_days, lear_config.target_tz)

    dataset = prepare_lear_operational_prediction_dataset(config=lear_config, forecast_date=forecast_day)
    _validate_point_base_dataset(
        X=dataset["X"],
        forecast_days=forecast_days,
        forecast_day=forecast_day,
        train_days=lear_config.train_days_rolling,
    )
    forecast_df, runtime_df, _, _, _ = rolling_point_forecast(
        X=dataset["X"],
        Y=dataset["Y"],
        forecast_days=forecast_days,
        train_days=lear_config.train_days_rolling,
        lars_start_date=pd.Timestamp(lear_config.lars_start_date),
        use_vst=lear_config.use_vst,
    )
    if forecast_day not in forecast_df.index.normalize().unique():
        raise RuntimeError(f"Point base forecast does not contain target day {forecast_day.date()}.")

    save_prediction_outputs(
        forecast_df=forecast_df,
        runtime_df=runtime_df,
        config={
            "experiment_name": lear_config.experiment_name,
            "mode": "daily_energy_arena_point_base",
            "forecast_date": forecast_day.date().isoformat(),
            "history_days": history_days,
            "model_metadata": build_model_metadata(lear_config),
        },
        export_dir=export_dir,
    )
    return export_dir / "forecast.csv"


def materialize_daily_configs(
    point_payload: dict[str, Any],
    quantile_payload: dict[str, Any],
    forecast_date: date,
    paths: DailyPaths,
    point_forecast_path: Path,
    point_source_name: str,
    submit: bool,
    repo_root: Path,
) -> tuple[Path, Path]:
    point_generated = build_point_submission_payload(
        point_payload=point_payload,
        forecast_date=forecast_date,
        point_forecast_path=point_forecast_path,
        point_source_name=point_source_name,
        submit=submit,
    )
    quantile_generated = build_quantile_submission_payload(
        quantile_payload=quantile_payload,
        forecast_date=forecast_date,
        point_forecast_path=point_forecast_path,
        sqra_export_dir=paths.sqra_export_dir,
        submit=submit,
        repo_root=repo_root,
    )
    _write_yaml(paths.point_submission_config, point_generated)
    _write_yaml(paths.quantile_submission_config, quantile_generated)
    return paths.point_submission_config, paths.quantile_submission_config


def _run_point_submission(config_path: Path, submit: bool) -> None:
    run_config = validate_config_payload(load_config_payload(config_path), RunConfig, repo_root=find_repo_root(config_path))
    run_from_config(run_config, submit_override=submit)


def _pixi_executable() -> str:
    pixi = shutil.which("pixi")
    if pixi is not None:
        return pixi
    return str(Path.home() / ".pixi" / "bin" / "pixi")


def _run_quantile_submission(config_path: Path, submit: bool, repo_root: Path) -> None:
    command = [_pixi_executable(), "run", "-e", "sqra", "da-price-forecast", "--config", str(config_path)]
    if submit:
        command.append("--submit")
    subprocess.run(command, cwd=repo_root, check=True)


def run_daily_energy_arena(
    point_config_path: Path = DEFAULT_POINT_CONFIG,
    quantile_config_path: Path = DEFAULT_QUANTILE_CONFIG,
    forecast_date: date | None = None,
    target_tz: str = "Europe/Berlin",
    work_root: Path | None = None,
    submit: bool = True,
    run_quantile: bool = True,
) -> DailyPaths:
    repo_root = find_repo_root()
    day = forecast_date or tomorrow_in_tz(target_tz)
    paths = dated_work_paths(repo_root=repo_root, forecast_date=day, work_root=work_root)
    point_payload, quantile_payload = load_daily_inputs(point_config_path, quantile_config_path, repo_root)
    lear_config = prepare_lear_config_for_point_base(point_payload, day, paths, repo_root)
    history_days = sqra_train_days(quantile_payload, repo_root)

    print(f"Daily Energy Arena run for forecast_date={day.isoformat()}")
    print(f"Working directory: {paths.work_dir}")
    point_forecast_path = run_point_base_forecasts(
        lear_config=lear_config,
        forecast_date=day,
        history_days=history_days,
        export_dir=paths.point_base_dir,
    )
    point_config, quantile_config = materialize_daily_configs(
        point_payload=point_payload,
        quantile_payload=quantile_payload,
        forecast_date=day,
        paths=paths,
        point_forecast_path=point_forecast_path,
        point_source_name=_model_name_from_lear_config(lear_config),
        submit=submit,
        repo_root=repo_root,
    )

    _run_point_submission(point_config, submit=submit)
    if run_quantile:
        _run_quantile_submission(quantile_config, submit=submit, repo_root=repo_root)

    with open(paths.work_dir / "daily_run.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "forecast_date": day.isoformat(),
                "submit": submit,
                "point_config": str(point_config),
                "quantile_config": str(quantile_config),
                "point_forecast_path": str(point_forecast_path),
            },
            handle,
            indent=2,
        )
    return paths


def _parse_retry_until(value: str | None) -> datetime_time | None:
    if value is None:
        return None
    try:
        parsed = datetime.strptime(value, "%H:%M").time()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--retry-until must use HH:MM, for example 12:30.") from exc
    return parsed


def _retry_deadline(now: datetime, retry_until: datetime_time | None, target_tz: str) -> datetime | None:
    if retry_until is None:
        return None
    tz = ZoneInfo(target_tz)
    current = now.astimezone(tz) if now.tzinfo is not None else now.replace(tzinfo=tz)
    return datetime.combine(current.date(), retry_until, tzinfo=tz)


def run_daily_energy_arena_with_retries(
    *,
    point_config_path: Path,
    quantile_config_path: Path,
    forecast_date: date | None,
    target_tz: str,
    work_root: Path | None,
    submit: bool,
    run_quantile: bool,
    retry_until: datetime_time | None,
    retry_interval_minutes: float,
) -> DailyPaths:
    deadline = _retry_deadline(datetime.now(ZoneInfo(target_tz)), retry_until, target_tz)
    interval_seconds = max(retry_interval_minutes, 0.1) * 60
    attempt = 1

    while True:
        try:
            return run_daily_energy_arena(
                point_config_path=point_config_path,
                quantile_config_path=quantile_config_path,
                forecast_date=forecast_date,
                target_tz=target_tz,
                work_root=work_root,
                submit=submit,
                run_quantile=run_quantile,
            )
        except Exception as exc:
            now = datetime.now(ZoneInfo(target_tz))
            if deadline is None or now + timedelta(seconds=interval_seconds) > deadline:
                print(f"Daily Energy Arena attempt {attempt} failed and no retries remain: {exc}", flush=True)
                raise

            next_attempt = now + timedelta(seconds=interval_seconds)
            print(
                f"Daily Energy Arena attempt {attempt} failed: {exc}\n"
                f"Retrying at {next_attempt.isoformat()} until {deadline.isoformat()}.",
                flush=True,
            )
            attempt += 1
            time.sleep(interval_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the daily Energy Arena EXAA-only point + SQRA submission workflow.")
    parser.add_argument("--point-config", type=Path, default=DEFAULT_POINT_CONFIG)
    parser.add_argument("--quantile-config", type=Path, default=DEFAULT_QUANTILE_CONFIG)
    parser.add_argument("--forecast-date", type=date.fromisoformat, default=None, help="Target date; defaults to tomorrow.")
    parser.add_argument("--target-tz", default="Europe/Berlin")
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Generate payloads but do not submit to Energy Arena.")
    parser.add_argument("--skip-quantile", action="store_true", help="Run only the point base and point submission.")
    parser.add_argument("--retry-until", type=_parse_retry_until, default=None, help="Retry failed attempts until HH:MM in target timezone.")
    parser.add_argument("--retry-interval-minutes", type=float, default=10.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_daily_energy_arena_with_retries(
        point_config_path=args.point_config,
        quantile_config_path=args.quantile_config,
        forecast_date=args.forecast_date,
        target_tz=args.target_tz,
        work_root=args.work_root,
        submit=not args.dry_run,
        run_quantile=not args.skip_quantile,
        retry_until=args.retry_until,
        retry_interval_minutes=args.retry_interval_minutes,
    )


if __name__ == "__main__":
    main()
