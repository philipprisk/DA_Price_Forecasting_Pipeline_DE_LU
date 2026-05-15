from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ...config import (
    EnergyArenaSubmissionConfig,
    ForecastFileEnergyArenaSource,
    LearOperationalConfig,
    LearOperationalEnergyArenaSource,
    LoadForecastModelConfig,
    LoadForecastModelEnergyArenaSource,
    RunConfig,
    SqraConfig,
    SqraEnergyArenaSource,
    TabpfnLocalConfig,
    TabpfnLocalEnergyArenaSource,
    TabpfnTsConfig,
    TabpfnTsEnergyArenaSource,
    load_config,
    load_config_payload,
    validate_config_payload,
)


@dataclass
class EnergyArenaForecastResult:
    forecast: pd.DataFrame
    source_name: str
    model_config: Any | None = None
    metadata: dict[str, Any] | None = None
    default_value_column: str | None = None
    default_quantile_columns: list[str] | None = None


def load_forecast_frame(path: Path, target_tz: str) -> pd.DataFrame:
    """Load a saved forecast CSV with a timezone-aware timestamp index."""
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(target_tz)
    return df


def _load_embedded_or_path_config(
    source: Any,
    model_cls: type[Any],
    submission_config: EnergyArenaSubmissionConfig,
) -> Any:
    if source.config_path is not None:
        payload = load_config_payload(source.config_path)
        if payload.get("kind") is not None:
            run_config = validate_config_payload(payload, RunConfig, repo_root=submission_config.repo_root)
            if run_config.config_path is not None:
                return load_config(run_config.config_path, model_cls)
            return validate_config_payload(run_config.config, model_cls, repo_root=run_config.repo_root)
        return validate_config_payload(payload, model_cls, repo_root=submission_config.repo_root)
    return validate_config_payload(source.config or {}, model_cls, repo_root=submission_config.repo_root)


def _source_config_metadata(source: Any) -> dict[str, Any]:
    if source.config_path is not None:
        return {"source_config_path": str(source.config_path)}
    return {"source_config_embedded": True}


def infer_source_name(submission_config: EnergyArenaSubmissionConfig) -> str:
    """Derive a stable artifact directory name from the configured source."""
    source = submission_config.source
    if isinstance(source, LearOperationalEnergyArenaSource):
        config = _load_embedded_or_path_config(source, LearOperationalConfig, submission_config)
        return config.experiment_name

    if isinstance(source, SqraEnergyArenaSource):
        config = _load_embedded_or_path_config(source, SqraConfig, submission_config)
        return config.experiment_name

    if isinstance(source, TabpfnTsEnergyArenaSource):
        config = _load_embedded_or_path_config(source, TabpfnTsConfig, submission_config)
        return config.resolved_experiment_name

    if isinstance(source, TabpfnLocalEnergyArenaSource):
        config = _load_embedded_or_path_config(source, TabpfnLocalConfig, submission_config)
        return config.resolved_experiment_name

    if isinstance(source, LoadForecastModelEnergyArenaSource):
        config = _load_embedded_or_path_config(source, LoadForecastModelConfig, submission_config)
        return config.export_dir.name

    if isinstance(source, ForecastFileEnergyArenaSource):
        return source.name or source.path.parent.name or source.path.stem

    raise ValueError(f"Unsupported Energy Arena source: {source!r}")


def _forecast_day(submission_config: EnergyArenaSubmissionConfig) -> pd.Timestamp:
    day = pd.Timestamp(submission_config.forecast_date)
    if day.tz is None:
        day = day.tz_localize(submission_config.target_tz)
    else:
        day = day.tz_convert(submission_config.target_tz)
    return day.normalize()


def _run_or_load_lear(
    source: LearOperationalEnergyArenaSource,
    submission_config: EnergyArenaSubmissionConfig,
    forecast_dir: Path,
) -> EnergyArenaForecastResult:
    from ...pipelines.lear import run_lear_operational_prediction_pipeline

    config = _load_embedded_or_path_config(source, LearOperationalConfig, submission_config)
    if source.run_before_submit:
        result = run_lear_operational_prediction_pipeline(
            config=config,
            forecast_date=_forecast_day(submission_config),
            save_outputs=True,
            export_dir=forecast_dir,
        )
        forecast = result["forecast"]
    else:
        forecast = load_forecast_frame(config.resolved_export_dir / "forecast.csv", config.target_tz)

    return EnergyArenaForecastResult(
        forecast=forecast,
        source_name=config.experiment_name,
        model_config=config,
        metadata={
            "source_kind": source.kind,
            "run_before_submit": source.run_before_submit,
        }
        | _source_config_metadata(source),
        default_value_column="y_pred",
    )


def _run_or_load_sqra(
    source: SqraEnergyArenaSource,
    submission_config: EnergyArenaSubmissionConfig,
    forecast_dir: Path,
) -> EnergyArenaForecastResult:
    from ...pipelines.sqra import run_sqra_pipeline

    config = _load_embedded_or_path_config(source, SqraConfig, submission_config)
    if source.run_before_submit:
        forecast_day = _forecast_day(submission_config)
        config.test_start = forecast_day.to_pydatetime()
        config.test_end = (forecast_day + pd.Timedelta(minutes=15 * 95)).to_pydatetime()
        config.export_dir = forecast_dir
        result = run_sqra_pipeline(config=config, save_outputs=True, plot=False)
        forecast = result["forecast"]
    else:
        forecast = load_forecast_frame(config.export_dir / "forecast.csv", config.target_tz)

    quantile_columns = source.quantile_columns or config.quantile_columns
    default_value_column = source.value_column
    if default_value_column is None and "q0.500" in quantile_columns:
        default_value_column = "q0.500"

    return EnergyArenaForecastResult(
        forecast=forecast,
        source_name=config.experiment_name,
        model_config=config,
        metadata={
            "source_kind": source.kind,
            "run_before_submit": source.run_before_submit,
        }
        | _source_config_metadata(source),
        default_value_column=default_value_column,
        default_quantile_columns=quantile_columns,
    )


def _load_forecast_file(
    source: ForecastFileEnergyArenaSource,
    submission_config: EnergyArenaSubmissionConfig,
) -> EnergyArenaForecastResult:
    return EnergyArenaForecastResult(
        forecast=load_forecast_frame(source.path, submission_config.target_tz),
        source_name=source.name or source.path.parent.name or source.path.stem,
        metadata={
            "source_kind": source.kind,
            "forecast_path": str(source.path),
        },
        default_value_column=source.value_column or "y_pred",
        default_quantile_columns=source.quantile_columns,
    )


def _run_or_load_tabpfn_ts(
    source: TabpfnTsEnergyArenaSource,
    submission_config: EnergyArenaSubmissionConfig,
    forecast_dir: Path,
) -> EnergyArenaForecastResult:
    from ...pipelines.tabpfn_ts import run_tabpfn_ts_pipeline

    config = _load_embedded_or_path_config(source, TabpfnTsConfig, submission_config)
    if source.run_before_submit:
        forecast_day = _forecast_day(submission_config)
        config.test_start = forecast_day.to_pydatetime()
        config.test_end = (forecast_day + pd.Timedelta(minutes=15 * (config.forecast_horizon - 1))).to_pydatetime()
        config.entsoe_end_date = (forecast_day - pd.Timedelta(days=1)).to_pydatetime()
        config.export_dir = forecast_dir
        result = run_tabpfn_ts_pipeline(config=config, save_outputs=True)
        forecast = result["forecast"]
    else:
        forecast = load_forecast_frame(config.resolved_export_dir / "forecast.csv", config.target_tz)

    quantile_columns = source.quantile_columns or config.quantile_columns
    default_value_column = source.value_column or "y_pred"

    return EnergyArenaForecastResult(
        forecast=forecast,
        source_name=config.resolved_experiment_name,
        model_config=config,
        metadata={
            "source_kind": source.kind,
            "run_before_submit": source.run_before_submit,
        }
        | _source_config_metadata(source),
        default_value_column=default_value_column,
        default_quantile_columns=quantile_columns,
    )


def _run_or_load_tabpfn_local(
    source: TabpfnLocalEnergyArenaSource,
    submission_config: EnergyArenaSubmissionConfig,
    forecast_dir: Path,
) -> EnergyArenaForecastResult:
    from ...pipelines.tabpfn_local import run_tabpfn_local_pipeline

    config = _load_embedded_or_path_config(source, TabpfnLocalConfig, submission_config)
    if source.run_before_submit:
        forecast_day = _forecast_day(submission_config)
        config.test_start = forecast_day.to_pydatetime()
        config.test_end = (forecast_day + pd.Timedelta(minutes=15 * (config.forecast_horizon - 1))).to_pydatetime()
        config.entsoe_end_date = (forecast_day - pd.Timedelta(days=1)).to_pydatetime()
        config.export_dir = forecast_dir
        result = run_tabpfn_local_pipeline(config=config, save_outputs=True)
        forecast = result["forecast"]
    else:
        forecast = load_forecast_frame(config.resolved_export_dir / "forecast.csv", config.target_tz)

    quantile_columns = source.quantile_columns or config.quantile_columns
    default_value_column = source.value_column or "y_pred"

    return EnergyArenaForecastResult(
        forecast=forecast,
        source_name=config.resolved_experiment_name,
        model_config=config,
        metadata={
            "source_kind": source.kind,
            "run_before_submit": source.run_before_submit,
        }
        | _source_config_metadata(source),
        default_value_column=default_value_column,
        default_quantile_columns=quantile_columns,
    )


def _run_or_load_load_forecast_model(
    source: LoadForecastModelEnergyArenaSource,
    submission_config: EnergyArenaSubmissionConfig,
    forecast_dir: Path,
) -> EnergyArenaForecastResult:
    from ...pipelines.load_forecast import run_load_forecast_pipeline

    config = _load_embedded_or_path_config(source, LoadForecastModelConfig, submission_config)
    source_name = config.export_dir.name
    if source.run_before_submit:
        forecast_day = _forecast_day(submission_config)
        config.test_start = forecast_day.date()
        config.test_end = forecast_day.date()
        config.entsoe_end_date = forecast_day.date()
        config.export_dir = forecast_dir
        result = run_load_forecast_pipeline(config=config, save_outputs=True)
        forecast = result["forecast"]
    else:
        forecast = load_forecast_frame(config.export_dir / "forecast.csv", config.target_tz)

    return EnergyArenaForecastResult(
        forecast=forecast,
        source_name=source_name,
        model_config=config,
        metadata={
            "source_kind": source.kind,
            "run_before_submit": source.run_before_submit,
        }
        | _source_config_metadata(source),
        default_value_column=source.value_column or "Load_Model_MW",
    )


def load_or_run_forecast_source(
    submission_config: EnergyArenaSubmissionConfig,
    forecast_dir: Path,
) -> EnergyArenaForecastResult:
    """Load or generate the forecast configured for an Energy Arena submission."""
    source = submission_config.source
    if isinstance(source, LearOperationalEnergyArenaSource):
        return _run_or_load_lear(source, submission_config, forecast_dir)

    if isinstance(source, SqraEnergyArenaSource):
        return _run_or_load_sqra(source, submission_config, forecast_dir)

    if isinstance(source, ForecastFileEnergyArenaSource):
        return _load_forecast_file(source, submission_config)

    if isinstance(source, TabpfnTsEnergyArenaSource):
        return _run_or_load_tabpfn_ts(source, submission_config, forecast_dir)

    if isinstance(source, TabpfnLocalEnergyArenaSource):
        return _run_or_load_tabpfn_local(source, submission_config, forecast_dir)

    if isinstance(source, LoadForecastModelEnergyArenaSource):
        return _run_or_load_load_forecast_model(source, submission_config, forecast_dir)

    raise ValueError(f"Unsupported Energy Arena source: {source!r}")
