from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from ..config import (
    VisualizationAncBarConfig,
    VisualizationAncHeatmapConfig,
    VisualizationArtifactTableConfig,
    VisualizationEvaluationReportConfig,
    VisualizationLoadForecastPlotConfig,
    VisualizationProbForecastConfig,
    VisualizationReportConfig,
)


def _configure_matplotlib_environment() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    cache_dir = Path(tempfile.gettempdir()) / "da_price_forecasting_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


def _load_visualization_script(repo_root: Path, module_name: str) -> ModuleType:
    _configure_matplotlib_environment()
    script_path = repo_root / "visualization" / f"{module_name}.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Visualization script not found: {script_path}")

    spec = importlib.util.spec_from_file_location(f"_da_price_forecasting_{module_name}", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load visualization script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_mae_table(config: VisualizationReportConfig, section: VisualizationArtifactTableConfig) -> list[Path]:
    module = _load_visualization_script(config.repo_root, "generate_mae_tables")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if section.artifacts:
        artifact_dirs = module._resolve_artifact_dirs(section.artifacts)
        if not artifact_dirs:
            raise ValueError("No artifact directories with metrics.csv found for MAE table.")
        output = section.output or (config.output_dir / "mae_artifacts.pdf")
        module.make_artifact_mae_figure(artifact_dirs=artifact_dirs, filename=output)
        return [output]

    output_paths = [
        config.output_dir / "mae_lear_fundamental.pdf",
        config.output_dir / "mae_lear_exaa_enriched.pdf",
        config.output_dir / "mae_exaa_only.pdf",
    ]
    vmin_fundamental, vmax_fundamental = module.get_vmin_vmax(module.lear_fundamental)
    vmin_exaa, vmax_exaa = module.get_vmin_vmax(module.lear_exaa_enriched)
    module.make_lear_figure(
        module.lear_fundamental,
        output_paths[0],
        vmin_fundamental,
        vmax_fundamental,
        module.CMAP_MINVAL_FUNDAMENTAL,
        module.CMAP_MAXVAL_FUNDAMENTAL,
    )
    module.make_lear_figure(
        module.lear_exaa_enriched,
        output_paths[1],
        vmin_exaa,
        vmax_exaa,
        module.CMAP_MINVAL_EXAA,
        module.CMAP_MAXVAL_EXAA,
    )
    module.make_exaa_only_figure(
        module.exaa_only,
        output_paths[2],
        vmin_exaa,
        vmax_exaa,
        module.CMAP_MINVAL_EXAA,
        module.CMAP_MAXVAL_EXAA,
    )
    return output_paths


def _run_runtime_table(config: VisualizationReportConfig, section: VisualizationArtifactTableConfig) -> list[Path]:
    module = _load_visualization_script(config.repo_root, "generate_comptime_tables")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if section.artifacts:
        artifact_dirs = module._resolve_artifact_dirs(section.artifacts)
        if not artifact_dirs:
            raise ValueError("No artifact directories with runtime.csv found for computation-time table.")
        output = section.output or (config.output_dir / "comptime_artifacts.pdf")
        module.make_artifact_runtime_figure(artifact_dirs=artifact_dirs, filename=output)
        return [output]

    output_paths = [
        config.output_dir / "comptime_lear_fundamental.pdf",
        config.output_dir / "comptime_lear_exaa_enriched.pdf",
        config.output_dir / "comptime_lear_exaa_only.pdf",
        config.output_dir / "comptime_sqra.pdf",
    ]
    vmin = 0
    vmax = module.get_vmax(module.lear_fundamental, module.lear_exaa_enriched, module.lear_exaa_only, module.sqra)
    module.make_lear_figure(module.lear_fundamental, output_paths[0], vmin, vmax, exaa_offset=0)
    module.make_lear_figure(module.lear_exaa_enriched, output_paths[1], vmin, vmax, exaa_offset=96)
    module.make_exaa_only_figure(module.lear_exaa_only, output_paths[2], vmin, vmax)
    module.make_sqra_figure(module.sqra, output_paths[3], vmin, vmax)
    return output_paths


def _run_prob_forecast(config: VisualizationReportConfig, section: VisualizationProbForecastConfig) -> list[Path]:
    module = _load_visualization_script(config.repo_root, "plot_prob_forecast_example")
    output = section.output or (config.output_dir / "plot_prob_forecast_example.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)

    df = module.load_forecast_csv(section.forecast_path)
    quantile_cols = module.infer_quantile_columns(df)
    module.plot_prob_forecast_paper(
        df_forecast=df,
        quantile_cols=quantile_cols,
        start_date=section.start,
        end_date=section.end,
        y_true_col=section.y_true_col,
        save_path=output,
        dpi=300,
        show=section.show,
    )
    return [output]


def _run_anc_bars(config: VisualizationReportConfig, section: VisualizationAncBarConfig) -> list[Path]:
    module = _load_visualization_script(config.repo_root, "plot_anc_bar")
    output_dir = section.output_dir or config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    if section.fundamental_csv is not None:
        features, values = module.load_anc_bars(section.fundamental_csv, top_n=section.top_n)
        output = output_dir / "anc_fundamental.pdf"
        module.save_anc_bar(features, values, output)
        output_paths.append(output)

    if section.exaa_csv is not None:
        features, values = module.load_anc_bars(section.exaa_csv, top_n=section.top_n)
        output = output_dir / "anc_exaa.pdf"
        module.save_anc_bar(features, values, output)
        output_paths.append(output)

    if output_paths:
        return output_paths

    output_paths = [output_dir / "anc_fundamental.pdf", output_dir / "anc_exaa.pdf"]
    module.save_anc_bar(module.features_fund, module.anc_fundamental, output_paths[0])
    module.save_anc_bar(module.features_exaa, module.anc_exaa, output_paths[1])
    return output_paths


def _run_anc_heatmaps(config: VisualizationReportConfig, section: VisualizationAncHeatmapConfig) -> list[Path]:
    module = _load_visualization_script(config.repo_root, "plot_cluster_heatmap")
    output_dir = section.output_dir or config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_gdf = module.build_cluster_polygons(section.cluster_parquet)
    germany = module.load_germany(section.shapefile, target_crs=cluster_gdf.crs)

    output_wind = output_dir / "anc_heatmap_wind.pdf"
    anc_wind = module.load_anc(section.anc_wind, train_days=section.train_days)
    module.plot_anc_heatmap(
        cluster_gdf=cluster_gdf,
        anc_df=anc_wind,
        germany=germany,
        title=f"Wind Cluster Importance by ANC (train_days={section.train_days})",
        output_path=output_wind,
        cmap=module.CMAP_WIND,
        dpi=module.DPI,
    )

    output_solar = output_dir / "anc_heatmap_solar.pdf"
    anc_solar = module.load_anc(section.anc_solar, train_days=section.train_days)
    module.plot_anc_heatmap(
        cluster_gdf=cluster_gdf,
        anc_df=anc_solar,
        germany=germany,
        title=f"Solar Cluster Importance by ANC (train_days={section.train_days})",
        output_path=output_solar,
        cmap=module.CMAP_SOLAR,
        dpi=module.DPI,
    )

    return [output_wind, output_solar]


def _load_panel_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation artifact not found: {path}")

    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Expected a timestamp column in {path}.")

    timestamps = pd.to_datetime(df.pop("timestamp"), utc=True).dt.tz_convert("Europe/Berlin")
    df.index = timestamps
    df.index.name = "timestamp"
    return df


def _load_metric_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation artifact not found: {path}")
    return pd.read_csv(path)


def _coerce_plot_timestamp(value: str | None, tz) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if tz is None:
        return timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(tz)
    return timestamp.tz_convert(tz)


def _filter_time_window(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    start_ts = _coerce_plot_timestamp(start, df.index.tz)
    end_ts = _coerce_plot_timestamp(end, df.index.tz)
    if start_ts is not None:
        df = df.loc[df.index >= start_ts]
    if end_ts is not None:
        df = df.loc[df.index <= end_ts]
    if df.empty:
        raise ValueError("The selected evaluation visualization window is empty.")
    return df


def _selected_models(available: list[str], requested: list[str]) -> list[str]:
    if not requested:
        return available

    missing = [model for model in requested if model not in available]
    if missing:
        raise ValueError(f"Requested model(s) not found in evaluation artifacts: {missing}")
    return [model for model in requested if model in available]


def _slugify(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in value)
    return "_".join(part for part in slug.split("_") if part)


def _apply_evaluation_style() -> None:
    _configure_matplotlib_environment()
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "grid.color": "#cccccc",
            "grid.linestyle": "--",
        }
    )


def _save_figure(
    fig,
    output_dir: Path,
    stem: str,
    formats: list[str],
    dpi: int,
    show: bool,
) -> list[Path]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for fmt in formats:
        output = output_dir / f"{stem}.{fmt}"
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
        output_paths.append(output)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_paths


def _save_figure_from_section(
    fig,
    *,
    output: Path | None,
    default_dir: Path,
    stem: str,
    formats: list[str],
    dpi: int,
    show: bool,
) -> list[Path]:
    if output is None:
        return _save_figure(fig, default_dir, stem, formats, dpi, show)
    if output.suffix == "":
        return _save_figure(fig, output.parent, output.name, formats, dpi, show)

    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return [output]


def _run_load_forecast_plot(
    config: VisualizationReportConfig,
    section: VisualizationLoadForecastPlotConfig,
) -> list[Path]:
    _apply_evaluation_style()
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    forecast = _load_panel_csv(section.forecast_path)
    forecast = _filter_time_window(forecast, section.start, section.end)
    required_cols = [section.actual_col, section.model_col]
    if section.benchmark_col is not None:
        required_cols.append(section.benchmark_col)
    missing = [column for column in required_cols if column not in forecast.columns]
    if missing:
        raise ValueError(f"Load forecast plot is missing required columns: {missing}")

    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    if section.benchmark_col is not None:
        ax.plot(
            forecast.index,
            forecast[section.benchmark_col],
            color="#8b949e",
            linewidth=1.2,
            label=section.benchmark_label,
            alpha=0.9,
            zorder=2,
        )
    ax.plot(
        forecast.index,
        forecast[section.model_col],
        color="#2f80ed",
        linewidth=1.35,
        label=section.model_label,
        zorder=3,
    )
    ax.plot(
        forecast.index,
        forecast[section.actual_col],
        color="#111111",
        linewidth=1.45,
        linestyle=(0, (3, 2)),
        label=section.actual_label,
        zorder=4,
    )

    model_rmse = _rmse(forecast[section.model_col], forecast[section.actual_col])
    title = section.title
    if section.benchmark_col is not None:
        benchmark_rmse = _rmse(forecast[section.benchmark_col], forecast[section.actual_col])
        title = f"{title}  |  RMSE: model {model_rmse:,.0f} MW vs ENTSO-E {benchmark_rmse:,.0f} MW"
    else:
        title = f"{title}  |  RMSE: {model_rmse:,.0f} MW"
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(section.ylabel)
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.grid(True, axis="both")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m", tz=forecast.index.tz))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    fig.autofmt_xdate(rotation=45, ha="right")
    fig.tight_layout()
    return _save_figure_from_section(
        fig,
        output=section.output,
        default_dir=config.output_dir,
        stem="load_forecast_comparison",
        formats=section.formats,
        dpi=section.dpi,
        show=section.show,
    )


def _rmse(prediction: pd.Series, actual: pd.Series) -> float:
    valid = pd.concat([prediction, actual], axis=1).dropna()
    if valid.empty:
        return float("nan")
    errors = valid.iloc[:, 0].to_numpy(dtype=float) - valid.iloc[:, 1].to_numpy(dtype=float)
    return float(np.sqrt(np.mean(errors**2)))


def _plot_point_forecast(section: VisualizationEvaluationReportConfig, output_dir: Path) -> list[Path]:
    _apply_evaluation_style()
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    point_panel = _load_panel_csv(section.evaluation_dir / "point_panel.csv")
    point_panel = _filter_time_window(point_panel, section.start, section.end)
    model_cols = [col for col in point_panel.columns if col != section.y_true_col]
    model_cols = _selected_models(model_cols, section.models)

    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.plot(
        point_panel.index,
        point_panel[section.y_true_col],
        color="#202020",
        linewidth=1.5,
        label="Realized price",
        zorder=4,
    )
    for model in model_cols:
        ax.plot(point_panel.index, point_panel[model], linewidth=1.2, label=model, alpha=0.9)

    ax.set_title("Point Forecast Evaluation")
    ax.set_xlabel("Date")
    ax.set_ylabel("Electricity price [EUR/MWh]")
    ax.grid(True, axis="y")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.9)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "point_forecast", section.formats, section.dpi, section.show)


def _infer_quantile_models(df: pd.DataFrame) -> dict[str, list[tuple[float, str]]]:
    models: dict[str, list[tuple[float, str]]] = {}
    for column in df.columns:
        if "_q" not in column:
            continue
        model, quantile_text = column.rsplit("_q", 1)
        try:
            quantile = float(quantile_text)
        except ValueError:
            continue
        models.setdefault(model, []).append((quantile, column))

    return {
        model: sorted(columns, key=lambda item: item[0])
        for model, columns in models.items()
        if len(columns) >= 3
    }


def _plot_quantile_fans(
    section: VisualizationEvaluationReportConfig, output_dir: Path, repo_root: Path
) -> list[Path]:
    module = _load_visualization_script(repo_root, "plot_prob_forecast_example")
    quantile_panel = _load_panel_csv(section.evaluation_dir / "quantile_panel.csv")
    quantile_models = _infer_quantile_models(quantile_panel)
    selected = _selected_models(list(quantile_models), section.models)
    output_paths: list[Path] = []

    for model in selected:
        df_plot = pd.DataFrame(index=quantile_panel.index)
        if section.y_true_col in quantile_panel.columns:
            df_plot[section.y_true_col] = quantile_panel[section.y_true_col]

        quantile_cols: list[str] = []
        for quantile, column in quantile_models[model]:
            target_col = f"q_{quantile:g}"
            df_plot[target_col] = quantile_panel[column]
            quantile_cols.append(target_col)

        for fmt in section.formats:
            output = output_dir / f"quantile_fan_{_slugify(model)}.{fmt}"
            module.plot_prob_forecast_paper(
                df_forecast=df_plot,
                quantile_cols=quantile_cols,
                start_date=section.start,
                end_date=section.end,
                y_true_col=section.y_true_col,
                save_path=output,
                dpi=section.dpi,
                show=section.show,
            )
            output_paths.append(output)

    return output_paths


def _format_metric_cell(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _plot_metric_summary(section: VisualizationEvaluationReportConfig, output_dir: Path) -> list[Path]:
    _apply_evaluation_style()
    import matplotlib.pyplot as plt

    rows: list[dict[str, object]] = []
    point_path = section.evaluation_dir / "point_metrics.csv"
    quantile_path = section.evaluation_dir / "quantile_metrics.csv"

    if point_path.exists():
        point_metrics = _load_metric_csv(point_path)
        if section.models:
            point_metrics = point_metrics[point_metrics["model"].isin(section.models)]
        for _, row in point_metrics.iterrows():
            rows.append(
                {
                    "type": "point",
                    "model": row["model"],
                    "mae": row.get("mae"),
                    "rmse": row.get("rmse"),
                    "bias": row.get("bias"),
                    "aps": pd.NA,
                    "n_obs": row.get("n_obs"),
                }
            )

    if quantile_path.exists():
        quantile_metrics = _load_metric_csv(quantile_path)
        if section.models:
            quantile_metrics = quantile_metrics[quantile_metrics["model"].isin(section.models)]
        for _, row in quantile_metrics.iterrows():
            rows.append(
                {
                    "type": "quantile",
                    "model": row["model"],
                    "mae": row.get("mae_median"),
                    "rmse": pd.NA,
                    "bias": pd.NA,
                    "aps": row.get("aps"),
                    "n_obs": row.get("n_obs"),
                }
            )

    if not rows:
        raise ValueError("No metric rows found for the requested evaluation visualization.")

    table_df = pd.DataFrame(rows)
    display_df = table_df[["type", "model", "mae", "rmse", "bias", "aps", "n_obs"]].map(_format_metric_cell)
    fig_height = max(2.4, 0.42 * (len(display_df) + 2))
    fig, ax = plt.subplots(figsize=(12.5, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=["Type", "Model", "MAE", "RMSE", "Bias", "APS", "N"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    table.auto_set_column_width(col=list(range(len(display_df.columns))))
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8edf3")
        elif row % 2 == 0:
            cell.set_facecolor("#f7f9fb")
    ax.set_title("Evaluation Metric Summary", fontsize=13, pad=16)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "metric_summary", section.formats, section.dpi, section.show)


def _plot_coverage(section: VisualizationEvaluationReportConfig, output_dir: Path) -> list[Path]:
    _apply_evaluation_style()
    import matplotlib.pyplot as plt
    import numpy as np

    coverage = _load_metric_csv(section.evaluation_dir / "quantile_coverage.csv")
    if section.models:
        coverage = coverage[coverage["model"].isin(section.models)]
    if coverage.empty:
        raise ValueError("No coverage rows found for the requested evaluation visualization.")

    labels = [
        row["interval"] if len(coverage["model"].unique()) == 1 else f"{row['model']}\n{row['interval']}"
        for _, row in coverage.iterrows()
    ]
    x = np.arange(len(coverage))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(7.0, len(labels) * 1.2), 4.6))
    ax.bar(x - width / 2, coverage["nominal_coverage"], width, label="Nominal", color="#8da0cb")
    ax.bar(x + width / 2, coverage["empirical_coverage"], width, label="Empirical", color="#66c2a5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Coverage")
    ax.set_title("Prediction Interval Coverage")
    ax.grid(True, axis="y")
    ax.legend(frameon=True, framealpha=0.9)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "coverage", section.formats, section.dpi, section.show)


def _plot_error_by_hour(section: VisualizationEvaluationReportConfig, output_dir: Path) -> list[Path]:
    _apply_evaluation_style()
    import matplotlib.pyplot as plt

    point_panel = _load_panel_csv(section.evaluation_dir / "point_panel.csv")
    model_cols = [col for col in point_panel.columns if col != section.y_true_col]
    model_cols = _selected_models(model_cols, section.models)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    hours = point_panel.index.hour
    for model in model_cols:
        abs_error = (point_panel[model] - point_panel[section.y_true_col]).abs()
        hourly = abs_error.groupby(hours).mean()
        ax.plot(hourly.index, hourly.values, marker="o", linewidth=1.4, label=model)

    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(0, 23)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean absolute error [EUR/MWh]")
    ax.set_title("Point Forecast Error by Hour")
    ax.grid(True, axis="y")
    ax.legend(loc="best", fontsize=9, frameon=True, framealpha=0.9)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "error_by_hour", section.formats, section.dpi, section.show)


def _run_evaluation_report(
    config: VisualizationReportConfig, section: VisualizationEvaluationReportConfig
) -> list[Path]:
    output_dir = section.output_dir or config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    known_plots = {
        "point_forecast": lambda: _plot_point_forecast(section, output_dir),
        "quantile_fan": lambda: _plot_quantile_fans(section, output_dir, config.repo_root),
        "metric_summary": lambda: _plot_metric_summary(section, output_dir),
        "coverage": lambda: _plot_coverage(section, output_dir),
        "error_by_hour": lambda: _plot_error_by_hour(section, output_dir),
    }
    unknown = [plot for plot in section.plots if plot not in known_plots]
    if unknown:
        raise ValueError(f"Unknown evaluation plot(s): {unknown}")

    generated: list[Path] = []
    for plot in section.plots:
        generated.extend(known_plots[plot]())
    return generated


def run_visualization_report(config: VisualizationReportConfig) -> list[Path]:
    """Generate configured report figures from saved pipeline artifacts."""
    generated: list[Path] = []

    if config.mae_table is not None and config.mae_table.enabled:
        generated.extend(_run_mae_table(config, config.mae_table))

    if config.runtime_table is not None and config.runtime_table.enabled:
        generated.extend(_run_runtime_table(config, config.runtime_table))

    if config.probabilistic_forecast is not None and config.probabilistic_forecast.enabled:
        generated.extend(_run_prob_forecast(config, config.probabilistic_forecast))

    if config.load_forecast_plot is not None and config.load_forecast_plot.enabled:
        generated.extend(_run_load_forecast_plot(config, config.load_forecast_plot))

    if config.anc_bars is not None and config.anc_bars.enabled:
        generated.extend(_run_anc_bars(config, config.anc_bars))

    if config.anc_heatmaps is not None and config.anc_heatmaps.enabled:
        generated.extend(_run_anc_heatmaps(config, config.anc_heatmaps))

    if config.evaluation_report is not None and config.evaluation_report.enabled:
        generated.extend(_run_evaluation_report(config, config.evaluation_report))

    if not generated:
        print("No visualization outputs were enabled in the config.")
    else:
        print(f"Generated {len(generated)} visualization artifact(s).")

    return generated
