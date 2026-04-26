from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path
from types import ModuleType

from ..config import (
    VisualizationAncBarConfig,
    VisualizationAncHeatmapConfig,
    VisualizationArtifactTableConfig,
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


def run_visualization_report(config: VisualizationReportConfig) -> list[Path]:
    """Generate configured report figures from saved pipeline artifacts."""
    generated: list[Path] = []

    if config.mae_table is not None and config.mae_table.enabled:
        generated.extend(_run_mae_table(config, config.mae_table))

    if config.runtime_table is not None and config.runtime_table.enabled:
        generated.extend(_run_runtime_table(config, config.runtime_table))

    if config.probabilistic_forecast is not None and config.probabilistic_forecast.enabled:
        generated.extend(_run_prob_forecast(config, config.probabilistic_forecast))

    if config.anc_bars is not None and config.anc_bars.enabled:
        generated.extend(_run_anc_bars(config, config.anc_bars))

    if config.anc_heatmaps is not None and config.anc_heatmaps.enabled:
        generated.extend(_run_anc_heatmaps(config, config.anc_heatmaps))

    if not generated:
        print("No visualization outputs were enabled in the config.")
    else:
        print(f"Generated {len(generated)} visualization artifact(s).")

    return generated
