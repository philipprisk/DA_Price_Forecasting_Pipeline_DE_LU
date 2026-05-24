from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from ..paths import resolve_path
from .base import RepoConfigModel


class VisualizationArtifactTableConfig(BaseModel):
    enabled: bool = False
    artifacts: list[str] = Field(default_factory=list)
    output: Path | None = None


class VisualizationProbForecastConfig(BaseModel):
    enabled: bool = False
    forecast_path: Path = Path("results/sqra_results/era5_fundamental/forecast.csv")
    output: Path | None = None
    start: str | None = "2026-02-02"
    end: str | None = "2026-02-05 23:45"
    y_true_col: str | None = "y_true"
    show: bool = False


class VisualizationLoadForecastPlotConfig(BaseModel):
    enabled: bool = False
    forecast_path: Path = Path("results/load_forecast_results/current_best/forecast.csv")
    output: Path | None = None
    formats: list[str] = Field(default_factory=lambda: ["png", "pdf"])
    start: str | None = None
    end: str | None = None
    actual_col: str = "Load_Actual_MW"
    model_col: str = "Load_Model_MW"
    benchmark_col: str | None = "Load_Benchmark_MW"
    actual_label: str = "Actual load"
    model_label: str = "Model forecast"
    benchmark_label: str = "ENTSO-E forecast"
    title: str = "Load Forecast Comparison"
    ylabel: str = "Load [MW]"
    dpi: int = 300
    show: bool = False

    @model_validator(mode="after")
    def _normalise_options(self) -> "VisualizationLoadForecastPlotConfig":
        self.formats = [fmt.removeprefix(".").lower() for fmt in self.formats]
        if not self.formats:
            raise ValueError("Load forecast plot requires at least one output format.")
        return self


class VisualizationAncBarConfig(BaseModel):
    enabled: bool = False
    fundamental_csv: Path | None = None
    exaa_csv: Path | None = None
    output_dir: Path | None = None
    top_n: int | None = None


class VisualizationAncHeatmapConfig(BaseModel):
    enabled: bool = False
    cluster_parquet: Path = Path("data/clustering/icon_d2_clustering_c5.parquet")
    anc_wind: Path = Path("results/lear_anc_results/era5/c5/d112/exaa/anc_wind_feature_results.csv")
    anc_solar: Path = Path("results/lear_anc_results/era5/c5/d112/exaa/anc_solar_feature_results.csv")
    shapefile: Path = Path("data/shapefile/ne_10m_admin_0_countries.shp")
    output_dir: Path | None = None
    train_days: int = 112


class VisualizationEvaluationReportConfig(BaseModel):
    enabled: bool = False
    evaluation_dir: Path = Path("results/evaluation/tabpfn_ts_march")
    output_dir: Path | None = None
    formats: list[str] = Field(default_factory=lambda: ["png", "pdf"])
    plots: list[str] = Field(
        default_factory=lambda: [
            "point_forecast",
            "quantile_fan",
            "metric_summary",
            "coverage",
            "error_by_hour",
        ]
    )
    models: list[str] = Field(default_factory=list)
    start: str | None = None
    end: str | None = None
    y_true_col: str = "y_true"
    dpi: int = 300
    show: bool = False

    @model_validator(mode="after")
    def _normalise_options(self) -> "VisualizationEvaluationReportConfig":
        self.formats = [fmt.removeprefix(".").lower() for fmt in self.formats]
        if not self.formats:
            raise ValueError("Evaluation visualization requires at least one output format.")
        self.plots = [plot.lower() for plot in self.plots]
        return self


class VisualizationReportConfig(RepoConfigModel):
    output_dir: Path = Path("output/figures")
    mae_table: VisualizationArtifactTableConfig | None = Field(default_factory=VisualizationArtifactTableConfig)
    runtime_table: VisualizationArtifactTableConfig | None = Field(default_factory=VisualizationArtifactTableConfig)
    probabilistic_forecast: VisualizationProbForecastConfig | None = Field(
        default_factory=VisualizationProbForecastConfig
    )
    load_forecast_plot: VisualizationLoadForecastPlotConfig | None = Field(
        default_factory=VisualizationLoadForecastPlotConfig
    )
    anc_bars: VisualizationAncBarConfig | None = Field(default_factory=VisualizationAncBarConfig)
    anc_heatmaps: VisualizationAncHeatmapConfig | None = Field(default_factory=VisualizationAncHeatmapConfig)
    evaluation_report: VisualizationEvaluationReportConfig | None = Field(
        default_factory=VisualizationEvaluationReportConfig
    )

    @model_validator(mode="after")
    def _resolve_paths(self) -> "VisualizationReportConfig":
        self.output_dir = resolve_path(self.output_dir, self.repo_root)

        if self.mae_table is not None:
            self.mae_table.artifacts = [
                str(resolve_path(Path(pattern), self.repo_root)) for pattern in self.mae_table.artifacts
            ]
            if self.mae_table.output is not None:
                self.mae_table.output = resolve_path(self.mae_table.output, self.repo_root)

        if self.runtime_table is not None:
            self.runtime_table.artifacts = [
                str(resolve_path(Path(pattern), self.repo_root)) for pattern in self.runtime_table.artifacts
            ]
            if self.runtime_table.output is not None:
                self.runtime_table.output = resolve_path(self.runtime_table.output, self.repo_root)

        if self.probabilistic_forecast is not None:
            self.probabilistic_forecast.forecast_path = resolve_path(
                self.probabilistic_forecast.forecast_path, self.repo_root
            )
            if self.probabilistic_forecast.output is not None:
                self.probabilistic_forecast.output = resolve_path(
                    self.probabilistic_forecast.output, self.repo_root
                )

        if self.load_forecast_plot is not None:
            self.load_forecast_plot.forecast_path = resolve_path(
                self.load_forecast_plot.forecast_path, self.repo_root
            )
            if self.load_forecast_plot.output is not None:
                self.load_forecast_plot.output = resolve_path(
                    self.load_forecast_plot.output, self.repo_root
                )

        if self.anc_bars is not None:
            if self.anc_bars.fundamental_csv is not None:
                self.anc_bars.fundamental_csv = resolve_path(self.anc_bars.fundamental_csv, self.repo_root)
            if self.anc_bars.exaa_csv is not None:
                self.anc_bars.exaa_csv = resolve_path(self.anc_bars.exaa_csv, self.repo_root)
            if self.anc_bars.output_dir is not None:
                self.anc_bars.output_dir = resolve_path(self.anc_bars.output_dir, self.repo_root)

        if self.anc_heatmaps is not None:
            self.anc_heatmaps.cluster_parquet = resolve_path(self.anc_heatmaps.cluster_parquet, self.repo_root)
            self.anc_heatmaps.anc_wind = resolve_path(self.anc_heatmaps.anc_wind, self.repo_root)
            self.anc_heatmaps.anc_solar = resolve_path(self.anc_heatmaps.anc_solar, self.repo_root)
            self.anc_heatmaps.shapefile = resolve_path(self.anc_heatmaps.shapefile, self.repo_root)
            if self.anc_heatmaps.output_dir is not None:
                self.anc_heatmaps.output_dir = resolve_path(self.anc_heatmaps.output_dir, self.repo_root)

        if self.evaluation_report is not None:
            self.evaluation_report.evaluation_dir = resolve_path(
                self.evaluation_report.evaluation_dir, self.repo_root
            )
            if self.evaluation_report.output_dir is not None:
                self.evaluation_report.output_dir = resolve_path(
                    self.evaluation_report.output_dir, self.repo_root
                )

        return self
