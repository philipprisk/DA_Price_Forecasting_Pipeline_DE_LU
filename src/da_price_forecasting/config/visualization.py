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


class VisualizationReportConfig(RepoConfigModel):
    output_dir: Path = Path("output/figures")
    mae_table: VisualizationArtifactTableConfig | None = Field(default_factory=VisualizationArtifactTableConfig)
    runtime_table: VisualizationArtifactTableConfig | None = Field(default_factory=VisualizationArtifactTableConfig)
    probabilistic_forecast: VisualizationProbForecastConfig | None = Field(
        default_factory=VisualizationProbForecastConfig
    )
    anc_bars: VisualizationAncBarConfig | None = Field(default_factory=VisualizationAncBarConfig)
    anc_heatmaps: VisualizationAncHeatmapConfig | None = Field(default_factory=VisualizationAncHeatmapConfig)

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

        return self
