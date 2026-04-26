from __future__ import annotations

from datetime import date
from pathlib import Path

from pydantic import Field, model_validator

from ..paths import resolve_path
from .base import RepoConfigModel


class Era5DownloadConfig(RepoConfigModel):
    start_date: date = date(2026, 1, 1)
    end_date: date = date(2026, 4, 23)
    output_dir: Path = Path("data/raw/era5")
    area: list[float] = Field(default_factory=lambda: [56.0, 2.0, 46.0, 17.0])
    force: bool = False
    append_missing: bool = False

    @model_validator(mode="after")
    def _resolve_paths(self) -> "Era5DownloadConfig":
        self.output_dir = resolve_path(self.output_dir, self.repo_root)
        return self


class Era5AggregationConfig(RepoConfigModel):
    main_file: Path = Path("data/raw/era5/era5_main.grib")
    solar_file: Path = Path("data/raw/era5/era5_solar.grib")
    cluster_file: Path = Path("data/clustering/icon_d2_clustering_c25.parquet")
    output_parent: Path = Path("data/processed/era5_aggregated")
    main_variables: list[str] = Field(default_factory=lambda: ["u10", "v10", "t2m", "sp", "sd"])
    solar_variables: list[str] = Field(default_factory=lambda: ["ssrd", "fdir"])
    start_time: str = "2026-01-01 00:00"
    end_time: str = "2026-04-23 23:45"

    @model_validator(mode="after")
    def _resolve_paths(self) -> "Era5AggregationConfig":
        self.main_file = resolve_path(self.main_file, self.repo_root)
        self.solar_file = resolve_path(self.solar_file, self.repo_root)
        self.cluster_file = resolve_path(self.cluster_file, self.repo_root)
        self.output_parent = resolve_path(self.output_parent, self.repo_root)
        return self


class IconAggregationConfig(RepoConfigModel):
    lsdf_base: Path = Path("/Volumes/iip-projects/energy/climateData/icon_by_Max_Kleinebrahm")
    output_parent: Path = Path("data/processed/icon_aggregated")
    shapefile_path: Path = Path("data/shapefile/ne_10m_admin_0_countries.shp")
    n_clusters: int = 5
    buffer_km: int = 50
    plot_clusters: bool = False
    skip_existing_output: bool = True
    only_day: str | None = None
    only_run_hour: str | None = "09"
    start_date: date = date(2026, 1, 3)
    variables: list[str] = Field(
        default_factory=lambda: ["t_2m", "p", "u_10m", "v_10m", "aswdir_s", "aswdifd_s", "h_snow"]
    )

    @model_validator(mode="after")
    def _resolve_paths(self) -> "IconAggregationConfig":
        self.shapefile_path = resolve_path(self.shapefile_path, self.repo_root)
        self.output_parent = resolve_path(self.output_parent, self.repo_root)
        return self

    @property
    def root_dir_pattern(self) -> str:
        return str(self.lsdf_base / "dwd_icon_daily_*")
