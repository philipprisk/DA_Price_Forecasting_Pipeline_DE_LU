from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from da_price_forecasting.scripts import dwd_icon_daily_update as daily_update


def test_tomorrow_in_tz_uses_local_day() -> None:
    now = datetime(2026, 5, 22, 10, 30, tzinfo=ZoneInfo("Europe/Berlin"))

    assert daily_update.tomorrow_in_tz("Europe/Berlin", now=now) == date(2026, 5, 23)


def test_daily_update_uses_one_target_forecast_day(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "load_config.yaml"
    config_path.write_text(
        f"""
kind: load_forecast_model
config:
  repo_root: "{tmp_path}"
  actual_load_file: "{tmp_path / 'actual_load.csv'}"
  entsoe_load_forecast_file: "{tmp_path / 'entsoe_load_forecast.csv'}"
  icon_dir: "{tmp_path / 'icon_aggregated_c25_run06'}"
  export_dir: "{tmp_path / 'export'}"
  include_holiday_features: false
  include_weather_features: true
  required_run: "06"
  dwd_folder_offset_date: "2025-10-26"
  dwd_icon_raw_dir: "{tmp_path / 'raw_dwd'}"
  dwd_icon_catch_up_missing_days: false
  dwd_icon_aggregation_n_clusters: 25
  dwd_icon_aggregation_cluster_source: mastr_solar_tso
  dwd_icon_aggregation_cluster_output_file: "{tmp_path / 'clusters.csv'}"
  dwd_icon_aggregation_capacity_file: "{tmp_path / 'capacity.csv'}"
  dwd_icon_aggregation_capacity_weighted: true
  dwd_icon_aggregation_shapefile_path: "{tmp_path / 'countries.shp'}"
""".lstrip(),
        encoding="utf-8",
    )
    calls = {}

    def fake_ensure_dwd_icon_weather(**kwargs):
        calls.update(kwargs)
        return [date(2026, 5, 22)]

    monkeypatch.setattr(daily_update, "ensure_dwd_icon_weather", fake_ensure_dwd_icon_weather)

    updated = daily_update.run_dwd_icon_daily_update(
        config_path=config_path,
        forecast_date=date(2026, 5, 23),
    )

    assert updated == [date(2026, 5, 22)]
    assert calls["forecast_start"] == date(2026, 5, 23)
    assert calls["forecast_end"] == date(2026, 5, 23)
    assert calls["run_hour"] == "06"
    assert calls["catch_up_missing_days"] is False
    assert calls["n_clusters"] == 25
    assert calls["cluster_source"] == "mastr_solar_tso"
    assert calls["cluster_output_file"] == tmp_path / "clusters.csv"
    assert calls["capacity_file"] == tmp_path / "capacity.csv"
    assert calls["capacity_weighted_aggregation"] is True
