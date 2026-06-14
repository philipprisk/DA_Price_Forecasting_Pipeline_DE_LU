from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from da_price_forecasting.config import RenewableGenerationModelConfig
from da_price_forecasting.pipelines.common import save_timestamp_csv
from da_price_forecasting.pipelines.renewable_generation import (
    _apply_rolling_bias_correction,
    _build_partial_actual_generation_features,
    _build_solar_physics_proxy_baselines,
    _target_baseline_column,
    build_renewable_generation_dataset,
)


def test_renewable_generation_dataset_keeps_future_proxy_rows_without_actuals(tmp_path: Path) -> None:
    target_tz = "Europe/Berlin"
    proxy_index = pd.date_range("2026-05-29T00:00:00+02:00", periods=3 * 96, freq="15min")
    actual_index = proxy_index[proxy_index < pd.Timestamp("2026-05-31T00:00:00+02:00")]

    proxy_file = tmp_path / "renewable_proxy.csv"
    actual_file = tmp_path / "actual_generation.csv"
    save_timestamp_csv(
        pd.DataFrame(
            {
                "solar_proxy_mw": np.linspace(0.0, 100.0, len(proxy_index)),
                "wind_proxy_mw": np.linspace(100.0, 200.0, len(proxy_index)),
            },
            index=proxy_index,
        ),
        proxy_file,
    )
    save_timestamp_csv(
        pd.DataFrame(
            {
                "Solar_Actual_MW": np.linspace(0.0, 100.0, len(actual_index)),
                "Wind_Total_Actual_MW": np.linspace(100.0, 200.0, len(actual_index)),
                "Renewable_Total_Actual_MW": np.linspace(100.0, 300.0, len(actual_index)),
            },
            index=actual_index,
        ),
        actual_file,
    )

    config = RenewableGenerationModelConfig(
        repo_root=tmp_path,
        target_tz=target_tz,
        entsoe_start_date=date(2026, 5, 1),
        entsoe_end_date=date(2026, 5, 30),
        actual_generation_file=actual_file,
        renewable_proxy_file=proxy_file,
        unavailability_file=tmp_path / "unavailability.csv",
        icon_dir=tmp_path / "icon",
        export_dir=tmp_path / "export",
        include_dwd_cluster_features=False,
        include_unavailability=False,
        target_columns=["Solar_Actual_MW", "Wind_Total_Actual_MW"],
        test_start=date(2026, 5, 31),
        test_end=date(2026, 5, 31),
    )

    dataset = build_renewable_generation_dataset(config)
    forecast_day = pd.Timestamp("2026-05-31T00:00:00+02:00")

    assert dataset.index.max() == pd.Timestamp("2026-05-31T23:45:00+02:00")
    assert len(dataset.loc[forecast_day:]) == 96
    assert dataset.loc[forecast_day:, "Solar_Actual_MW"].isna().all()
    assert dataset.loc[forecast_day:, "solar_proxy_mw"].notna().all()


def test_partial_actual_generation_features_use_previous_morning() -> None:
    target_tz = "Europe/Berlin"
    actual_index = pd.date_range("2026-04-20T00:00:00+02:00", "2026-04-30T23:45:00+02:00", freq="15min")
    actual = pd.DataFrame(
        {
            "Solar_50Hertz_Actual_MW": np.arange(len(actual_index), dtype=float),
        },
        index=actual_index,
    )
    target_index = pd.date_range("2026-04-29T00:00:00+02:00", periods=96, freq="15min")

    features = _build_partial_actual_generation_features(
        actual,
        target_index,
        columns=["Solar_50Hertz_Actual_MW"],
        reference_day=1,
        comparison_lag_days=7,
        morning_end_hour=10,
        morning_end_minute=0,
    )

    source = actual.loc[
        pd.Timestamp("2026-04-28T00:00:00+02:00"):pd.Timestamp("2026-04-28T10:00:00+02:00"),
        "Solar_50Hertz_Actual_MW",
    ]
    comparison = actual.loc[
        pd.Timestamp("2026-04-21T00:00:00+02:00"):pd.Timestamp("2026-04-21T10:00:00+02:00"),
        "Solar_50Hertz_Actual_MW",
    ]
    timestamp = pd.Timestamp("2026-04-29T12:00:00+02:00")

    assert np.isclose(features.loc[timestamp, "partial_gen_solar_50hertz_d1_00_1000_mean"], source.mean())
    assert np.isclose(features.loc[timestamp, "partial_gen_solar_50hertz_d1_00_1000_last"], source.iloc[-1])
    assert np.isclose(
        features.loc[timestamp, "partial_gen_solar_50hertz_d1_00_1000_mean_diff_d7"],
        source.mean() - comparison.mean(),
    )


def test_rolling_bias_correction_supports_explicit_overrides(tmp_path: Path) -> None:
    index_history = pd.date_range("2026-04-20T00:00:00+02:00", periods=96, freq="15min")
    index_current = pd.date_range("2026-04-21T00:00:00+02:00", periods=96, freq="15min")
    history = pd.DataFrame(
        {
            "Solar_Model_MW": np.full(len(index_history), 110.0),
            "Solar_Actual_MW": np.full(len(index_history), 100.0),
        },
        index=index_history,
    )
    current = pd.DataFrame(
        {
            "Solar_Model_MW": np.full(len(index_current), 120.0),
            "Solar_Actual_MW": np.full(len(index_current), 100.0),
        },
        index=index_current,
    )
    config = RenewableGenerationModelConfig(
        repo_root=tmp_path,
        renewable_proxy_file=tmp_path / "proxy.csv",
        actual_generation_file=tmp_path / "actual.csv",
        unavailability_file=tmp_path / "unavailable.csv",
        icon_dir=tmp_path / "icon",
        export_dir=tmp_path / "export",
    )

    _apply_rolling_bias_correction(
        current,
        [history],
        day=pd.Timestamp("2026-04-21T00:00:00+02:00"),
        pred_col="Solar_Model_MW",
        true_col="Solar_Actual_MW",
        upper_bound=None,
        config=config,
        window_days=7,
        group="global",
        min_observations=24,
        shrinkage=0.5,
    )

    assert np.allclose(current["Solar_Model_MW"], 115.0)


def test_solar_physics_proxy_baselines_follow_tso_regions(tmp_path: Path) -> None:
    index = pd.date_range("2026-05-01T10:00:00+02:00", periods=2, freq="15min")
    proxy = pd.DataFrame(
        {
            "solar_50hertz_proxy_mw": [1000.0, 2000.0],
            "solar_50hertz_irradiance_cap_weighted_W_m2": [800.0, 900.0],
            "solar_50hertz_t2m_cap_weighted_K": [293.15, 303.15],
            "solar_amprion_proxy_mw": [500.0, 600.0],
            "solar_amprion_irradiance_cap_weighted_W_m2": [700.0, 800.0],
            "solar_amprion_t2m_cap_weighted_K": [288.15, 293.15],
        },
        index=index,
    )
    config = RenewableGenerationModelConfig(
        repo_root=tmp_path,
        renewable_proxy_file=tmp_path / "proxy.csv",
        actual_generation_file=tmp_path / "actual.csv",
        unavailability_file=tmp_path / "unavailable.csv",
        icon_dir=tmp_path / "icon",
        export_dir=tmp_path / "export",
    )

    features = _build_solar_physics_proxy_baselines(proxy, config)

    assert "solar_50hertz_physics_proxy_mw" in features.columns
    assert "solar_amprion_physics_proxy_mw" in features.columns
    assert np.allclose(
        features["Renewable_Solar_Physics_Proxy_MW"],
        features["solar_50hertz_physics_proxy_mw"] + features["solar_amprion_physics_proxy_mw"],
    )
    assert not np.allclose(
        features["solar_50hertz_physics_proxy_mw"],
        proxy["solar_50hertz_proxy_mw"],
    )


def test_solar_physics_baseline_column_maps_control_area_target(tmp_path: Path) -> None:
    config = RenewableGenerationModelConfig(
        repo_root=tmp_path,
        renewable_proxy_file=tmp_path / "proxy.csv",
        actual_generation_file=tmp_path / "actual.csv",
        unavailability_file=tmp_path / "unavailable.csv",
        icon_dir=tmp_path / "icon",
        export_dir=tmp_path / "export",
        target_baseline_mode="solar_physics_proxy",
    )

    baseline = _target_baseline_column(
        "Solar_TenneT_Actual_MW",
        config,
        [
            "solar_50hertz_physics_proxy_mw",
            "solar_tennet_physics_proxy_mw",
            "wind_onshore_proxy_mw",
        ],
    )

    assert baseline == "solar_tennet_physics_proxy_mw"
    assert _target_baseline_column("Wind_Total_Actual_MW", config, ["wind_onshore_proxy_mw"]) is None
