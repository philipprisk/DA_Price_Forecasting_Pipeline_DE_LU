from __future__ import annotations

import numpy as np
import pandas as pd

from da_price_forecasting.config import TabpfnLocalConfig
from da_price_forecasting.features import covariates as shared_covariates
from da_price_forecasting.pipelines import tabpfn_ts
from da_price_forecasting.pipelines.tabpfn_local import (
    _select_training_context,
    _transform_target,
    build_tabpfn_local_features,
)


def _config(**overrides) -> TabpfnLocalConfig:
    payload = {
        "repo_root": ".",
        "target_tz": "Europe/Berlin",
        "entsoe_start_date": "2025-01-01T00:00:00+01:00",
        "entsoe_end_date": "2026-02-03T23:45:00+01:00",
        "test_start": "2026-02-01T00:00:00+01:00",
        "test_end": "2026-02-03T23:45:00+01:00",
    } | overrides
    return TabpfnLocalConfig.model_validate(payload)


def test_curve_features_add_neighbor_and_summary_columns() -> None:
    index = pd.date_range("2026-02-01T00:00:00+01:00", periods=96, freq="15min")
    history = pd.date_range("2026-01-20T00:00:00+01:00", periods=13 * 96, freq="15min")
    prices = pd.DataFrame({"price_da": np.linspace(50, 100, len(history))}, index=history)
    covariates = pd.DataFrame(
        {
            "price_exaa": np.arange(len(index), dtype=float),
            "load_fc": 1000.0 + np.arange(len(index), dtype=float),
        },
        index=index,
    )
    config = _config(
        add_curve_summary_features=True,
        curve_neighbor_offsets=[-1, 1],
        add_curve_features=False,
    )

    features = build_tabpfn_local_features(index, prices, covariates, config)

    assert "price_exaa_offset_-1mtu" in features.columns
    assert "price_exaa_offset_+1mtu" in features.columns
    assert "price_exaa_daily_spread" in features.columns
    assert "load_fc_evening_peak_mean" in features.columns


def test_curve_feature_sources_accept_semantic_covariate_names() -> None:
    index = pd.date_range("2026-02-01T00:00:00+01:00", periods=96, freq="15min")
    history = pd.date_range("2026-01-20T00:00:00+01:00", periods=13 * 96, freq="15min")
    prices = pd.DataFrame({"price_da": np.linspace(50, 100, len(history))}, index=history)
    covariates = pd.DataFrame(
        {
            "price_exaa": np.arange(len(index), dtype=float),
            "NTC_Net_MW": 5000.0 + np.arange(len(index), dtype=float),
        },
        index=index,
    )
    config = _config(
        add_curve_features=True,
        add_curve_summary_features=False,
        curve_neighbor_offsets=[],
        curve_feature_sources=["exaa", "ntc"],
    )

    features = build_tabpfn_local_features(index, prices, covariates, config)

    assert "price_exaa_curve_mtu_00" in features.columns
    assert "NTC_Net_MW_curve_mtu_00" in features.columns


def test_optional_engineered_features_add_price_stats_ramps_and_interactions() -> None:
    index = pd.date_range("2026-02-08T00:00:00+01:00", periods=96, freq="15min")
    history = pd.date_range("2026-01-29T00:00:00+01:00", periods=10 * 96, freq="15min")
    prices = pd.DataFrame({"price_da": np.arange(len(history), dtype=float)}, index=history)
    covariates = pd.DataFrame(
        {
            "price_exaa": 1000.0 + np.arange(len(index), dtype=float),
            "load_fc": 2000.0 + np.arange(len(index), dtype=float),
        },
        index=index,
    )
    config = _config(
        price_lag_days=[1],
        engineered_features={
            "holiday": False,
            "price_daily_stats": True,
            "price_rolling_stats": {
                "enabled": True,
                "windows_days": [7],
            },
            "covariate_ramps": {
                "sources": ["exaa", "load_forecast"],
                "horizons_mtu": [4, 12],
            },
            "interactions": {
                "seq2_minus_price_lag_d1": True,
            },
        },
    )

    features = build_tabpfn_local_features(index, prices, covariates, config)
    previous_day = pd.Timestamp("2026-02-07", tz="Europe/Berlin")
    previous_prices = prices.loc[previous_day:previous_day + pd.Timedelta(days=1) - pd.Timedelta(minutes=15), "price_da"]

    assert features.loc[index[0], "price_lastday_min"] == previous_prices.min()
    assert features.loc[index[0], "price_lastday_max"] == previous_prices.max()
    assert features.loc[index[0], "price_lastday_close"] == previous_prices.iloc[-1]
    assert "price_roll_7d_mean" in features.columns
    assert "price_roll_7d_std" in features.columns
    assert features.loc[index[4], "price_exaa_ramp_4mtu"] == 4.0
    assert features.loc[index[12], "load_fc_ramp_12mtu"] == 12.0
    assert features.loc[index[0], "seq2_minus_price_lag1d"] == (
        features.loc[index[0], "price_exaa"] - features.loc[index[0], "price_lag1d"]
    )


def test_feature_config_selects_covariates_and_syncs_legacy_flags() -> None:
    config = _config(
        features={
            "covariates": ["ntc", "generation_unavailability"],
            "ntc_neighbors": ["FR", "NL"],
        }
    )

    assert config.features is not None
    assert config.features.covariates == ["ntc", "generation_unavailability"]
    assert config.features.ntc_neighbors == ["FR", "NL"]
    assert config.use_exaa is False
    assert config.use_load_forecast is False
    assert "ntc_unavail" in config.resolved_experiment_name


def test_legacy_covariate_flags_populate_feature_config() -> None:
    config = _config(use_exaa=True, use_load_forecast=False)

    assert config.features is not None
    assert config.features.covariates == ["exaa"]
    assert config.use_exaa is True
    assert config.use_load_forecast is False


def test_tabpfn_covariate_builder_uses_selected_fetchers(monkeypatch) -> None:
    index = pd.date_range("2026-02-01T00:00:00+01:00", periods=2, freq="15min")
    calls = []

    def fake_fetch_prices_exaa(**kwargs):
        calls.append(("exaa", kwargs))
        return pd.DataFrame({"price_exaa": [1.0, 2.0]}, index=index)

    def fake_fetch_load_forecast(**kwargs):
        calls.append(("load_forecast", kwargs))
        return pd.DataFrame({"load_fc": [3.0, 4.0]}, index=index)

    def fake_fetch_ntc_data(**kwargs):
        calls.append(("ntc", kwargs))
        return pd.DataFrame({"NTC_Net_MW": [5.0, 6.0]}, index=index)

    def fake_fetch_generation_unavailability(**kwargs):
        calls.append(("generation_unavailability", kwargs))
        return pd.DataFrame({"Unavail_Total_MW": [7.0, 8.0]}, index=index)

    def fake_fetch_foreign_day_ahead_prices(**kwargs):
        calls.append(("foreign_day_ahead_prices", kwargs))
        return pd.DataFrame({"price_da_FR": [11.0, 12.0]}, index=index)

    def fake_fetch_yfinance_commodity_prices(**kwargs):
        calls.append(("commodities", kwargs))
        return pd.DataFrame({"gas_ttf": [9.0, 10.0]}, index=index)

    def fake_fetch_investiny_commodity_prices(**kwargs):
        calls.append(("commodities", kwargs))
        return pd.DataFrame({"gas_ttf": [9.0, 10.0]}, index=index)

    monkeypatch.setattr(
        shared_covariates,
        "_require_entsoe_fetchers",
        lambda: (
            fake_fetch_prices_exaa,
            fake_fetch_load_forecast,
            fake_fetch_ntc_data,
            fake_fetch_generation_unavailability,
            fake_fetch_foreign_day_ahead_prices,
        ),
    )
    monkeypatch.setattr(
        "da_price_forecasting.data.commodities.fetch_yfinance_commodity_prices",
        fake_fetch_yfinance_commodity_prices,
    )
    monkeypatch.setattr(
        "da_price_forecasting.data.commodities.fetch_investiny_commodity_prices",
        fake_fetch_investiny_commodity_prices,
    )
    config = _config(
        features={
            "covariates": ["exaa", "load_forecast", "ntc", "generation_unavailability", "commodities"],
            "ntc_neighbors": ["FR"],
            "commodities": {
                "provider": "investiny",
                "lag_days": 2,
                "instruments": [
                    {
                        "name": "gas_ttf",
                        "investing_id": 1178699,
                    }
                ],
            },
        }
    )

    covariates = tabpfn_ts._build_tabpfn_covariates(
        config,
        start_day=pd.Timestamp("2026-02-01", tz="Europe/Berlin"),
        end_day=pd.Timestamp("2026-02-01", tz="Europe/Berlin"),
    )

    assert covariates.columns.tolist() == ["price_exaa", "load_fc", "NTC_Net_MW", "Unavail_Total_MW", "gas_ttf"]
    assert covariates.index.name == "timestamp"
    assert calls[2][1]["neighbors"] == ["FR"]
    assert calls[4][1]["instruments"][0]["name"] == "gas_ttf"
    assert calls[4][1]["instruments"][0]["investing_id"] == 1178699
    assert [call[0] for call in calls] == [
        "exaa",
        "load_forecast",
        "ntc",
        "generation_unavailability",
        "commodities",
    ]


def test_recent_weekday_even_context_keeps_recent_and_older_rows() -> None:
    index = pd.date_range("2026-01-01T00:00:00+01:00", periods=40 * 96, freq="15min")
    train = pd.DataFrame({"x": np.arange(len(index), dtype=float)}, index=index)
    y_train = pd.Series(np.arange(len(index), dtype=float), index=index)
    forecast_day = pd.Timestamp("2026-02-10T00:00:00+01:00")
    config = _config(
        context_selection="recent_weekday_even",
        context_recent_days=3,
        context_same_weekday_fraction=0.25,
        context_even_history_fraction=0.25,
        max_train_rows=500,
    )

    selected, selected_y = _select_training_context(train, y_train, forecast_day, config)

    assert len(selected) == 500
    assert selected.index.equals(selected_y.index)
    assert selected.index.min() < forecast_day - pd.Timedelta(days=10)
    assert selected.index.max() == train.index.max()


def test_asinh_robust_target_transform_is_finite() -> None:
    y = pd.Series([-100.0, -10.0, 0.0, 20.0, 1000.0])
    transformed, params = _transform_target(y, _config(target_transform="asinh_robust"))

    assert params["kind"] == "asinh_robust"
    assert np.isfinite(transformed).all()
