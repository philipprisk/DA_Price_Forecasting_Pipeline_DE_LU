from __future__ import annotations

from typing import Any

import pandas as pd
from dotenv import load_dotenv

from ..config import LearAncConfig, LearOperationalConfig, WeatherSource
from ..data.entsoe import fetch_load_forecast, fetch_prices, fetch_prices_exaa
from ..data.weather import load_dwd, load_era5
from ..features.engineering import (
    build_dwd_features,
    build_era5_features,
    build_load_features,
    build_price_features,
    build_temporal_features,
    build_y_matrix,
    merge_all_features,
)
from ..models.lear import (
    evaluate_and_plot_forecast_from_df,
    rolling_anc_feature_importance,
    rolling_point_forecast,
    save_anc_outputs,
    save_experiment_outputs,
    save_prediction_outputs,
    summarize_feature_group_anc,
    summarize_solar_cluster_anc,
    summarize_wind_cluster_anc,
)


def _load_env(repo_root) -> None:
    load_dotenv(repo_root / ".env")


def _load_weather_features(
    weather_source: WeatherSource,
    era5_dirs,
    icon_dir,
    start_folder_date,
    required_run,
    skip_dates,
    folder_offset_date,
    target_tz: str,
):
    if weather_source == WeatherSource.ERA5:
        df_era5 = load_era5(dirs=era5_dirs, target_tz=target_tz)
        return build_era5_features(df_era5=df_era5)

    df_dwd_hourly, df_dwd_qh = load_dwd(
        icon_dir=icon_dir,
        start_folder_date=start_folder_date,
        required_run=required_run,
        skip_dates=set(skip_dates),
        folder_offset_date=folder_offset_date,
        target_tz=target_tz,
    )
    return build_dwd_features(df_hourly=df_dwd_hourly, df_qh=df_dwd_qh, tz_local=target_tz)


def prepare_lear_operational_dataset(config: LearOperationalConfig) -> dict[str, Any]:
    """Load data and construct X/Y for the operational LEAR run."""
    _load_env(config.repo_root)

    entsoe_start = pd.Timestamp(config.entsoe_start_date)
    entsoe_end = pd.Timestamp(config.entsoe_end_date)

    df_prices_15 = fetch_prices(
        start_day=entsoe_start,
        end_day=entsoe_end,
        country_code=config.country_code_entsoe,
        api_key_env=config.entsoe_api_key_env,
        target_tz=config.target_tz,
    )

    df_prices_exaa_15 = None
    if config.use_exaa or config.use_exaa_only:
        df_prices_exaa_15 = fetch_prices_exaa(
            start_day=entsoe_start,
            end_day=entsoe_end,
            country_code=config.country_code_entsoe,
            api_key_env=config.entsoe_api_key_env,
            target_tz=config.target_tz,
        )

    if config.use_exaa_only:
        X_lear_op = build_price_features(
            df_prices=df_prices_15,
            exaa_only=True,
            df_prices_exaa_15=df_prices_exaa_15,
        )
        dropped_X_lear_op = pd.DataFrame()
    else:
        df_load_fc_entso_15 = fetch_load_forecast(
            start_day=entsoe_start,
            end_day=entsoe_end,
            country_code=config.country_code_entsoe,
            api_key_env=config.entsoe_api_key_env,
            target_tz=config.target_tz,
        )
        df_weather_features = _load_weather_features(
            weather_source=config.weather_source,
            era5_dirs=config.era5_dirs,
            icon_dir=config.icon_dir,
            start_folder_date=config.start_folder_date,
            required_run=config.required_run,
            skip_dates=config.skip_dates,
            folder_offset_date=config.dwd_folder_offset_date,
            target_tz=config.target_tz,
        )
        daily_index = df_prices_15.index.normalize().unique().sort_values()
        X_lear_op, dropped_X_lear_op = merge_all_features(
            df_weather_features=df_weather_features,
            df_price_features=build_price_features(
                df_prices=df_prices_15,
                exaa_vector=config.use_exaa,
                df_prices_exaa_15=df_prices_exaa_15 if config.use_exaa else None,
            ),
            df_load_features=build_load_features(df_load_fc_entso_15),
            df_time_features=build_temporal_features(
                daily_index=daily_index,
                post_regime_start=pd.Timestamp(config.post_regime_start),
            ),
        )

    Y_lear_op = build_y_matrix(df_prices_15, X_lear_op.index)
    valid_mask = X_lear_op.notna().all(axis=1)
    X_lear_op = X_lear_op.loc[valid_mask]
    Y_lear_op = Y_lear_op.loc[valid_mask]

    return {
        "X": X_lear_op,
        "Y": Y_lear_op,
        "dropped_rows": dropped_X_lear_op,
        "prices": df_prices_15,
        "prices_exaa": df_prices_exaa_15,
    }


def run_lear_operational_pipeline(
    config: LearOperationalConfig,
    save_outputs: bool = True,
    plot: bool = False,
) -> dict[str, Any]:
    """Run the operational LEAR pipeline outside of the notebook."""
    dataset = prepare_lear_operational_dataset(config)
    forecast_days = pd.date_range(
        start=pd.Timestamp(config.test_start).normalize(),
        end=pd.Timestamp(config.test_end).normalize(),
        freq="D",
        tz=config.target_tz,
    )

    forecast_df, runtime_df, coef_df, intercept_df, degenerate_df = rolling_point_forecast(
        X=dataset["X"],
        Y=dataset["Y"],
        forecast_days=forecast_days,
        train_days=config.train_days_rolling,
        lars_start_date=pd.Timestamp(config.lars_start_date),
        use_vst=config.use_vst,
    )

    export_payload = {
        "experiment_name": config.experiment_name,
        "use_exaa": config.use_exaa,
        "use_exaa_only": config.use_exaa_only,
        "n_clusters": None if config.use_exaa_only else config.n_clusters,
        "weather_source": None if config.use_exaa_only else config.weather_source.value,
        "train_days": config.train_days_rolling,
        "use_vst": config.use_vst,
        "lars_start_date": str(pd.Timestamp(config.lars_start_date).date()),
        "test_start": str(pd.Timestamp(config.test_start).date()),
        "test_end": str(pd.Timestamp(config.test_end).date()),
    }

    if save_outputs:
        save_experiment_outputs(
            name=config.experiment_name,
            forecast_df=forecast_df,
            runtime_df=runtime_df,
            config=export_payload,
            export_dir=config.resolved_export_dir,
        )

    if plot:
        evaluate_and_plot_forecast_from_df(
            forecasts=forecast_df,
            start=str(pd.Timestamp(config.test_start).date()),
            end=str(pd.Timestamp(config.test_end).date()),
            model_label=config.experiment_name,
            title=f"Rolling Point Forecast vs Actual Prices - {config.experiment_name}",
        )

    return {
        "forecast": forecast_df,
        "runtime": runtime_df,
        "coefficients": coef_df,
        "intercepts": intercept_df,
        "degenerate_ssrd": degenerate_df,
        "dataset": dataset,
    }


def prepare_lear_operational_prediction_dataset(
    config: LearOperationalConfig,
    forecast_date: pd.Timestamp,
) -> dict[str, Any]:
    """Prepare a predict-only operational LEAR dataset for a future day."""
    _load_env(config.repo_root)

    entsoe_start = pd.Timestamp(config.entsoe_start_date)
    history_end = forecast_date - pd.Timedelta(days=1)
    feature_start = entsoe_start
    if feature_start.tz is None:
        feature_start = feature_start.tz_localize(config.target_tz)
    else:
        feature_start = feature_start.tz_convert(config.target_tz)
    daily_index = pd.date_range(
        start=feature_start.normalize(),
        end=forecast_date.normalize(),
        freq="D",
    )

    df_prices_15 = fetch_prices(
        start_day=entsoe_start,
        end_day=history_end,
        country_code=config.country_code_entsoe,
        api_key_env=config.entsoe_api_key_env,
        target_tz=config.target_tz,
    )

    df_prices_exaa_15 = None
    if config.use_exaa or config.use_exaa_only:
        df_prices_exaa_15 = fetch_prices_exaa(
            start_day=entsoe_start,
            end_day=forecast_date,
            country_code=config.country_code_entsoe,
            api_key_env=config.entsoe_api_key_env,
            target_tz=config.target_tz,
        )

    if config.use_exaa_only:
        X_lear_op = build_price_features(
            df_prices=df_prices_15,
            exaa_only=True,
            df_prices_exaa_15=df_prices_exaa_15,
            daily_index=daily_index,
        )
        dropped_X_lear_op = pd.DataFrame()
    else:
        df_load_fc_entso_15 = fetch_load_forecast(
            start_day=entsoe_start,
            end_day=forecast_date,
            country_code=config.country_code_entsoe,
            api_key_env=config.entsoe_api_key_env,
            target_tz=config.target_tz,
        )
        df_weather_features = _load_weather_features(
            weather_source=config.weather_source,
            era5_dirs=config.era5_dirs,
            icon_dir=config.icon_dir,
            start_folder_date=config.start_folder_date,
            required_run=config.required_run,
            skip_dates=config.skip_dates,
            folder_offset_date=config.dwd_folder_offset_date,
            target_tz=config.target_tz,
        )
        X_lear_op, dropped_X_lear_op = merge_all_features(
            df_weather_features=df_weather_features,
            df_price_features=build_price_features(
                df_prices=df_prices_15,
                exaa_vector=config.use_exaa,
                df_prices_exaa_15=df_prices_exaa_15 if config.use_exaa else None,
                daily_index=daily_index,
            ),
            df_load_features=build_load_features(df_load_fc_entso_15),
            df_time_features=build_temporal_features(
                daily_index=daily_index,
                post_regime_start=pd.Timestamp(config.post_regime_start),
            ),
        )

    Y_lear_op = build_y_matrix(df_prices_15, X_lear_op.index)

    valid_mask = X_lear_op.notna().all(axis=1)
    X_lear_op = X_lear_op.loc[valid_mask]
    Y_lear_op = Y_lear_op.loc[valid_mask]

    return {
        "X": X_lear_op,
        "Y": Y_lear_op,
        "dropped_rows": dropped_X_lear_op,
        "prices": df_prices_15,
        "prices_exaa": df_prices_exaa_15,
    }


def run_lear_operational_prediction_pipeline(
    config: LearOperationalConfig,
    forecast_date: pd.Timestamp,
    save_outputs: bool = True,
    export_dir=None,
) -> dict[str, Any]:
    """Run a forecast-only LEAR pipeline for a single operational target day."""
    forecast_day = pd.Timestamp(forecast_date)
    if forecast_day.tz is None:
        forecast_day = forecast_day.tz_localize(config.target_tz)
    else:
        forecast_day = forecast_day.tz_convert(config.target_tz)
    forecast_day = forecast_day.normalize()
    dataset = prepare_lear_operational_prediction_dataset(config=config, forecast_date=forecast_day)

    forecast_df, runtime_df, coef_df, intercept_df, degenerate_df = rolling_point_forecast(
        X=dataset["X"],
        Y=dataset["Y"],
        forecast_days=[forecast_day],
        train_days=config.train_days_rolling,
        lars_start_date=pd.Timestamp(config.lars_start_date),
        use_vst=config.use_vst,
    )

    forecast_payload = {
        "experiment_name": config.experiment_name,
        "mode": "predict_only",
        "forecast_date": str(forecast_day.date()),
        "use_exaa": config.use_exaa,
        "use_exaa_only": config.use_exaa_only,
        "n_clusters": None if config.use_exaa_only else config.n_clusters,
        "weather_source": None if config.use_exaa_only else config.weather_source.value,
        "train_days": config.train_days_rolling,
        "use_vst": config.use_vst,
        "lars_start_date": str(pd.Timestamp(config.lars_start_date).date()),
    }

    if save_outputs:
        save_prediction_outputs(
            forecast_df=forecast_df,
            runtime_df=runtime_df,
            config=forecast_payload,
            export_dir=export_dir or config.resolved_export_dir,
        )

    return {
        "forecast": forecast_df,
        "runtime": runtime_df,
        "coefficients": coef_df,
        "intercepts": intercept_df,
        "degenerate_ssrd": degenerate_df,
        "dataset": dataset,
    }


def prepare_lear_anc_dataset(config: LearAncConfig) -> dict[str, Any]:
    """Load data and construct X/Y for the ANC LEAR run."""
    _load_env(config.repo_root)

    entsoe_start = pd.Timestamp(config.entsoe_start_date)
    entsoe_end = pd.Timestamp(config.entsoe_end_date)

    df_prices_15 = fetch_prices(
        start_day=entsoe_start,
        end_day=entsoe_end,
        country_code=config.country_code_entsoe,
        api_key_env=config.entsoe_api_key_env,
        target_tz=config.target_tz,
    )

    df_prices_exaa_15 = None
    if config.use_exaa or config.use_exaa_only:
        df_prices_exaa_15 = fetch_prices_exaa(
            start_day=entsoe_start,
            end_day=entsoe_end,
            country_code=config.country_code_entsoe,
            api_key_env=config.entsoe_api_key_env,
            target_tz=config.target_tz,
        )

    if config.use_exaa_only:
        X_lear_anc = build_price_features(
            df_prices=df_prices_15,
            exaa_only=True,
            df_prices_exaa_15=df_prices_exaa_15,
        )
        dropped_X_lear_anc = pd.DataFrame()
    else:
        df_load_fc_entso_15 = fetch_load_forecast(
            start_day=entsoe_start,
            end_day=entsoe_end,
            country_code=config.country_code_entsoe,
            api_key_env=config.entsoe_api_key_env,
            target_tz=config.target_tz,
        )
        df_weather_features_anc = _load_weather_features(
            weather_source=config.weather_source,
            era5_dirs=config.era5_dirs,
            icon_dir=config.icon_dir,
            start_folder_date=config.start_folder_date,
            required_run=config.required_run,
            skip_dates=config.skip_dates,
            folder_offset_date=config.dwd_folder_offset_date,
            target_tz=config.target_tz,
        )
        daily_index = df_prices_15.index.normalize().unique().sort_values()
        X_lear_anc, dropped_X_lear_anc = merge_all_features(
            df_weather_features=df_weather_features_anc,
            df_price_features=build_price_features(
                df_prices=df_prices_15,
                exaa_vector=config.use_exaa,
                df_prices_exaa_15=df_prices_exaa_15 if config.use_exaa else None,
            ),
            df_load_features=build_load_features(df_load_fc_entso_15),
            df_time_features=build_temporal_features(daily_index=daily_index),
        )

    Y_lear_anc = build_y_matrix(df_prices_15, X_lear_anc.index)
    valid_mask = X_lear_anc.notna().all(axis=1)
    X_lear_anc = X_lear_anc.loc[valid_mask]
    Y_lear_anc = Y_lear_anc.loc[valid_mask]

    return {
        "X": X_lear_anc,
        "Y": Y_lear_anc,
        "dropped_rows": dropped_X_lear_anc,
        "prices": df_prices_15,
        "prices_exaa": df_prices_exaa_15,
    }


def run_lear_anc_pipeline(config: LearAncConfig, save_outputs: bool = True) -> dict[str, Any]:
    """Run the ANC LEAR pipeline outside of the notebook."""
    dataset = prepare_lear_anc_dataset(config)
    forecast_days = pd.date_range(
        start=pd.Timestamp(config.test_start).normalize(),
        end=pd.Timestamp(config.test_end).normalize(),
        freq="D",
        tz=config.target_tz,
    )

    anc_df = rolling_anc_feature_importance(
        X=dataset["X"],
        Y=dataset["Y"],
        forecast_days=forecast_days,
        train_days=config.train_days_rolling,
        lars_start_date=pd.Timestamp(config.lars_start_date),
    )

    _, anc_summary_df = summarize_feature_group_anc(anc_df)
    wind_anc_summary, wind_anc_export = summarize_wind_cluster_anc(anc_df, config.mtu_window_wind)
    solar_anc_summary, solar_anc_export = summarize_solar_cluster_anc(anc_df, config.mtu_window_solar)

    export_payload = {
        "experiment_name": config.experiment_name,
        "use_exaa": config.use_exaa,
        "use_exaa_only": config.use_exaa_only,
        "n_clusters": None if config.use_exaa_only else config.n_clusters,
        "weather_source": None if config.use_exaa_only else config.weather_source.value,
        "train_days": config.train_days_rolling,
        "lars_start_date": str(pd.Timestamp(config.lars_start_date).date()),
        "test_start": str(pd.Timestamp(config.test_start).date()),
        "test_end": str(pd.Timestamp(config.test_end).date()),
    }

    if save_outputs:
        save_anc_outputs(
            export_dir=config.resolved_export_dir,
            anc_summary_df=anc_summary_df,
            wind_anc_export=wind_anc_export,
            solar_anc_export=solar_anc_export,
            config=export_payload,
        )

    return {
        "anc_raw": anc_df,
        "anc_summary": anc_summary_df,
        "wind_anc_summary": wind_anc_summary,
        "wind_anc_export": wind_anc_export,
        "solar_anc_summary": solar_anc_summary,
        "solar_anc_export": solar_anc_export,
        "dataset": dataset,
    }
