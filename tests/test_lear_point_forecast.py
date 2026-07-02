from __future__ import annotations

import numpy as np
import pandas as pd

from da_price_forecasting.models import lear


class _RecordingRegressor:
    fit_lengths: list[int] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def fit(self, X, y):
        assert np.isfinite(y).all()
        self.fit_lengths.append(len(y))
        self.coef_ = np.zeros(X.shape[1], dtype=float)
        self.intercept_ = float(np.mean(y))
        self.alpha_ = 0.0
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.intercept_, dtype=float)


def test_rolling_point_forecast_drops_nan_training_targets(monkeypatch) -> None:
    index = pd.date_range("2026-01-01", periods=9, freq="D", tz="Europe/Berlin")
    X = pd.DataFrame(
        {
            "exaa_d0_mtu_00": np.linspace(10.0, 18.0, len(index)),
            "weekday_0": (index.weekday == 0).astype(float),
        },
        index=index,
    )
    Y = pd.DataFrame(
        {
            mtu: np.linspace(40.0 + mtu, 48.0 + mtu, len(index))
            for mtu in range(96)
        },
        index=index,
    )
    Y.loc[index[3], :] = np.nan
    Y.loc[index[-1], :] = np.nan
    _RecordingRegressor.fit_lengths = []

    monkeypatch.setattr(lear, "LassoLarsCV", _RecordingRegressor)

    forecast, *_ = lear.rolling_point_forecast(
        X=X,
        Y=Y,
        forecast_days=[index[-1]],
        train_days=8,
        lars_start_date=pd.Timestamp("2025-01-01", tz="Europe/Berlin"),
    )

    assert len(forecast) == 96
    assert forecast["y_pred"].notna().all()
    assert set(_RecordingRegressor.fit_lengths) == {7}
