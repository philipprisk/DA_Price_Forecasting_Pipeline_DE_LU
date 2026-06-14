from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from da_price_forecasting.config import IconAggregationConfig
from da_price_forecasting.preprocessing.icon_d2_aggregation import (
    aggregate_cluster_values,
    cluster_mastr_solar_capacity_coordinates,
    cluster_mastr_solar_tso_capacity_coordinates,
)


def test_aggregate_cluster_values_uses_capacity_weights() -> None:
    values = np.array([10.0, 20.0, 100.0])
    labels = np.array([0, 0, 1])
    weights = np.array([1.0, 3.0, 0.0])

    aggregated = aggregate_cluster_values(values, labels, n_clusters=2, cluster_weights=weights)

    assert aggregated[0] == pytest.approx(17.5)
    assert aggregated[1] == pytest.approx(100.0)


def test_mastr_solar_clustering_maps_capacity_to_dwd_grid(tmp_path: Path) -> None:
    capacity_file = tmp_path / "installed_capacity.csv"
    pd.DataFrame(
        {
            "technology": ["pv", "pv", "wind_onshore"],
            "capacity_mw": [10.0, 30.0, 999.0],
            "lat": [50.0, 51.0, 52.0],
            "lon": [8.0, 9.0, 10.0],
            "operating_status": ["35", "35", "35"],
        }
    ).to_csv(capacity_file, index=False)

    config = IconAggregationConfig(
        repo_root=tmp_path,
        capacity_file=capacity_file,
        cluster_source="mastr_solar",
        n_clusters=2,
    )
    latitudes = np.array([49.5, 50.0, 51.0, 51.5])
    longitudes = np.array([7.5, 8.0, 9.0, 9.5])
    mask = np.ones((len(latitudes), len(longitudes)), dtype=bool)

    coords, labels, centroids, weights = cluster_mastr_solar_capacity_coordinates(
        latitudes,
        longitudes,
        mask,
        config,
    )

    assert len(coords) == mask.sum()
    assert set(labels) == {0, 1}
    assert centroids.shape == (2, 2)
    assert weights.sum() == pytest.approx(40.0)
    assert np.count_nonzero(weights) == 2


def test_mastr_solar_tso_clustering_builds_clusters_per_proxy_region(tmp_path: Path) -> None:
    capacity_file = tmp_path / "installed_capacity.csv"
    pd.DataFrame(
        {
            "technology": ["pv", "pv", "pv", "pv"],
            "capacity_mw": [10.0, 20.0, 30.0, 40.0],
            "lat": [52.0, 51.0, 49.0, 48.5],
            "lon": [13.0, 7.0, 11.0, 9.0],
            "operating_status": ["35"] * 4,
            "federal_state_code": [1400, 1409, 1403, 1402],
        }
    ).to_csv(capacity_file, index=False)

    config = IconAggregationConfig(
        repo_root=tmp_path,
        capacity_file=capacity_file,
        cluster_source="mastr_solar_tso",
        n_clusters=1,
    )
    latitudes = np.array([48.5, 49.0, 51.0, 52.0])
    longitudes = np.array([7.0, 9.0, 11.0, 13.0])
    mask = np.ones((len(latitudes), len(longitudes)), dtype=bool)

    coords, labels, centroids, weights, regions = cluster_mastr_solar_tso_capacity_coordinates(
        latitudes,
        longitudes,
        mask,
        config,
    )

    assert len(coords) == mask.sum()
    assert set(labels) == {0, 1, 2, 3}
    assert centroids.shape == (4, 2)
    assert regions.tolist() == [
        "solar_50hertz",
        "solar_amprion",
        "solar_tennet",
        "solar_transnetbw",
    ]
    assert weights.sum() == pytest.approx(100.0)
