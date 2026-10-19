import pytest
from ranking import (recommend, nearest_cluster, cluster_centroids,
                     scaled_table, make_profile, PRESETS, SODIUM_COLUMN)


@pytest.mark.parametrize("preset", list(PRESETS))
@pytest.mark.parametrize("daily_limit", [1500, 2300])
def test_presets_return_ten_foods_under_cap(preset, daily_limit):
    result = recommend(make_profile(scaled_table, PRESETS[preset], daily_limit), 10)
    assert len(result) == 10
    assert (result[SODIUM_COLUMN] <= daily_limit / 3).all()


def test_scores_go_down_within_each_cluster():
    result = recommend(make_profile(scaled_table, PRESETS["low_carb"], 1500), 10)
    for _, group in result.groupby("cluster", sort=False):
        assert group["scores"].is_monotonic_decreasing


def test_short_cluster_is_filled_from_next_cluster():
    # Top vitamin D lands in a 1-food cluster; the old code returned 1 food
    profile = make_profile(scaled_table, {"Vitamin D (D2 + D3)": 1.0}, 1500)
    result = recommend(profile, 10)
    assert len(result) == 10
    assert result["cluster"].iloc[0] == nearest_cluster(cluster_centroids, profile)
    assert result["cluster"].nunique() > 1
    assert (result[SODIUM_COLUMN] <= 500).all()


def test_empty_cluster_does_not_crash():
    # Top cholesterol at 150 mg/day used to raise ValueError
    profile = make_profile(scaled_table, {"Cholesterol": 1.0}, 150)
    result = recommend(profile, 10)
    assert len(result) == 10
    assert (result[SODIUM_COLUMN] <= 50).all()


def test_nothing_passes_returns_empty_table():
    profile = make_profile(scaled_table, PRESETS["low_carb"], -3)
    result = recommend(profile, 10)
    assert len(result) == 0
    assert list(result.columns) == ["scores", "cluster", "description", SODIUM_COLUMN]


def test_no_duplicate_foods():
    profile = make_profile(scaled_table, {"Vitamin D (D2 + D3)": 1.0}, 1500)
    result = recommend(profile, 10)
    assert result.index.is_unique