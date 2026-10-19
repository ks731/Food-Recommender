import pandas as pd
import pytest
from constraints import filter_by_sodium, selected_table, SODIUM_COLUMN


def toy_table():
    """Five fake foods with sodium right around the 500 mg cap"""
    return pd.DataFrame({SODIUM_COLUMN: [0.0, 499.0, 500.0, 500.5, 2000.0]},
                        index=[1, 2, 3, 4, 5])


def test_food_exactly_at_cap_is_kept():
    kept = filter_by_sodium(toy_table(), 1500)
    assert 3 in kept.index


def test_food_just_over_cap_is_removed():
    kept = filter_by_sodium(toy_table(), 1500)
    assert 4 not in kept.index
    assert 5 not in kept.index


def test_keeps_exactly_the_right_foods():
    kept = filter_by_sodium(toy_table(), 1500)
    assert list(kept.index) == [1, 2, 3]


def test_beta_changes_the_cap():
    kept = filter_by_sodium(toy_table(), 1500, beta=1/2)
    assert list(kept.index) == [1, 2, 3, 4]


@pytest.mark.parametrize("daily_limit", [1500, 2300])
def test_nothing_over_cap_on_real_data(daily_limit):
    kept = filter_by_sodium(selected_table, daily_limit)
    assert (kept[SODIUM_COLUMN] <= daily_limit / 3).all()


@pytest.mark.parametrize("daily_limit", [1500, 2300])
def test_no_safe_food_is_dropped(daily_limit):
    kept = filter_by_sodium(selected_table, daily_limit)
    should_keep = selected_table[selected_table[SODIUM_COLUMN] <= daily_limit / 3]
    assert set(kept.index) == set(should_keep.index)