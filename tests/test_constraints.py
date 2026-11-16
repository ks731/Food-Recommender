import pandas as pd
import pytest
from constraints import filter_by_sodium, load_serving_grams, selected_table, SODIUM_COLUMN

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

def test_bigger_portion_tightens_cap():
    # 200 g portion: 500 mg meal cap means at most 250 mg per 100 g
    kept = filter_by_sodium(toy_table(), 1500, grams=200)
    assert list(kept.index) == [1]


def test_smaller_portion_loosens_cap():
    # 50 g portion: at most 1,000 mg per 100 g
    kept = filter_by_sodium(toy_table(), 1500, grams=50)
    assert list(kept.index) == [1, 2, 3, 4]


def test_default_portion_is_100g():
    assert filter_by_sodium(toy_table(), 1500).equals(filter_by_sodium(toy_table(), 1500, grams=100))

def test_each_food_uses_its_own_serving():
    # 1,500 mg plan, 500 mg meal cap. Food 2 (499 mg/100 g) in a 200 g serving has 998 mg -> dropped.
    # Food 5 (2,000 mg/100 g) in a 10 g serving has 200 mg -> kept.
    grams = pd.Series({1: 100, 2: 200, 3: 100, 4: 100, 5: 10})
    kept = filter_by_sodium(toy_table(), 1500, grams=grams)
    assert list(kept.index) == [1, 3, 5]


def test_food_without_serving_falls_back_to_100g():
    grams = pd.Series({1: 100, 2: 100, 3: 100, 4: 100})     # food 5 has no serving
    kept = filter_by_sodium(toy_table(), 1500, grams=grams)
    assert 5 not in kept.index


def test_serving_grams_are_positive_and_one_per_food():
    serving = load_serving_grams()
    assert serving.index.is_unique
    assert (serving > 0).all()


@pytest.mark.parametrize("daily_limit", [1500, 2300])
def test_nothing_over_cap_per_serving_on_real_data(daily_limit):
    serving = load_serving_grams()
    kept = filter_by_sodium(selected_table, daily_limit, grams=serving)
    grams = serving.reindex(kept.index).fillna(100)
    assert (kept[SODIUM_COLUMN] * grams / 100 <= daily_limit / 3 + 1e-9).all()