import numpy as np
import pytest
from constraints import (filter_by_kidney, kidney_table, selected_table, load_serving_grams,
                         KIDNEY_CAPS, PHOSPHORUS_COLUMN, SODIUM_COLUMN)
from ranking import recommend, scaled_table, make_profile, PRESETS
from user_profiles import add_kidney


def portion_amounts(result, grams=100):
    """Each capped nutrient in the portion actually eaten, for the recommended foods"""
    return kidney_table.loc[result.index].mul(grams / 100)


def test_phosphorus_is_loaded_but_not_clustered():
    assert PHOSPHORUS_COLUMN in kidney_table.columns
    assert PHOSPHORUS_COLUMN not in scaled_table.columns
    assert kidney_table[PHOSPHORUS_COLUMN].notna().sum() > 7000


def test_kidney_filter_keeps_only_foods_under_every_cap():
    kept = filter_by_kidney(selected_table)
    amounts = kidney_table.loc[kept.index]
    for nutrient, cap in KIDNEY_CAPS.items():
        assert (amounts[nutrient] <= cap).all()


def test_food_with_no_phosphorus_value_is_dropped():
    missing = kidney_table.index[kidney_table[PHOSPHORUS_COLUMN].isna()]
    assert len(missing) > 0
    kept = filter_by_kidney(selected_table.loc[missing])
    assert kept.empty


def test_bigger_portion_keeps_fewer_foods():
    assert len(filter_by_kidney(selected_table, grams=200)) < len(filter_by_kidney(selected_table))


def test_only_chosen_caps_are_used():
    sodium_only = filter_by_kidney(selected_table, {SODIUM_COLUMN: 500})
    assert len(sodium_only) == (selected_table[SODIUM_COLUMN] <= 500).sum()


def test_add_kidney_turns_on_caps_without_changing_the_goal():
    profile = make_profile(scaled_table, PRESETS["low_carb"], 1500)
    ckd = add_kidney(profile)
    assert ckd["condition"] == "ckd"
    assert ckd["kidney_caps"] == KIDNEY_CAPS
    assert ckd["u"].equals(profile["u"])
    assert "kidney_caps" not in profile


def test_add_kidney_with_some_caps():
    profile = make_profile(scaled_table, PRESETS["low_carb"], 1500)
    ckd = add_kidney(profile, [SODIUM_COLUMN, PHOSPHORUS_COLUMN])
    assert ckd["kidney_caps"] == {SODIUM_COLUMN: 500, PHOSPHORUS_COLUMN: 250}


def test_profile_without_kidney_matches_old_behavior():
    profile = make_profile(scaled_table, PRESETS["high_fiber"], 2300)
    off = dict(profile, kidney_caps=None)
    assert recommend(profile, 10).equals(recommend(off, 10))


@pytest.mark.parametrize("preset", list(PRESETS))
@pytest.mark.parametrize("daily_limit", [1500, 2300])
def test_kidney_recommendations_stay_under_every_cap(preset, daily_limit):
    profile = add_kidney(make_profile(scaled_table, PRESETS[preset], daily_limit))
    result = recommend(profile, 10)
    assert len(result) == 10
    amounts = portion_amounts(result)
    for nutrient, cap in KIDNEY_CAPS.items():
        assert (amounts[nutrient] <= cap).all()
    assert (result[SODIUM_COLUMN] <= daily_limit / 3).all()


@pytest.mark.parametrize("grams", [50, 200])
def test_kidney_caps_scale_with_portion(grams):
    profile = add_kidney(make_profile(scaled_table, PRESETS["low_calorie"], 1500, grams=grams))
    result = recommend(profile, 10)
    assert len(result) == 10
    amounts = portion_amounts(result, grams)
    for nutrient, cap in KIDNEY_CAPS.items():
        assert (amounts[nutrient] <= cap).all()


def test_kidney_caps_work_with_per_food_servings():
    servings = load_serving_grams()
    profile = add_kidney(make_profile(scaled_table, PRESETS["low_calorie"], 1500, grams=servings))
    result = recommend(profile, 10)
    g = servings.reindex(result.index).fillna(100).to_numpy()
    amounts = kidney_table.loc[result.index].mul(g / 100, axis=0)
    for nutrient, cap in KIDNEY_CAPS.items():
        assert (amounts[nutrient] <= cap).all()


def test_soft_mode_ignores_kidney_caps():
    profile = make_profile(scaled_table, PRESETS["low_carb"], 1500)
    assert recommend(add_kidney(profile), 10, hard=False).equals(recommend(profile, 10, hard=False))
