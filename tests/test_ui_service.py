import json
import pytest
import pandas as pd
from ui_service import (meta, run, validate, goal_targets, kept_count, sodium_cap, PLANS, TOP_N_CHOICES, food_groups)
from ranking import recommend, cluster_labels, scaled_table, make_profile, PRESETS
from constraints import selected_table, SODIUM_COLUMN

GOALS = list(PRESETS)
LIMITS = [plan["limit"] for plan in PLANS]


def test_meta_lists_every_preset_and_both_plans():
    info = meta()
    assert [g["id"] for g in info["goals"]] == GOALS
    assert [p["limit"] for p in info["plans"]] == [2300, 1500]
    assert info["foods"] == len(selected_table)
    assert info["nutrients"] == scaled_table.shape[1]


def test_meta_default_choice_is_valid():
    d = meta()["default"]
    validate(d["goal"], d["limit"], d["top_n"])


@pytest.mark.parametrize("goal", GOALS)
@pytest.mark.parametrize("limit", LIMITS)
@pytest.mark.parametrize("top_n", TOP_N_CHOICES)
def test_run_returns_full_lists_under_the_cap(goal, limit, top_n):
    data = run(goal, limit, top_n)
    assert len(data["foods"]) == top_n
    assert all(food["sodium"] <= sodium_cap(limit) for food in data["foods"])
    assert not any(food["over_cap"] for food in data["foods"])
    assert [food["rank"] for food in data["foods"]] == list(range(1, top_n + 1))


@pytest.mark.parametrize("goal", GOALS)
def test_run_matches_the_recommender(goal):
    data = run(goal, 1500, 10)
    expected = recommend(make_profile(scaled_table, PRESETS[goal], 1500), 10)
    assert [food["fdc_id"] for food in data["foods"]] == list(expected.index)


@pytest.mark.parametrize("goal", GOALS)
@pytest.mark.parametrize("limit", LIMITS)
def test_step_numbers_add_up(goal, limit):
    data = run(goal, limit, 5)
    steps = data["steps"]
    members = cluster_labels[cluster_labels == steps["nearest_cluster"]].index
    over = (selected_table.loc[members, SODIUM_COLUMN] > sodium_cap(limit)).sum()
    assert steps["cluster_size"] == len(members)
    assert steps["removed"] == over
    assert steps["removed"] == len(members) - len(members.intersection(selected_table[selected_table[SODIUM_COLUMN] <= sodium_cap(limit)].index))


@pytest.mark.parametrize("limit", LIMITS)
def test_kept_count_agrees_with_the_cap(limit):
    assert kept_count(limit) == int((selected_table[SODIUM_COLUMN] <= sodium_cap(limit)).sum())


@pytest.mark.parametrize("goal", GOALS)
def test_blocked_foods_are_over_the_cap_in_the_nearest_cluster_best_first(goal):
    data = run(goal, 1500, 5)
    cap = sodium_cap(1500)
    assert len(data["blocked"]) == min(5, data["steps"]["removed"])
    assert all(food["over_cap"] and food["sodium"] > cap for food in data["blocked"])
    assert all(food["cluster"] == data["steps"]["nearest_cluster"] for food in data["blocked"])
    matches = [food["match"] for food in data["blocked"]]
    assert matches == sorted(matches, reverse=True)


def test_blocked_and_recommended_foods_do_not_overlap():
    data = run("high_protein", 1500, 10)
    shown = {food["fdc_id"] for food in data["foods"]}
    assert not shown & {food["fdc_id"] for food in data["blocked"]}


@pytest.mark.parametrize("goal", GOALS)
def test_target_values_are_real_units_of_the_same_percentile(goal):
    for target in goal_targets(goal):
        column = target["nutrient"]
        scaled_value = scaled_table[column].quantile(target["pct"] / 100)
        mean, std = selected_table[column].mean(), selected_table[column].std(ddof=0)
        assert target["value"] == pytest.approx(scaled_value * std + mean, rel=1e-6, abs=1e-6)


def test_targeted_nutrients_are_flagged_on_the_chips():
    data = run("high_protein", 1500, 5)
    flagged = {chip["label"] for chip in data["foods"][0]["chips"] if chip["targeted"]}
    assert flagged == {"Protein", "Carbs"}


def test_result_can_be_sent_as_json():
    assert json.loads(json.dumps(run("low_carb", 2300, 10)))["profile"]["limit"] == 2300


def test_food_groups_are_a_series_even_when_the_category_file_is_missing():
    assert isinstance(food_groups, pd.Series)


@pytest.mark.parametrize("goal, limit, top_n", [("nope", 1500, 5), ("low_carb", 999, 5), ("low_carb", 1500, 7)])
def test_bad_choices_are_rejected(goal, limit, top_n):
    with pytest.raises(ValueError):
        run(goal, limit, top_n)
