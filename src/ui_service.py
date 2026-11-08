"""Logic behind the web UI (ui/server.py). Turns a profile choice (goal, sodium plan, list
size) into plain dictionaries that the page can draw. It reuses the existing pipeline
(make_profile, recommend, ...) and has no web code, so it can be tested on its own.
"""
import pandas as pd
from pathlib import Path
from ranking import (recommend, rank_by_similarity, clusters_by_distance, cluster_centroids,
                     cluster_labels, scaled_table, make_profile, food_descriptions, PRESETS)
from constraints import selected_table, SODIUM_COLUMN, filter_by_sodium

here = Path(__file__).resolve()
PROJECT_ROOT = here.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"

BETA = 1 / 3                       # share of the daily limit one food may use (as in constraints.py)
PLANS = [                          # DASH sodium plans, in the order the page shows them
    {"limit": 2300, "name": "Standard"},
    {"limit": 1500, "name": "Lower sodium"},
]
TOP_N_CHOICES = (5, 10)
DEFAULT_GOAL, DEFAULT_LIMIT, DEFAULT_TOP_N = "high_protein", 1500, 5

# column in the nutrient table -> (short label, unit) for the target vector and the chips
NUTRIENT_DISPLAY = {
    "Energy (KCAL)": ("Energy", "kcal"),
    "Protein": ("Protein", "g"),
    "Carbohydrate, by difference": ("Carbs", "g"),
    "Fiber, total dietary": ("Fiber", "g"),
    "Calcium, Ca": ("Calcium", "mg"),
    SODIUM_COLUMN: ("Sodium", "mg"),
}
CHIP_COLUMNS = ["Energy (KCAL)", "Protein", "Carbohydrate, by difference", "Fiber, total dietary", "Calcium, Ca"]


def load_food_groups():
    """Returns a food group name per fdc_id (e.g. 'Poultry Products') from the raw USDA tables.
    Returns an empty series when food_category.csv or the category column is not in data/raw/"""
    try:
        food = pd.read_csv(DATA_RAW / "food.csv", usecols=["fdc_id", "food_category_id"], index_col="fdc_id")
        categories = pd.read_csv(DATA_RAW / "food_category.csv")
    except (FileNotFoundError, ValueError):
        return pd.Series(dtype=object)
    category_ids = pd.to_numeric(food["food_category_id"], errors="coerce")
    for key in ("id", "code"):                     # the id column is standard; code is a fallback
        if key in categories.columns and "description" in categories.columns:
            names = categories.set_index(key)["description"]
            groups = category_ids.map(names)
            if groups.notna().mean() > 0.5:
                return groups
    return pd.Series(dtype=object)

food_groups = load_food_groups()


def display_name(column):
    """Returns (label, unit) for a nutrient column"""
    if column in NUTRIENT_DISPLAY:
        return NUTRIENT_DISPLAY[column]
    return column.split(",")[0], ""


def goal_label(goal):
    """'high_protein' -> 'High protein'"""
    return goal.replace("_", " ").capitalize()


def sodium_cap(limit):
    """Returns the per-food sodium cap in mg per 100 g"""
    return limit * BETA


def goal_targets(goal):
    """Returns the targeted nutrients of a preset in real units, as the target-vector panel shows them.
    Percentiles are the same on the raw and scaled tables, so the raw quantile is the target value"""
    rows = []
    for nutrient, pct in PRESETS[goal].items():
        label, unit = display_name(nutrient)
        rows.append({"nutrient": nutrient, "label": label, "unit": unit,
                     "pct": int(round(pct * 100)), "weight": 3,
                     "value": float(selected_table[nutrient].quantile(pct))})
    return rows


def meta():
    """Returns what the page needs before any recommendation: data size, goals, plans, defaults"""
    n_nutrients = int(scaled_table.shape[1])
    goals = []
    for goal in PRESETS:
        goals.append({"id": goal, "label": goal_label(goal), "targets": goal_targets(goal),
                      "other_nutrients": n_nutrients - len(PRESETS[goal])})
    return {"title": "DASH Food Recommender", "source": "USDA SR Legacy",
            "foods": int(len(selected_table)), "nutrients": n_nutrients,
            "goals": goals, "plans": PLANS, "top_n_choices": list(TOP_N_CHOICES),
            "beta": BETA, "default": {"goal": DEFAULT_GOAL, "limit": DEFAULT_LIMIT, "top_n": DEFAULT_TOP_N}}


def validate(goal, limit, top_n):
    """Raises ValueError for choices the recommender does not offer"""
    if goal not in PRESETS:
        raise ValueError("Unknown goal: %s" % goal)
    if limit not in [plan["limit"] for plan in PLANS]:
        raise ValueError("Sodium limit must be one of %s" % [plan["limit"] for plan in PLANS])
    if top_n not in TOP_N_CHOICES:
        raise ValueError("top_n must be one of %s" % list(TOP_N_CHOICES))


def describe_foods(ids, scores, clusters, targeted, limit):
    """Returns one dictionary per food (in the order given) with the numbers the page shows"""
    cap = sodium_cap(limit)
    foods = []
    for rank, fdc_id in enumerate(ids, start=1):
        row = selected_table.loc[fdc_id]
        sodium = float(row[SODIUM_COLUMN])
        group = food_groups.get(fdc_id) if len(food_groups) else None
        foods.append({
            "rank": rank,
            "fdc_id": int(fdc_id),
            "name": str(food_descriptions.loc[fdc_id, "description"]),
            "group": None if group is None or pd.isna(group) else str(group),
            "cluster": int(clusters[fdc_id]),
            "match": float(scores[fdc_id]),
            "sodium": sodium,
            "over_cap": bool(sodium > cap),
            "chips": [{"label": display_name(c)[0], "unit": display_name(c)[1],
                       "value": float(row[c]), "targeted": c in targeted}
                      for c in CHIP_COLUMNS if c in selected_table.columns],
        })
    return foods


def run(goal=DEFAULT_GOAL, limit=DEFAULT_LIMIT, top_n=DEFAULT_TOP_N):
    """Runs the recommender for one profile and returns everything the page shows:
    the profile, the four pipeline steps, the recommended foods, and the best foods the filter blocked"""
    validate(goal, limit, top_n)
    targets = PRESETS[goal]
    targeted = set(targets)
    profile = make_profile(scaled_table, targets, limit)
    cap = sodium_cap(limit)

    result = recommend(profile, top_n)
    nearest = int(clusters_by_distance(cluster_centroids, profile)[0])
    members = cluster_labels[cluster_labels == nearest].index
    over = members[selected_table.loc[members, SODIUM_COLUMN] > cap]

    recommended = describe_foods(result.index, result["scores"], result["cluster"], targeted, limit)

    blocked = []
    if len(over):
        blocked_scores = rank_by_similarity(scaled_table.loc[over], profile["u"], top_n)
        blocked = describe_foods(blocked_scores.index, blocked_scores,
                                 pd.Series(nearest, index=blocked_scores.index), targeted, limit)

    return {
        "profile": {"goal": goal, "label": goal_label(goal), "limit": limit, "cap": cap,
                    "targets": goal_targets(goal), "other_nutrients": int(scaled_table.shape[1]) - len(targets)},
        "steps": {"nearest_cluster": nearest, "cluster_size": int(len(members)),
                  "removed": int(len(over)), "top_n": top_n,
                  "clusters_used": int(result["cluster"].nunique()) if len(result) else 0},
        "foods": recommended,
        "blocked": blocked,
    }


def kept_count(limit):
    """Returns how many foods pass the sodium filter at this limit (used by tests and the page)"""
    return int(len(filter_by_sodium(selected_table, limit)))
