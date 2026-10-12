import pandas as pd
from pathlib import Path

here = Path(__file__).resolve()
PROJECT_ROOT = here.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

scaled_table = pd.read_csv(DATA_PROCESSED / "nutrients_scaled.csv", index_col="fdc_id")

def baseline_profile(df):
    """Returns a series of 23 nutrient columns' medians"""
    medians = df.median()
    return medians

def build_profile(df,targets):
    """Returns 23 target numbers"""
    profile = baseline_profile(df)
    for nutrient, pct in targets.items():
        profile[nutrient] = df[nutrient].quantile(pct)
    return profile

def build_weights(df,targets):
    """returns weights for profile"""
    weights = pd.Series(1.0,index=df.columns)
    nutrients = list(targets)
    weights[nutrients] = 3.0
    return weights

def make_profile(df, targets, sodium_limit):
    """Returns profile with target vector,weights, and sodium limit"""
    profile = {
        "u": build_profile(df,targets),
        "weights": build_weights(df,targets),
        "sodium_limit": sodium_limit,
    }
    return profile

PRESETS = {
    "high_protein": {"Protein": 0.80, "Carbohydrate, by difference": 0.10},
    "low_carb": {"Carbohydrate, by difference": 0.10},
    "high_fiber": {"Fiber, total dietary": 0.80},
    "high_calcium": {"Calcium, Ca": 0.80},
    "low_calorie": {"Energy (KCAL)": 0.20},
}
