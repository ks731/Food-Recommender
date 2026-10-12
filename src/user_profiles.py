import pandas as pd
from pathlib import Path

here = Path(__file__).resolve()
PROJECT_ROOT = here.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

scaled_table = pd.read_csv(DATA_PROCESSED / "nutrients_scaled.csv", index_col="fdc_id")

def baseline_profile(df):
    """Returns a series of columns' medians"""
    medians = df.median()
    return medians

HIGH_PROTEIN = {"Protein": 0.80, "Carbohydrate, by difference": 0.10}

def build_profile(df,targets):
    """Overwrites each nutrient with its percentile and returns profile"""
    profile = baseline_profile(df)
    for nutrient, pct in targets.items():
        profile[nutrient] = df[nutrient].quantile(pct)
    return profile

print(build_profile(scaled_table, HIGH_PROTEIN))

