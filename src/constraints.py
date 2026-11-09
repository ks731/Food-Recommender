import pandas as pd
from pathlib import Path

here = Path(__file__).resolve()
PROJECT_ROOT = here.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

selected_table = pd.read_csv(DATA_PROCESSED / "nutrients_selected.csv", index_col="fdc_id")
SODIUM_COLUMN = "Sodium, Na"

def sodium_cap_per_100g(daily_limit, beta=1/3, grams=100):
    """Returns the most sodium per 100 g a food can have so a `grams` portion stays under the per-meal cap"""
    return daily_limit * beta * 100 / grams

def filter_by_sodium(df, daily_limit, beta=1/3, grams=100):
    """Keeps foods whose sodium in a `grams` portion is at most daily_limit * beta"""
    cap = sodium_cap_per_100g(daily_limit, beta, grams)
    mask = df[SODIUM_COLUMN] <= cap
    kept = df[mask]
    return kept

result = filter_by_sodium(selected_table,1500)
