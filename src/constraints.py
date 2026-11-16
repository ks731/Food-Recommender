import pandas as pd
from pathlib import Path

here = Path(__file__).resolve()
PROJECT_ROOT = here.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_RAW = PROJECT_ROOT / "data" / "raw"

selected_table = pd.read_csv(DATA_PROCESSED / "nutrients_selected.csv", index_col="fdc_id")
SODIUM_COLUMN = "Sodium, Na"

def sodium_cap_per_100g(daily_limit, beta=1/3, grams=100):
    """Returns the most sodium per 100 g a food can have so a `grams` portion stays under the per-meal cap"""
    return daily_limit * beta * 100 / grams

def load_serving_grams(path=DATA_RAW / "food_portion.csv"):
    """Returns each food's typical serving in grams: the gram weight of its first listed portion"""
    portions = pd.read_csv(path, usecols=["fdc_id", "seq_num", "gram_weight"])
    portions = portions[portions["gram_weight"] > 0]
    first = portions.sort_values(["fdc_id", "seq_num"]).drop_duplicates("fdc_id")
    return first.set_index("fdc_id")["gram_weight"]

def portion_grams(index, grams):
    """Returns the grams to check for these foods: one number for all, or each food's own serving.
    Foods with no serving data fall back to 100 g"""
    if isinstance(grams, pd.Series):
        return grams.reindex(index).fillna(100)
    return grams

def filter_by_sodium(df, daily_limit, beta=1/3, grams=100):
    """Keeps foods whose sodium in a `grams` portion is at most daily_limit * beta.
    grams can be one number for every food, or a series of grams per food (e.g. load_serving_grams())"""
    cap = sodium_cap_per_100g(daily_limit, beta, portion_grams(df.index, grams))
    mask = df[SODIUM_COLUMN] <= cap
    kept = df[mask]
    return kept

result = filter_by_sodium(selected_table,1500)
