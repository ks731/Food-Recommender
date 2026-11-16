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

PHOSPHORUS_ID = 1091   # "Phosphorus, P" (mg) in data/raw/nutrient.csv
PHOSPHORUS_COLUMN = "Phosphorus, P"

# Per-meal caps for CKD stages 3-4, not on dialysis (National Kidney Foundation,
# "Creating a Kidney-Friendly Plate"). Sodium in mg, potassium mg, phosphorus mg, protein g.
KIDNEY_CAPS = {
    SODIUM_COLUMN: 500,
    "Potassium, K": 450,
    PHOSPHORUS_COLUMN: 250,
    "Protein": 15,
}

def load_phosphorus(path=DATA_RAW / "food_nutrient.csv"):
    """Returns phosphorus (mg per 100 g) for every food in selected_table; NaN where USDA has no value.
    Kept out of the clustering: it is only used by the kidney filter"""
    rows = pd.read_csv(path, usecols=["fdc_id", "nutrient_id", "amount"])
    phosphorus = rows[rows["nutrient_id"] == PHOSPHORUS_ID].set_index("fdc_id")["amount"]
    return phosphorus.reindex(selected_table.index).rename(PHOSPHORUS_COLUMN)

kidney_table = selected_table[[SODIUM_COLUMN, "Potassium, K", "Protein"]].join(load_phosphorus())

def filter_by_kidney(df, caps=KIDNEY_CAPS, grams=100):
    """Keeps foods in df whose amount of every capped nutrient in a `grams` portion is at most its cap.
    A food with no value for a capped nutrient is dropped, since it can't be shown to be safe"""
    g = portion_grams(df.index, grams)
    values = kidney_table.loc[df.index, list(caps)]
    portion = values.mul(g / 100, axis=0)
    mask = portion.le(pd.Series(caps)).all(axis=1)
    return df[mask]

def filter_by_sodium(df, daily_limit, beta=1/3, grams=100):
    """Keeps foods whose sodium in a `grams` portion is at most daily_limit * beta.
    grams can be one number for every food, or a series of grams per food (e.g. load_serving_grams())"""
    cap = sodium_cap_per_100g(daily_limit, beta, portion_grams(df.index, grams))
    mask = df[SODIUM_COLUMN] <= cap
    kept = df[mask]
    return kept

result = filter_by_sodium(selected_table,1500)
