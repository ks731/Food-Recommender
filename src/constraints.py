import pandas as pd
from pathlib import Path

here = Path(__file__).resolve()
PROJECT_ROOT = here.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

selected_table = pd.read_csv(DATA_PROCESSED / "nutrients_selected.csv", index_col="fdc_id")

def filter_by_sodium(df, daily_limit, beta=1/3):
    daily_limit = 2300
    beta = 1/3
    cap = daily_limit * beta

    mask = df["Protein"] <= cap
    kept = df[mask]