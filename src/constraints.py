import pandas as pd
from pathlib import Path

here = Path(__file__).resolve()
PROJECT_ROOT = here.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

selected_table = pd.read_csv(DATA_PROCESSED / "nutrients_selected.csv", index_col="fdc_id")
SODIUM_COLUMN = "Sodium, Na"

def filter_by_sodium(df, daily_limit, beta=1/3):
    """filters coloumn by sodium limit """
    cap = daily_limit * beta
    mask = df[SODIUM_COLUMN] <= cap
    kept = df[mask]
    return kept

result = filter_by_sodium(selected_table,1500)
