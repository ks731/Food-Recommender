"""
Loads the raw USDA SR Legacy CSV tables and prepares them for clustering
"""
from pathlib import Path
import pandas as pd

here = Path(__file__).resolve()
PROJECT_ROOT = here.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"

def load_table(filepath):
    """ Load a CSV file into a pandas dataframe"""
    df = pd.read_csv(filepath)
    return df


def merge_nutrient_names(food_nutrient_df, nutrient_df):
    """Attach nutrient name and unit to each food_nutrient row"""
    merged = pd.merge(food_nutrient_df, nutrient_df,left_on="nutrient_id",right_on="id",how="left")
    return merged

food_nutrient_csv = load_table(DATA_RAW / "food_nutrient.csv")
nutrients_csv = load_table(DATA_RAW / "nutrient.csv")
merged_nutrient_names = merge_nutrient_names(food_nutrient_csv, nutrients_csv)

print(merged_nutrient_names.head())

def merge_food_descriptions(nutrient_df, food_df):
    """ """
    merged = pd.merge(nutrient_df, food_df, on="fdc_id",how="left")
    return merged

food_csv = load_table(DATA_RAW / "food.csv")
merged_food_descriptions = merge_food_descriptions(merged_nutrient_names, food_csv)
print(merged_food_descriptions.head())
    