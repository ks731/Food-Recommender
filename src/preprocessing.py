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

nutrients_csv = load_table(DATA_RAW / "nutrient.csv")

def differentiate_nutrient_names(nutrient_df):
    """Finds duplicate nutrient names with different units and renames those rows"""
    bool_series = nutrient_df["name"].duplicated(keep=False)
    nutrient_df.loc[bool_series, "name"] = nutrient_df["name"] + " (" + nutrient_df["unit_name"] + ")"
    return nutrient_df

nutrients_csv = differentiate_nutrient_names(nutrients_csv)

def merge_nutrient_names(food_nutrient_df, nutrient_df):
    """Attach nutrient name and unit to each food_nutrient row"""
    merged = pd.merge(food_nutrient_df, nutrient_df,left_on="nutrient_id",right_on="id",how="left")
    return merged

food_nutrient_csv = load_table(DATA_RAW / "food_nutrient.csv")
merged_nutrient_names = merge_nutrient_names(food_nutrient_csv, nutrients_csv)

def merge_food_descriptions(merged_nutrient_names_df, food_df):
    """Match fdc_id for nutrient_df and food_df """
    merged = pd.merge(merged_nutrient_names_df, food_df, on="fdc_id",how="left")
    return merged

food_csv = load_table(DATA_RAW / "food.csv")
merged_food_descriptions = merge_food_descriptions(merged_nutrient_names, food_csv)

def pivot_to_wide(merged_food_desc_df):
    """Takes dataframe and tranforms it into a wide table"""
    tablepivoted = merged_food_desc_df.pivot_table(index="fdc_id", columns = "name",
                                                     values = "amount")
    widetable = tablepivoted
    return widetable
wide_table = pivot_to_wide(merged_food_descriptions)

def cleaning_missing_values(df, threshold = 0.5):
    """Returns dataframe with only columns 50% or more filled and fills missing values with 0"""
    clean_df = df.loc[:, df.isna().mean() <= threshold]
    clean_df = clean_df.fillna(0)
    return clean_df

wide_table = cleaning_missing_values(wide_table, threshold=0.5)
