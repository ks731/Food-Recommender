import pandas as pd
from sklearn.preprocessing import StandardScaler
from preprocessing import wide_table

CLUSTER_FEATURES = [
    "Protein", "Total lipid (fat)", "Carbohydrate, by difference", "Fiber, total dietary", "Sugars, Total",
    "Energy (KCAL)",
    "Fatty acids, total saturated", "Fatty acids, total monounsaturated", "Fatty acids, total polyunsaturated", "Cholesterol",
    "Calcium, Ca", "Iron, Fe", "Magnesium, Mg", "Potassium, K", "Sodium, Na", "Zinc, Zn",
    "Vitamin C, total ascorbic acid", "Vitamin A, RAE", "Vitamin D (D2 + D3)", "Vitamin E (alpha-tocopherol)",
    "Vitamin K (phylloquinone)", "Vitamin B-12", "Folate, total",
]

def select_features(df, features):
    """Returns a table with a select amount of columns"""
    selected_table = df[features]
    return selected_table

selected_wide_table = select_features(wide_table, CLUSTER_FEATURES)

def scale_features(df):
    """Scales table and returns a Dataframe"""
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df)
    scaled = pd.DataFrame(scaled_array, index=df.index, columns=df.columns)
    return scaled

scaled_wide_table = scale_features(selected_wide_table)

print(scaled_wide_table.head())