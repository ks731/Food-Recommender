import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from preprocessing import wide_table

here = Path(__file__).resolve()
PROJECT_ROOT = here.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
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

def cluster_foods(df, n_clusters):
    """Runs K-means on scaled table and returns a series of cluster labels"""
    km = KMeans(n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(df)
    result = pd.Series(labels,index=df.index,name="cluster")
    return result

clustered_scaled = cluster_foods(scaled_wide_table,n_clusters=8)

def profile_clusters(df,labels):
    """Returns the average of each nutrient per cluster"""
    profile = df.groupby(labels).mean()
    return profile.round(1)

cluster_profile = profile_clusters(selected_wide_table, clustered_scaled)

def evaluate_clusters(df, labels):
    """Returns silhouette score"""
    score = silhouette_score(df,labels)
    return score

sil_score = evaluate_clusters(scaled_wide_table,clustered_scaled)

def save_table(df,filename):
    """Writes table into a file and sets fdc_id as index"""
    df.to_csv(DATA_PROCESSED / filename)

save_table(selected_wide_table, "nutrients_selected.csv")
save_table(scaled_wide_table, "nutrients_scaled.csv")
save_table(clustered_scaled, "cluster_labels.csv")

#Check:
back = pd.read_csv(DATA_PROCESSED / "nutrients_selected.csv", index_col="fdc_id")
print(back.shape)
back2 = pd.read_csv(DATA_PROCESSED / "nutrients_scaled.csv",index_col="fdc_id")
print(back2.shape)
back3 = pd.read_csv(DATA_PROCESSED / "cluster_labels.csv",index_col="fdc_id")
print(back3.shape)