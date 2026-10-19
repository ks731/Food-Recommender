import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from constraints import filter_by_sodium, selected_table, SODIUM_COLUMN
from user_profiles import make_profile, PRESETS


here = Path(__file__).resolve()
PROJECT_ROOT = here.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_RAW = PROJECT_ROOT / "data" / "raw"

food_descriptions = pd.read_csv(DATA_RAW / "food.csv", index_col="fdc_id", usecols=["fdc_id","description"])
scaled_table = pd.read_csv(DATA_PROCESSED / "nutrients_scaled.csv", index_col="fdc_id")
cluster_labels = pd.read_csv(DATA_PROCESSED / "cluster_labels.csv", index_col="fdc_id").squeeze("columns")

def compute_centroids(df, labels):
    """Returns the average of the foods in each cluster"""
    centroids = df.groupby(labels).mean()
    return centroids
cluster_centroids = compute_centroids(scaled_table, cluster_labels)

def nearest_cluster(centroids, profile):
    """Returns a profile's nearest centroid"""
    diff = centroids - profile["u"]
    sq = diff ** 2
    weighted = sq.mul(profile["weights"])
    dist = weighted.sum(axis=1) ** 0.5
    return dist.idxmin()

def rank_by_similarity(foods, u, top_n=10):
    """Returns a series of top n foods, best first """
    sims = cosine_similarity(foods, u.values.reshape(1,-1))
    scores = pd.Series(sims.ravel(), index=foods.index)
    top_scores = scores.sort_values(ascending=False).head(top_n)
    return top_scores

def recommend(profile, top_n=10):
    """Recommends the top n foods considering sodium limit and nearest cluster"""
    kept = filter_by_sodium(selected_table, profile["sodium_limit"])
    c = nearest_cluster(cluster_centroids,profile)
    in_cluster = cluster_labels[cluster_labels == c].index
    candidates = scaled_table.loc[in_cluster.intersection(kept.index)]
    ranked = rank_by_similarity(candidates, profile["u"], top_n)
    ranked = ranked.to_frame(name="scores").join(food_descriptions).join(selected_table[SODIUM_COLUMN])
    return ranked

test_profile = make_profile(scaled_table, PRESETS["low_carb"], 1500)
strict = make_profile(scaled_table, PRESETS["low_carb"], 300)
print(recommend(test_profile, 10))







