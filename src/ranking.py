import pandas as pd
from pathlib import Path
from user_profiles import make_profile, PRESETS


here = Path(__file__).resolve()
PROJECT_ROOT = here.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

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





