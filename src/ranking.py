import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from constraints import filter_by_sodium, sodium_cap_per_100g, selected_table, SODIUM_COLUMN
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
    """Returns the single nearest cluster"""
    return clusters_by_distance(centroids,profile)[0]

def clusters_by_distance(centroids, profile):
    """Returns all clusters from nearest to farthest from the profile"""
    diff = centroids - profile["u"]
    sq = diff ** 2
    weighted = sq.mul(profile["weights"])
    dist = weighted.sum(axis=1) ** 0.5
    order = dist.sort_values().index
    return order

def rank_by_similarity(foods, u, top_n=10):
    """Returns a series of top n foods, best first """
    sims = cosine_similarity(foods, u.values.reshape(1,-1))
    scores = pd.Series(sims.ravel(), index=foods.index)
    top_scores = scores.sort_values(ascending=False).head(top_n)
    return top_scores

def sodium_penalty_scores(foods, u, daily_limit, penalty, beta=1/3, grams=100):
    """Returns (similarity, similarity minus penalty * portion sodium / cap) for every food"""
    sims = rank_by_similarity(foods, u, top_n=len(foods))
    cap = sodium_cap_per_100g(daily_limit, beta, grams)
    sodium = selected_table.loc[sims.index, SODIUM_COLUMN]
    adjusted = sims - penalty * sodium / cap if penalty else sims
    return sims, adjusted

def recommend(profile, top_n=10, hard=True, sodium_penalty=0.0, labels=None):
    """Recommends the top n foods, filling from the next-nearest cluster if needed.
    hard=True drops foods over the sodium cap; sodium_penalty > 0 lowers each food's
    rank by sodium_penalty * sodium / cap (the soft baseline). "scores" is always the
    plain similarity, so fidelity stays comparable across settings.
    labels: a cluster label per food (e.g. from a k = 4 run); None uses the saved k = 8 clusters.
    The sodium check uses the profile's portion size ("grams", default 100)."""
    if labels is None:
        labels, centroids = cluster_labels, cluster_centroids
    else:
        centroids = compute_centroids(scaled_table, labels)
    grams = profile.get("grams", 100)
    kept = filter_by_sodium(selected_table, profile["sodium_limit"], grams=grams) if hard else selected_table
    pieces = []
    found = 0
    for c in clusters_by_distance(centroids, profile):
        in_cluster = labels[labels == c].index
        candidates = scaled_table.loc[in_cluster.intersection(kept.index)]
        if candidates.empty:
            continue
        sims, adjusted = sodium_penalty_scores(candidates, profile["u"], profile["sodium_limit"],
                                             sodium_penalty, grams=grams)
        top = adjusted.sort_values(ascending=False, kind="stable").head(top_n - found).index
        piece = sims[top].to_frame(name="scores")
        piece["cluster"] = c
        pieces.append(piece)
        found += len(piece)
        if found >= top_n:
            break
    if not pieces:
        return pd.DataFrame(columns=["scores","cluster", "description", SODIUM_COLUMN])
    ranked = pd.concat(pieces)
    ranked = ranked.join(food_descriptions).join(selected_table[SODIUM_COLUMN])
    return ranked

if __name__ == "__main__":
    test_profile = make_profile(scaled_table, PRESETS["low_carb"], 1500)
    strict = make_profile(scaled_table, PRESETS["low_carb"], 300)
    print(recommend(test_profile, 10))







