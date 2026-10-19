import numpy as np
from ranking import recommend, scaled_table, make_profile, PRESETS
from scipy.spatial.distance import pdist
from ranking import SODIUM_COLUMN

TIERS = [1500, 2300]

def make_profiles(n_random=95, seed=42):
    """Returns a list of (name, profile): every preset at both tiers, plus random profiles at both tiers"""
    rng = np.random.default_rng(seed)
    nutrients = [n for n in scaled_table.columns if n != "Sodium, Na"]
    profiles = []
    for name, targets in PRESETS.items():
        for tier in TIERS:
            profiles.append((f"{name}_{tier}", make_profile(scaled_table, targets, tier)))
    for i in range(n_random):
        k = rng.integers(1, 3)
        chosen = rng.choice(nutrients, size=k, replace=False)
        targets = {n: float(rng.uniform(0.05, 0.95)) for n in chosen}
        for tier in TIERS:
            profiles.append((f"random{i}_{tier}", make_profile(scaled_table, targets, tier)))
    return profiles

def violation_rate(result, daily_limit, beta=1/3):
    """Returns the fraction of recommended foods over the sodium cap"""
    if len(result) == 0:
        return np.nan
    cap = daily_limit * beta
    return (result[SODIUM_COLUMN] > cap).mean()

def fidelity(result):
    """Returns the mean cosine similarity between the profile and the recommended foods"""
    if len(result) == 0:
        return np.nan
    return result["scores"].mean()

def diversity(result):
    """Returns the mean pairwise nutrient distance between the recommended foods"""
    if len(result) < 2:
        return np.nan
    vectors = scaled_table.loc[result.index]
    return pdist(vectors).mean()

def jaccard(result_a, result_b):
    """Returns how much two lists overlap: shared foods / all foods"""
    a, b = set(result_a.index), set(result_b.index)
    if not a | b:
        return np.nan
    return len(a & b) / len(a | b)