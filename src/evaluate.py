import numpy as np
import time
import pandas as pd
from ranking import recommend, rank_by_similarity, scaled_table, make_profile, PRESETS
from user_profiles import add_low_sodium
from scipy.spatial.distance import pdist
from ranking import SODIUM_COLUMN
from pathlib import Path

here = Path(__file__).resolve()
PROJECT_ROOT = here.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

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

def evaluate_profile(name, profile, top_n=10):
    """Runs one profile with and without the sodium filter and returns its metrics"""
    limit = profile["sodium_limit"]
    start = time.perf_counter()
    hard = recommend(profile, top_n)
    seconds = time.perf_counter() - start
    free = recommend(dict(profile, sodium_limit=float("inf")), top_n)
    return {
        "profile": name,
        "tier": limit,
        "n_hard": len(hard),
        "clusters_used": hard["cluster"].nunique(),
        "vr_hard": violation_rate(hard, limit),
        "vr_free": violation_rate(free, limit),
        "sodium_hard": hard[SODIUM_COLUMN].mean(),
        "sodium_free": free[SODIUM_COLUMN].mean(),
        "fidelity_hard": fidelity(hard),
        "fidelity_free": fidelity(free),
        "diversity_hard": diversity(hard),
        "diversity_free": diversity(free),
        "jaccard": jaccard(hard, free),
        "seconds": seconds,
    }

def run_evaluation(profiles, top_n=10):
    """Evaluates every profile and returns one row per profile"""
    rows = [evaluate_profile(name, p, top_n) for name, p in profiles]
    return pd.DataFrame(rows).set_index("profile")

PENALTIES = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

def metrics_row(name, limit, result, hard, **setting):
    """Returns one result row: the run's setting plus its metrics compared to the hard filter"""
    return {
        "profile": name,
        "tier": limit,
        **setting,
        "vr": violation_rate(result, limit),
        "violated": violation_rate(result, limit) > 0,
        "sodium": result[SODIUM_COLUMN].mean(),
        "fidelity": fidelity(result),
        "diversity": diversity(result),
        "jaccard_vs_hard": jaccard(result, hard),
    }

def evaluate_soft(profiles, penalties=PENALTIES, top_n=10):
    """Runs every profile with the hard filter and with the soft penalty at each strength.
    Returns one row per (profile, method); the hard filter's rows have penalty NaN."""
    rows = []
    for name, profile in profiles:
        limit = profile["sodium_limit"]
        hard = recommend(profile, top_n)
        rows.append(metrics_row(name, limit, hard, hard, method="hard", penalty=np.nan))
        for lam in penalties:
            soft = recommend(profile, top_n, hard=False, sodium_penalty=lam)
            rows.append(metrics_row(name, limit, soft, hard, method="soft", penalty=lam))
    return pd.DataFrame(rows)

# Sarker & Tanjim's low-sodium preset: sodium as one more target (percentile, weight)
LOW_SODIUM_SETTINGS = [(0.05, 1.0), (0.05, 3.0), (0.10, 1.0), (0.10, 3.0), (0.25, 1.0), (0.25, 3.0)]

def evaluate_low_sodium(profiles, settings=LOW_SODIUM_SETTINGS, top_n=10):
    """Runs every profile three ways: no sodium handling, Sarker & Tanjim's low-sodium
    preference at each (percentile, weight), and the hard filter"""
    rows = []
    for name, profile in profiles:
        limit = profile["sodium_limit"]
        hard = recommend(profile, top_n)
        free = recommend(profile, top_n, hard=False)
        rows.append(metrics_row(name, limit, free, hard, method="none", sodium_pct=np.nan, sodium_weight=np.nan))
        for pct, weight in settings:
            pref = recommend(add_low_sodium(scaled_table, profile, pct, weight), top_n, hard=False)
            # Score against the user's original goal, not the sodium-adjusted one, so fidelity is comparable
            pref["scores"] = rank_by_similarity(scaled_table.loc[pref.index], profile["u"], len(pref))
            rows.append(metrics_row(name, limit, pref, hard, method="low_sodium_pref", sodium_pct=pct, sodium_weight=weight))
        rows.append(metrics_row(name, limit, hard, hard, method="hard", sodium_pct=np.nan, sodium_weight=np.nan))
    return pd.DataFrame(rows)

def summarize_low_sodium(results):
    """Averages the low-sodium comparison per tier, method and setting"""
    keys = ["tier", "method", "sodium_pct", "sodium_weight"]
    grouped = results.groupby(keys, dropna=False, sort=False)
    summary = grouped[["vr", "sodium", "fidelity", "diversity", "jaccard_vs_hard"]].mean()
    summary["violating_profiles"] = grouped["violated"].sum()
    summary["profiles"] = grouped.size()
    return summary

def summarize_soft(results):
    """Averages the soft sweep per tier, method and penalty"""
    keys = ["tier", "method", "penalty"]
    grouped = results.groupby(keys, dropna=False)
    summary = grouped[["vr", "sodium", "fidelity", "diversity", "jaccard_vs_hard"]].mean()
    summary["violating_profiles"] = grouped["violated"].sum()
    summary["profiles"] = grouped.size()
    return summary

def summarize(results):
    """Averages every metric per tier and counts the cases that matter"""
    results = results.copy()
    results["fidelity_change_pct"] = 100 * (results["fidelity_hard"] - results["fidelity_free"]) / results["fidelity_free"]
    by_tier = results.groupby("tier")
    summary = by_tier.mean(numeric_only=True)
    summary["profiles"] = by_tier.size()
    summary["free_violating_profiles"] = by_tier["vr_free"].apply(lambda s: (s > 0).sum())
    summary["filled_profiles"] = by_tier["clusters_used"].apply(lambda s: (s > 1).sum())
    summary["worst_fidelity_change_pct"] = by_tier["fidelity_change_pct"].min()
    return summary


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    results = run_evaluation(make_profiles())
    results.to_csv(RESULTS_DIR / "evaluation_results.csv")
    summary = summarize(results)
    summary.to_csv(RESULTS_DIR / "evaluation_summary.csv")
    print(summary.round(4).T)

    soft = evaluate_soft(make_profiles())
    soft.to_csv(RESULTS_DIR / "soft_penalty_results.csv", index=False)
    soft_summary = summarize_soft(soft)
    soft_summary.to_csv(RESULTS_DIR / "soft_penalty_summary.csv")
    print(soft_summary.round(4))

    low = evaluate_low_sodium(make_profiles())
    low.to_csv(RESULTS_DIR / "low_sodium_pref_results.csv", index=False)
    low_summary = summarize_low_sodium(low)
    low_summary.to_csv(RESULTS_DIR / "low_sodium_pref_summary.csv")
    print(low_summary.round(4))