import numpy as np
from ranking import recommend, scaled_table, make_profile, PRESETS

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
