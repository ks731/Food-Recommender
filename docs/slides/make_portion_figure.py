"""Makes the portion-size figure for the slides.
Run from the repo root:  python3 docs/slides/make_portion_figure.py
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evaluate import make_profiles, make_salty_profiles, recommend, violation_rate  # noqa: E402
from constraints import selected_table, filter_by_sodium  # noqa: E402

OUT = Path(__file__).resolve().parent
TIER = 1500                       # mg/day plan shown
GRAMS = [50, 100, 150, 200, 300]  # portion sizes compared

INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SAFE, OVER, OVER_LIGHT = "#c3c2b7", "#d03b3b", "#eba29a"
plt.rcParams.update({
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 12,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def unsafe_users(profiles):
    """For each portion size, counts users whose unfiltered top 10 has a food over the meal cap,
    and checks that the hard filter has none"""
    free = [recommend(dict(p, sodium_limit=float("inf")), 10) for p in profiles]
    counts = []
    for g in GRAMS:
        counts.append(sum(violation_rate(r, TIER, grams=g) > 0 for r in free))
        hard = [recommend(dict(p, grams=g), 10) for p in profiles]
        assert all(violation_rate(r, TIER, grams=g) == 0 for r in hard)
    return counts


def portion_figure():
    foods = [len(filter_by_sodium(selected_table, TIER, grams=g)) for g in GRAMS]
    regular = unsafe_users([p for n, p in make_profiles() if n.endswith(f"_{TIER}")])
    salty = unsafe_users([p for n, p in make_salty_profiles() if n.endswith(f"_{TIER}")])
    print("foods allowed:", foods)
    print("regular users unsafe without filter:", regular)
    print("salty-goal users unsafe without filter:", salty)

    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 5.6))
    x = np.arange(len(GRAMS))
    labels = [f"{g} g" for g in GRAMS]
    cap = TIER / 3

    # Left: how many foods pass the filter
    a.bar(x, foods, width=0.62, color=[INK2 if g == 100 else SAFE for g in GRAMS], zorder=3)
    for i, v in enumerate(foods):
        a.text(i, v + 120, f"{v:,}", ha="center", fontsize=11, color=INK)
    i100 = GRAMS.index(100)
    a.text(i100, foods[i100] / 2, "old\nassumption", ha="center", va="center", fontsize=10, color="white")
    a.set_xticks(x, labels)
    a.set_ylim(0, len(selected_table) * 1.1)
    a.set_ylabel(f"Foods allowed (of {len(selected_table):,})")
    a.set_xlabel("Portion size")
    a.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    a.set_title(f"Bigger portions → fewer foods pass\ncap per 100 g = {cap:.0f} mg × 100 / grams",
                loc="left", fontsize=13, fontweight="bold")

    # Right: users who would get an unsafe food without the filter
    w = 0.36
    b.bar(x - w / 2 - 0.01, regular, width=w, color=OVER_LIGHT, label="Regular users", zorder=3)
    b.bar(x + w / 2 + 0.01, salty, width=w, color=OVER, label="Salty-goal users", zorder=3)
    for i in range(len(GRAMS)):
        b.text(i - w / 2, regular[i] + 1.5, str(regular[i]), ha="center", fontsize=10, color=INK)
        b.text(i + w / 2, salty[i] + 1.5, str(salty[i]), ha="center", fontsize=10, color=INK)
    b.set_xticks(x, labels)
    b.set_ylim(0, 100)
    b.set_ylabel("Users out of 100")
    b.set_xlabel("Portion size")
    b.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    b.legend(frameon=False, loc="upper left")
    b.set_title("Without the filter, more users get unsafe food\nwith the hard filter: 0 at every portion size",
                loc="left", fontsize=13, fontweight="bold")

    fig.suptitle(f"Scaling sodium to the portion eaten  ({TIER:,} mg/day plan, {cap:.0f} mg per meal)",
                 x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "portion_scaling.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "portion_scaling.png")


if __name__ == "__main__":
    portion_figure()
