"""Makes the before/after portion figure for the slides: one user's top 10 at 100 g and 200 g.
Run from the repo root:  python3 docs/slides/make_portion_before_after.py
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evaluate import make_profiles, recommend, SODIUM_COLUMN  # noqa: E402

OUT = Path(__file__).resolve().parent
USER = "random33_1500"   # wants high-calorie foods, 1,500 mg/day plan (same user as the case study)
PORTIONS = [100, 200]

GONE = "#d03b3b"   # in the 100 g list but removed at 200 g
NEW = "#3b6fd0"    # new in the 200 g list
SAFE = "#c3c2b7"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 12,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def before_after():
    profile = dict(make_profiles())[USER]
    cap = profile["sodium_limit"] / 3
    before, after = (recommend(dict(profile, grams=g), 10) for g in PORTIONS)
    gone = set(before.index) - set(after.index)
    new = set(after.index) - set(before.index)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), sharex=True)
    for ax, g, result in zip(axes, PORTIONS, (before, after)):
        portion = (result[SODIUM_COLUMN] * g / 100).to_numpy()[::-1]
        ids = list(result.index)[::-1]
        names = [d if len(d) <= 38 else d[:37] + "…" for d in result["description"]][::-1]
        colors = [GONE if i in gone else NEW if i in new else SAFE for i in ids]
        ax.barh(range(len(portion)), portion, color=colors, height=0.7, zorder=3)
        ax.set_yticks(range(len(portion)), names, fontsize=9)
        ax.axvline(cap, color=INK, linestyle="--", linewidth=1.2, zorder=4)
        ax.text(cap + 10, len(portion) - 0.4, f"{cap:.0f} mg meal cap", fontsize=9, color=INK)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_xlabel(f"Sodium in a {g} g portion (mg)")
        if g == PORTIONS[0]:
            sub = f"{len(gone)} of 10 (red) would be over the cap at {PORTIONS[1]} g"
        else:
            sub = f"cap per 100 g drops to {cap * 100 / g:.0f} mg; {len(new)} new foods (blue)"
        ax.set_title(f"{g} g portion\n{sub}", loc="left", fontsize=12, fontweight="bold")
    axes[0].set_xlim(0, cap * 1.15)
    fig.suptitle("Same user, bigger portion: high-calorie goal, 1,500 mg/day plan",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "portion_before_after.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("removed at 200 g:", list(before.loc[sorted(gone), "description"]))
    print("new at 200 g:", list(after.loc[sorted(new), "description"]))
    print("wrote", OUT / "portion_before_after.png")


if __name__ == "__main__":
    before_after()
