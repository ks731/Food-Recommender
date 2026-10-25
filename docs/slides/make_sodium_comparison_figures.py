"""Makes the sodium-method case-study figure for the slides.
Run from the repo root:  python3 docs/slides/make_sodium_comparison_figures.py
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evaluate import make_profiles, recommend, scaled_table, SODIUM_COLUMN  # noqa: E402
from user_profiles import add_low_sodium  # noqa: E402

OUT = Path(__file__).resolve().parent
USER = "random33_1500"   # wants high-calorie foods, 1,500 mg/day plan
PENALTY = 0.02           # sodium penalty strength shown in the middle panel

OVER = "#d03b3b"   # food over the sodium cap
SAFE = "#c3c2b7"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def case_study():
    profile = dict(make_profiles())[USER]
    cap = profile["sodium_limit"] / 3
    runs = [
        ("Low-sodium preset (Sarker & Tanjim)", "",
         recommend(add_low_sodium(scaled_table, profile, 0.10, 3.0), 10, hard=False)),
        (f"Sodium penalty, λ = {PENALTY:g} (ours)", "needs λ = 0.05 to reach 0",
         recommend(profile, 10, hard=False, sodium_penalty=PENALTY)),
        ("Hard filter (ours)", "",
         recommend(profile, 10)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), sharex=True)
    for ax, (title, note, result) in zip(axes, runs):
        sodium = result[SODIUM_COLUMN].to_numpy()[::-1]
        names = [d if len(d) <= 34 else d[:33] + "…" for d in result["description"]][::-1]
        ax.barh(range(len(sodium)), sodium, color=[OVER if v > cap else SAFE for v in sodium], height=0.7, zorder=3)
        ax.set_yticks(range(len(sodium)), names, fontsize=9)
        ax.axvline(cap, color=INK, linestyle="--", linewidth=1.2, zorder=4)
        ax.text(cap + 15, len(sodium) - 0.4, f"{cap:.0f} mg cap", fontsize=9, color=INK)
        n_over = int((sodium > cap).sum())
        subtitle = f"{n_over} of 10 over the cap" + (f"  ({note})" if note else "")
        ax.set_title(f"{title}\n{subtitle}", loc="left", fontsize=12, fontweight="bold")
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_xlabel("Sodium per 100 g (mg)")
    fig.suptitle("Example user who wants high-calorie foods, 1,500 mg/day plan   (red = over the cap)",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "sodium_case_study.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "sodium_case_study.png")


if __name__ == "__main__":
    case_study()
