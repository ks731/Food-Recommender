"""Makes the kidney (CKD stages 3-4) figures for the slides.
Run from the repo root:  python3 docs/slides/make_kidney_figures.py
  kidney_caps.png          what each cap removes, and which caps catch unsafe DASH-only lists
  kidney_before_after.png  one user's top 10, hypertension only vs. with the kidney caps
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evaluate import make_profiles, make_salty_profiles, recommend  # noqa: E402
from constraints import kidney_table, KIDNEY_CAPS, PHOSPHORUS_COLUMN  # noqa: E402
from user_profiles import add_kidney  # noqa: E402

OUT = Path(__file__).resolve().parent
TIER = 1500
LABELS = {"Sodium, Na": "Sodium < 500 mg", "Potassium, K": "Potassium < 450 mg",
          PHOSPHORUS_COLUMN: "Phosphorus < 250 mg", "Protein": "Protein < 15 g"}
SHORT = {"Sodium, Na": "Na", "Potassium, K": "K", PHOSPHORUS_COLUMN: "P", "Protein": "protein"}

OVER, OVER_LIGHT, NEW, SAFE = "#d03b3b", "#eba29a", "#3b6fd0", "#c3c2b7"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 12,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

# True where a food (per 100 g) is over a cap; a missing value counts as over
over = pd.DataFrame({n: ~(kidney_table[n] <= cap) for n, cap in KIDNEY_CAPS.items()})


def users_caught(profiles):
    """Per cap: how many users' hypertension-only top 10 has a food over that cap"""
    counts = dict.fromkeys(KIDNEY_CAPS, 0)
    for _, p in profiles:
        o = over.loc[recommend(p, 10).index]
        for n in KIDNEY_CAPS:
            counts[n] += int(o[n].any())
    return counts


def caps_figure():
    order = list(over.sum().sort_values().index)   # smallest at the bottom
    alone = over.sum()[order]
    only = over[over.sum(axis=1) == 1].sum()[order]
    regular = users_caught([(n, p) for n, p in make_profiles() if n.endswith(f"_{TIER}")])
    salty = users_caught([(n, p) for n, p in make_salty_profiles() if n.endswith(f"_{TIER}")])
    print("removed by each cap:", alone.to_dict())
    print("removed only by each cap:", only.to_dict())
    print("regular users caught:", regular)
    print("salty-goal users caught:", salty)

    fig, (a, b) = plt.subplots(1, 2, figsize=(16, 5.6))
    y = np.arange(len(order))
    names = [LABELS[n] for n in order]

    # Left: foods removed by each cap
    a.barh(y, alone, height=0.62, color=SAFE, zorder=3, label="Also over another cap")
    a.barh(y, only, height=0.62, color=INK2, zorder=3, label="Only this cap")
    for i, n in enumerate(order):
        a.text(alone[n] + 40, i, f"{alone[n]:,}", va="center", fontsize=11)
    a.set_yticks(y, names)
    a.set_xlim(0, alone.max() * 1.15)
    a.set_xlabel(f"Foods removed (of {len(over):,})")
    a.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    a.legend(frameon=False, loc="lower right")
    kept = (~over.any(axis=1)).sum()
    a.set_title(f"What each cap removes\nall four together leave {kept:,} foods",
                loc="left", fontsize=13, fontweight="bold")

    # Right: users whose hypertension-only list breaks each cap
    h = 0.36
    a_reg = [regular[n] for n in order]
    a_salty = [salty[n] for n in order]
    b.barh(y + h / 2 + 0.01, a_reg, height=h, color=OVER_LIGHT, label="Regular users", zorder=3)
    b.barh(y - h / 2 - 0.01, a_salty, height=h, color=OVER, label="Salty-goal users", zorder=3)
    for i in range(len(order)):
        b.text(a_reg[i] + 0.8, i + h / 2, str(a_reg[i]), va="center", fontsize=10)
        b.text(a_salty[i] + 0.8, i - h / 2, str(a_salty[i]), va="center", fontsize=10)
    b.set_yticks(y, names)
    b.set_xlim(0, 100)
    b.set_xlabel("Users out of 100 whose hypertension-only list breaks this cap")
    b.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    b.legend(frameon=False, loc="lower right")
    b.set_title("What each cap catches\nwith the kidney caps on: 0 for every cap",
                loc="left", fontsize=13, fontweight="bold")

    fig.suptitle(f"Kidney caps per meal (CKD stages 3–4, National Kidney Foundation), {TIER:,} mg/day plan",
                 x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "kidney_caps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "kidney_caps.png")


# Each food's amount as a % of each cap (per 100 g); missing values stay NaN
pct_of_cap = kidney_table[list(KIDNEY_CAPS)].div(pd.Series(KIDNEY_CAPS)) * 100


def pick_user():
    """The regular user whose hypertension-only list has the most foods measured over a kidney cap
    (real values, not missing ones). Users who target protein are skipped: they clash with the
    protein cap by design, so they aren't a typical example"""
    best, most = None, 0
    measured_over = (pct_of_cap > 100).any(axis=1)
    for name, p in make_profiles():
        if name.endswith(f"_{TIER}") and p["weights"]["Protein"] == 1:
            n = int(measured_over.loc[recommend(p, 10).index].sum())
            if n > most:
                best, most = (name, p), n
    return best


def before_after_figure():
    name, profile = pick_user()
    dash, ckd = recommend(profile, 10), recommend(add_kidney(profile), 10)
    gone, new = set(dash.index) - set(ckd.index), set(ckd.index) - set(dash.index)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), sharex=True)
    for ax, title, result in zip(axes, ["Hypertension only", "With kidney caps"], (dash, ckd)):
        ids = list(result.index)[::-1]
        worst = pct_of_cap.loc[ids].max(axis=1).to_numpy()
        names = []
        for i, d in zip(ids, result["description"][::-1]):
            d = d if len(d) <= 34 else d[:33] + "…"
            broken = [SHORT[n] for n in KIDNEY_CAPS if over.loc[i, n]]
            missing = [SHORT[n] for n in KIDNEY_CAPS if pd.isna(kidney_table.loc[i, n])]
            tag = f"  (over: {', '.join(broken)})" if broken else ""
            tag += f"  (no {', '.join(missing)} data)" if missing else ""
            names.append(d + tag)
        colors = [OVER if i in gone else NEW if i in new else SAFE for i in ids]
        ax.barh(range(len(worst)), np.nan_to_num(worst), color=colors, height=0.7, zorder=3)
        ax.set_yticks(range(len(worst)), names, fontsize=9)
        ax.axvline(100, color=INK, linestyle="--", linewidth=1.2, zorder=4)
        ax.text(103, len(worst) - 0.4, "cap", fontsize=9)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_xlabel("Strictest kidney cap used (% of cap, 100 g portion)")
        sub = (f"{len(gone)} of 10 (red) break a kidney cap" if result is dash
               else f"{len(new)} new foods (blue), all under every cap")
        ax.set_title(f"{title}\n{sub}", loc="left", fontsize=12, fontweight="bold")
    fig.suptitle(f"Same user, kidney caps on  ({name}, {TIER:,} mg/day plan)",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "kidney_before_after.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("user:", name, "targets:", {k: round(v, 2) for k, v in profile["u"].items()
                                       if profile["weights"][k] > 1})
    print("removed:", list(dash.loc[sorted(gone), "description"]))
    print("new:", list(ckd.loc[sorted(new), "description"]))
    print("wrote", OUT / "kidney_before_after.png")


if __name__ == "__main__":
    caps_figure()
    before_after_figure()
