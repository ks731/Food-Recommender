"""Makes the before/after scaling figures for the progress-report slides.
Run from the repo root:  python3 docs/slides/make_scaling_figures.py
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from clustering import selected_wide_table, scaled_wide_table  # noqa: E402
from preprocessing import food_csv  # noqa: E402

OUT = Path(__file__).resolve().parent
COLS = ["Sodium, Na", "Vitamin B-12"]
LABELS = ["Sodium (mg)", "Vitamin B-12 (µg)"]
COLORS = ["#2a6fdb", "#e07a2a"]

raw = selected_wide_table[COLS]
scaled = scaled_wide_table[COLS]

# ---- Figure 1: box plots before vs after scaling ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
for ax, data, title, ylabel in [
    (ax1, raw, "Before scaling (raw units)", "Amount per 100 g"),
    (ax2, scaled, "After scaling (StandardScaler)", "Standard deviations from mean"),
]:
    bp = ax.boxplot([data[c] for c in COLS], showfliers=False, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("black")
    ax.set_xticks([1, 2], LABELS)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
fig.suptitle(f"Sodium vs. vitamin B-12 across {len(raw):,} foods", fontsize=14)
fig.text(0.5, 0.01, "Outliers hidden for readability", ha="center", fontsize=9, color="gray")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(OUT / "scaling_boxplots.png", dpi=200)

# ---- Figure 2: before/after table for three familiar foods ----
FOODS = {173414: "Cheddar cheese", 175168: "Salmon (cooked)", 168462: "Spinach (raw)"}
rows = []
for fdc_id, name in FOODS.items():
    rows.append([
        name,
        f"{raw.loc[fdc_id, COLS[0]]:,.0f} mg", f"{scaled.loc[fdc_id, COLS[0]]:+.2f}",
        f"{raw.loc[fdc_id, COLS[1]]:.2f} µg", f"{scaled.loc[fdc_id, COLS[1]]:+.2f}",
    ])
headers = ["Food", "Sodium (raw)", "Sodium (scaled)", "B-12 (raw)", "B-12 (scaled)"]
pd.DataFrame(rows, columns=headers).to_csv(OUT / "scaling_table.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 2.2))
ax.axis("off")
table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#333333")
        cell.set_text_props(color="white", fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "scaling_table.png", dpi=200, bbox_inches="tight")

print(pd.DataFrame(rows, columns=headers).to_string(index=False))
