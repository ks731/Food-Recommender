"""Checks how many clusters (k) the food table supports.
Run from the project root:  python3 src/k_check.py
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib import font_manager

here = Path(__file__).resolve()
PROJECT_ROOT = here.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
SLIDES_DIR = PROJECT_ROOT / "docs" / "slides"

installed_fonts = {f.name for f in font_manager.fontManager.ttflist}
if "IBM Plex Sans" in installed_fonts:
    plt.rcParams["font.family"] = "IBM Plex Sans"


def scan_k(scaled, ks):
    """Runs K-means for each k and returns inertia, silhouette and cluster sizes"""
    rows = []
    for k in ks:
        km = KMeans(k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled)
        sizes = pd.Series(labels).value_counts()
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(scaled, labels),
            "smallest_cluster": sizes.min(),
            "largest_cluster": sizes.max(),
        })
    return pd.DataFrame(rows).set_index("k")


def plot_elbow(scan, marks=(4, 8)):
    """Draws inertia and silhouette against k, with dashed lines at the marked k values"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col, title in [
        (ax1, "inertia", "Elbow: inertia (lower = tighter clusters)"),
        (ax2, "silhouette", "Silhouette (higher = better separated)"),
    ]:
        ax.plot(scan.index, scan[col], color="black", marker="o", markersize=4)
        for k in marks:
            ax.axvline(k, color="gray", linestyle="--", linewidth=1)
            ax.text(k, ax.get_ylim()[1], f" k = {k}", va="top", fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Number of clusters (k)")
        ax.set_xticks(list(scan.index))
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    scaled = pd.read_csv(DATA_PROCESSED / "nutrients_scaled.csv", index_col="fdc_id")
    scan = scan_k(scaled, range(2, 16))
    print(scan.round(3))
    RESULTS_DIR.mkdir(exist_ok=True)
    scan.to_csv(RESULTS_DIR / "k_check.csv")
    fig = plot_elbow(scan)
    fig.savefig(SLIDES_DIR / "elbow_silhouette.png", dpi=200)