"""Checks how many clusters (k) the food table supports.
Run from the project root:  python3 src/k_check.py
"""
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
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

def fit_labels(scaled, k):
    """Runs K-means once and returns each food's cluster label"""
    km = KMeans(k, random_state=42, n_init=10)
    labels = km.fit_predict(scaled)
    return pd.Series(labels, index=scaled.index, name="cluster")


def pca_2d(scaled):
    """Squeezes the 23 scaled nutrients into 2 coordinates for plotting"""
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled)
    coords = pd.DataFrame(coords, index=scaled.index, columns=["PC1", "PC2"])
    return coords, pca.explained_variance_ratio_


def plot_pca_clusters(coords, labels, explained, ncols=4):
    """One small panel per cluster: that cluster's foods in black, every other food in gray"""
    k = labels.nunique()
    nrows = math.ceil(k / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    sizes = labels.value_counts()
    for c, ax in zip(sorted(sizes.index), axes.flat):
        ax.scatter(coords["PC1"], coords["PC2"], color="lightgray", s=2, linewidths=0)
        members = coords[labels == c]
        ax.scatter(members["PC1"], members["PC2"], color="black", s=3, linewidths=0)
        word = "food" if sizes[c] == 1 else "foods"
        ax.set_title(f"Cluster {c}: {sizes[c]:,} {word}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel(f"PC1 ({explained[0]:.0%} of variance)")
    fig.supylabel(f"PC2 ({explained[1]:.0%} of variance)")
    fig.suptitle(f"k = {k}: each cluster in black, all other foods in gray")
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
    coords, explained = pca_2d(scaled)
    print(f"PC1 + PC2 explain {explained.sum():.1%} of the variance")
    saved_k8 = pd.read_csv(DATA_PROCESSED / "cluster_labels.csv", index_col="fdc_id").squeeze("columns")
    for k in (8, 4):
        labels = fit_labels(scaled, k)
        if k == 8:
            print("k = 8 labels match cluster_labels.csv:", (labels == saved_k8.reindex(labels.index)).all())
        if k == 4:
            labels.to_csv(DATA_PROCESSED / "cluster_labels_k4.csv")
        fig = plot_pca_clusters(coords, labels, explained)
        fig.savefig(SLIDES_DIR / f"pca_clusters_k{k}.png", dpi=200)