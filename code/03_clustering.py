"""
03_clustering.py
------------------
Unsupervised structure discovery via KMeans, with the number of clusters
k chosen by an explicit, reported model-selection procedure (elbow of
inertia + silhouette + Davies-Bouldin), rather than fixed a priori as in
the original coursework (which fixed k=5 without justification). We
scan k in [2, 10], report all three criteria, and select k by silhouette
maximization, breaking ties toward the elbow location.

We also report agreement between the discovered clusters and the true
7-class obesity label via the Adjusted Rand Index (ARI) and Normalized
Mutual Information (NMI), to make explicit how much of the natural
unsupervised structure aligns with the clinical labeling scheme (and how
much does not).
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              adjusted_rand_score, normalized_mutual_info_score)
from sklearn.decomposition import PCA

from config import RESULTS_DIR, FIGURES_DIR, RANDOM_STATE, CLASS_ORDER
from preprocessing import get_clean_engineered, build_preprocessing_pipeline, TARGET_COL
from viz_style import set_style, PALETTE

set_style()


def main():
    df, _ = get_clean_engineered()
    pre = build_preprocessing_pipeline()
    X = pre.fit_transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()

    y_true = df[TARGET_COL].values

    # --- model selection over k ---
    rows = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
            "ARI_vs_true_class": adjusted_rand_score(y_true, labels),
            "NMI_vs_true_class": normalized_mutual_info_score(y_true, labels),
        })
    selection_df = pd.DataFrame(rows)
    selection_df.to_csv(RESULTS_DIR / "clustering_model_selection.csv", index=False)

    best_k = int(selection_df.loc[selection_df["silhouette"].idxmax(), "k"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(selection_df.k, selection_df.inertia, "o-", color=PALETTE[0])
    axes[0].set_title("Elbow: inertia vs k")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
    axes[1].plot(selection_df.k, selection_df.silhouette, "o-", color=PALETTE[4])
    axes[1].axvline(best_k, color="grey", ls="--", lw=1)
    axes[1].set_title("Silhouette score vs k")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette")
    axes[2].plot(selection_df.k, selection_df.davies_bouldin, "o-", color=PALETTE[5])
    axes[2].set_title("Davies-Bouldin index vs k (lower is better)")
    axes[2].set_xlabel("k"); axes[2].set_ylabel("DB index")
    fig.suptitle("KMeans model selection", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig05_clustering_model_selection.png")
    plt.close(fig)

    # --- final clustering with selected k -----------------------------------
    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = km.fit_predict(X)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sc0 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="tab10", s=10, alpha=0.7)
    axes[0].set_title(f"PCA projection colored by KMeans cluster (k={best_k})")
    axes[0].set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
    axes[0].set_ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
    plt.colorbar(sc0, ax=axes[0], label="Cluster")

    class_to_int = {c: i for i, c in enumerate(CLASS_ORDER)}
    y_int = df[TARGET_COL].map(class_to_int).values
    sc1 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_int, cmap="viridis", s=10, alpha=0.7)
    axes[1].set_title("PCA projection colored by true obesity class")
    axes[1].set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
    axes[1].set_ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
    cbar = plt.colorbar(sc1, ax=axes[1], ticks=range(len(CLASS_ORDER)))
    cbar.ax.set_yticklabels([c.replace("_", " ") for c in CLASS_ORDER], fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig06_pca_clusters_vs_class.png")
    plt.close(fig)

    summary = {
        "selected_k": best_k,
        "silhouette_at_selected_k": float(selection_df.loc[selection_df.k == best_k, "silhouette"].iloc[0]),
        "davies_bouldin_at_selected_k": float(selection_df.loc[selection_df.k == best_k, "davies_bouldin"].iloc[0]),
        "ARI_vs_true_class": float(selection_df.loc[selection_df.k == best_k, "ARI_vs_true_class"].iloc[0]),
        "NMI_vs_true_class": float(selection_df.loc[selection_df.k == best_k, "NMI_vs_true_class"].iloc[0]),
        "pca_explained_variance_ratio_2d": evr.tolist(),
    }
    RESULTS_DIR.joinpath("clustering_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
