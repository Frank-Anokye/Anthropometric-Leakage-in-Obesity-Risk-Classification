"""
01_eda.py
Exploratory data analysis: descriptive statistics, distribution plots,
and correlation structure. Writes tables to results/ and figures to
figures/. Run as: python 01_eda.py  (from anywhere; paths are relative
to this file via config.py).
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import FIGURES_DIR, RESULTS_DIR, CLASS_ORDER
from preprocessing import get_clean_engineered, NUMERIC_COLS, TARGET_COL
from viz_style import set_style, PALETTE

set_style()


def main():
    df, clean_report = get_clean_engineered()
    RESULTS_DIR.joinpath("cleaning_report.json").write_text(json.dumps(clean_report, indent=2))

    # --- Descriptive statistics table ---
    desc = df[NUMERIC_COLS + ["BMI"]].describe().T
    desc.to_csv(RESULTS_DIR / "descriptive_statistics.csv")

    # --- Class distribution ---
    counts = df[TARGET_COL].value_counts().reindex(CLASS_ORDER)
    counts.to_csv(RESULTS_DIR / "class_distribution.csv", header=["count"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(counts)), counts.values, color=PALETTE[0])
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([c.replace("_", " ") for c in counts.index], rotation=40, ha="right")
    ax.set_ylabel("Number of records")
    ax.set_title("Class distribution of obesity level (target)")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 3, str(v), ha="center", fontsize=8)
    fig.savefig(FIGURES_DIR / "fig01_class_distribution.png")
    plt.close(fig)

    # --- Histograms of numeric features ---
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, col in zip(axes.ravel(), NUMERIC_COLS):
        ax.hist(df[col], bins=25, color=PALETTE[1], edgecolor="white")
        ax.set_title(col)
    fig.suptitle("Distributions of numeric predictors", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig02_histograms.png")
    plt.close(fig)

    # --- Boxplots by class for BMI and Weight ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col in zip(axes, ["BMI", "FAF"]):
        order = CLASS_ORDER
        data = [df.loc[df[TARGET_COL] == c, col].values for c in order]
        bp = ax.boxplot(data, tick_labels=[c.replace("_", "\n") for c in order], patch_artist=True)
        for patch, color in zip(bp["boxes"], PALETTE * 2):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(f"{col} by obesity class")
        ax.tick_params(axis="x", labelsize=7, rotation=45)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig03_boxplots_by_class.png")
    plt.close(fig)

    # --- Correlation heatmap (numeric + BMI) ---
    corr = df[NUMERIC_COLS + ["BMI"]].corr(method="spearman")
    corr.to_csv(RESULTS_DIR / "spearman_correlation.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                cbar_kws={"label": "Spearman \u03c1"})
    ax.set_title("Spearman correlation among numeric predictors and BMI")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig04_correlation_heatmap.png")
    plt.close(fig)

    print("EDA complete.")
    print(desc)


if __name__ == "__main__":
    main()
