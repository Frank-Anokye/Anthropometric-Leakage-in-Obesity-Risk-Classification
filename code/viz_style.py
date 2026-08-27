"""Consistent, publication-style matplotlib settings used by every script."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#1b4965", "#5fa8d3", "#bee9e8", "#cae9ff", "#e07a5f", "#81b29a", "#f2cc8f"]

def set_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#333333",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
