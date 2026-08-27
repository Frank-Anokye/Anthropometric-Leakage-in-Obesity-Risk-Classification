"""
config.py
---------
Central path configuration. All paths are resolved *relative to this file's
location on disk*, not to the current working directory. This means the
entire `code/` package can be run from any machine, any working directory,
and any absolute install path (e.g. after `git clone`) without editing a
single path string.

Directory layout expected (created automatically if missing):

    <repo_root>/
        code/            <- this file lives here
        data/            <- raw CSV input
        results/         <- CSV/JSON numeric outputs (tables, metrics)
        figures/         <- PNG figures for the paper/slides
"""
from pathlib import Path

# Resolve repo root as the parent of the directory containing this file.
CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent

DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"

RAW_CSV = DATA_DIR / "ObesityDataSet_raw_and_data_sinthetic.csv"

RANDOM_STATE = 42

# Canonical ordering of the ordinal target classes (WHO/CDC-consistent
# ordering from underweight to most severe obesity). Used for ordinal
# encoding, confusion-matrix axis ordering, and plot ordering.
CLASS_ORDER = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
