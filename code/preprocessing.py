"""
preprocessing.py
-----------------
Loads the raw obesity dataset and performs documented, reproducible
cleaning and feature engineering. Every transformation is a pure function
of the input CSV, so results are deterministic given RANDOM_STATE.

Source dataset: Palechor & de la Hoz Manotas (2019), "Estimation of obesity
levels based on eating habits and physical condition", Data in Brief, 25,
104344. Distributed via the UCI Machine Learning Repository
(https://doi.org/10.24432/C5H31Z). 77% of records are synthetically
generated (SMOTE-based) from an original survey of 498 respondents from
Colombia, Peru, and Mexico; 23% are original survey responses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import RAW_CSV, CLASS_ORDER

# Columns and their semantics, retained for documentation / the paper.
NUMERIC_COLS = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
BINARY_COLS = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
ORDINAL_COLS = ["CAEC", "CALC"]  # Never, Sometimes, Frequently, Always
NOMINAL_COLS = ["MTRANS"]
TARGET_COL = "NObeyesdad"

ORDINAL_LEVELS = ["no", "Sometimes", "Frequently", "Always"]


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Deduplicate and type-check. Returns (clean_df, report_dict)."""
    report = {}
    report["n_rows_raw"] = int(len(df))
    report["n_missing_cells"] = int(df.isnull().sum().sum())

    n_dupes = int(df.duplicated().sum())
    report["n_exact_duplicates"] = n_dupes
    # NOTE: the original coursework report claimed zero duplicate rows.
    # A direct pandas.duplicated() check on the released CSV finds
    # n_dupes fully-identical rows. Because ~77% of the dataset is
    # synthetically generated via SMOTE interpolation over a small
    # (n=498) base survey, a small number of coincident duplicate rows
    # is plausible and is documented here rather than silently dropped
    # or silently ignored.
    df_clean = df.drop_duplicates().reset_index(drop=True)
    report["n_rows_after_dedup"] = int(len(df_clean))

    # Range sanity checks
    report["age_range"] = [float(df_clean.Age.min()), float(df_clean.Age.max())]
    report["height_range_m"] = [float(df_clean.Height.min()), float(df_clean.Height.max())]
    report["weight_range_kg"] = [float(df_clean.Weight.min()), float(df_clean.Weight.max())]

    return df_clean, report


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add BMI (the clinically standard continuous obesity indicator) and
    an ordinal-encoded target used for ordinal-aware evaluation."""
    df = df.copy()
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)

    class_to_rank = {c: i for i, c in enumerate(CLASS_ORDER)}
    df["ObesityRank"] = df[TARGET_COL].map(class_to_rank)
    return df


def get_clean_engineered() -> tuple[pd.DataFrame, dict]:
    raw = load_raw()
    clean_df, report = clean(raw)
    full_df = engineer_features(clean_df)
    return full_df, report


def build_preprocessing_pipeline(numeric_cols=None):
    """Returns an sklearn ColumnTransformer implementing:
      - numeric: median impute (safety net) + z-score standardization
      - binary (yes/no, gender): one-hot (drop='if_binary')
      - ordinal (CAEC, CALC): explicit ordinal encoding (no/Sometimes/
        Frequently/Always -> 0..3), since these have a natural order and
        collapsing them to unordered dummies discards information.
      - nominal (MTRANS): one-hot, no natural order.
    All fitting happens inside a scikit-learn Pipeline / cross-validation
    loop elsewhere in this codebase, so there is no leakage from test
    folds into the scaler or encoders.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    binary_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
    ])
    ordinal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(categories=[ORDINAL_LEVELS] * len(ORDINAL_COLS),
                                handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    nominal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    cols = numeric_cols if numeric_cols is not None else NUMERIC_COLS
    pre = ColumnTransformer([
        ("num", numeric_pipe, cols),
        ("bin", binary_pipe, BINARY_COLS),
        ("ord", ordinal_pipe, ORDINAL_COLS),
        ("nom", nominal_pipe, NOMINAL_COLS),
    ])
    return pre


if __name__ == "__main__":
    df, report = get_clean_engineered()
    print(report)
    print(df.head())
