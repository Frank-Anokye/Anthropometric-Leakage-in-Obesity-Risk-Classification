"""
05_imbalance_study.py
-----------------------
Robustness check for class imbalance. The target is only mildly
imbalanced (majority:minority ratio ~1.29:1 -- 351 Obesity_Type_I vs
272 Insufficient_Weight), so large gains from imbalance-correction
techniques are not expected a priori; we test this expectation rather
than assume it, on the harder and more meaningful lifestyle-only task
(see 04_classification.py, feature_mode='lifestyle').

Three strategies are compared under identical 5-fold stratified CV on
the training split, using the RandomForest architecture selected as the
strongest lifestyle-only base learner in 04_classification.py:
  (a) baseline           - no correction
  (b) class_weight='balanced' - inverse-frequency reweighting in the loss
  (c) SMOTE oversampling  - synthetic minority oversampling applied
      *inside* each CV training fold only (via imblearn.pipeline.Pipeline),
      which avoids the common leakage bug of oversampling before the
      train/test or CV split.
Per-class recall is reported explicitly, since the practical harm of
imbalance is under-detection of minority classes, which macro accuracy
can mask.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, recall_score, f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from config import RESULTS_DIR, FIGURES_DIR, RANDOM_STATE, CLASS_ORDER
from preprocessing import get_clean_engineered, build_preprocessing_pipeline, TARGET_COL, NUMERIC_COLS
from viz_style import set_style, PALETTE

set_style()

ANTHROPOMETRIC_LEAKAGE_COLS = ["Weight", "Height", "BMI"]


def per_class_recall_scorer(class_label):
    def _scorer(y_true, y_pred):
        return recall_score(y_true, y_pred, labels=[class_label], average="macro", zero_division=0)
    return make_scorer(_scorer)


def main():
    df, _ = get_clean_engineered()
    X = df.drop(columns=[TARGET_COL, "ObesityRank"] + ANTHROPOMETRIC_LEAKAGE_COLS)
    y = df[TARGET_COL]
    numeric_cols = [c for c in NUMERIC_COLS if c in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    strategies = {}
    pre = build_preprocessing_pipeline(numeric_cols=numeric_cols)
    strategies["baseline"] = ImbPipeline([
        ("pre", pre), ("clf", RandomForestClassifier(n_estimators=200, max_depth=15, random_state=RANDOM_STATE)),
    ])
    pre2 = build_preprocessing_pipeline(numeric_cols=numeric_cols)
    strategies["class_weight_balanced"] = ImbPipeline([
        ("pre", pre2), ("clf", RandomForestClassifier(n_estimators=200, max_depth=15,
                                                        class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    pre3 = build_preprocessing_pipeline(numeric_cols=numeric_cols)
    strategies["smote_oversampling"] = ImbPipeline([
        ("pre", pre3), ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=15, random_state=RANDOM_STATE)),
    ])

    scoring = {"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy"}
    for c in CLASS_ORDER:
        scoring[f"recall_{c}"] = per_class_recall_scorer(c)

    rows = []
    for strat_name, pipe in strategies.items():
        cvres = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        row = {"strategy": strat_name}
        for key in scoring:
            row[key] = np.mean(cvres[f"test_{key}"])
        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(RESULTS_DIR / "imbalance_strategy_comparison.csv", index=False)

    # --- Figure: per-class recall by strategy ---
    recall_cols = [f"recall_{c}" for c in CLASS_ORDER]
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.25
    x = np.arange(len(CLASS_ORDER))
    for i, strat_name in enumerate(result_df["strategy"]):
        vals = result_df.loc[result_df.strategy == strat_name, recall_cols].values.flatten()
        ax.bar(x + (i - 1) * width, vals, width, label=strat_name, color=PALETTE[i * 2 % len(PALETTE)])
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in CLASS_ORDER], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Per-class recall (5-fold CV mean)")
    ax.set_title("Class-imbalance correction strategies: per-class recall\n(lifestyle-only feature set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig09_imbalance_strategy_comparison.png")
    plt.close(fig)

    print(result_df[["strategy", "macro_f1", "balanced_accuracy"]])
    RESULTS_DIR.joinpath("imbalance_study_summary.json").write_text(
        json.dumps(result_df.to_dict(orient="records"), indent=2)
    )


if __name__ == "__main__":
    main()
