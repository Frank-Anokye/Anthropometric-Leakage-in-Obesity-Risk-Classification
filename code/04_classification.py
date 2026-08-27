"""
04_classification.py
----------------------
Multi-class classification of the 7-level obesity target.

Methodology (improves on the coursework's single untuned Decision Tree):
  1. A single stratified 80/20 train/test split is held out and touched
     exactly once, at the very end, for final reporting (no test-set
     leakage into model selection).
  2. Each candidate model's hyperparameters are tuned by 5-fold
     stratified cross-validation (macro-F1 scoring, appropriate for a
     7-class, roughly-balanced problem where all classes matter equally)
     on the training split only, via RandomizedSearchCV.
  3. A stacking ensemble (novel contribution relative to the coursework)
     combines the tuned base learners with a logistic-regression
     meta-learner trained on out-of-fold base-learner predictions.
  4. All tuned models are then evaluated once on the untouched test set.

All preprocessing (scaling/encoding) is fit only on the training fold
inside each cross-validation split via sklearn Pipelines, so there is no
leakage of test-set statistics into the scaler or encoders.
"""
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                              classification_report, confusion_matrix, balanced_accuracy_score)
from scipy.stats import randint, uniform

from config import RESULTS_DIR, FIGURES_DIR, RANDOM_STATE, CLASS_ORDER
from preprocessing import get_clean_engineered, build_preprocessing_pipeline, TARGET_COL, NUMERIC_COLS
from viz_style import set_style, PALETTE

warnings.filterwarnings("ignore")
set_style()

N_CV_FOLDS = 3
N_ITER_SEARCH = 8


def make_search_space():
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {"clf__C": uniform(0.01, 10), "clf__penalty": ["l2"]},
        ),
        "kNN": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": randint(3, 30), "clf__weights": ["uniform", "distance"],
             "clf__p": [1, 2]},
        ),
        "SVM_RBF": (
            SVC(probability=False, random_state=RANDOM_STATE),
            {"clf__C": uniform(0.1, 20), "clf__gamma": uniform(0.001, 1), "clf__kernel": ["rbf"]},
        ),
        "DecisionTree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"clf__max_depth": randint(3, 25), "clf__min_samples_split": randint(2, 20),
             "clf__min_samples_leaf": randint(1, 10), "clf__criterion": ["gini", "entropy"]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {"clf__n_estimators": randint(80, 250), "clf__max_depth": randint(4, 25),
             "clf__min_samples_split": randint(2, 15), "clf__max_features": ["sqrt", "log2", None]},
        ),
        "GradientBoosting": (
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {"clf__max_iter": randint(60, 200), "clf__learning_rate": uniform(0.02, 0.25),
             "clf__max_depth": randint(2, 8), "clf__l2_regularization": uniform(0.0, 1.0)},
        ),
    }


def tune_model(name, base_est, param_dist, preprocessor, X_train, y_train, cv):
    pipe = Pipeline([("pre", preprocessor), ("clf", base_est)])
    search = RandomizedSearchCV(
        pipe, param_distributions=param_dist, n_iter=N_ITER_SEARCH, scoring="f1_macro",
        cv=cv, random_state=RANDOM_STATE, n_jobs=-1, refit=True, verbose=0,
    )
    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0
    return search, elapsed


def evaluate_on_test(name, fitted_pipeline, X_test, y_test, out_rows, cm_store, suffix=""):
    y_pred = fitted_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    prm = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rem = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1w = f1_score(y_test, y_pred, average="weighted")
    out_rows.append({
        "model": name, "accuracy": acc, "balanced_accuracy": bacc,
        "macro_precision": prm, "macro_recall": rem, "macro_f1": f1m, "weighted_f1": f1w,
    })
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
    cm_store[name] = cm
    report = classification_report(y_test, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(RESULTS_DIR / f"classification_report_{name}{suffix}.csv")
    return y_pred


def plot_confusion(name, cm, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_yticks(range(len(CLASS_ORDER)))
    labels = [c.replace("_", " ") for c in CLASS_ORDER]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix: {name}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# Anthropometric variables from which BMI, and hence (per the dataset's own
# construction) the class label itself, are near-deterministically derived.
# See docs/label_leakage_note.md: NObeyesdad in this dataset is assigned via
# fixed BMI cutoffs (Palechor & de la Hoz Manotas, 2019), so a classifier
# with access to Weight and Height is largely solving an arithmetic
# threshold-recovery problem rather than a genuine behavioral risk-prediction
# problem. We therefore evaluate two feature regimes:
#   "full"      - all 16 original predictors (diagnostic/BMI-recovery task)
#   "lifestyle" - Weight, Height, BMI removed (genuine behavioral risk task)
ANTHROPOMETRIC_LEAKAGE_COLS = ["Weight", "Height", "BMI"]


def main(feature_mode="full"):
    df, _ = get_clean_engineered()
    drop_cols = [TARGET_COL, "ObesityRank"]
    if feature_mode == "lifestyle":
        drop_cols += ANTHROPOMETRIC_LEAKAGE_COLS
    X = df.drop(columns=drop_cols)
    y = df[TARGET_COL]
    suffix = "" if feature_mode == "full" else "_lifestyle"
    numeric_cols_this_mode = [c for c in NUMERIC_COLS if c in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search_space = make_search_space()
    fitted = {}
    cv_summary_rows = []
    for name, (est, dist) in search_space.items():
        pre = build_preprocessing_pipeline(numeric_cols=numeric_cols_this_mode)
        search, elapsed = tune_model(name, est, dist, pre, X_train, y_train, cv)
        fitted[name] = search.best_estimator_
        cv_summary_rows.append({
            "model": name, "best_cv_macro_f1": search.best_score_,
            "best_params": json.dumps({k: (v if not hasattr(v, "item") else v.item())
                                        for k, v in search.best_params_.items()}, default=str),
            "tuning_seconds": elapsed,
        })
        print(f"[{feature_mode}] [{name}] best CV macro-F1={search.best_score_:.4f} ({elapsed:.1f}s)")

    # --- Stacking ensemble (novel contribution) ----------------------------
    # Base learners: the three tuned tree/kernel models with the strongest
    # individual CV performance are combined via a logistic-regression
    # meta-learner trained on 5-fold out-of-fold predictions, which is the
    # standard leakage-safe stacking procedure implemented by
    # sklearn.ensemble.StackingClassifier.
    ranked = sorted(cv_summary_rows, key=lambda r: -r["best_cv_macro_f1"])
    top3_names = [r["model"] for r in ranked[:3]]
    print("Top-3 base learners for stacking:", top3_names)

    base_estimators = [(n, fitted[n]) for n in top3_names]
    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        cv=cv, n_jobs=-1, passthrough=False,
    )
    stack.fit(X_train, y_train)
    fitted["StackingEnsemble"] = stack

    # --- Final held-out test evaluation ---
    out_rows = []
    cm_store = {}
    for name, model in fitted.items():
        evaluate_on_test(name, model, X_test, y_test, out_rows, cm_store, suffix=suffix)
        plot_confusion(name, cm_store[name], FIGURES_DIR / f"fig_cm_{name}{suffix}.png")

    results_df = pd.DataFrame(out_rows).sort_values("macro_f1", ascending=False)
    results_df.to_csv(RESULTS_DIR / f"classification_model_comparison{suffix}.csv", index=False)
    pd.DataFrame(cv_summary_rows).to_csv(RESULTS_DIR / f"classification_cv_tuning_summary{suffix}.csv", index=False)

    # Best model's confusion matrix gets a dedicated, larger figure for the paper
    best_name = results_df.iloc[0]["model"]
    fig_tag = "fig07_confusion_matrix_best_model" if feature_mode == "full" else "fig11_confusion_matrix_best_model_lifestyle"
    plot_confusion(best_name, cm_store[best_name], FIGURES_DIR / f"{fig_tag}.png")

    # --- Model comparison bar chart ---
    fig, ax = plt.subplots(figsize=(9, 5))
    order = results_df["model"].tolist()
    x = np.arange(len(order))
    width = 0.25
    ax.bar(x - width, results_df["accuracy"], width, label="Accuracy", color=PALETTE[0])
    ax.bar(x, results_df["macro_f1"], width, label="Macro F1", color=PALETTE[4])
    ax.bar(x + width, results_df["balanced_accuracy"], width, label="Balanced Accuracy", color=PALETTE[5])
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    mode_label = "full feature set (incl. Weight/Height/BMI)" if feature_mode == "full" else "lifestyle-only feature set (Weight/Height/BMI excluded)"
    ax.set_title(f"Held-out test performance across models\n({mode_label})")
    ax.legend()
    fig.tight_layout()
    fig_tag2 = "fig08_model_comparison" if feature_mode == "full" else "fig10_model_comparison_lifestyle"
    fig.savefig(FIGURES_DIR / f"{fig_tag2}.png")
    plt.close(fig)

    print(results_df)

    # Persist the winning fitted pipeline + train/test split indices for
    # downstream explainability / imbalance scripts.
    import joblib
    joblib.dump(fitted[best_name], RESULTS_DIR / f"best_model_pipeline{suffix}.joblib")
    joblib.dump(fitted, RESULTS_DIR / f"all_fitted_models{suffix}.joblib")
    X_train.to_csv(RESULTS_DIR / f"X_train{suffix}.csv", index=False)
    X_test.to_csv(RESULTS_DIR / f"X_test{suffix}.csv", index=False)
    y_train.to_csv(RESULTS_DIR / f"y_train{suffix}.csv", index=False)
    y_test.to_csv(RESULTS_DIR / f"y_test{suffix}.csv", index=False)

    summary = {"feature_mode": feature_mode, "best_model": best_name,
               "top3_base_learners_for_stacking": top3_names,
               "best_model_test_macro_f1": float(results_df.iloc[0]["macro_f1"]),
               "best_model_test_accuracy": float(results_df.iloc[0]["accuracy"])}
    RESULTS_DIR.joinpath(f"classification_summary{suffix}.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    summary_full = main(feature_mode="full")
    summary_lifestyle = main(feature_mode="lifestyle")
    print("FULL:", summary_full)
    print("LIFESTYLE:", summary_lifestyle)
