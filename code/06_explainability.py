"""
06_explainability.py
----------------------
Model-agnostic explainability via SHAP (SHapley Additive exPlanations,
Lundberg & Lee, 2017) applied to the best lifestyle-only classifier
(RandomForest; see 04_classification.py). We deliberately explain the
*lifestyle-only* model rather than the full-feature model, because the
full-feature model's importances would trivially collapse onto
Weight/Height (see docs/label_leakage_note.md) and add no scientific
insight; the lifestyle-only explanation identifies which *modifiable*
behavioral factors drive predicted obesity risk, which is the
practically and scientifically interesting question.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

from config import RESULTS_DIR, FIGURES_DIR
from preprocessing import get_clean_engineered, TARGET_COL
from viz_style import set_style

set_style()

ANTHROPOMETRIC_LEAKAGE_COLS = ["Weight", "Height", "BMI"]


def get_feature_names(preprocessor):
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            names.extend(cols)
        else:
            # each sub-pipeline's final step is a fitted encoder with get_feature_names_out
            encoder = trans.named_steps[list(trans.named_steps.keys())[-1]]
            if hasattr(encoder, "get_feature_names_out"):
                names.extend(encoder.get_feature_names_out(cols).tolist())
            else:
                names.extend(cols)
    return names


def main():
    df, _ = get_clean_engineered()
    X = df.drop(columns=[TARGET_COL, "ObesityRank"] + ANTHROPOMETRIC_LEAKAGE_COLS)
    y = df[TARGET_COL]

    fitted = joblib.load(RESULTS_DIR / "all_fitted_models_lifestyle.joblib")
    summary = json.loads((RESULTS_DIR / "classification_summary_lifestyle.json").read_text())
    best_name = summary["best_model"]
    pipeline = fitted[best_name]
    if best_name == "StackingEnsemble":
        # Explain the strongest individual base learner instead, since
        # StackingClassifier is not directly tree-explainable end-to-end.
        base_names = summary["top3_base_learners_for_stacking"]
        tree_based = [n for n in base_names if n in ("RandomForest", "GradientBoosting", "DecisionTree")]
        explain_name = tree_based[0] if tree_based else base_names[0]
        pipeline = fitted[explain_name]
    else:
        explain_name = best_name

    pre = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]

    X_test = pd.read_csv(RESULTS_DIR / "X_test_lifestyle.csv")
    X_test_t = pre.transform(X_test)
    if hasattr(X_test_t, "toarray"):
        X_test_t = X_test_t.toarray()
    feat_names = get_feature_names(pre)

    # Subsample for computational tractability on a single CPU core.
    rng = np.random.default_rng(42)
    n_sample = min(300, X_test_t.shape[0])
    idx = rng.choice(X_test_t.shape[0], n_sample, replace=False)
    X_sample = X_test_t[idx]

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)

    # shap_values shape for multi-class: (n_samples, n_features, n_classes) in
    # recent SHAP versions, or a list of arrays in older ones. Normalize to a
    # single (n_samples, n_features) global-importance matrix by averaging
    # |SHAP| across classes.
    if isinstance(shap_values, list):
        abs_stack = np.stack([np.abs(sv) for sv in shap_values], axis=0)
        mean_abs = abs_stack.mean(axis=0).mean(axis=0)  # (n_features,)
    else:
        sv = np.asarray(shap_values)
        if sv.ndim == 3:
            mean_abs = np.abs(sv).mean(axis=2).mean(axis=0)
        else:
            mean_abs = np.abs(sv).mean(axis=0)

    imp_df = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )
    imp_df.to_csv(RESULTS_DIR / "shap_feature_importance_lifestyle.csv", index=False)

    top = imp_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#1b4965")
    ax.set_xlabel("Mean |SHAP value| (average impact on model output magnitude)")
    ax.set_title(f"Global feature importance ({explain_name}, lifestyle-only model)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig12_shap_feature_importance.png")
    plt.close(fig)

    print(imp_df.head(15))


if __name__ == "__main__":
    main()
