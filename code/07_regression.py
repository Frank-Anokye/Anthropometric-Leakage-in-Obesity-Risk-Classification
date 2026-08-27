"""
07_regression.py
------------------
Continuous-outcome regression: predicting Body Mass Index (BMI) from
lifestyle/behavioral predictors only (Weight and Height are excluded,
since BMI = Weight / Height^2 by definition, and predicting BMI from its
own algebraic constituents is a tautology, not a research question).

This corrects two issues in the original coursework's regression
analysis: (1) it regressed the *ordinal class label* (0-6) with plain
OLS, which is a poor match for an ordinal, non-interval target, and (2)
it used a small feature subset without cross-validation or regularized
alternatives, yielding a weak, unstable R^2 (0.207 reported).

Here we predict the continuous, clinically standard BMI outcome, using
5-fold cross-validation, and compare OLS, Ridge, Lasso and Random Forest
regression, each with the same lifestyle-only predictor set used in the
classification track for consistency.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_validate, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import uniform, randint

from config import RESULTS_DIR, FIGURES_DIR, RANDOM_STATE
from preprocessing import get_clean_engineered, build_preprocessing_pipeline, TARGET_COL, NUMERIC_COLS
from viz_style import set_style, PALETTE

set_style()

ANTHROPOMETRIC_LEAKAGE_COLS = ["Weight", "Height"]  # BMI itself is the target


def main():
    df, _ = get_clean_engineered()
    X = df.drop(columns=[TARGET_COL, "ObesityRank", "BMI"] + ANTHROPOMETRIC_LEAKAGE_COLS)
    y = df["BMI"]
    numeric_cols = [c for c in NUMERIC_COLS if c in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "OLS": (LinearRegression(), None),
        "Ridge": (Ridge(random_state=RANDOM_STATE), {"reg__alpha": uniform(0.01, 50)}),
        "Lasso": (Lasso(random_state=RANDOM_STATE, max_iter=5000), {"reg__alpha": uniform(0.001, 2)}),
        "RandomForestRegressor": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
            {"reg__n_estimators": randint(80, 250), "reg__max_depth": randint(3, 20),
             "reg__min_samples_split": randint(2, 15)},
        ),
    }

    rows = []
    fitted = {}
    for name, (est, dist) in models.items():
        pre = build_preprocessing_pipeline(numeric_cols=numeric_cols)
        pipe = Pipeline([("pre", pre), ("reg", est)])
        if dist is not None:
            search = RandomizedSearchCV(pipe, param_distributions=dist, n_iter=15, cv=cv,
                                          scoring="r2", random_state=RANDOM_STATE, n_jobs=1)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            cv_r2 = search.best_score_
        else:
            cvres = cross_validate(pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=1)
            cv_r2 = np.mean(cvres["test_score"])
            pipe.fit(X_train, y_train)
            model = pipe
        fitted[name] = model

        y_pred = model.predict(X_test)
        rows.append({
            "model": name,
            "cv_r2_train": cv_r2,
            "test_r2": r2_score(y_test, y_pred),
            "test_mae": mean_absolute_error(y_test, y_pred),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        })
        print(f"{name}: CV R2={cv_r2:.4f}  test R2={rows[-1]['test_r2']:.4f}  "
              f"test RMSE={rows[-1]['test_rmse']:.3f} kg/m^2")

    results_df = pd.DataFrame(rows).sort_values("test_r2", ascending=False)
    results_df.to_csv(RESULTS_DIR / "regression_model_comparison.csv", index=False)

    best_name = results_df.iloc[0]["model"]
    best_model = fitted[best_name]
    y_pred_best = best_model.predict(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_test, y_pred_best, alpha=0.4, s=15, color=PALETTE[0])
    lims = [min(y_test.min(), y_pred_best.min()), max(y_test.max(), y_pred_best.max())]
    axes[0].plot(lims, lims, "--", color="grey", lw=1)
    axes[0].set_xlabel("True BMI (kg/m$^2$)")
    axes[0].set_ylabel("Predicted BMI (kg/m$^2$)")
    axes[0].set_title(f"Predicted vs. true BMI ({best_name})\nTest $R^2$={results_df.iloc[0]['test_r2']:.3f}")

    residuals = y_test.values - y_pred_best
    axes[1].scatter(y_pred_best, residuals, alpha=0.4, s=15, color=PALETTE[4])
    axes[1].axhline(0, color="grey", ls="--", lw=1)
    axes[1].set_xlabel("Predicted BMI (kg/m$^2$)")
    axes[1].set_ylabel("Residual (True $-$ Predicted)")
    axes[1].set_title("Residual plot")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig13_regression_bmi_prediction.png")
    plt.close(fig)

    # Bar chart comparing all models
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(results_df["model"], results_df["test_r2"], color=PALETTE[0])
    ax.set_ylabel("Test $R^2$")
    ax.set_title("BMI regression: model comparison (lifestyle-only predictors)")
    ax.set_ylim(0, max(0.05, results_df["test_r2"].max() * 1.2))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig14_regression_model_comparison.png")
    plt.close(fig)

    print(results_df)
    RESULTS_DIR.joinpath("regression_summary.json").write_text(
        json.dumps({"best_model": best_name, **results_df.iloc[0].to_dict()}, indent=2, default=float)
    )


if __name__ == "__main__":
    main()
