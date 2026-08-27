"""
02_statistical_tests.py
-----------------------
Formal inferential statistics linking predictors to the obesity class.

Numeric predictors: one-way ANOVA (F-test) across the 7 obesity classes,
with eta-squared effect size, and a Kruskal-Wallis robustness check
(non-parametric, distribution-free alternative -- the original coursework
report used ANOVA alone without checking its normality/homoscedasticity
assumptions; we verify degradation-robustness here).

Categorical predictors: chi-square test of independence with Cramer's V.

Multiple comparisons: Benjamini-Hochberg false discovery rate correction
across all tests performed (numeric + categorical), since testing many
features and reporting only nominal p-values inflates the family-wise
type-I error rate.
"""
import numpy as np
import pandas as pd
from scipy import stats

from config import RESULTS_DIR, CLASS_ORDER
from preprocessing import get_clean_engineered, NUMERIC_COLS, TARGET_COL, BINARY_COLS, ORDINAL_COLS, NOMINAL_COLS


def eta_squared(groups):
    all_vals = np.concatenate(groups)
    grand_mean = all_vals.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = sum(((g - grand_mean) ** 2).sum() for g in groups)
    return ss_between / ss_total if ss_total > 0 else np.nan


def cramers_v(confusion_matrix: np.ndarray) -> float:
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    r, k = confusion_matrix.shape
    phi2 = chi2 / n
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    return np.sqrt(phi2corr / denom) if denom > 0 else np.nan


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    bh = ranked * n / (np.arange(1, n + 1))
    bh_monotone = np.minimum.accumulate(bh[::-1])[::-1]
    bh_monotone = np.clip(bh_monotone, 0, 1)
    out = np.empty(n)
    out[order] = bh_monotone
    return out


def main():
    df, _ = get_clean_engineered()

    numeric_rows = []
    for col in NUMERIC_COLS + ["BMI"]:
        groups = [df.loc[df[TARGET_COL] == c, col].values for c in CLASS_ORDER]
        f_stat, p_anova = stats.f_oneway(*groups)
        h_stat, p_kw = stats.kruskal(*groups)
        eta2 = eta_squared(groups)
        numeric_rows.append({
            "feature": col, "test": "ANOVA", "statistic": f_stat, "p_value": p_anova,
            "eta_squared": eta2, "kruskal_H": h_stat, "kruskal_p": p_kw,
        })
    numeric_df = pd.DataFrame(numeric_rows)

    cat_rows = []
    for col in BINARY_COLS + ORDINAL_COLS + NOMINAL_COLS:
        ct = pd.crosstab(df[col], df[TARGET_COL])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        v = cramers_v(ct.values)
        cat_rows.append({"feature": col, "test": "Chi-square", "statistic": chi2,
                          "p_value": p, "dof": dof, "cramers_v": v})
    cat_df = pd.DataFrame(cat_rows)

    combined = pd.concat([
        numeric_df[["feature", "test", "statistic", "p_value"]],
        cat_df[["feature", "test", "statistic", "p_value"]],
    ], ignore_index=True)
    combined["p_bh_adjusted"] = benjamini_hochberg(combined["p_value"].values)
    combined["significant_at_0.05_BH"] = combined["p_bh_adjusted"] < 0.05

    numeric_df.to_csv(RESULTS_DIR / "anova_numeric_features.csv", index=False)
    cat_df.to_csv(RESULTS_DIR / "chisquare_categorical_features.csv", index=False)
    combined.to_csv(RESULTS_DIR / "hypothesis_tests_bh_corrected.csv", index=False)

    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
