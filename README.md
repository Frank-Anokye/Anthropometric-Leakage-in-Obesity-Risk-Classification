# Quantifying Anthropometric Leakage in Obesity Risk Classification and It's Implications for Threshold-Defined Labels in Clinical Machine Learning

**Prepint:** https://doi.org/10.13140/RG.2.2.29908.56962

Frank Anokye, University of L'Aquila, Italy, L,Aquila, Italy & Silesian University of Technology, Poland, Gliwice, August 2026

---

## Short version

Machine learning studies on the popular Palechor and de la Hoz Manotas obesity dataset routinely report 90% to 96% classification accuracy. Most of that number is not what it looks like. This repository shows that about 12% of that accuracy comes from a single fact that almost no published study on this dataset states directly: the target label is just a fixed cut-off on BMI, and BMI can be computed exactly from two of the predictors already in the data, Weight and Height. A model given those two features is not learning about diet or lifestyle. It is mostly just doing arithmetic. This project separates the two things that get conflated. One task keeps every feature and measures how well a model can recover the BMI cut-off rule. The other removes Weight, Height, and BMI, and measures what lifestyle and demographic factors alone can actually predict. Both are tested under the exact same rigorous process, so the size of the gap between them is measured, not assumed.

| | Full feature set (diagnostic) | Lifestyle only (behavioral risk) |
|---|---|---|
| Best model | Stacking Ensemble | Random Forest |
| Test accuracy | **97.8%** | **86.1%** |
| Test macro F1 | 0.978 | 0.857 |

## Project overview

This project began as coursework for the course Data Analytics and Data Driven Decision Making (2 July 2024), and has since been rebuilt and substantially extended into a full research study.

The central contribution is methodological, not just a bigger analysis. The dataset's seven class obesity label (NObeyesdad) is assigned using fixed cutoffs on BMI, and BMI is simply Weight divided by Height squared. Since Weight and Height are already predictors in the dataset, a classifier given access to them can mostly solve an arithmetic recovery problem rather than genuinely learning to predict behavior. This project states that problem clearly, measures its exact size in a controlled experiment, and then reruns the entire analysis, classification, regression, clustering, statistical testing, explainability, and class imbalance testing, under two separate conditions: one using the full feature set (a diagnostic task), and one using lifestyle information only (a genuine behavioral risk task).

Full details, math, and discussion are in [`paper/paper.pdf`](paper/paper.pdf) (the IEEE style preprint).

## Repository structure

```
.
├── code/                          # All analysis code (see "Reproducing" below)
│   ├── config.py                  # Central path setup, resolves paths relative to this file
│   ├── preprocessing.py           # Data loading, cleaning, sklearn preprocessing pipeline
│   ├── viz_style.py                # Shared matplotlib style
│   ├── 01_eda.py                  # Basic statistics and exploratory figures
│   ├── 02_statistical_tests.py    # ANOVA, chi square, Benjamini Hochberg correction
│   ├── 03_clustering.py           # KMeans model selection and cluster validity
│   ├── 04_classification.py       # Task A and B: tuning, stacking ensemble, testing
│   ├── 05_imbalance_study.py      # Class imbalance robustness check
│   ├── 06_explainability.py       # SHAP analysis on the behavioral risk model
│   ├── 07_regression.py           # BMI regression from lifestyle only features
│  
├── data/
│   └── ObesityDataSet_raw_and_data_sinthetic.csv   # Palechor and de la Hoz Manotas (2019)
├── results/                       # All numeric outputs (CSV and JSON tables, fitted models)
├── figures/                       # All generated figures (PNG, 300dpi)
├── paper/                         # IEEE style preprint (LaTeX source and compiled PDF)
│   ├── obesity.tex,
│   ├── references.bib
├── requirements.txt
├── LICENSE
└── README.md                      # This file

```

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/[username]/obesity-risk-leakage.git
cd obesity-risk-leakage
pip install -r requirements.txt
```

No other setup is needed. **Every script in `code/` finds its own input and output paths based on where it is saved on disk** (using pathlib in `config.py`)

## Dataset

**Source:** [UC IRVINE MACHINE LEARNING REPOSITORY]
(https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)

- 2,111 records (2,087 after removing duplicates, see thesis Chapter 4), 17 attributes.
- 23 percent directly surveyed (about 498 people), 77 percent SMOTE synthesized to balance the classes.
- **This project did not collect any new data.** No human subjects research was performed here.

## Usage: reproducing every result

Run each script in `code/` in numeric order (each one is independent, prints its own results, and writes tables to `results/` and figures to `figures/`):

```bash
cd code
python 01_eda.py                  # basic statistics and exploratory figures
python 02_statistical_tests.py    # ANOVA and chi square, adjusted for multiple tests
python 03_clustering.py           # KMeans model selection (chooses k=4)
python 04_classification.py       # Task A and Task B: 6 tuned models plus stacking ensemble
python 05_imbalance_study.py      # class weighting vs SMOTE vs baseline
python 06_explainability.py       # SHAP on the Task B (behavioral) model
python 07_regression.py           # BMI regression, lifestyle only predictors
```

Every random process is seeded (`RANDOM_STATE = 42` in `config.py`), so every run reproduces the exact numbers reported in the paper. The full pipeline takes under five minutes to run on a single CPU core. This was tested directly: the entire pipeline was copied to a different folder and run from an unrelated working directory, and it produced identical results.

To regenerate the research, and slides themselves (requires Node.js with and `pptxgenjs`, and a LaTeX installation): For example

```bash
cd code
cd ../paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Method summary

| Analysis | Method | Key result |
|---|---|---|
| Statistical testing | ANOVA and chi square, Benjamini Hochberg correction | All 16 predictors significant at 0.05 |
| Classification | 6 tuned classifiers (LR, kNN, SVM, DT, RF, HistGB) plus a stacking ensemble | Task A 97.8%, Task B 86.1% accuracy |
| Class imbalance | Baseline vs class weighting vs SMOTE, applied only inside cross validation folds | Small, steady gains (mild 1.29 to 1 imbalance) |
| Explainability | SHAP (TreeExplainer) on the Task B model | Top drivers: FCVC, Age, Gender, family history |
| Regression | OLS, Ridge, Lasso, Random Forest, predicting BMI from lifestyle only | R squared 0.869 (Random Forest) vs 0.207 (original baseline) |
| Cluster validity | KMeans, number of clusters chosen using silhouette and Davies Bouldin scores, k from 2 to 10 | k=4 chosen; weak label match (ARI 0.179) |

Full mathematical detail for every method is in the thesis (Chapter 5) and the preprint (Sections IV and V).

## Results summary

- **Anthropometric leakage effect:** about 11.7% points of accuracy (12.1 of macro F1) come from Weight and Height being both predictors and the exact building blocks of the label.
- **Behavioral drivers** of obesity risk classification (with Weight, Height, and BMI removed): vegetable consumption frequency, age, gender, and family history of overweight matter most. Physical activity frequency ranks lower than commonly assumed.
- **Class imbalance** is mild (1.29 to 1), and correcting for it gives only modest gains. This dataset does not need aggressive rebalancing.
- **The natural cluster structure** of the data lines up only weakly with the clinical 7 class label (adjusted Rand index 0.179), which confirms the classes are cutoffs chosen based on BMI, not groups that occur naturally.
- **Continuous BMI can be predicted** from lifestyle factors alone with an R squared of 0.869 (Random Forest), a large improvement over linear models (about 0.46) and the original coursework baseline (0.207).

## License

**Code and written content in this repository** (everything except the dataset CSV) is released under the **MIT License**, see [`LICENSE`](LICENSE). It creates no ambiguity for academic reuse, verification, or citation by any other researcher.

## Ethics statement

This project uses a public, de-identified, secondary dataset collected under the original authors' institutional protocol. No new data was collected from human subjects for this work. 

## Citation

If you use this repository, please cite both the original dataset paper and this work.

```bibtex
@article{palechor2019dataset,
  author  = {Palechor, Fabio Mendoza and de la Hoz Manotas, Alexis},
  title   = {Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico},
  journal = {Data in Brief},
  volume  = {25}, pages = {104344}, year = {2019},
  doi     = {10.1016/j.dib.2019.104344}
}

Anokye, Frank. (2026). Quantifying Anthropometric Leakage in Obesity Risk Classification and It's Implications for Threshold-Defined Labels in Clinical Machine Learning. 10.13140/RG.2.2.29908.56962.
```

## Acknowledgements

This project grew out of coursework originally submitted for Data Analytics and Data Driven Decision Making (University of L'Aquila, July 2, 2024). Thanks to the original project group, Group 19, for their work on the first version of this analysis, and to the course instructor for proposing the original project.
All extensions, analyses, and results presented in this study, including the identification and measurement of anthropometric leakage, the revised classification and regression pipeline, and the accompanying codebase, were developed independently by the author.
