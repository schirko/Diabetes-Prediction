# Diabetes Prediction

A machine-learning project on the Pima Indians Diabetes dataset (768 women, 8 measurements, diagnosis within five years).
It started as a college notebook that reported 81% accuracy. This repo rebuilds it with an evaluation that can be trusted,
and grows milestone by milestone into deeper machine learning.

**Project page and interactive model:** https://schirko.com/scott/diabetes-prediction/

**New here?** Start with [docs/OVERVIEW.md](docs/OVERVIEW.md), the whole project in plain English in five minutes.
Every technical term is explained in [docs/GLOSSARY.md](docs/GLOSSARY.md).

## Headline results

- The original 81% accuracy was a lucky split; the same method averages 77% over 200 splits.
- The published model (logistic regression with curves for glucose, BMI and age) scores AUC 0.85 on women it never saw.
- In a fair, tuned contest with nested cross-validation, nine rivals (including XGBoost, a neural network and a stacked ensemble) didn't beat it.
- The cut-off comes from a stated cost: a missed case counts as 5 false alarms, giving 0.17. Per 100 women it catches 33 of 35 cases with 30 false alarms (the page also shows 2 times and 10 times). At the standard 0.50 it misses 39%.
- Rebalancing the data (class weights, SMOTE) doesn't help: moving the cut-off does the same job without distorting the risks.
- Glucose does most of the work. Insulin and skinfold are half missing and add nothing, so they are left out.
- Glucose, BMI and age are the only inputs it clearly relies on; a boosted-tree model ranks them the same way, and SHAP matches the model's exact contributions. The page has a "What would change this?" box for flagged women.
- It serves women 45 and over less well. The cost-based cut-off also fixes the drop in cases caught under 30, with no age-specific rules.

Every finding, organised by question and marked by how solid it is, is in [docs/FINDINGS.md](docs/FINDINGS.md).

## Layout

| Path | What it is |
|---|---|
| `data/diabetes.csv` | The dataset as published |
| `src/data.py` | Loading; turns impossible zeros into missing values |
| `src/modeling.py` | The candidate models, each in a leak-free pipeline |
| `src/missing.py` | Milestone 2: missing-data strategies, recovery test, multiple imputation |
| `src/nonlinear.py` | Milestone 3: curved effects, boosted models, insulin re-test; the published model and its export |
| `src/contest.py` | Milestone 4: tuned model contest (nested cross-validation, Optuna), corrected t-test, stacked ensemble |
| `src/decision.py` | Milestone 5: rebalancing comparison, cost-based cut-offs, decision curve, age-band policies |
| `src/explain.py` | Milestone 6: permutation importance, BMI plateau test, SHAP check, boosted-tree comparison, counterfactuals |
| `src/uncertainty.py` | Milestone 7: bootstrap ranges exported for the page, spread and coverage checks, conformal prediction |
| `src/structure.py` | Milestone 9: maps (PCA, UMAP), cluster search, odd-record checks, and the distance used by the page's unusual-inputs notice |
| `src/external.py` | Milestone 8: builds the NHANES table and tests the frozen model on it (ranking, risks, cut-offs, subgroups) |
| `src/evaluate.py` | Runs the whole evaluation, writes `reports/` |
| `src/figures.py` | Charts shared by the notebooks and the site |
| `src/run_notebooks.py` | Re-runs every notebook top to bottom |
| `src/build_site.py` | Builds the static project site into `SITE_DIR` |
| `notebooks/` | The analysis, in order. Code, result, how to read it, what it means |
| `site_template/index.html` | Template for the project page and the in-browser model |
| `docs/OVERVIEW.md` | Start here: the project in plain English |
| `docs/GLOSSARY.md` | Every technical term, with an example from this project |
| `docs/FINDINGS.md` | What the project has established, one question at a time |
| `docs/MILESTONES.md` | The milestone plan |
| `docs/ENHANCEMENTS_AND_BUGS.md` | Numbered tracker (E01..., B01...) |
| `tools/make_notebooks.py` | Scaffold that writes the notebooks' structure and text |
| `docs/DECISIONS.md` | Why things are the way they are |
| `docs/MODEL_CARD.md` | The model on one page: intended use, limits, performance by group (Milestone 10) |
| `src/tracking.py` | Milestone 10: logs each evaluation run with MLflow |
| `.github/workflows/ci.yml` | Milestone 10: re-runs the tests and rebuilds the page on every push |
| `tests/` | Checks, including that the web page's maths equals scikit-learn's |
| `Diabetes.ipynb` | The original college notebook, kept unchanged as the "before" |

## Run it

```
pip install -r requirements.txt
python -m src.evaluate          # scores every model, writes reports/
python -m src.run_notebooks     # re-runs the notebooks so outputs match the code
python -m pytest                # checks
python -m src.build_site        # writes the site to SITE_DIR (see .env.example)
mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse past evaluation runs (after running evaluate)
```

## Limits

The data is 768 women of Pima heritage aged 21 and over, collected near Phoenix, Arizona and published in 1988.
The model was run on one outside group (US women in a national survey): it ranked well but its risk numbers were far too high. It is not a medical tool.

Data source: National Institute of Diabetes and Digestive and Kidney Diseases; Smith et al., 1988.

Outside data (Milestone 8): NHANES 2013 to 2014, US Centers for Disease Control and Prevention. To rebuild `data/nhanes_women.csv`
from scratch, download `DEMO_H`, `OGTT_H`, `BMX_H`, `BPX_H`, `RHQ_H`, `GHB_H` and `DIQ_H` (`.xpt` files) from
`https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/` into `data/nhanes/`, delete the CSV and run `python -m src.evaluate`.
The small CSV is committed, so the tests and the notebooks run without the raw files.
