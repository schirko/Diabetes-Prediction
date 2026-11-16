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
- At a screening cut-off of 0.28 it catches 85% of cases; at the standard 0.50 it misses 39%.
- Glucose does most of the work. Insulin and skinfold are half missing and add nothing, so they are left out.
- It serves women 45 and over less well, and at the shared cut-off catches fewer cases under 30 (open for Milestone 5).

Every finding, organised by question and marked by how solid it is, is in [docs/FINDINGS.md](docs/FINDINGS.md).

## Layout

| Path | What it is |
|---|---|
| `data/diabetes.csv` | The dataset as published |
| `src/data.py` | Loading; turns impossible zeros into missing values |
| `src/modeling.py` | The candidate models, each in a leak-free pipeline |
| `src/missing.py` | Milestone 2: missing-data strategies, recovery test, multiple imputation |
| `src/nonlinear.py` | Milestone 3: curved effects, boosted models, insulin re-test; the published model and its export |
| `src/evaluate.py` | Runs the whole evaluation, writes `reports/` |
| `src/figures.py` | Charts shared by the notebooks and the site |
| `src/run_notebooks.py` | Re-runs every notebook top to bottom |
| `src/build_site.py` | Builds the static project site into `SITE_DIR` |
| `notebooks/` | The analysis, in order. Code, result, how to read it, what it means |
| `site_template/index.html` | Template for the project page and the in-browser model |
| `docs/OVERVIEW.md` | Start here: the project in plain English |
| `docs/GLOSSARY.md` | Every technical term, with an example from this project |
| `docs/FINDINGS.md` | What the project has established, one question at a time |
| `docs/MILESTONES.md` | The ten-milestone plan |
| `docs/ENHANCEMENTS_AND_BUGS.md` | Numbered tracker (E01..., B01...) |
| `tools/make_notebooks.py` | Scaffold that writes the notebooks' structure and text |
| `docs/DECISIONS.md` | Why things are the way they are |
| `tests/` | Checks, including that the web page's maths equals scikit-learn's |
| `Diabetes.ipynb` | The original college notebook, kept unchanged as the "before" |

## Run it

```
pip install -r requirements.txt
python -m src.evaluate          # scores every model, writes reports/
python -m src.run_notebooks     # re-runs the notebooks so outputs match the code
python -m pytest                # checks
python -m src.build_site        # writes the site to SITE_DIR (see .env.example)
```

## Limits

The data is 768 women of Pima heritage aged 21 and over, collected near Phoenix, Arizona and published in 1988.
The model has not been tested on anyone else and is not a medical tool.

Data source: National Institute of Diabetes and Digestive and Kidney Diseases; Smith et al., 1988.
