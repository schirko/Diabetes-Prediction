# Diabetes Prediction

A machine-learning project on the Pima Indians Diabetes dataset (768 women, 8 measurements, diagnosis within five years).
It started as a college notebook that reported 81% accuracy. This repo rebuilds it with an evaluation that can be trusted,
and grows milestone by milestone into deeper machine learning.

**Project page and interactive model:** https://schirko.com/scott/diabetes-prediction/

## What Milestone 1 found

- The original 81% accuracy was a lucky train/test split. The same method over 200 random splits averages 77% (range 68% to 84%). Always answering "no diabetes" scores 65%.
- 374 of 768 insulin values (49%) and 227 skinfold values (30%) are zeros that mean "not measured".
- Logistic regression (AUC 0.835) ties a random forest (0.838) on the same 50 cross-validation splits. Glucose alone reaches 0.791.
- At the default 0.50 cut-off the model misses 116 of 268 women with diabetes. A screening cut-off of 0.24 catches 85%, with 188 false alarms instead of 61.
- Predicted probabilities are well calibrated (slope 0.96).
- Weak spot: for women 45 and over the model clears only 22% of those without diabetes (tracked as B10, still open).

## What Milestone 2 found

- Seven ways of handling the missing readings score the same: AUC 0.832 to 0.838 for the six logistic versions.
- Hidden insulin readings can't be guessed back. The best method recovers 29% of their variation.
- Filling the gaps 30 plausible ways moves 53 of 768 women (7%) from one side of the cut-off to the other.
- The published model now leaves insulin and skinfold out: six inputs, AUC 0.838, cut-off 0.23, 86% of cases caught.

## Layout

| Path | What it is |
|---|---|
| `data/diabetes.csv` | The dataset as published |
| `src/data.py` | Loading; turns impossible zeros into missing values |
| `src/modeling.py` | The candidate models, each in a leak-free pipeline |
| `src/missing.py` | Milestone 2: missing-data strategies, recovery test, multiple imputation |
| `src/evaluate.py` | Runs the whole evaluation, writes `reports/` |
| `src/figures.py` | Charts shared by the notebooks and the site |
| `src/run_notebooks.py` | Re-runs every notebook top to bottom |
| `src/build_site.py` | Builds the static project site into `SITE_DIR` |
| `notebooks/` | The analysis, in order. Code, result, how to read it, what it means |
| `site_template/index.html` | Template for the project page and the in-browser model |
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
