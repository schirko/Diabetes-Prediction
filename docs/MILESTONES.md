# Milestones

The plan for the Diabetes Prediction & Classification project, written before the work is done.
Each milestone ends with notebooks that run cleanly, an updated project page, and its tracker items closed.
Item numbers (E08, B10) refer to [ENHANCEMENTS_AND_BUGS.md](ENHANCEMENTS_AND_BUGS.md).

| # | Milestone | Status | Main question | Notebooks |
|---|---|---|---|---|
| M1 | Honest baseline and first app | Done 2026-11-15 | How good is a simple model once the evaluation is fair? | 01, 02, 03 |
| M2 | Missing data | Next | Half the insulin values are missing. Does the way we fill them change the answer? | 04 |
| M3 | Non-linear effects and feature engineering | Planned | Can curved effects and interactions fix the weak results for women 45 and over? | 05 |
| M4 | Model zoo, tuning and nested cross-validation | Planned | Given a fair, tuned contest, does any model beat logistic regression? | 06, 07 |
| M5 | Class imbalance and cost-based decisions | Planned | Do resampling tricks help, or is moving the cut-off enough? | 08 |
| M6 | Explainability | Planned | Why did the model give this woman this number? | 09 |
| M7 | Uncertainty | Planned | How sure is the model about one person's estimate? | 10 |
| M8 | External validation | Planned | Does it work on people who are not in this dataset? | 11 |
| M9 | Unsupervised structure | Planned | Are there natural groups or odd records the labels don't show? | 12 |
| M10 | Production polish | Planned | Could someone else rebuild, test and trust this? | none |

## M1. Honest baseline and first app (done)

Rebuilt the college notebook from the data up.

- Zeros that mean "not measured" are treated as missing (E01).
- Every model runs in a leak-free pipeline and is scored on the same 50 cross-validation splits (E02, E03).
- Showed the original 81% accuracy was a lucky split; the same recipe averages 77% (E04).
- Chose a screening cut-off, checked calibration, and found the weak spot for women 45 and over (E05, B10).
- Built the in-browser risk explorer and the project page (E06, E07).

**Exit check, met:** `python -m src.evaluate`, `python -m src.run_notebooks`, `python -m pytest` and
`python -m src.build_site` all run clean; the web page's risk matches scikit-learn to 6 decimal places.

## M2. Missing data

Insulin is missing for 49% of women and skinfold for 30%. M1 fills gaps with the median, the simplest choice.

- E08: compare median fill, median plus "was missing" flags, nearest-neighbour fill, iterative (MICE-style) fill, models that accept gaps directly, and dropping the two worst columns. Same 50 splits.
- E09: multiple imputation, to show how much a person's risk moves purely because of what was guessed for her insulin.

**Exit check:** one recommended strategy with evidence; the app's "not measured" switch uses it.

## M3. Non-linear effects and feature engineering

M1 assumes every input pushes risk along a straight line. The age results (B10) suggest that is too simple.

- E10: splines for glucose, BMI and age; an age-by-glucose interaction; log transforms for skewed inputs.
- E11: Explainable Boosting Machine and boosting with monotonic constraints (risk may only rise with glucose).

**Exit check:** specificity for women 45 and over improves at the same sensitivity, or we show clearly that it can't with this data. B10 closed either way.

## M4. Model zoo, tuning and nested cross-validation

- E12: support vector machine, k-nearest neighbours, XGBoost or LightGBM, and a small neural network, each tuned with Optuna inside nested cross-validation so tuning can't flatter the score.
- E13: formal comparison of models (corrected resampled t-test), so "tied" and "better" are statements with evidence.
- E14: a stacked ensemble of the best few.

**Exit check:** a ranked table with intervals, and a decision to keep or replace logistic regression.

## M5. Class imbalance and cost-based decisions

- E15: class weights, SMOTE and plain cut-off moving, compared honestly (resampling inside the folds only).
- E16: decision-curve analysis: at which costs of a missed case versus a false alarm is the model worth using?

**Exit check:** the cut-off is justified by a stated cost ratio, not a round number.

## M6. Explainability

- E17: SHAP values, permutation importance, partial-dependence and ICE curves for the M4 winner.
- E18: counterfactuals: the smallest realistic change that would move a woman below the cut-off.

**Exit check:** the app explains any model, not only a linear one.

## M7. Uncertainty

- E19: a bootstrap interval around each individual estimate.
- E20: conformal prediction, with a "too uncertain to call" outcome.

**Exit check:** the app shows a range, and the stated coverage is verified on held-out data.

## M8. External validation

- E21: test the frozen model on a different population (for example NHANES survey data with matching measurements).
- E22: measure the shift between populations and recalibrate; extend the subgroup checks beyond age.

**Exit check:** an honest "works here, fails there" table.

## M9. Unsupervised structure

- E23: clustering and low-dimensional maps (PCA, UMAP) to look for distinct patient profiles.
- E24: anomaly detection as a data-quality check (the insulin value of 846, skinfold of 99).

**Exit check:** findings either feed a new feature or are reported as "nothing useful here".

## M10. Production polish

- E25: a model card (intended use, limits, performance by group).
- E26: GitHub Actions that re-run the tests and notebooks on every push.
- E27: experiment tracking with MLflow.
- E28: optional Streamlit or FastAPI version on the VPS.

## How milestones are run

1. Open the milestone's tracker items and mark them "In progress".
2. Add the notebook(s). Every section follows the house pattern: what we're looking for, code, result, how to read it, what it means, what's next and why.
3. Move reusable code into `src/` and add a test for it.
4. Run `python -m src.evaluate`, `python -m src.run_notebooks`, `python -m pytest`, `python -m src.build_site`.
5. Update this file, the tracker, and DECISIONS.md. Commit.
