# Milestones

The plan for the Diabetes Prediction project, written before the work is done.
Each milestone ends with notebooks that run cleanly, an updated project page, and its tracker items closed.
Item numbers (E08, B10) refer to [ENHANCEMENTS_AND_BUGS.md](ENHANCEMENTS_AND_BUGS.md).

| # | Milestone | Status | Main question | Notebooks |
|---|---|---|---|---|
| M1 | Honest baseline and first app | Done 2026-11-15 | How good is a simple model once the evaluation is fair? | 01, 02, 03 |
| M2 | Missing data | Done 2026-11-15 | Half the insulin values are missing. Does the way we fill them change the answer? | 04 |
| M3 | Non-linear effects and feature engineering | Done 2026-11-16 | Can curved effects and interactions fix the weak results for women 45 and over? | 05 |
| M4 | Model zoo, tuning and nested cross-validation | Next | Given a fair, tuned contest, does any model beat logistic regression? | 06, 07 |
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
`python -m src.build_site` all run clean; the web page's risk matches scikit-learn exactly on the test cases.

## M2. Missing data (done)

Insulin is missing for 49% of women and skinfold for 30%. M1 filled gaps with the median, the simplest choice.

- E08: compared seven strategies on the same 50 splits. The six logistic versions score AUC 0.832 to 0.838, a range one fifth the size of the split-to-split wobble (0.027). "Was missing" flags add nothing.
- E08: hid known readings and asked each method to guess them back. The best recovers 29% of the variation in insulin and 40% in skinfold. Insulin can't be reconstructed.
- E09: filled the gaps 30 plausible ways. A woman with no insulin reading typically moves 5.3 percentage points; 53 of 768 women (7%) land on both sides of the cut-off.
- D09: the published model now leaves insulin and skinfold out. Six inputs score AUC 0.838 against 0.836 for eight, and women with any filled-in value fall from 374 to 44.

**Exit check, met:** one recommended strategy with evidence (leave the two columns out, median fill for the few remaining gaps); the app runs the six-input model and its "not measured" switch uses the median.

## M3. Non-linear effects and feature engineering (done)

M1 and M2 assumed every input pushes risk along a straight line. B10 said that broke down for older women.

- Diagnosis: B10 was two problems. Half of women 45 and over have diabetes, so a shared cut-off flags most of them whatever the model does (base rate). But the straight line was also the wrong shape: the real rate peaks at 71% at ages 50 to 54 and falls to 28% at 60 and over, while the model kept predicting about 50%.
- E10: compared log scales, an age curve, curves for glucose, BMI and age, an age-by-glucose interaction, and curves plus interaction. Curves for glucose, BMI and age won: AUC 0.853 against 0.838, winning 86% of the 50 splits, and 0.854 on a fresh set of splits. The interaction and log scales added nothing.
- Extreme values are held at the 1st and 99th percentiles before the curves. Without that, the BMI curve multiplied the odds by 82 at the top of its range.
- E11: an Explainable Boosting Machine (0.841) and boosting constrained so risk only rises with glucose, BMI and family history (0.832). Neither beat the curves.
- E29: on the 394 women with a real insulin reading, insulin added nothing in any form (raw, log, or log of glucose times insulin). D09 stands.
- D11: the page now runs the curves model. At the same 85% catch rate, false alarms fall from 194 to 154 and healthy women 45 and over cleared rise from 22% to 43%, for 2 more missed cases.
- B10 closed as improved. New B12: under 30, the shared cut-off now catches 71% of cases instead of 80%.

**Exit check, met:** specificity for women 45 and over improved at the same overall sensitivity (22% to 43%), with the remaining gap explained (base rate, and weaker separation among older women: AUC 0.79 against 0.85 under 30).

## M4. Model zoo, tuning and nested cross-validation

- E12: support vector machine, k-nearest neighbours, XGBoost or LightGBM, and a small neural network, each tuned with Optuna inside nested cross-validation so tuning can't flatter the score.
- E13: formal comparison of models (corrected resampled t-test), so "tied" and "better" are statements with evidence.
- E14: a stacked ensemble of the best few.

**Exit check:** a ranked table with intervals, and a decision to keep or replace the M3 curves model.

## M5. Class imbalance and cost-based decisions

- E15: class weights, SMOTE and plain cut-off moving, compared honestly (resampling inside the folds only).
- E16: decision-curve analysis: at which costs of a missed case versus a false alarm is the model worth using?
- E31: cut-offs set separately by age band, to fix B12 without losing the M3 gain for older women.

**Exit check:** the cut-off is justified by a stated cost ratio, not a round number, and B12 is resolved or explained.

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
5. Update the documents: FINDINGS.md (rewrite any answer the milestone changed, plain-language line first), OVERVIEW.md
   (the five-minute version), GLOSSARY.md (any new term), this file, the tracker and DECISIONS.md. Give the new notebook a
   plain-language summary. Commit.
