# Milestones

The plan for the Diabetes Prediction project, written before the work is done.
Each milestone ends with notebooks that run cleanly, an updated project page, and its tracker items closed.
Item numbers (E08, B10) refer to [ENHANCEMENTS_AND_BUGS.md](ENHANCEMENTS_AND_BUGS.md).

| # | Milestone | Status | Main question | Notebooks |
|---|---|---|---|---|
| M1 | Honest baseline and first app | Done 2026-11-15 | How good is a simple model once the evaluation is fair? | 01, 02, 03 |
| M2 | Missing data | Done 2026-11-15 | Half the insulin values are missing. Does the way we fill them change the answer? | 04 |
| M3 | Non-linear effects and feature engineering | Done 2026-11-16 | Can curved effects and interactions fix the weak results for women 45 and over? | 05 |
| M4 | Model zoo, tuning and nested cross-validation | Done 2026-11-17 | Given a fair, tuned contest, does any model beat logistic regression? | 06 |
| M5 | Class imbalance and cost-based decisions | Done 2026-11-23 | Do resampling tricks help, or is moving the cut-off enough? | 07 |
| M6 | Explainability | Done 2026-11-23 | Why did the model give this woman this number? | 08 |
| M7 | Uncertainty | Next | How sure is the model about one person's estimate? | 09 |
| M8 | External validation | Planned | Does it work on people who are not in this dataset? | 10 |
| M9 | Unsupervised structure | Planned | Are there natural groups or odd records the labels don't show? | 11 |
| M10 | Production polish | Planned | Could someone else rebuild, test and trust this? | none |
| M11 | Presentation | Planned, after M10 | Can the whole project be told in 12 slides to a non-analyst? | none |

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

## M4. Model zoo, tuning and nested cross-validation (done)

M1 to M3 compared models mostly on default settings, which can be unfair to models that need tuning.

- E12: nine model types, eight of them tuned by Optuna (30 trials) inside nested cross-validation: 15 outer splits for scoring, 3 inner folds for tuning, so the test women never influence the settings. Contestants: the published curves model, a tuned curves model, tuned straight lines, k-nearest neighbours, a support vector machine, a random forest, gradient boosting, XGBoost and a neural network.
- Results (AUC): tuned curves 0.854, published curves 0.852, stacked ensemble 0.849, SVM 0.840, random forest 0.840, straight lines 0.839, XGBoost 0.837, neural network 0.835, gradient boosting 0.832, k-nearest neighbours 0.827.
- E13: each model compared with the published one by the corrected resampled t-test. Nothing is better; only k-nearest neighbours is clearly worse (p = 0.02). The tuned curves model's +0.002 is noise (p = 0.36).
- The corrected test also re-examined M3's "curves beat straight lines": the lead is consistent (87% of splits) but not proven (p = 0.16, against 0.006 from a naive test). That finding is now rated "likely".
- Tuning preferred smoother curves (stronger smoothing in every split), supporting the suspicion that BMI's flat stretch is noise. Noted for M6.
- E14: a stacked ensemble of four model types scored 0.849, no better than the curves model alone.
- D14: the page keeps the M3 curves model.

**Exit check, met:** a ranked table with intervals and formal comparisons, and a decision to keep the published model.

## M5. Class imbalance and cost-based decisions

- E15: class weights, SMOTE and plain cut-off moving, compared honestly (resampling inside the folds only).
- E16: decision-curve analysis: at which costs of a missed case versus a false alarm is the model worth using?
- E31: cut-offs set separately by age band, to fix B12 without losing the M3 gain for older women.

**Exit check, met:** the cut-off is justified by a stated cost ratio, not a round number, and B12 is resolved.

What came out of it:
- Rebalancing (class weights, SMOTE) found nothing that moving the cut-off doesn't, and it pushed average predicted risk from 35% to 43 to 44% (D18).
- The cut-off is now 1 / (1 + cost ratio). The page defaults to a missed case being 5 times as bad as a false alarm (cut-off 0.17) and shows 2 times and 10 times beside it (D17).
- At 5 times, per 100 women: 33 of 35 cases caught, 2 missed, 30 false alarms. The old 85% rule (0.28) implied a ratio of about 2.6.
- Cut-offs per age band did not beat the single shared cut-off (D19). B12 is fixed because the shared cut-off is lower.

## M6. Explainability

- E17: permutation importance, partial-dependence and ICE curves, and SHAP values for the published model (the M4 winner was kept, D14).
- E18: counterfactuals: the smallest realistic change that would move a woman below the cut-off.

**Exit check, met:** the page explains any woman's number, and every explanation tool is checked against something.

What came out of it:
- Glucose, BMI and age are the only inputs the model clearly relies on (AUC lost when scrambled: 0.17, 0.07, 0.04; the other three 0.01 or less, ranges including zero).
- SHAP reproduces the model's exact contributions to within 0.002 log-odds; a boosted-tree model ranks the inputs in the same order (rank correlation 1.0).
- BMI's flat stretch is real but not provably flat: from BMI 30 to 42 the odds multiply by 1.2 (range 0.7 to 2.3), where a steady rise would be about 29. A straight line for BMI scores 0.006 AUC lower (p = 0.15), so the curve's benefit is likely, not proven (D21).
- At the 5 times cut-off, 477 of 768 women are flagged, and a realistic change in glucose or BMI would clear only 29% of them (D22). The box on the page says it explains the model and is not advice.

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

## M11. Presentation (after all other milestones)

Same approach as the meat-prices project.

- E34: write `docs/PRESENTATION_PLAN.md` first: about 12 slides plus an appendix, each with one headline a
  non-analyst can read on its own, one visual, and what to say over it. Story arc: the question → the lucky 81% →
  what the model does (100 women) → what drives risk → the model contest → the cut-off trade-off → who it serves
  less well → does it work elsewhere → verdict and limits.
- E34: build the deck from the plan and publish it as HTML, PowerPoint (.pptx) and PNG slide images (PDF too), linked
  from the project page. Every number is checked against FINDINGS.md at build time.

**Exit check:** the deck downloads in every format, opens cleanly, and every number matches FINDINGS.md.

## How milestones are run

1. Open the milestone's tracker items and mark them "In progress".
2. Add the notebook(s). Every section follows the house pattern: what we're looking for, code, result, how to read it, what it means, what's next and why.
3. Move reusable code into `src/` and add a test for it.
4. Run `python -m src.evaluate`, `python -m src.run_notebooks`, `python -m pytest`, `python -m src.build_site`.
5. Update the documents: FINDINGS.md (rewrite any answer the milestone changed, plain-language line first), OVERVIEW.md
   (the five-minute version), GLOSSARY.md (any new term), this file, the tracker and DECISIONS.md. Give the new notebook a
   plain-language summary. Commit.
