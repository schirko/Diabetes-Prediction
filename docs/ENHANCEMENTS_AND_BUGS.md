# Enhancements and bugs

One numbered list for everything done and planned. Refer to items by number (E08, B10).
Numbers are never reused. Milestones are described in [MILESTONES.md](MILESTONES.md).

Last updated: 2026-11-15

## Enhancements

| ID | Enhancement | Milestone | Status |
|---|---|---|---|
| E01 | Treat impossible zeros (glucose, blood pressure, skinfold, insulin, BMI) as missing values | M1 | Done |
| E02 | Leak-free pipelines: gap filling and rescaling are learned inside each training fold | M1 | Done |
| E03 | Score every model on the same 50 cross-validation splits; report AUC, Brier score and sensitivity, not accuracy alone | M1 | Done |
| E04 | "Lucky split" demonstration: the original recipe on 200 random splits | M1 | Done |
| E05 | Screening cut-off, calibration check, glucose-rule benchmark, results by age band | M1 | Done |
| E06 | In-browser risk explorer (model runs in JavaScript, no server) | M1 | Done |
| E07 | Project page, lean notebook export, one-command build, parity tests | M1 | Done |
| E08 | Compare missing-data strategies: median, missing flags, nearest neighbours, iterative, native, drop columns | M2 | Planned |
| E09 | Multiple imputation: how much one person's risk moves because of guessed insulin | M2 | Planned |
| E10 | Splines, interactions (age by glucose) and log transforms | M3 | Planned |
| E11 | Explainable Boosting Machine and monotonic-constraint boosting | M3 | Planned |
| E12 | Tuned SVM, k-NN, XGBoost/LightGBM and neural network in nested cross-validation (Optuna) | M4 | Planned |
| E13 | Formal model comparison (corrected resampled t-test) | M4 | Planned |
| E14 | Stacked ensemble | M4 | Planned |
| E15 | Class weights vs SMOTE vs cut-off moving | M5 | Planned |
| E16 | Decision-curve analysis and a cost-based cut-off | M5 | Planned |
| E17 | SHAP, permutation importance, partial-dependence and ICE curves | M6 | Planned |
| E18 | Counterfactual explanations in the app | M6 | Planned |
| E19 | Bootstrap interval around each individual estimate | M7 | Planned |
| E20 | Conformal prediction with a "too uncertain to call" outcome | M7 | Planned |
| E21 | External validation on a different population | M8 | Planned |
| E22 | Population-shift measurement, recalibration, wider subgroup checks | M8 | Planned |
| E23 | Clustering and low-dimensional maps (PCA, UMAP) | M9 | Planned |
| E24 | Anomaly detection as a data-quality check | M9 | Planned |
| E25 | Model card | M10 | Planned |
| E26 | GitHub Actions: tests and notebooks re-run on every push | M10 | Planned |
| E27 | Experiment tracking with MLflow | M10 | Planned |
| E28 | Optional Streamlit or FastAPI version on the VPS | M10 | Planned |

## Bugs

B01 to B09 were found in the original college notebook (`Diabetes.ipynb`) and are fixed in the rebuild.
The original file is kept unchanged as the "before".

| ID | Bug | Found | Status |
|---|---|---|---|
| B01 | Zeros were filled using statistics from the whole table before the train/test split (a leak). The insulin fill value (30.5) was the median of a column that was half zeros. | M1 | Fixed by E01, E02 |
| B02 | "Training Score", "Mean Squared Error" and "R2" printed under every model were the logistic regression's numbers (`reg.score`, a stale `y_pred`), not that model's. | M1 | Fixed: one scoring function for all models |
| B03 | The SVM grid search was fitted on all rows, test rows included (`grid_search.fit(X, y)`), then scored on those test rows. | M1 | Fixed by E02 |
| B04 | Precision was described as "out of all the people in the dataset the model says 77% had diabetes". Precision is the share of flagged women who really have diabetes. | M1 | Fixed in notebook text |
| B05 | The "About this Project" text was carried over from the life-expectancy project ("predict various life expectancies by country"). | M1 | Fixed |
| B06 | DiabetesPedigreeFunction was described as a glucose test. It is a family-history score. | M1 | Fixed |
| B07 | Exploratory notes said glucose, insulin, family history and pregnancies were lower among women with diabetes. All four are higher. | M1 | Fixed, see notebook 01 section 1.5 |
| B08 | Text said 624 training rows. An 80/20 split of 768 gives 614 and 154. | M1 | Fixed |
| B09 | Headline accuracies (79% to 81%) came from one favourable split; R2 and mean squared error were used to judge a yes/no classifier. | M1 | Fixed by E03, E04 |
| B10 | One cut-off for everyone clears only 22% of healthy women aged 45 and over, so for that group the model flags nearly everyone. | M1 | Open, target M3 (E10, E11) |
| B11 | The Git repo tracks seven old notebook copies (about 50 MB) and an unrelated `yield_df.csv`. | M1 | Open, waiting on Scott's cleanup commands |

## How to use this file

- New idea: add the next E number with status "Planned" and a milestone (or "Unassigned").
- New problem: add the next B number with where it was found.
- Status values: Planned, In progress, Done, Dropped (with a reason), Open, Fixed.
