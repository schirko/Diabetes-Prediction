# Model card

A short, standard description of the published model: what it is for, what it is not for, how well it works, and where it
breaks. Written for a reader who has not seen the rest of the project. The numbers come from
[FINDINGS.md](FINDINGS.md) and are checked against `reports/metrics.json` by a test, so they cannot drift.

Last updated: 2026-11-24, after Milestone 10.

## The model in one paragraph

A logistic regression that takes six measurements for one woman (pregnancies, two-hour glucose, blood pressure, BMI,
family-history score and age) and returns the chance, from 0% to 100%, that she is diagnosed with diabetes within five
years. Glucose, BMI and age are allowed to curve; the other three are straight lines. Extreme values are held at the 1st
and 99th percentile of the training women. Insulin and skinfold thickness are not used (D09). The risk is a sum of one
effect per measurement, so each measurement's contribution can be shown exactly.

## Intended use

- **Demonstration.** Showing how a risk model is built, tested and judged, for readers who are not analysts.
- **A worked example of screening logic.** Turning a risk into a "send for a follow-up test" decision with a cut-off that
  comes from a stated cost: a missed case is treated as 5 times as bad as an unnecessary test, so the cut-off is 17%. The
  page also shows 2 times (33%) and 10 times (9%).

## Not intended for

- **Diagnosing or advising any real person.** It is not a medical device and was never validated clinically.
- **Women outside the data's population.** On a US national survey sample the risk percentages were several times too high
  (see below).
- **Men, children, or anyone under 21.** None are in the data.
- **Telling anyone what would lower their risk.** The "What would change this?" box explains the model's arithmetic. The
  model describes patterns, not causes.

## Data

- 768 women of Pima heritage aged 21 and over, from one community near Phoenix, Arizona. Collected by the US National
  Institute of Diabetes and Digestive and Kidney Diseases and published in 1988.
- 268 (35%) were diagnosed with diabetes within five years, far higher than in most populations.
- Impossible zeros (glucose, blood pressure, skinfold, insulin, BMI) are treated as missing. Insulin is missing for 49% of
  women and skinfold for 30%, which is why they are left out.
- Outside check: 961 women aged 21 and over from the US NHANES survey, 2013 to 2014 (37 with diabetes, 3.9%).

## Performance

All figures are for women the model had not seen during training (repeated 5-fold cross-validation, 50 splits).

| Measure | Result |
|---|---|
| Ranking (AUC) | 0.85 (95% interval 0.83 to 0.88) |
| At the 17% cut-off, per 100 women (35 with diabetes) | 33 caught, 2 missed, 30 false alarms, 35 correctly cleared |
| Calibration slope (1.0 is ideal) | 0.94 |
| Against a "glucose of 140 or more" rule | 21 more cases found for about the same number of false alarms |
| Against nine tuned rival model types | None beat it |

### By group

| Group | Women | AUC |
|---|---|---|
| 21 to 29 | 396 | 0.85 |
| 30 to 44 | 239 | 0.80 |
| 45 and over | 133 | 0.79 |

Half of the women 45 and over have diabetes, so any shared cut-off flags most of them. The model also separates them less
well. This group is small, so its numbers are uncertain.

### On women outside the data (NHANES)

| Measure | Result |
|---|---|
| Ranking (AUC) | 0.88 (0.82 to 0.93); glucose alone scores 0.90 |
| Average predicted risk against actual | 25% against 3.9% |
| At the 17% cut-off | catches 89% of cases (33 of 37), flags 45% of women, 7.6% of those flagged have diabetes |

The ranking travels; the percentages do not. Only 37 women have diabetes in this group, and the outcome is defined
differently from the Pima data (D26), so treat this as behaviour away from home, not a repeat of the same test.

## Uncertainty

- **Range.** Each estimate carries a 95% bootstrap range, typically 19 percentage points wide. In a simulation it held the
  true risk 91% of the time, so the page calls it "about 90%".
- **"Too uncertain to call."** A conformal rule says no, yes or too uncertain, with a guarantee that holds on unseen women
  (about 90% for women with and without diabetes separately). At the 90% level a third of women are too uncertain, and a
  confident "yes" is still wrong 25% of the time.
- **Unusual inputs.** The page shows a notice when a woman's numbers are further from the typical Pima woman than 95 in 100
  Pima women are. About a quarter of the NHANES women would trigger it. The notice does not change the risk.

## Known limits

- One community, one era (data published in 1988). Nothing here shows it works in other populations or today.
- The outcome is a diagnosis within five years, set by a glucose threshold. It is not "has diabetes now".
- The cost ratio of 5 is a judgment. Change it and the cut-off changes.
- Women 45 and over are served less well, and the oldest are slightly over-predicted at the extremes.
- Odd records (insulin of 846, skinfold of 99, BMI of 67) are kept. Removing them changed nothing measurable (D30).
- Glucose does most of the work. Outside the Pima data the other five inputs added nothing measurable beyond it.

## How it is kept honest

- Every score comes from women the model did not train on, on the same 50 splits.
- A test confirms the page's JavaScript gives the same risk as scikit-learn.
- A test confirms the numbers in this card match `reports/metrics.json`.
- GitHub Actions re-runs the tests and rebuilds the page on every push (E26). Each evaluation run is logged with MLflow
  (E27).

## Rebuilding it

```
pip install -r requirements.txt
python -m src.evaluate        # scores everything, writes reports/
python -m pytest              # checks, including this card's numbers
python -m src.run_notebooks   # re-runs every notebook
python -m src.build_site      # writes the project page into SITE_DIR
```

Questions about "why" are answered in [DECISIONS.md](DECISIONS.md); every term is in the [glossary](GLOSSARY.md).
