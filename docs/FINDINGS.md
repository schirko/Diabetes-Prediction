# Findings

What this project has established so far, one question at a time. Each answer opens with a plain-language
version; the details and numbers follow for readers who want them. New to the project? Read the
[overview](OVERVIEW.md) first, and use the [glossary](GLOSSARY.md) for any unfamiliar word.

Each answer is the current one; when a later milestone changes it, the answer is rewritten and the old one is noted
under "Changed". Evidence points to the notebooks; item numbers (E10, B12, D11) refer to the
[tracker](ENHANCEMENTS_AND_BUGS.md) and [decisions](DECISIONS.md).

Last updated: 2026-11-16, after Milestone 3.

**How solid each answer is:**

- **Firm**: holds on repeated splits, a fresh set of splits, or a direct test.
- **Likely**: the direction is clear, but the size is uncertain because the group is small.
- **Open**: a known gap, with a milestone planned for it.

All numbers come from predictions for women the model had not seen during training (repeated 5-fold
cross-validation, 50 train/test splits), unless stated otherwise.

## Summary

| Question | Answer | How solid |
|---|---|---|
| How good is the model? | AUC 0.85 (95% interval 0.83 to 0.88) | Firm |
| Was the original 81% accuracy real? | No, a lucky split. The same method averages 77% | Firm |
| Which model? | Logistic regression with curves for glucose, BMI and age | Firm |
| What drives the prediction? | Glucose, far ahead of anything else | Firm |
| Do insulin and skinfold help? | No. Half missing, can't be rebuilt, add nothing even when measured | Firm |
| Can the probabilities be believed? | Yes, for women like these (calibration slope 0.94) | Firm |
| Where should the cut-off go? | 0.28 catches 85% of cases; the 85% target is a judgment call | Open (M5) |
| Who does it serve less well? | Women 45 and over (weaker separation); women under 30 at the shared cut-off | Likely / Open (M5) |
| Does it beat a simple glucose rule? | Yes: 21 more cases found for about the same number of false alarms | Firm |
| Does it work outside this data? | Unknown | Open (M8) |

## 1. How good is the model?

**In plain terms.** Out of every 100 women like these, about 35 will develop diabetes. The model flags 30 of
them for a follow-up test and misses 5. It also sends 20 healthy women for tests they didn't need.

**Answer.** It ranks a woman who develops diabetes above one who doesn't 85% of the time (AUC 0.854, 95% interval
0.83 to 0.88). For comparison, a coin flip scores 0.50, and always answering "no diabetes" is right 65% of the
time while catching nobody.

At the screening cut-off of 0.28 it catches 229 of 268 women with diabetes (85%) and correctly clears 346 of 500
without it (69%). Of the women it flags, 60% do have diabetes. At the standard 0.50 cut-off it misses 104 of 268
(39%).

**How solid.** Firm. Averaged over the 50 splits the AUC is 0.853; on a fresh set of 50 splits it is 0.854.

**Evidence.** Notebooks 02, 03 and 05 (section 5.9).

**Changed.** M1: AUC 0.836 (eight inputs, straight lines). M2: 0.838 (six inputs). M3: 0.854 (curves).

## 2. Was the original 81% accuracy real?

**In plain terms.** No. The college version was tested once and got a good draw. Tested fairly, many times, the
same method scores 77%.

**Answer.** No. The college notebook scored 81% on one random train/test split. Repeating the same recipe on 200
different splits gives anything from 68% to 84%, averaging 77%; the original split beat 88% of them. The original
also filled missing values using the whole table before splitting, which lets test data leak into training (B01).
Nine bugs from the original are listed as B01 to B09.

**How solid.** Firm. Reproduced exactly, then repeated 200 times.

**Evidence.** Notebook 02, sections 2.1 and 2.2.

## 3. Which model should be used?

**In plain terms.** A simple, explainable model that lets three measurements bend instead of forcing them into
straight lines. More complicated methods did not do better on 768 women.

**Answer.** Logistic regression with curves (splines) for glucose, BMI and age, straight lines for the other three
inputs, and extreme values held at the 1st and 99th percentiles (D11). It is the best of every model tried, and it
remains a sum of one effect per input, so each measurement's contribution can be shown exactly.

| Model | AUC |
|---|---|
| Curves for glucose, BMI and age (published) | 0.853 |
| Curves plus an age-by-glucose interaction | 0.853 |
| Curve for age only | 0.847 |
| Explainable Boosting Machine | 0.841 |
| Logistic regression, straight lines (six inputs) | 0.838 |
| Random forest (eight inputs, M1) | 0.838 |
| Gradient boosting, risk forced to rise with glucose and BMI | 0.832 |
| Decision tree (eight inputs, M1) | 0.807 |

The curves beat straight lines on 86% of the 50 splits and on a fresh set of splits. With 768 women, a few
well-chosen curves do better than models that must discover every shape themselves (D12).

**How solid.** Firm for "curves beat straight lines". The gap between the curves and the Explainable Boosting
Machine is small, and Milestone 4 retests every model type with proper tuning.

**Evidence.** Notebook 02 (section 2.4) and notebook 05 (sections 5.3 and 5.4).

**Changed.** M1 and M2: straight-line logistic regression, chosen because it tied the random forest.

## 4. What drives the prediction?

**In plain terms.** Blood sugar after a sugary drink, far more than anything else. Weight and age come next.
Risk rises with age until about 50, then falls.

**Answer.** Glucose two hours after a sugar drink, by a wide margin. Glucose alone reaches AUC 0.79; the other five
inputs together add about 0.06.

Moving each input from a low value to a high one (10th to 90th percentile of the women in the data), with the rest
held at typical values, multiplies the odds by:

| Input | Low to high | Odds |
|---|---|---|
| Glucose | 86 to 167 mg/dL | ×25.6 |
| BMI | 24 to 42 | ×6.2 |
| Age | 22 to 51 years | ×5.5 |
| Family-history score | 0.16 to 0.88 | ×1.8 |
| Pregnancies | 0 to 9 | ×1.6 |
| Blood pressure | 58 to 88 mm Hg | ×0.7 |

Age does not rise steadily: the diabetes rate peaks at 71% at ages 50 to 54 and falls to 28% at 60 and over, and
the model's age curve follows that. BMI rises steeply up to about 30, is roughly flat from 30 to 42, then rises
again; the flat stretch may be noise.

These are associations in this group of women, not causes.

**How solid.** Firm for glucose. Likely for the exact curve shapes, especially at the edges, where few women sit.

**Evidence.** Notebook 01 (section 1.5), notebook 02 (section 2.4) and notebook 05 (sections 5.1 and 5.5).

## 5. Do insulin and skinfold thickness help?

**In plain terms.** No. Half the insulin readings were never taken, the gaps can't be filled reliably, and even
real readings add nothing that blood sugar and weight don't already tell us. So the model doesn't ask for them.

**Answer.** No, and the published model leaves both out (D09). Insulin is recorded as zero, meaning not measured,
for 374 of 768 women (49%), and skinfold for 227 (30%).

- Seven ways of filling the gaps all score the same (AUC 0.832 to 0.838).
- When known insulin readings are hidden, the best method recovers only 29% of their variation.
- Filling the gaps 30 plausible ways moves 53 women (7%) from one side of the cut-off to the other.
- On the 394 women who do have an insulin reading, adding it in any form (raw, log, or combined with glucose as a
  rough insulin-resistance measure) lowers AUC slightly.

**How solid.** Firm. Four separate tests point the same way.

**Evidence.** Notebook 04 and notebook 05 (section 5.8).

**Changed.** M1 used all eight inputs, with gaps filled by the median.

## 6. Can the probabilities be believed?

**In plain terms.** Yes, for women like these. When the model says 30%, about 30 in 100 such women did develop
diabetes.

**Answer.** Yes, for a population like this one. When the model says 30%, about 30% of such women had diabetes
(calibration slope 0.94, where 1.00 is ideal). The curved model also fixed the over-prediction for women 60 and
over (predicted 30%, actual 28%; the straight-line model said 50%).

**How solid.** Firm within this data. The probabilities assume a population where about 35% have diabetes; in a
group where it is rarer, they would run high.

**Evidence.** Notebook 03 (section 3.4) and notebook 05 (section 5.1 and the age chart in 5.6).

## 7. Where should the cut-off go?

**In plain terms.** There's no single right answer. Flagging more women catches more cases but means more
unnecessary tests. The project currently flags anyone at 28% risk or more, which catches 85% of cases. Milestone 5
will set it from the real costs instead.

**Answer.** For screening, 0.28: the highest cut-off that still catches at least 85% of cases. The cost is 154 false
alarms among 500 healthy women, and half of all women get flagged. At 0.50, false alarms drop to 68, but 104 cases
are missed.

The 85% target is a judgment, not something the data decides (D03). The right cut-off depends on what a missed case
costs compared with a follow-up blood test.

**How solid.** Open. Milestone 5 sets the cut-off from stated costs (E16) and tests cut-offs by age group (E31).

**Evidence.** Notebook 03 (section 3.2) and the cut-off explorer on the project page.

**Changed.** M1: 0.24. M2: 0.23. M3: 0.28, because the curved model's risks are spread differently.

## 8. Who does the model serve less well?

**In plain terms.** Older women, partly because half of them have diabetes, which makes them hard to tell apart.
And young women lose out a little at the current cut-off: fewer of their cases are caught than with the earlier
model.

**Answer.** Two groups, for different reasons.

- **Women 45 and over.** Half of them have diabetes, so a shared cut-off flags most of them; that part is the base
  rate, not a fault. The model also separates them less well (AUC 0.79, against 0.85 under 30). The curved model
  improved this group: healthy women 45 and over who are cleared rose from 22% to 43% (B10, improved).
- **Women under 30, at the shared cut-off.** The curved model gives them lower, more accurate risks, so more of
  their cases fall under the cut-off: 60 of 84 caught (71%), against 80% for the straight-line model (B12).

**How solid.** Likely. Only 133 women are 45 and over (67 of them healthy); the gain for them is clear in direction
but uncertain in size (95% interval from slightly worse to much better). B12 is open for Milestone 5.

**Evidence.** Notebook 03 (section 3.5) and notebook 05 (sections 5.2, 5.6 and 5.7).

## 9. Does it beat a simple rule?

**In plain terms.** Yes. Compared with flagging everyone whose blood sugar is 140 or more, the model finds 21 more
women with diabetes for about the same number of false alarms.

**Answer.** Yes, modestly. Flagging every woman with a two-hour glucose of 140 mg/dL or more catches 135 cases with
62 false alarms. The model, set to raise about the same number of false alarms (64), catches 156: 21 more women
found.

**How solid.** Firm.

**Evidence.** Notebook 03, section 3.3 (shown there for the M1 model, which found 17 more).

## 10. Does it work outside this data?

**In plain terms.** We don't know yet. Every woman in the data comes from one community studied decades ago.

**Answer.** Unknown. All 768 women are of Pima heritage, aged 21 and over, from one community near Phoenix, Arizona,
in data published in 1988. Diabetes is unusually common in this group (35%). Nothing here shows the model works on
anyone else, and the page says so.

**How solid.** Open. Milestone 8 tests the model on a different population (E21).

## What didn't work

Ideas tested and dropped, with the result, so they aren't retried without a new reason. Most are ways of making the
model more complicated; none of them helped.

| Idea | Result | Evidence |
|---|---|---|
| Dropping four inputs chosen by correlation (as the original did) | Same AUC (0.833 vs 0.835) | Notebook 02, 2.5 |
| "Was missing" flags for the gaps | No gain (0.832) | Notebook 04, 4.1 |
| Nearest-neighbour or iterative filling of gaps | No gain (0.836 to 0.837) | Notebook 04, 4.1 |
| Age-by-glucose interaction | No gain, alone or with curves | Notebook 05, 5.3 |
| Log scale for skewed inputs | No gain | Notebook 05, 5.3 |
| Explainable Boosting Machine | Lower than the curves (0.841) | Notebook 05, 5.3 |
| Boosting forced to rise with glucose, BMI, family history | Lowest of the nine (0.832) | Notebook 05, 5.3 |
| Insulin, raw or combined with glucose | Slightly lower AUC | Notebook 05, 5.8 |
