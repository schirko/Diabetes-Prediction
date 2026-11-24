# Findings

What this project has established so far, one question at a time. Each answer opens with a plain-language
version; the details and numbers follow for readers who want them. New to the project? Read the
[overview](OVERVIEW.md) first, and use the [glossary](GLOSSARY.md) for any unfamiliar word.

Each answer is the current one; when a later milestone changes it, the answer is rewritten and the old one is noted
under "Changed". Evidence points to the notebooks; item numbers (E10, B12, D11) refer to the
[tracker](ENHANCEMENTS_AND_BUGS.md) and [decisions](DECISIONS.md).

Last updated: 2026-11-23, after Milestone 9.

**How solid each answer is:**

- **Firm**: holds on repeated splits, a fresh set of splits, or a direct test.
- **Likely**: the direction is clear, but the size is uncertain because the group is small.
- **Open**: a known gap, with a milestone planned for it.

All numbers come from predictions for women the model had not seen during training (repeated 5-fold
cross-validation, 50 train/test splits), unless stated otherwise. The Milestone 4 contest uses 15 splits with tuning
kept separate (nested cross-validation).

## Summary

| Question | Answer | How solid |
|---|---|---|
| How good is the model? | AUC 0.85 (95% interval 0.83 to 0.88) | Firm |
| Was the original 81% accuracy real? | No, a lucky split. The same method averages 77% | Firm |
| Which model? | Logistic regression with curves for glucose, BMI and age. Nine tuned rivals didn't beat it | Firm |
| What drives the prediction? | Glucose, far ahead of anything else | Firm |
| Do insulin and skinfold help? | No. Half missing, can't be rebuilt, add nothing even when measured | Firm |
| Can the probabilities be believed? | Yes, for women like these (calibration slope 0.94) | Firm |
| Where should the cut-off go? | From a stated cost. At 5 times, 0.17: 33 of 35 cases caught, 30 false alarms per 100 | Firm on method; the cost ratio is a judgment |
| Who does it serve less well? | Women 45 and over (weaker separation). Women under 30 are fixed by the cost-based cut-off | Likely |
| Why did it give her this number? | Glucose, then BMI and age; the same order in a boosted-tree model; exact explanation | Firm |
| Can a flagged woman be cleared by a small change? | Rarely: about 3 in 10 at the page's cut-off | Firm |
| How sure is the model about one woman? | Typical 95% range 19 points wide; it holds the true risk about 91% of the time | Firm (simulation) |
| Can it say "too uncertain"? | Yes. Guarantee holds on unseen women; 33% uncertain at 90%, 7% at 80% | Firm |
| Does rebalancing help? | No: same ranking, distorted risks | Firm |
| Does it beat a simple glucose rule? | Yes: 21 more cases found for about the same number of false alarms | Firm |
| Does it work outside this data? | The ranking does (AUC 0.88 on 961 US women). The risk numbers do not: 25% predicted against 3.9% actual | Likely (only 37 cases) |
| Are there hidden groups or odd records? | No stable groups; odd records found but harmless; groups and oddness scores don't help the model | Firm |

## 1. How good is the model?

**In plain terms.** Out of every 100 women like these, about 35 will develop diabetes. The model flags 33 of
them for a follow-up test and misses 2. It also sends 30 healthy women for tests they didn't need.

**Answer.** It ranks a woman who develops diabetes above one who doesn't 85% of the time (AUC 0.854, 95% interval
0.83 to 0.88). For comparison, a coin flip scores 0.50, and always answering "no diabetes" is right 65% of the
time while catching nobody.

At the page's cut-off of 0.17 it catches 252 of 268 women with diabetes (94%) and correctly clears 269 of 500
without it (54%). Of the women it flags, 52% do have diabetes. At the standard 0.50 cut-off it misses 104 of 268
(39%). (Before Milestone 5 the cut-off was 0.28, which caught 229 of 268.)

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
straight lines. Nine other kinds of model were each given a fair, fully tuned chance, and none did better. Combining
several models didn't help either.

**Answer.** Logistic regression with curves (splines) for glucose, BMI and age, straight lines for the other three
inputs, and extreme values held at the 1st and 99th percentiles (D11, kept in D14). It is a sum of one effect per
input, so each measurement's contribution can be shown exactly.

In Milestone 4 every model type had its settings tuned automatically (Optuna, 30 tries) using only training women,
and was then scored on women it had never seen (nested cross-validation, 15 splits):

| Model | AUC | Compared with the published model |
|---|---|---|
| Curves, settings tuned | 0.854 | +0.002, could be chance (p = 0.36) |
| **Curves, as published** | **0.852** | |
| Stacked ensemble of four model types | 0.849 | -0.003, could be chance |
| Support vector machine | 0.840 | -0.011, probably worse |
| Random forest | 0.840 | -0.012, probably worse |
| Straight lines (logistic regression) | 0.839 | -0.013, probably worse |
| XGBoost | 0.837 | -0.015, probably worse |
| Neural network | 0.835 | -0.017, probably worse |
| Gradient boosting | 0.832 | -0.020, probably worse |
| k-nearest neighbours | 0.827 | -0.025, clearly worse (p = 0.02) |

"Probably worse" means the model lost to the published one on 73% to 93% of the splits, but the formal test can't
rule out chance for that model alone. Every reasonable method lands between 0.83 and 0.85: with 768 women and six
measurements, the limit is the information in the data, not the choice of algorithm.

**How solid.** Firm that nothing tried beats the published model. Likely, not firm, that curves beat straight lines:
they win on 87% of splits and the gain is concentrated in older women, but the corrected test gives p = 0.16 (a naive
test would have said 0.006, which is why the corrected one is used; D15).

**Evidence.** Notebook 05 (sections 5.3 and 5.4) and notebook 06.

**Changed.** M1 and M2: straight-line logistic regression, chosen because it tied the random forest. M3: switched to
curves. M4: kept after the tuned contest; "curves beat straight lines" downgraded from firm to likely.

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
again. Milestone 6 tested the flat stretch: it is real, but the data can't say whether it is exactly flat (section 11).

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
unnecessary tests. What decides it is how bad a missed case is compared with an unnecessary blood test. The project
says 5 times as bad, which puts the cut-off at 17%.

**Answer.** If a missed case costs R times a false alarm, the cheapest cut-off is 1 / (1 + R). Per 100 women (35 with
diabetes):

| A missed case is... | Cut-off | Caught | Missed | False alarms |
|---|---|---|---|---|
| 2 times a false alarm | 0.33 | 28 | 7 | 17 |
| **5 times (the page's default)** | **0.17** | **33** | **2** | **30** |
| 10 times | 0.09 | 34 | 1 | 39 |

At 5 times the cost per 100 women is 40 false-alarm units. Testing everybody costs 65, the old 85% rule (0.28) 45, a
glucose-of-140 rule 95 and testing nobody 175. The old rule quietly assumed a ratio of about 2.6. The formula's cut-off
falls inside the 95% range of the cheapest cut-off found in the data at all three ratios. The cost curves are flat near
their lowest point, so the cut-off need not be exact, but the cost ratio matters. At a ratio of 10 the model barely beats
testing everybody (net benefit 0.305 against 0.285), so it mostly spares the quarter of women with very low risk.

**How solid.** Firm on the method. The ratio itself is a judgment, which is why the page shows three.

**Evidence.** Notebook 07 (sections 7.4 to 7.7) and the cut-off explorer on the project page.

**Changed.** M1: 0.24. M2: 0.23. M3: 0.28, because the curved model's risks are spread differently. M5: 0.17, from a
stated cost (D17).

## 8. Who does the model serve less well?

**In plain terms.** Older women, partly because half of them have diabetes, which makes them hard to tell apart.
Young women lost out under the old 85% rule, which caught fewer of their cases than the earlier model did. Choosing the cut-off from costs fixed that.

**Answer.** Two groups, for different reasons.

- **Women 45 and over.** Half of them have diabetes, so a shared cut-off flags most of them; that part is the base
  rate, not a fault. The model also separates them less well (AUC 0.79, against 0.85 under 30). The curved model
  improved this group: healthy women 45 and over who are cleared rose from 22% to 43% (B10, improved).
- **Women under 30, under the old rule.** The curved model gives them lower, more accurate risks, so more of
  their cases fell under the 0.28 cut-off: 60 of 84 caught (71%), against 80% for the straight-line model (B12). At the
  cost-based cut-off of 0.17, 85% of their cases are caught (62% at 2 times, 95% at 10 times). Separate cut-offs per age
  band did no better than one shared cut-off (42 against 40 per 100 women, D19).

**How solid.** Likely. Only 133 women are 45 and over (67 of them healthy); the gain for them is clear in direction
but uncertain in size (95% interval from slightly worse to much better). B12 is fixed. The risks are well
calibrated in every age band on average (slope 1.02, 0.91 and 0.77 from youngest to oldest), a little too extreme for
the oldest.

**Evidence.** Notebook 03 (section 3.5), notebook 05 (sections 5.2, 5.6 and 5.7) and notebook 07 (sections 7.8 and 7.9).

## 9. Does it beat a simple rule?

**In plain terms.** Yes. Compared with flagging everyone whose blood sugar is 140 or more, the model finds 21 more
women with diabetes for about the same number of false alarms.

**Answer.** Yes, modestly. Flagging every woman with a two-hour glucose of 140 mg/dL or more catches 135 cases with
62 false alarms. The model, set to raise about the same number of false alarms (64), catches 156: 21 more women
found.

**How solid.** Firm.

**Evidence.** Notebook 03, section 3.3 (shown there for the M1 model, which found 17 more).

## 10. Does it work outside this data?

**In plain terms.** Partly. On women from a US national survey, the model still ranks who is more likely to have diabetes
about as well as on the original women, but the risk percentages it gives are far too high, and the cut-off would flag
nearly half of all women.

**Answer.** The frozen model was run, unchanged, on 961 women aged 21 and over from NHANES 2013 to 2014 who had the same two-hour
glucose test. Only 37 (3.9%) have diabetes by the definition used (HbA1c of 6.5% or higher, or a doctor's diagnosis), against 35% of
the Pima women, so every number below is uncertain by a lot.

- **The groups differ.** Median age 48 against 29, BMI 27.8 against 32.3, 2 pregnancies against 3. A simple model separates
  them with AUC 0.92 (0.50 would mean identical).
- **The ranking holds.** AUC 0.88 (95% range 0.82 to 0.93), against 0.85 on unseen Pima women. The glucose reading alone scores
  0.90 (0.82 to 0.96), so the model adds nothing measurable beyond glucose here.
- **The risks do not.** Average predicted risk 25%, actual 3.9%; the top band (average 78%) is 19% in fact. The model's error
  score (Brier) is 0.124, against 0.037 for quoting everyone the average. Counting glucose of 200 or more as diabetes too raises
  the actual share only to 7.1%. A single shift fitted on held-out folds brings the error score to 0.032.
- **The cut-off.** At the 5 times cut-off the model catches 33 of 37 cases (89%; 94% on the Pima women) but flags 45% of
  women (62% on Pima), about 42 in 100 get an unnecessary test, and 7.6% of flagged women have diabetes.
- **Subgroups.** The ranking is fine wherever it can be measured (AUC 0.84 to 0.98 across age and ethnic groups), with 3 to 17
  cases per group, so this is weak evidence. The risks are too high in every group.

NHANES is a one-time snapshot, not five-year follow-up, and the Pima label was set by a glucose threshold, so differences in how
diabetes was defined cannot be separated from differences between the women. Nothing here tests the model on men, children or
present-day patients elsewhere.

**How solid.** Likely for the ranking (the range overlaps 0.85 but spans 0.82 to 0.93 because there are few cases). Firm that
the risk level does not transfer.

**Evidence.** Notebook 10 and Story 12 on the project page.

**Changed.** M1 to M7: unknown. M8: one outside group tested.

## 11. Why did the model give this woman this number?

**In plain terms.** Mostly her blood sugar, then her weight and age. A very different kind of model ranks the measurements
the same way, so that ranking belongs to the data. For women the model flags, most are too far above the cut-off for
a small, realistic change to clear them.

**Answer.**

- **What it relies on.** Shuffling one input among women the model hasn't seen costs this much AUC: glucose 0.17, BMI 0.07,
  age 0.04, family history 0.01, pregnancies 0.003, blood pressure 0.001. The last three have ranges that include zero.
- **Does another model agree?** Boosted trees (XGBoost), read with SHAP, rank the inputs in the same order (rank
  correlation 1.0), with glucose at 35% of total importance against 36% for the published model.
- **Is the explanation tool right?** SHAP reproduces the published model's exact contributions to within 0.002 on the
  log-odds scale (average gap under a millionth), so it can be trusted on models where no exact answer exists.
- **BMI's flat stretch.** From BMI 24 to 30 the odds multiply by 5.4 (range 3.1 to 10.9). From 30 to 42 they multiply by
  1.2 (range 0.7 to 2.3), where a steady continuation would give about 29. The levelling-off is real; whether it is
  exactly flat is unknown. A straight line for BMI scores 0.006 AUC lower, which could be chance (p = 0.15).
- **What would change her result?** At the 0.17 cut-off, 477 of 768 women are flagged, with a median risk of 49%. Lowering
  glucose by up to 20 mg/dL clears 28% of them, lowering BMI by up to 10% clears 13%, either one 29%. For flagged
  women whose risk is under 30%, 94% can be cleared. At 0.33 the figure is 35%, at 0.09 it is 24%.

These are explanations of the model, which describes patterns in 768 women. They do not show that changing a reading would
change anyone's real risk.

**How solid.** Firm for the importance ranking and the SHAP check. Likely for the BMI curve's benefit (p = 0.15).

**Evidence.** Notebook 08, and the "What would change this?" box on the project page.

## 12. How sure is the model about one woman?

**In plain terms.** An estimate is not exact. Refitting the model on resamples shows a typical range of 19 points around
each risk, and a separate rule can decline to call a woman when it can't be sure, at a cost.

**Answer.**

- **Size of the range.** Median 95% range 19 points (10% of women under 3.6, 10% over 34.3). By risk: under 10%, 6 points;
  10% to 30%, 17; 30% to 60%, 27; 60% and over, 25. The widest ranges belong to older women with high glucose (median
  age 60, glucose 152.5).
- **Does the cut-off matter?** The cut-off lies inside the range for 24.7% of women at 0.33, 21.9% at 0.17 and 22.4% at 0.09.
- **Is the spread believable?** It agrees with the spread across the 50 cross-validation fits (median ratio 1.08, correlation
  0.96).
- **Is the 95% honest?** In a simulation of 60 studies the range held the true risk 91.4% of the time (78.9% to 99.1%), so it
  is a little too narrow. The estimates are not biased (average error -0.002).
- **The conformal rule.** Checked on 300 random splits: at the 90% level it covered 89.9% of women without diabetes and
  90.8% of women with it; at the 80% level, 80.2% and 79.3%. The worst single splits fell well below (65% for women with
  diabetes at 80%).
- **The price.** At 90%: 41.6% confident no, 33.3% too uncertain, 25.1% confident yes; a confident "no" is wrong for 7.4%, a
  confident "yes" for 25.3%. At 80%: 54.9%, 6.9%, 37.7%; wrong for 12.6% and 32.9%. The page's cut-offs for the 90% level
  are 0.22 and 0.56, and for 80% they are 0.33 and 0.39.

The range and the call describe the model's estimate. Neither says what will happen to one woman.

**How solid.** Firm for the coverage checks. The conformal guarantee is about averages over many groups of women, not any one
group.

**Evidence.** Notebook 09, and the range on the risk meter and the "How sure can the model be?" box on the project page.

## 13. Is there hidden structure in the data?

**In plain terms.** No. The women form one continuous cloud rather than separate types, and the odd records are real but
make no difference to the model. One useful thing came out of it: a way of telling when a woman's numbers are unlike the
women the model learned from.

**Answer.** All of this was done without using the diabetes label, which was only added afterwards.

- **The map.** The first two directions of greatest spread keep 54% of the variation (the first alone 33%), so six inputs
  are six separate things. UMAP keeps each woman's neighbours better than PCA (0.94 against 0.83) and draws filaments, but no
  islands. Colour sorts left to right on the PCA map with a broad overlap in the middle.
- **Groups.** The best split is two groups (separation 0.23, against 0.19 for data with no groups in it). Groups reappear
  when the women are resampled at most 76% of the time (0.8 would be reliable), falling to about 50% by six groups. A second
  method prefers four groups, which the stability check does not support (0.70).
- **What the groups are.** Two groups: 419 younger women (median age 25, 18% with diabetes) and 349 older ones (age 41, 55%).
  Three groups add younger women with a high BMI (median 37.8, 49% with diabetes). They restate age, weight and glucose.
- **Do they help the model?** No. AUC with group membership 0.852, with distances 0.851, with an oddness score 0.852, against
  0.853. None of the ranges excludes zero.
- **Odd records.** Three methods (isolation forest, local outlier factor, Mahalanobis distance) agree with each other
  (0.81 to 0.95). With all eight columns the skinfold of 99 is number 3 of 768, the insulin of 846 number 4, the BMI of
  67.1 number 8, 17 pregnancies number 26 and blood pressure 24 number 29. On the six model inputs these rank lower (13 to
  165), because insulin and skinfold are not among them. Rule-of-thumb breaks are few: 3 insulin readings over 600, 2
  skinfolds over 60, 4 blood pressures under 40.
- **Harm.** In two of three methods the unusual 5% (39 women) were predicted a little worse (AUC 0.74 against 0.86); the
  third saw no difference, and across all women the link between oddness and error is near zero. Removing the 15 most unusual
  from training changes AUC by -0.0001 (-0.003 to +0.002).
- **The page notice.** Distance from the typical Pima woman is computed on the page. A quarter of NHANES women (26%) are
  beyond the line that 95 in 100 Pima women stay inside, 11% beyond the 99th. Unusual Pima women have wider ranges (28 points
  against 19). But the notice does not fix the risk level: unusual and typical NHANES women are both predicted at about 25%
  while 7.9% and 2.4% have diabetes.

**How solid.** Firm that no useful groups exist and that cleaning odd records does not help. The statement that unusual
women are predicted worse is weak (39 women, two of three methods).

**Evidence.** Notebook 11 and Story 13 on the project page.

**Changed.** New in M9.

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
| Tuned support vector machine, random forest, XGBoost, gradient boosting, neural network | 0.832 to 0.840, all below the curves | Notebook 06, 6.2 |
| Tuned k-nearest neighbours | 0.827, clearly worse | Notebook 06, 6.3 |
| Stacked ensemble of four model types | 0.849, no better than the curves alone | Notebook 06, 6.6 |
| Tuning the curves model's own settings | +0.002, within noise | Notebook 06, 6.3 and 6.5 |
| Class weights or SMOTE instead of moving the cut-off | Same ranking (AUC 0.851 to 0.853), same false alarms to catch 85% (152 to 154), risks too high (43 to 44% against 35%) | Notebook 07, 7.2 and 7.3 |
| A separate cut-off for each age band | No better than one shared cut-off (42 against 40 per 100 women at 5 times) | Notebook 07, 7.9 |
| Rescaling the risks for a new population | Fixes the average (error score 0.124 to 0.032) but needs outside outcomes, and adds little over quoting the average (0.037) | Notebook 10, 10.5 |
| Cluster membership, distance to cluster centres or an oddness score as extra inputs | AUC 0.851 to 0.852 against 0.853 | Notebook 11, 11.4 |
| Dropping the 15 most unusual women from training | AUC change -0.0001 | Notebook 11, 11.7 |
| Straight line for BMI instead of a curve | 0.006 AUC lower, could be chance (p = 0.15), so the curve stays | Notebook 08, 8.4 |
