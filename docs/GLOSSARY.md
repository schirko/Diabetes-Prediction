# Glossary

Every technical word used in this project, in plain language, with what it means here.
Start with the [overview](OVERVIEW.md) if you haven't read it.

## Judging a model

**Accuracy.** The share of women the model labels correctly. It sounds like the obvious measure, but it can mislead:
here, answering "no diabetes" for everyone is 65% accurate and catches nobody. This project reports accuracy only
alongside the numbers below.

**AUC.** How well the model ranks people. Take one woman who developed diabetes and one who didn't: the AUC is the
chance the model gives the first one the higher risk. 0.5 is a coin flip; 1.0 is perfect. *Here: 0.85.* (It stands
for "area under the curve", which refers to a chart nobody needs to look at to use the number.)

**Sensitivity (cases caught).** Of the women who really have diabetes, the share the model flags. *Here: 85% at the
screening cut-off, 30 of every 35.*

**Specificity (healthy women cleared).** Of the women who don't have diabetes, the share the model correctly leaves
alone. *Here: 69% at the screening cut-off, 45 of every 65.*

**False alarm (false positive).** A healthy woman the model flags. The cost is an unnecessary follow-up test.

**Missed case (false negative).** A woman with diabetes the model doesn't flag. The cost is a later diagnosis.

**Precision (positive predictive value).** Of the women the model flags, the share who really have diabetes.
*Here: 60% at the screening cut-off.*

**Calibration.** Whether the risks mean what they say. A well-calibrated model that says "30%" is right about 30 times
in 100 for women like that. The **calibration slope** summarises it: 1.0 is ideal. *Here: 0.94.*

**Brier score.** The average squared gap between the predicted risk and what happened (0 or 1). Lower is better.
Used to compare models' probabilities.

## Making decisions

**Risk (predicted probability).** The model's estimate, from 0% to 100%, that a woman like this is diagnosed within
five years.

**Cut-off (threshold).** The risk at which someone is flagged. A low cut-off catches more cases and raises more false
alarms; a high cut-off does the reverse. Choosing it is a judgment about costs, not something the data decides.
*Here: 0.17 (17%), chosen from a stated cost (see cost ratio). The earlier rule, 0.28, caught 85% of cases.*

**Cost ratio.** How many times worse a missed case is than a false alarm. If a missed case is 5 times as bad, the
cheapest cut-off is 1 / (1 + 5), about 0.17. *Here: the page shows 2 times (0.33), 5 times (0.17, the default) and 10
times (0.09).*

**Cost per 100 women.** Missed cases times the cost ratio, plus false alarms, per 100 women screened. Lower is better.
It is measured in false-alarm units.

**Net benefit and decision curve.** Cases found minus false alarms weighted by the cost ratio, drawn across every
cut-off. A model is worth using where its line sits above both "test nobody" (zero) and "test everybody".

**Class weights.** Telling the model to count each woman in the smaller group more heavily when it learns.

**SMOTE.** Synthetic minority over-sampling: inventing extra cases for the smaller group by blending real ones. Done
inside each training set only, so test women never leak in. *Here: neither it nor class weights helped.*

**Screening.** Testing people who have no symptoms, to find who needs a closer look. A screening tool flags people
for a proper test; it does not diagnose.

**Base rate (prevalence).** How common the condition is in a group. *Here: 35% overall, 21% under 30, 50% at 45 and
over.* When the base rate is high, even a good model flags many people in that group.

**Glucose rule.** The simple alternative the model is compared against: flag everyone whose two-hour glucose is 140
or more, a standard line for raised blood sugar.

## Testing fairly

**Training and testing.** A model learns from one set of women (training) and is scored on women it has never seen
(testing). Scoring it on the women it learned from would flatter it.

**Train/test split.** Dividing the data once into a training part and a testing part, usually 80% and 20%. The
original college project did this once, which is why its score depended on luck.

**Cross-validation.** Splitting the data into five parts, training on four and testing on the fifth, and rotating
until every woman has been tested once. *Here it is repeated 10 times with different random splits: 50 tests in all.*

**Held-out prediction.** A risk for one woman made by a model that did not see her during training. All results in
this project use these.

**Data leakage.** When information from the test women slips into training, making the score look better than it
really is. The original project filled missing values using averages from all 768 women, test women included.

**Pipeline.** A bundle of preparation steps and the model, run together so that every step learns only from the
training women. It is how this project prevents leakage.

**Overfitting.** When a model learns the quirks of its training data instead of the real pattern, so it does worse
on new people. Complex models on small datasets are prone to it.

**Fresh splits.** Re-running a comparison on a new random set of splits. If the winner is still the winner, the
result is not an accident of one set of splits.

**Tuning (hyperparameters).** Many models have settings that must be chosen before they learn, such as how many
neighbours to consult or how deep a tree may grow. These are called hyperparameters. Tuning means trying many
settings and keeping the one that scores best. *Here it is done by Optuna, trying 30 combinations per model.*

**Optuna.** A tool that tunes settings efficiently: instead of trying combinations at random, it learns from earlier
tries which areas look promising.

**Nested cross-validation.** Cross-validation inside cross-validation. The outer splits score the model; inside each
outer training set, a second split is used only to tune its settings. The outer test women never influence the
settings, so tuning can't flatter the score. *Used for the Milestone 4 contest.*

**p-value.** If two models were really equally good, the chance of seeing a difference at least as large as the one
measured. Below 0.05 is the usual bar for "probably not chance". A large p-value doesn't prove the models are equal;
it means the data can't tell them apart.

**Corrected resampled t-test.** A test of whether one model beats another across the same cross-validation splits.
An ordinary t-test treats each split as an independent experiment, but the splits share most of their women, so it
is overconfident. The correction (Nadeau and Bengio, 2003) allows for that. *Here an ordinary test gave p = 0.006
where the corrected one gives 0.16.*

**95% interval.** The range the true value probably lies in, given how much the result would vary with different
data. *Here the AUC is 0.85, with a 95% interval of 0.83 to 0.88.*

**Standard deviation ("wobble").** How much a score varies from one split to the next. Used to tell a real difference
between models from noise.

**Percentile.** The value below which a given share of women fall. The 10th percentile of glucose (86) is lower than
90% of women's readings.

## The models

**Model.** A formula, learned from the data, that turns measurements into a risk.

**Input (feature).** One of the measurements the model uses. *The current model uses six.*

**Logistic regression.** A classic model that adds up each input's effect and converts the total into a risk between
0% and 100%. Easy to explain, because each input's contribution can be shown on its own.

**Straight line vs curve.** A straight-line model assumes each extra unit of an input always adds the same amount of
risk. A curve (here, a **spline**) can rise, level off or fall. *The current model lets glucose, BMI and age curve.*

**Spline.** A smooth curve built from a few joined pieces, so a model can follow a shape like "rises until 50, then
falls".

**Interaction.** When the effect of one input depends on another, such as glucose mattering more at some ages than
others. *Tested in Milestone 3; it added nothing.*

**Decision tree.** A model that asks a series of yes/no questions ("is glucose above 128?") to reach a risk.

**Random forest.** Hundreds of decision trees, each trained a little differently, averaged together.

**Gradient boosting.** Many small decision trees built one after another, each correcting the last one's mistakes.

**XGBoost.** A popular, fast version of gradient boosting, often a winner in data-science competitions.

**k-nearest neighbours.** Predicts a woman's risk from the women most similar to her: if 20 of her 50 closest
neighbours developed diabetes, her risk is 40%.

**Support vector machine (SVM).** A model that finds the boundary separating the two groups with the widest possible
margin, and can bend that boundary.

**Neural network.** Layers of simple connected units that together can learn complicated patterns. Powerful with
lots of data; on 768 women it has little to work with. *Here: one hidden layer, tuned for size.*

**Stacked ensemble.** Several different models trained side by side, with a final model that learns how much to
trust each one's prediction. *Tested in Milestone 4: it did no better than the best single model.*

**Explainable Boosting Machine (EBM).** A boosting model that learns one curve per input and keeps them separate, so it
stays explainable.

**Odds.** Another way to express risk: a 50% risk is odds of 1 to 1; a 20% risk is odds of 1 to 4. "Multiplies the
odds by 2" roughly doubles a small risk.

## Explaining a model

**Permutation importance.** How much worse the model gets on women it hasn't seen when one input is shuffled among them,
which breaks its link to the outcome. A bigger fall means the model relies more on that input. *Here: glucose costs
0.17 of AUC, BMI 0.07, age 0.04.*

**Partial dependence.** The model's average predicted risk as one input moves across its range, with every woman keeping
her other values.

**ICE curve (individual conditional expectation).** The same idea for one woman: how her own predicted risk would move
if only that one input changed. Plotting many shows whether everyone responds alike.

**SHAP.** A method that shares credit for a woman's prediction among the inputs by averaging each input's effect over
every order the inputs could be added. *Here it is checked against the model's exact contributions and agrees to within
0.002 on the log-odds scale.*

**Contribution.** How much one input adds to a woman's log-odds compared with a typical woman. The page shows it as an odds
multiplier ("x2.0" doubles her odds).

**Log-odds.** Odds on a log scale, the scale a logistic regression adds effects on. Adding 0.69 doubles the odds.

**Bootstrap.** Refitting the model on many resamples of the same women (drawn with replacement) to see how much a result
would vary. *Here: 300 refits to test whether BMI's flat stretch is real.*

**Counterfactual.** The smallest change to the inputs that would change the model's answer, for example how much lower
a woman's glucose would need to be to fall below the cut-off. It explains the model and does not say that making the
change would lower her real risk.

## Uncertainty

**Bootstrap range (95% range).** The middle 95% of a woman's risks across 200 refits of the model, each on a resample of the
768 women. It shows how far her estimate could move had a different group been studied. It describes the model's
estimate, not whether she will get diabetes. Here it holds the true risk about 91% of the time in a simulation.

**Coverage.** The share of the time a range or a rule contains the truth. A range labelled 95% should have coverage of 95%.

**Conformal prediction.** A way of turning a risk into a call (confident no, confident yes, or too uncertain) with a
guarantee about how often the call is right, which holds on women the rule has never seen.

**Calibration set.** Women set aside, with known outcomes, to measure how far off the model's risks are, so that the
conformal rule's guarantee holds on new women.

**Too uncertain to call.** The outcome when both "no" and "yes" fit at the chosen level. It is the price of the guarantee.

## Testing on new data

**External validation.** Testing a finished model, unchanged, on a different group of people from the one it was built on.
It answers a question no amount of re-splitting the original data can: does it travel?

**NHANES.** The US National Health and Nutrition Examination Survey, run by the CDC. A representative sample of people who
are interviewed and examined. *Here: 961 women aged 21 and over from the 2013 to 2014 round who had the two-hour glucose test.*

**HbA1c.** A blood test that reflects average blood sugar over about three months. 6.5% or higher counts as diabetes. *Used here
to define who has diabetes in NHANES, because it is a different test from the glucose drink the model uses as an input.*

**Dataset shift.** When the new group differs from the group the model learned from, in who is in it, how often the condition
occurs, or how it was measured. *Here: the NHANES women are older and lighter, and 4% have diabetes against 35%.*

**Recalibration.** Adjusting a model's risks so they match what happens in a new group, without changing how it ranks women. *Here:
one shift applied to every risk, fitted on part of the NHANES women and tested on the rest.*

## Finding structure

**Unsupervised.** Looking for patterns without using the answer (here, without using who has diabetes). The opposite of the
prediction work in the other notebooks, which learns from the answer.

**PCA (principal component analysis).** A way of flattening many measurements onto a few directions, chosen to keep as much of
the spread between women as possible. *Here: the first two directions keep 54% of the spread.*

**UMAP.** Another way of drawing a map of the data, which instead tries to keep each woman's nearest neighbours near her. It can
draw shapes that look like groups even when none exist, so a pattern on a UMAP is a reason to check, not a finding.

**Cluster / clustering.** Splitting people into groups of similar people. *Here: k-means, which gathers women around group
centres.* It always returns groups, whether or not any exist.

**Silhouette (separation score).** How well clustered the groups are: 1 is tight and far apart, near 0 is overlapping. It is
only meaningful compared with data that has no groups in it.

**Stability.** Whether clustering two resamples of the same women gives the same groups (1 is identical). Groups that don't
reappear are not groups.

**Anomaly detection (outlier detection).** Scoring how unusual each record is. *Used here as a data-quality check: three methods
(isolation forest, local outlier factor and Mahalanobis distance) are compared.*

**Mahalanobis distance.** How far a woman is from the typical woman, counting measurements that move together (such as age and
pregnancies) as one rather than several. *The page's "unusual measurements" notice uses it.*

## Missing data

**Missing value.** A measurement that was never taken. *In this data, a missing value is recorded as zero, which is
impossible for glucose, blood pressure, skin thickness, insulin and BMI.*

**Filling in (imputation).** Replacing a missing value with an estimate, such as the middle value of that measurement.
*Here the model fills the few remaining gaps with the median.*

**Median.** The middle value: half the women are above it and half below.

**Multiple imputation.** Filling the gaps many different plausible ways, to see how much the answer depends on the
guesses. *Here, 53 women landed on different sides of the cut-off depending on how their gaps were filled.*

## The measurements

**Glucose (two-hour).** Blood sugar measured two hours after drinking a standard sugary drink (an oral glucose
tolerance test), in mg/dL. Higher values mean the body is clearing sugar slowly.

**Insulin.** Insulin level from the same test, in mu U/mL. Missing for 49% of women here.

**BMI (body mass index).** Weight in kilograms divided by height in metres squared. 25 to 30 counts as overweight,
30 and over as obese.

**Skin thickness.** Triceps skinfold: a pinch of skin at the back of the upper arm, in mm, a rough measure of body fat.
Missing for 30% of women here.

**Family-history score (Diabetes Pedigree Function).** A score built by the original researchers from how many
relatives had diabetes and how closely related they are. Higher means more family history.

**Insulin resistance.** When the body needs more insulin than usual to control blood sugar; a step towards type 2
diabetes. *Tested in Milestone 3 as glucose times insulin; it added nothing.*

## The project

**Notebook.** A document mixing code, its results and explanations. Each step here follows the same pattern: what
we're looking for, the code, the result, how to read it, what it means, and what's next.

**Milestone.** One planned stage of the project with its own question and a test for when it's finished. See the
[roadmap](MILESTONES.md).

**E and B numbers.** Items in the [enhancements and bugs list](ENHANCEMENTS_AND_BUGS.md): E for an improvement, B for a
problem found. **D numbers** are entries in the [decisions log](DECISIONS.md).
